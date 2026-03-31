"""Train a PyTorch MLP on BERT embeddings — a true deep-learning classifier.

Supports both 3-class and binary (3-star-as-negative) modes.
Includes 5-fold Stratified CV for fair comparison with LR/SVM.

Usage:
    python src/train_mlp.py
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "results", "models")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentMLP(nn.Module):
    """3-layer MLP: 1024 → 256 → 64 → n_classes."""

    def __init__(self, input_dim=1024, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_one_fold(X_train, y_train, X_val, y_val, n_classes=3,
                   epochs=50, lr=1e-3, batch_size=128):
    """Train MLP on one fold, return val metrics and trained model."""
    # Convert to tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SentimentMLP(input_dim=X_train.shape[1], n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=n_classes)
    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        logits = model(X_v.to(DEVICE))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    y_val_np = y_val
    metrics = {
        "accuracy": accuracy_score(y_val_np, preds),
        "f1_macro": f1_score(y_val_np, preds, average="macro"),
        "precision_macro": precision_score(y_val_np, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_val_np, preds, average="macro", zero_division=0),
    }

    return model, metrics, probs, preds


def cross_validate_mlp(X, y, n_classes=3, n_splits=5, **kwargs):
    """5-fold Stratified CV for MLP."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        _, metrics, _, _ = train_one_fold(X_tr, y_tr, X_val, y_val,
                                          n_classes=n_classes, **kwargs)
        all_metrics.append(metrics)
        print(f"  Fold {fold+1}: acc={metrics['accuracy']:.4f}, "
              f"f1={metrics['f1_macro']:.4f}")

    summary = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        summary[key] = {
            "test_mean": np.mean(vals),
            "test_std": np.std(vals),
        }
    return summary


def train_final_model(X_train, y_train, X_test, y_test, n_classes=3,
                      model_name="mlp_bert", **kwargs):
    """Train on full training set, evaluate on test set, save model + plots."""
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.preprocessing import label_binarize
    from evaluate import save_all_plots, save_metrics

    model, _, probs, preds = train_one_fold(
        X_train, y_train, X_test, y_test,
        n_classes=n_classes, epochs=80, **kwargs
    )

    # Compute metrics manually to handle binary/multiclass
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_macro": float(f1_score(y_test, preds, average="macro")),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
    }
    label_names = ["negative", "neutral", "positive"] if n_classes == 3 else ["dissatisfied", "satisfied"]
    metrics["classification_report"] = classification_report(
        y_test, preds, target_names=label_names, output_dict=True
    )
    if n_classes == 3:
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        metrics["auroc_macro"] = float(
            roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
        )

    # Save plots (only for 3-class where evaluate.py expects 3 labels)
    if n_classes == 3:
        tag = model_name.upper()
        save_all_plots(y_test, preds, probs, tag)
    save_metrics(metrics, model_name.upper())

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    return model, metrics


def main():
    # Load BERT embeddings
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train_bert.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test_bert.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    print(f"Device: {DEVICE}")
    print(f"X shape: {X_full.shape}, classes: {np.unique(y_full)}")

    # ── 3-class ──
    print("\n" + "=" * 60)
    print("  MLP 5-fold CV — 3-class")
    print("=" * 60)
    cv_3class = cross_validate_mlp(X_full, y_full, n_classes=3)
    for metric, vals in cv_3class.items():
        print(f"  {metric}: {vals['test_mean']:.4f} ± {vals['test_std']:.4f}")

    # Train final 3-class model
    print("\nTraining final 3-class MLP...")
    _, metrics_3c = train_final_model(X_train, y_train, X_test, y_test,
                                       n_classes=3, model_name="mlp_bert")

    # ── Binary (3-star as negative) ──
    print("\n" + "=" * 60)
    print("  MLP 5-fold CV — Binary (3-star → negative)")
    print("=" * 60)
    # Remap: 0=neg, 1=neutral→neg, 2=pos→1
    y_full_bin = np.where(y_full == 2, 1, 0)
    y_train_bin = np.where(y_train == 2, 1, 0)
    y_test_bin = np.where(y_test == 2, 1, 0)
    print(f"  Binary distribution: {np.bincount(y_full_bin)}")

    cv_binary = cross_validate_mlp(X_full, y_full_bin, n_classes=2)
    for metric, vals in cv_binary.items():
        print(f"  {metric}: {vals['test_mean']:.4f} ± {vals['test_std']:.4f}")

    # Train final binary model
    print("\nTraining final binary MLP...")
    _, metrics_bin = train_final_model(X_train, y_train_bin, X_test, y_test_bin,
                                        n_classes=2, model_name="mlp_bert_binary")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  3-class MLP:  acc={metrics_3c['accuracy']:.4f}, "
          f"f1={metrics_3c['f1_macro']:.4f}")
    print(f"  Binary MLP:   acc={metrics_bin['accuracy']:.4f}, "
          f"f1={metrics_bin['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
