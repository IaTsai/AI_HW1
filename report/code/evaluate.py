"""Evaluation: metrics computation + save_all_plots() for one-click figure generation."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
        ),
    }
    if y_prob is not None:
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        metrics["auroc_macro"] = float(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
        )
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc_curve(y_true, y_prob, model_name, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, name in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pr_curve(y_true, y_prob, model_name, output_dir=None):
    output_dir = output_dir or FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))

    for i, name in enumerate(LABEL_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"pr_curve_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def save_all_plots(y_true, y_pred, y_prob, model_name, output_dir=None):
    """Generate all evaluation plots in one call."""
    output_dir = output_dir or FIGURES_DIR
    print(f"\nGenerating plots for {model_name}...")
    plot_confusion_matrix(y_true, y_pred, model_name, output_dir)
    if y_prob is not None:
        plot_roc_curve(y_true, y_prob, model_name, output_dir)
        plot_pr_curve(y_true, y_prob, model_name, output_dir)


def save_metrics(metrics, model_name, output_dir=None):
    output_dir = output_dir or os.path.join(ROOT, "results")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}")


def evaluate_model(model, X_test, y_test, model_name):
    """Full evaluation: compute metrics, save plots and metrics JSON."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_prob)
    save_all_plots(y_test, y_pred, y_prob, model_name)
    save_metrics(metrics, model_name)

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    if "auroc_macro" in metrics:
        print(f"  AUROC:     {metrics['auroc_macro']:.4f}")

    return metrics


if __name__ == "__main__":
    import pickle

    from preprocess import load_processed

    X_train, X_test, y_train, y_test, _, _ = load_processed()

    models_dir = os.path.join(ROOT, "results", "models")
    for mt in ["lr", "svm"]:
        path = os.path.join(models_dir, f"{mt}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            evaluate_model(model, X_test, y_test, mt.upper())
