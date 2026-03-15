"""Train SVM on BERT embeddings and compare with TF-IDF SVM + BERT zero-shot.

Usage:
    python src/train_bert_svm.py
"""

import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

MODELS_DIR = os.path.join(ROOT, "results", "models")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")


def train_bert_svm():
    """Train SVM + LR on BERT embeddings, compare with TF-IDF models."""
    from bert_features import preprocess_bert, load_processed_bert
    from train import cross_validate_model, train_and_save
    from evaluate import compute_metrics, save_all_plots, save_metrics

    # Step 1: Extract BERT embeddings (or load if already saved)
    bert_train_path = os.path.join(PROCESSED_DIR, "X_train_bert.npy")
    if os.path.exists(bert_train_path):
        print("Loading existing BERT embeddings...")
        X_train, X_test, y_train, y_test = load_processed_bert()
    else:
        print("Extracting BERT embeddings (this uses GPU)...")
        csv_path = os.path.join(PROCESSED_DIR, "reviews_clean.csv")
        X_train, X_test, y_train, y_test = preprocess_bert(csv_path)

    print(f"\nBERT embeddings: X_train={X_train.shape}, X_test={X_test.shape}")

    # Step 2: Train and evaluate SVM on BERT embeddings
    print("\n" + "=" * 60)
    print("Training SVM on BERT embeddings")
    print("=" * 60)

    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])

    # Cross-validation
    for model_type in ["svm", "lr"]:
        print(f"\n--- {model_type.upper()} on BERT embeddings ---")
        cv_results = cross_validate_model(X_full, y_full, model_type=model_type)
        for metric, vals in cv_results.items():
            print(f"  {metric}: test={vals['test_mean']:.4f}±{vals['test_std']:.4f}")

    # Train final SVM model on BERT embeddings
    from sklearn.svm import SVC
    svm_bert = SVC(kernel="rbf", probability=True, random_state=42)
    svm_bert.fit(X_train, y_train)

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "svm_bert_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(svm_bert, f)
    print(f"\nSaved BERT-SVM model: {model_path}")

    # Evaluate on test set
    y_pred = svm_bert.predict(X_test)
    y_prob = svm_bert.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    save_all_plots(y_test, y_pred, y_prob, "BERT_SVM")
    save_metrics(metrics, "BERT_SVM")

    # Step 3: Also train LR on BERT embeddings
    from sklearn.linear_model import LogisticRegression
    lr_bert = LogisticRegression(max_iter=1000, random_state=42)
    lr_bert.fit(X_train, y_train)

    lr_model_path = os.path.join(MODELS_DIR, "lr_bert_model.pkl")
    with open(lr_model_path, "wb") as f:
        pickle.dump(lr_bert, f)

    y_pred_lr = lr_bert.predict(X_test)
    y_prob_lr = lr_bert.predict_proba(X_test)
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)
    save_all_plots(y_test, y_pred_lr, y_prob_lr, "BERT_LR")
    save_metrics(metrics_lr, "BERT_LR")

    # Step 4: Print comparison table
    # Load TF-IDF model results for comparison
    from preprocess import load_processed
    _, X_test_tfidf, _, y_test_tfidf, _, _ = load_processed()

    tfidf_svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(tfidf_svm_path, "rb") as f:
        svm_tfidf = pickle.load(f)
    y_pred_tfidf = svm_tfidf.predict(X_test_tfidf)
    y_prob_tfidf = svm_tfidf.predict_proba(X_test_tfidf)
    metrics_tfidf = compute_metrics(y_test_tfidf, y_pred_tfidf, y_prob_tfidf)

    print("\n" + "=" * 60)
    print("  COMPARISON: TF-IDF SVM vs BERT-SVM vs BERT-LR")
    print("=" * 60)
    header = f"  {'Model':<20} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10}"
    print(header)
    print(f"  {'-' * 62}")

    for name, m in [
        ("TF-IDF + SVM", metrics_tfidf),
        ("BERT + SVM", metrics),
        ("BERT + LR", metrics_lr),
    ]:
        print(
            f"  {name:<20} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
            f"{m['precision_macro']:>10.4f} {m['recall_macro']:>10.4f}"
        )

    print("=" * 60)

    return metrics, metrics_lr


if __name__ == "__main__":
    train_bert_svm()
