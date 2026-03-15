"""BERT zero-shot sentiment baseline using GPU."""

import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CANDIDATE_LABELS = ["正面評價", "中性評價", "負面評價"]
# Maps BERT label index → our label encoding (0=neg, 1=neutral, 2=pos)
LABEL_MAP = {"正面評價": 2, "中性評價": 1, "負面評價": 0}


def load_classifier(model_name="joeddav/xlm-roberta-large-xnli", device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    return clf


def predict_batch(clf, texts, batch_size=32):
    """Classify a list of texts, return predicted labels and probabilities."""
    preds = []
    probs_all = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results = clf(batch, candidate_labels=CANDIDATE_LABELS, batch_size=batch_size)
        if not isinstance(results, list):
            results = [results]
        for r in results:
            # r['labels'] is sorted by score descending
            top_label = r["labels"][0]
            pred = LABEL_MAP[top_label]
            preds.append(pred)
            # Build probability array in order [neg, neutral, pos]
            prob = [0.0, 0.0, 0.0]
            for label, score in zip(r["labels"], r["scores"]):
                prob[LABEL_MAP[label]] = score
            probs_all.append(prob)

        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(preds), np.array(probs_all)


def run_baseline(input_csv=None, model_name="joeddav/xlm-roberta-large-xnli",
                  max_samples=1000, batch_size=8):
    """Run BERT zero-shot baseline on the dataset.

    For large datasets, stratified-sample max_samples reviews to avoid OOM.
    """
    from evaluate import compute_metrics, save_all_plots, save_metrics

    if input_csv is None:
        input_csv = os.path.join(ROOT, "data", "dummy", "reviews.csv")

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["review_text"]).reset_index(drop=True)

    # Stratified subsample for large datasets
    if len(df) > max_samples:
        print(f"  Subsampling {max_samples} from {len(df)} reviews (stratified)...")
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(
            df, test_size=max_samples, stratify=df["label"], random_state=42,
        )
        df = df.reset_index(drop=True)

    # Truncate long reviews to avoid tokenizer issues
    df["review_text"] = df["review_text"].str[:512]

    texts = df["review_text"].tolist()
    y_true = df["label"].values

    print(f"Running BERT zero-shot baseline on {len(texts)} reviews...")
    print(f"  Model: {model_name}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    clf = load_classifier(model_name)
    y_pred, y_prob = predict_batch(clf, texts, batch_size=batch_size)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    save_all_plots(y_true, y_pred, y_prob, "BERT_ZeroShot")
    save_metrics(metrics, "BERT_ZeroShot")

    print(f"\nBERT Zero-Shot Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:  {metrics['f1_macro']:.4f}")
    if "auroc_macro" in metrics:
        print(f"  AUROC:     {metrics['auroc_macro']:.4f}")

    return metrics


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_baseline(csv_path)
