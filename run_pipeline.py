"""End-to-end pipeline runner: preprocess → train → evaluate → BERT baseline."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess import preprocess
from train import train_all
from evaluate import evaluate_model


def main(input_csv=None, run_bert=True):
    if input_csv is None:
        input_csv = os.path.join(ROOT, "data", "dummy", "reviews.csv")

    if not os.path.exists(input_csv):
        print("Generating dummy data first...")
        from generate_dummy_data import generate
        generate()

    # Step 1: Preprocess
    print("\n" + "=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test, tfidf, svd, df = preprocess(input_csv)

    # Step 2: Train
    print("\n" + "=" * 60)
    print("STEP 2: Training (LR + SVM)")
    print("=" * 60)
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    models, cv_results = train_all(X_train, y_train, X_full, y_full)

    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation")
    print("=" * 60)
    all_metrics = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name.upper())
        all_metrics[name] = metrics

    # Step 4: BERT Zero-Shot Baseline (GPU)
    if run_bert:
        print("\n" + "=" * 60)
        print("STEP 4: BERT Zero-Shot Baseline (GPU)")
        print("=" * 60)
        try:
            from bert_baseline import run_baseline
            bert_metrics = run_baseline(input_csv)
            all_metrics["bert_zeroshot"] = bert_metrics
        except Exception as e:
            print(f"BERT baseline failed: {e}")
            print("Skipping BERT baseline.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {os.path.join(ROOT, 'results')}")

    return all_metrics


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    bert = "--no-bert" not in sys.argv
    main(csv_path, run_bert=bert)
