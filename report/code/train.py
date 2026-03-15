"""Model training: Logistic Regression + SVM with cross-validation."""

import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "results", "models")


def make_model(model_type="lr", **kwargs):
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
    elif model_type == "svm":
        return SVC(kernel="rbf", probability=True, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def cross_validate_model(X, y, model_type="lr", n_splits=5, **kwargs):
    """Run stratified k-fold CV and return results dict."""
    model = make_model(model_type, **kwargs)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, return_train_score=True
    )
    summary = {}
    for metric in scoring:
        test_key = f"test_{metric}"
        train_key = f"train_{metric}"
        summary[metric] = {
            "test_mean": results[test_key].mean(),
            "test_std": results[test_key].std(),
            "train_mean": results[train_key].mean(),
            "train_std": results[train_key].std(),
        }
    return summary


def train_and_save(X_train, y_train, model_type="lr", save=True, **kwargs):
    """Train a model on full training set and optionally save it."""
    model = make_model(model_type, **kwargs)
    model.fit(X_train, y_train)

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {model_type} model to {path}")

    return model


def train_all(X_train, y_train, X_full=None, y_full=None):
    """Train LR + SVM, run CV, return models and CV results."""
    if X_full is None:
        X_full = X_train
    if y_full is None:
        y_full = y_train

    results = {}
    models = {}

    for mt in ["lr", "svm"]:
        print(f"\n{'='*50}")
        print(f"Training {mt.upper()}")
        print(f"{'='*50}")

        cv_results = cross_validate_model(X_full, y_full, model_type=mt)
        results[mt] = cv_results
        for metric, vals in cv_results.items():
            print(f"  {metric}: test={vals['test_mean']:.4f}±{vals['test_std']:.4f}")

        model = train_and_save(X_train, y_train, model_type=mt)
        models[mt] = model

    return models, results


if __name__ == "__main__":
    from preprocess import load_processed

    X_train, X_test, y_train, y_test, _, _ = load_processed()
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    train_all(X_train, y_train, X_full, y_full)
