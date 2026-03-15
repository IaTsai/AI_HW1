"""Experiment wrappers for all ablation studies."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, learning_curve

from evaluate import LABEL_NAMES, compute_metrics, save_all_plots
from train import make_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(ROOT, "results", "figures")
TABLES_DIR = os.path.join(ROOT, "results", "tables")


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


# ─── Experiment 1: Learning Curve ────────────────────────────────────────────

def exp_learning_curve(X, y, model_type="lr", train_sizes=None):
    """Plot learning curves at different training set fractions."""
    _ensure_dirs()
    if train_sizes is None:
        train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

    model = make_model(model_type)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv,
        scoring="f1_macro", n_jobs=-1, random_state=42,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes_abs, train_scores.mean(axis=1), "o-", label="Train F1")
    ax.fill_between(
        train_sizes_abs,
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.15,
    )
    ax.plot(train_sizes_abs, test_scores.mean(axis=1), "o-", label="Val F1")
    ax.fill_between(
        train_sizes_abs,
        test_scores.mean(axis=1) - test_scores.std(axis=1),
        test_scores.mean(axis=1) + test_scores.std(axis=1),
        alpha=0.15,
    )
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1 Macro", fontsize=12)
    ax.set_title(f"Learning Curve — {model_type.upper()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"learning_curve_{model_type}.png"), dpi=150)
    plt.close(fig)

    # Save table
    df = pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_f1_mean": train_scores.mean(axis=1),
        "train_f1_std": train_scores.std(axis=1),
        "val_f1_mean": test_scores.mean(axis=1),
        "val_f1_std": test_scores.std(axis=1),
    })
    df.to_latex(os.path.join(TABLES_DIR, f"learning_curve_{model_type}.tex"),
                index=False, float_format="%.4f", caption=f"Learning Curve — {model_type.upper()}")
    print(f"[Exp1] Learning curve done for {model_type.upper()}")
    return df


# ─── Experiment 2: Class Balance ─────────────────────────────────────────────

def exp_class_balance(X_train, y_train, X_test, y_test, model_type="lr"):
    """Compare original vs class_weight='balanced' vs SMOTE."""
    _ensure_dirs()
    results = {}

    # Original
    model = make_model(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    results["original"] = compute_metrics(y_test, y_pred, y_prob)

    # Balanced class weights
    model_b = make_model(model_type, class_weight="balanced")
    model_b.fit(X_train, y_train)
    y_pred_b = model_b.predict(X_test)
    y_prob_b = model_b.predict_proba(X_test) if hasattr(model_b, "predict_proba") else None
    results["balanced"] = compute_metrics(y_test, y_pred_b, y_prob_b)

    # SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        model_s = make_model(model_type)
        model_s.fit(X_res, y_res)
        y_pred_s = model_s.predict(X_test)
        y_prob_s = model_s.predict_proba(X_test) if hasattr(model_s, "predict_proba") else None
        results["smote"] = compute_metrics(y_test, y_pred_s, y_prob_s)
    except ImportError:
        print("  [Warning] imblearn not installed, skipping SMOTE")

    # Build comparison table
    rows = []
    for strategy, m in results.items():
        rows.append({
            "strategy": strategy,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
        })
    df = pd.DataFrame(rows)
    df.to_latex(os.path.join(TABLES_DIR, f"class_balance_{model_type}.tex"),
                index=False, float_format="%.4f", caption=f"Class Balance — {model_type.upper()}")
    print(f"[Exp2] Class balance done for {model_type.upper()}")
    return df, results


# ─── Experiment 3: Dimensionality Reduction ──────────────────────────────────

def exp_svd_dimensions(X_tfidf, y, model_type="lr", dims=None):
    """Compare different SVD dimensions + plot cumulative explained variance."""
    _ensure_dirs()
    if dims is None:
        dims = [50, 100, 150, 200]

    # Cumulative explained variance plot
    max_dim = min(max(dims), X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd_full = TruncatedSVD(n_components=max_dim, random_state=42)
    svd_full.fit(X_tfidf)

    cumvar = np.cumsum(svd_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, max_dim + 1), cumvar, "-")
    for d in dims:
        if d <= max_dim:
            ax.axvline(d, color="red", linestyle="--", alpha=0.5)
            ax.annotate(f"d={d}: {cumvar[d-1]:.3f}", (d, cumvar[d-1]),
                        fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Number of SVD Components", fontsize=12)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
    ax.set_title("TruncatedSVD — Cumulative Explained Variance", fontsize=13)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "svd_cumulative_variance.png"), dpi=150)
    plt.close(fig)

    # Compare accuracy at different dims
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for d in dims:
        d_actual = min(d, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        svd = TruncatedSVD(n_components=d_actual, random_state=42)
        X_svd = svd.fit_transform(X_tfidf)

        model = make_model(model_type)
        from sklearn.model_selection import cross_validate
        scores = cross_validate(model, X_svd, y, cv=cv,
                                scoring=["accuracy", "f1_macro"], n_jobs=-1)
        rows.append({
            "svd_dim": d,
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "f1_mean": scores["test_f1_macro"].mean(),
            "f1_std": scores["test_f1_macro"].std(),
            "var_explained": cumvar[d_actual - 1] if d_actual <= max_dim else None,
        })

    # No SVD baseline
    model = make_model(model_type)
    scores = cross_validate(model, X_tfidf, y, cv=cv,
                            scoring=["accuracy", "f1_macro"], n_jobs=-1)
    rows.insert(0, {
        "svd_dim": "None",
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std": scores["test_accuracy"].std(),
        "f1_mean": scores["test_f1_macro"].mean(),
        "f1_std": scores["test_f1_macro"].std(),
        "var_explained": 1.0,
    })

    df = pd.DataFrame(rows)
    df.to_latex(os.path.join(TABLES_DIR, f"svd_dims_{model_type}.tex"),
                index=False, float_format="%.4f", caption=f"SVD Dimensions — {model_type.upper()}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [str(r["svd_dim"]) for _, r in df.iterrows()]
    f1s = df["f1_mean"].values
    ax.bar(labels, f1s, color="steelblue")
    ax.set_xlabel("SVD Dimensions", fontsize=12)
    ax.set_ylabel("F1 Macro (CV)", fontsize=12)
    ax.set_title(f"SVD Dimension Comparison — {model_type.upper()}", fontsize=13)
    ax.tick_params(labelsize=10)
    for i, v in enumerate(f1s):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"svd_dims_{model_type}.png"), dpi=150)
    plt.close(fig)

    print(f"[Exp3] SVD dimensions done for {model_type.upper()}")
    return df


# ─── Experiment 4: Neutral Class Handling ────────────────────────────────────

def exp_neutral_class(df_raw, model_type="lr", max_features=5000, n_components=150):
    """Compare: 3-class vs binary (remove neutral) vs 3-star-as-negative vs 3-star-as-positive."""
    _ensure_dirs()
    from preprocess import tokenize

    scenarios = {}

    def _run_scenario(df, name, label_names):
        tfidf = TfidfVectorizer(tokenizer=tokenize, max_features=max_features, token_pattern=None)
        X = tfidf.fit_transform(df["review_text"])
        y = df["label"].values
        n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        X_svd = svd.fit_transform(X)

        model = make_model(model_type)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        from sklearn.model_selection import cross_validate
        scores = cross_validate(model, X_svd, y, cv=cv,
                                scoring=["accuracy", "f1_macro"], n_jobs=-1)
        scenarios[name] = {
            "n_samples": len(df),
            "n_classes": len(label_names),
            "accuracy_mean": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "f1_mean": scores["test_f1_macro"].mean(),
            "f1_std": scores["test_f1_macro"].std(),
        }

    # (a) Original 3-class
    _run_scenario(df_raw, "3-class", LABEL_NAMES)

    # (b) Binary: remove neutral
    df_binary = df_raw[df_raw["label"] != 1].copy()
    df_binary["label"] = df_binary["label"].map({0: 0, 2: 1})
    _run_scenario(df_binary, "binary (no neutral)", ["negative", "positive"])

    # (c) 3-star as negative
    df_neg = df_raw.copy()
    df_neg["label"] = df_neg["label"].map({0: 0, 1: 0, 2: 1})
    _run_scenario(df_neg, "3-star → negative", ["negative", "positive"])

    # (d) 3-star as positive
    df_pos = df_raw.copy()
    df_pos["label"] = df_pos["label"].map({0: 0, 1: 1, 2: 1})
    _run_scenario(df_pos, "3-star → positive", ["negative", "positive"])

    df = pd.DataFrame(scenarios).T
    df.index.name = "scenario"
    df.to_latex(os.path.join(TABLES_DIR, f"neutral_class_{model_type}.tex"),
                float_format="%.4f", caption=f"Neutral Class Handling — {model_type.upper()}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    ax.bar(x, df["f1_mean"], yerr=df["f1_std"], color="steelblue", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("F1 Macro (CV)", fontsize=12)
    ax.set_title(f"Neutral Class Experiment — {model_type.upper()}", fontsize=13)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"neutral_class_{model_type}.png"), dpi=150)
    plt.close(fig)

    print(f"[Exp4] Neutral class done for {model_type.upper()}")
    return df


# ─── Experiment 5: Data Augmentation ─────────────────────────────────────────

def exp_augmentation(X_train, y_train, X_test, y_test, model_type="lr"):
    """Compare original vs simple augmentation (noise injection on SVD features)."""
    _ensure_dirs()
    results = {}

    # Original
    model = make_model(model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    results["original"] = compute_metrics(y_test, y_pred, y_prob)

    # Augment: add Gaussian noise to minority classes
    unique, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    X_aug, y_aug = [X_train.copy()], [y_train.copy()]
    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            n_needed = max_count - cnt
            idx = np.where(y_train == cls)[0]
            chosen = np.random.RandomState(42).choice(idx, size=n_needed, replace=True)
            noise = np.random.RandomState(42).normal(0, 0.01, size=(n_needed, X_train.shape[1]))
            X_aug.append(X_train[chosen] + noise)
            y_aug.append(np.full(n_needed, cls))
    X_aug = np.vstack(X_aug)
    y_aug = np.concatenate(y_aug)

    model_a = make_model(model_type)
    model_a.fit(X_aug, y_aug)
    y_pred_a = model_a.predict(X_test)
    y_prob_a = model_a.predict_proba(X_test) if hasattr(model_a, "predict_proba") else None
    results["augmented"] = compute_metrics(y_test, y_pred_a, y_prob_a)

    rows = []
    for strategy, m in results.items():
        rows.append({
            "strategy": strategy,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"],
        })
    df = pd.DataFrame(rows)
    df.to_latex(os.path.join(TABLES_DIR, f"augmentation_{model_type}.tex"),
                index=False, float_format="%.4f", caption=f"Data Augmentation — {model_type.upper()}")

    print(f"[Exp5] Augmentation done for {model_type.upper()}")
    return df, results


# ─── Run All Experiments ─────────────────────────────────────────────────────

def run_all(input_csv=None, model_types=None):
    """Run all experiments end-to-end."""
    from preprocess import preprocess, tokenize

    if input_csv is None:
        input_csv = os.path.join(ROOT, "data", "dummy", "reviews.csv")
    if model_types is None:
        model_types = ["lr", "svm"]

    print("=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print("=" * 60)

    # Preprocess
    X_train, X_test, y_train, y_test, tfidf, svd, df_raw = preprocess(input_csv)

    # Rebuild TF-IDF matrix for SVD experiment
    tfidf_full = TfidfVectorizer(tokenizer=tokenize, max_features=5000, token_pattern=None)
    X_tfidf_full = tfidf_full.fit_transform(df_raw["review_text"])
    y_full = df_raw["label"].values

    for mt in model_types:
        print(f"\n{'─'*60}")
        print(f"Model: {mt.upper()}")
        print(f"{'─'*60}")

        X_full_svd = np.vstack([X_train, X_test])
        y_full_concat = np.concatenate([y_train, y_test])

        exp_learning_curve(X_full_svd, y_full_concat, model_type=mt)
        exp_class_balance(X_train, y_train, X_test, y_test, model_type=mt)
        exp_svd_dimensions(X_tfidf_full, y_full, model_type=mt)
        exp_neutral_class(df_raw, model_type=mt)
        exp_augmentation(X_train, y_train, X_test, y_test, model_type=mt)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(csv)
