"""Text preprocessing: jieba tokenization + TF-IDF + TruncatedSVD."""

import os
import pickle

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERDICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_dict.txt")
STOPWORDS_PATH = os.path.join(ROOT, "data", "stopwords.txt")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

_userdict_loaded = False


def _load_userdict():
    global _userdict_loaded
    if not _userdict_loaded and os.path.exists(USERDICT_PATH):
        jieba.load_userdict(USERDICT_PATH)
        _userdict_loaded = True


def _load_stopwords():
    if os.path.exists(STOPWORDS_PATH):
        with open(STOPWORDS_PATH, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


STOPWORDS = None


def _get_stopwords():
    global STOPWORDS
    if STOPWORDS is None:
        STOPWORDS = _load_stopwords()
    return STOPWORDS


def tokenize(text):
    _load_userdict()
    stopwords = _get_stopwords()
    words = jieba.cut(text, cut_all=False)
    return [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in stopwords]


def preprocess(
    input_csv,
    max_features=5000,
    n_components=150,
    test_size=0.2,
    random_state=42,
    save=True,
):
    """Load CSV, build TF-IDF + SVD features, split train/test.

    Returns (X_train, X_test, y_train, y_test, tfidf, svd, df).
    """
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["review_text"]).reset_index(drop=True)

    # TF-IDF
    tfidf = TfidfVectorizer(tokenizer=tokenize, max_features=max_features, token_pattern=None)
    X_tfidf = tfidf.fit_transform(df["review_text"])

    # TruncatedSVD
    n_comp = min(n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_svd = svd.fit_transform(X_tfidf)

    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_svd, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
        with open(os.path.join(PROCESSED_DIR, "tfidf.pkl"), "wb") as f:
            pickle.dump(tfidf, f)
        with open(os.path.join(PROCESSED_DIR, "svd.pkl"), "wb") as f:
            pickle.dump(svd, f)
        print(f"Saved processed data to {PROCESSED_DIR}")
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"  SVD components: {n_comp}, explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

    return X_train, X_test, y_train, y_test, tfidf, svd, df


def load_processed(processed_dir=None):
    """Load saved processed data."""
    d = processed_dir or PROCESSED_DIR
    X_train = np.load(os.path.join(d, "X_train.npy"))
    X_test = np.load(os.path.join(d, "X_test.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    with open(os.path.join(d, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(d, "svd.pkl"), "rb") as f:
        svd = pickle.load(f)
    return X_train, X_test, y_train, y_test, tfidf, svd


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "dummy", "reviews.csv")
    preprocess(csv_path)
