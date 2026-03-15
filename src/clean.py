"""Data cleaning: filter short, non-Chinese, duplicate reviews."""

import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# At least one CJK character
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def clean(input_csv, output_csv=None, min_length=5):
    """Clean raw reviews:
    - Remove empty/NaN text
    - Remove reviews shorter than min_length characters
    - Remove reviews without Chinese characters
    - Remove exact duplicates
    - Remove reviews with rating=0 (extraction failures)
    """
    if output_csv is None:
        output_csv = os.path.join(ROOT, "data", "processed", "reviews_clean.csv")

    df = pd.read_csv(input_csv)
    n_orig = len(df)

    # Drop NaN
    df = df.dropna(subset=["review_text"])

    # Drop too short
    df = df[df["review_text"].str.len() >= min_length]

    # Must contain Chinese
    df = df[df["review_text"].apply(lambda t: bool(CJK_RE.search(t)))]

    # Drop exact duplicate text
    df = df.drop_duplicates(subset=["review_text"])

    # Drop rating=0 (extraction failure)
    df = df[df["rating"] > 0]

    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Cleaned: {n_orig} → {len(df)} reviews")
    print(f"  Removed: {n_orig - len(df)}")
    print(f"  Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"  Saved to: {output_csv}")

    return df


if __name__ == "__main__":
    import sys

    in_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "raw", "reviews.csv")
    clean(in_csv)
