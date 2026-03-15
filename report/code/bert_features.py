"""Extract BERT sentence embeddings for SVM training.

Uses the same xlm-roberta model as bert_baseline.py, but as a feature
extractor (mean pooling of last hidden state) rather than a classifier.

Usage:
    python src/bert_features.py [input_csv]
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

MODEL_NAME = "joeddav/xlm-roberta-large-xnli"
MAX_LENGTH = 128  # most reviews are short; 128 tokens is enough


class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def extract_embeddings(texts, model_name=MODEL_NAME, batch_size=64, device=None):
    """Extract mean-pooled sentence embeddings from BERT.

    Returns: np.ndarray of shape (n_texts, hidden_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    dataset = ReviewDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Mean pooling: average over non-padding tokens
            hidden = outputs.last_hidden_state  # (B, seq_len, hidden)
            mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, hidden)
            counts = mask.sum(dim=1).clamp(min=1)  # (B, 1)
            embeddings = summed / counts  # (B, hidden)

            all_embeddings.append(embeddings.cpu().numpy())

            done = min((batch_idx + 1) * batch_size, len(texts))
            print(f"  Encoded {done}/{len(texts)}")

    return np.vstack(all_embeddings)


def preprocess_bert(input_csv, test_size=0.2, random_state=42, save=True):
    """Full pipeline: load CSV → extract BERT embeddings → train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["review_text"]).reset_index(drop=True)

    texts = df["review_text"].tolist()
    y = df["label"].values

    embeddings = extract_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        np.save(os.path.join(PROCESSED_DIR, "X_train_bert.npy"), X_train)
        np.save(os.path.join(PROCESSED_DIR, "X_test_bert.npy"), X_test)
        np.save(os.path.join(PROCESSED_DIR, "y_train_bert.npy"), y_train)
        np.save(os.path.join(PROCESSED_DIR, "y_test_bert.npy"), y_test)
        print(f"Saved to {PROCESSED_DIR}: X_train_bert.npy, X_test_bert.npy, ...")
        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def load_processed_bert(processed_dir=None):
    """Load saved BERT-processed data."""
    d = processed_dir or PROCESSED_DIR
    X_train = np.load(os.path.join(d, "X_train_bert.npy"))
    X_test = np.load(os.path.join(d, "X_test_bert.npy"))
    y_train = np.load(os.path.join(d, "y_train_bert.npy"))
    y_test = np.load(os.path.join(d, "y_test_bert.npy"))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "data", "processed", "reviews_clean.csv"
    )
    preprocess_bert(csv_path)
