# Taiwan Food Review Sentiment Classification

NYCU AI Capstone 2026 Spring — Project #1

## Overview

Sentiment classification of Taiwan restaurant reviews from Google Maps. Reviews are classified into **positive**, **neutral**, and **negative** using TF-IDF features with Logistic Regression and SVM.

## Dataset

- **Source**: Google Maps restaurant reviews (Taiwan)
- **Labels**: Positive (4-5★), Neutral (3★), Negative (1-2★)
- **Size**: 1000+ reviews
- **Fields**: `review_text`, `rating`, `restaurant`, `area`, `date`, `likes`, `label`

## Methods

| Method | Description |
|--------|-------------|
| **Preprocessing** | jieba tokenization (custom food dictionary) → TF-IDF → TruncatedSVD |
| **Logistic Regression** | Text classification baseline |
| **SVM (RBF)** | Non-linear classifier |
| **BERT Zero-Shot** | `xlm-roberta-large-xnli` baseline (no training) |

## Project Structure

```
AI_HW1/
├── data/
│   ├── raw/              # Scraped reviews
│   ├── processed/        # TF-IDF + SVD features
│   └── dummy/            # Pipeline test data
├── src/
│   ├── collect.py        # Google Maps scraper (Playwright)
│   ├── preprocess.py     # jieba + TF-IDF + TruncatedSVD
│   ├── train.py          # Model training + CV
│   ├── evaluate.py       # Metrics + save_all_plots()
│   ├── experiments.py    # Ablation studies
│   ├── bert_baseline.py  # BERT zero-shot (GPU)
│   └── clean.py          # Data cleaning
├── notebooks/EDA.ipynb
├── results/
│   ├── figures/          # All plots
│   └── tables/           # LaTeX tables
├── run_pipeline.py       # End-to-end runner
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py                          # Run with dummy data
python run_pipeline.py data/raw/reviews.csv     # Run with real data
python -c "import sys; sys.path.insert(0,'src'); from experiments import run_all; run_all('data/raw/reviews.csv')"
```

## Experiments

1. **Learning Curve** — Performance vs training data size
2. **Class Balance** — Original vs balanced weights vs SMOTE
3. **SVD Dimensions** — Cumulative explained variance + optimal dimensions
4. **Neutral Class** — 3-class vs binary, 3-star reclassification
5. **Data Augmentation** — Noise injection for minority classes
