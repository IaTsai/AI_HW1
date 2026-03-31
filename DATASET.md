# Dataset Documentation

## Taiwan Restaurant Reviews from Google Maps

**Author:** Ian Tsai (313553058)
**Date Created:** March 2026
**License:** For academic use only (NYCU AI Capstone Spring 2026)

---

## Data Type

Text classification dataset. Each sample is a restaurant review written in **Traditional Chinese**, paired with a star rating (1--5) that serves as the label.

**Fields per sample:**

| Column | Type | Description |
|--------|------|-------------|
| `review_text` | string | Review content in Traditional Chinese |
| `rating` | int (1--5) | Star rating given by the reviewer |
| `restaurant` | string | Restaurant name |
| `area` | string | City or region (e.g., night market name) |
| `date` | string | Relative date as displayed on Google Maps |
| `likes` | int | Number of "helpful" votes from other users |
| `label` | int (0/1/2) | Derived sentiment: 0 = negative (1--2 stars), 1 = neutral (3 stars), 2 = positive (4--5 stars) |

**File format:** CSV (UTF-8 with BOM), comma-separated.

---

## External Source

All reviews were crawled from **Google Maps** (maps.google.com). Reviews are user-generated content posted publicly on restaurant pages.

---

## Amount and Composition

| Attribute | Value |
|-----------|-------|
| Raw reviews collected | 6,547 |
| After cleaning | 6,509 (99.4% retention) |
| Unique restaurants | 218 |
| Regions covered | 18 |
| Positive (4--5 stars) | 5,162 (79.3%) |
| Neutral (3 stars) | 664 (10.2%) |
| Negative (1--2 stars) | 683 (10.5%) |
| Average review length | 106.2 characters |
| Median review length | 72 characters |
| Mean star rating | 4.18 |

The dataset exhibits **severe class imbalance**: positive reviews dominate at 79.3%, while neutral and negative reviews each constitute approximately 10%.

---

## Conditions for Data Collection

- **Geographic scope:** 10 cities across Taiwan (Taipei, Taichung, Tainan, Kaohsiung, etc.) and 8 night markets (e.g., Shilin, Raohe, Garden Night Market).
- **Search queries:** 45 queries were constructed combining city/region names with food-related keywords (e.g., "Tainan restaurants", "Shilin Night Market food").
- **Selection criteria:** Restaurants appearing in Google Maps search results for these queries; no minimum review count or rating threshold was applied.
- **Time period:** Reviews span multiple years as displayed on Google Maps (the platform shows all historical reviews).
- **Language:** Only reviews containing Chinese characters were retained during cleaning.
- **Minimum length:** Reviews shorter than 5 characters were removed.

---

## Process of Data Collection

### Software and Tools

| Tool | Purpose |
|------|---------|
| **Python 3.9** | Programming language |
| **Playwright** (Microsoft) | Browser automation for headless Chromium |
| **jieba** | Chinese word segmentation |
| **CUDA 12.1** + **NVIDIA RTX 4090** | GPU-accelerated BERT embedding extraction |
| **XLM-RoBERTa-large** (`joeddav/xlm-roberta-large-xnli`) | Pretrained multilingual BERT model |
| **scikit-learn** | Classical ML classifiers (LR, SVM) and evaluation |
| **PyTorch** | MLP deep learning classifier |

### Crawling Procedure

1. The crawler (`src/collect.py`) launches a headless Chromium browser via Playwright.
2. For each of the 45 search queries, it searches Google Maps and iterates through restaurant results.
3. For each restaurant, it navigates to the reviews tab, scrolls to load all reviews, and clicks "More" buttons to expand truncated review text.
4. Structured data (review text, rating, restaurant name, area, date, likes count) is extracted and saved to CSV.
5. **Anti-blocking measures:** random delays (2--5 seconds between actions), restaurant rotation, randomized query order, and checkpoint saving for resumable crawling.

### Data Cleaning

The cleaning step (`src/clean.py`) removed 38 reviews (0.6% of raw data):
- Empty or NaN text entries
- Reviews shorter than 5 characters
- Reviews without any Chinese characters
- Exact duplicate reviews
- Entries with rating = 0 (extraction failures)

---

## Examples

Below are sample reviews from the dataset (original Traditional Chinese text):

### Negative (1 star)
> **Restaurant:** 二師兄古早味滷味（台南花園夜市）| **Area:** 花園夜市
>
> 路過很香、排隊買了雞翅雞爪跟米血，忍不住對鍋子拍了照，結果被戴眼鏡的一位老杯杯瞪。雞翅米血跟雞翅都算是軟q入味還不錯吃，但態度真的可以不用這麼差，不然再好吃都不會想再去

### Neutral (3 stars)
> **Restaurant:** 台南花園夜市 | **Area:** 花園夜市
>
> 夜市裡面如同大家說的垃圾桶很少，應該是根本沒有吧！最後只有在男廁旁邊看到3個垃圾車。然後廁所大排長龍。在裡面買了4種小吃...

### Positive (5 stars)
> **Restaurant:** 台南花園夜市 | **Area:** 花園夜市
>
> 好久沒來花園夜市了，人潮少的有點讓我震驚，跟兩三年前記憶中的人擠人完全不一樣...

### Positive (4 stars)
> **Restaurant:** 台南花園夜市 | **Area:** 花園夜市
>
> 第一次來～～週四來很好逛！不會太擁擠！滿載而歸買到好多好物，且大多都可以LINE PAY

---

## File Locations

- **Raw dataset:** `data/raw/reviews.csv` (6,547 reviews)
- **Cleaned dataset:** `data/processed/reviews_clean.csv` (6,509 reviews)

---

## Label Mapping

### 3-class (default)
| Stars | Label | Class |
|-------|-------|-------|
| 1--2 | 0 | Negative |
| 3 | 1 | Neutral |
| 4--5 | 2 | Positive |

### Binary (best performance)
| Stars | Label | Class |
|-------|-------|-------|
| 1--3 | 0 | Negative (dissatisfied) |
| 4--5 | 1 | Positive (satisfied) |

The binary mapping reflects a cultural insight: in Taiwan, a 3-star rating on a 5-point scale typically signals dissatisfaction rather than true neutrality.
