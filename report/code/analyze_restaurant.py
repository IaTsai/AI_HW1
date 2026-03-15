"""Restaurant sentiment analyzer — practical inference application.

Usage:
    python src/analyze_restaurant.py <google_maps_url> [--model svm|lr] [--max-reviews 200]

Example:
    python src/analyze_restaurant.py "https://www.google.com/maps/place/鼎泰豐..." --model svm
"""

import argparse
import os
import pickle
import re
import sys
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from collect import _extract_reviews_from_page, _extract_rating, _handle_consent, _dismiss_overlays, _random_sleep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "results", "models")
OUTPUT_DIR = os.path.join(ROOT, "results", "analysis")

LABEL_NAMES = {0: "負面", 1: "中性", 2: "正面"}
LABEL_NAMES_BINARY = {0: "不滿意", 1: "滿意"}
LABEL_COLORS = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
LABEL_COLORS_BINARY = {0: "#e74c3c", 1: "#2ecc71"}


# ---------------------------------------------------------------------------
# Date parsing — Google Maps shows relative dates like "2 週前", "3 個月前"
# ---------------------------------------------------------------------------

_RELATIVE_DATE_PATTERNS = [
    (re.compile(r"(\d+)\s*分鐘前"), lambda m: timedelta(minutes=int(m.group(1)))),
    (re.compile(r"(\d+)\s*小時前"), lambda m: timedelta(hours=int(m.group(1)))),
    (re.compile(r"(\d+)\s*天前"), lambda m: timedelta(days=int(m.group(1)))),
    (re.compile(r"(\d+)\s*週前"), lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*個月前"), lambda m: timedelta(days=int(m.group(1)) * 30)),
    (re.compile(r"(\d+)\s*年前"), lambda m: timedelta(days=int(m.group(1)) * 365)),
    # English fallbacks
    (re.compile(r"(\d+)\s*minute"), lambda m: timedelta(minutes=int(m.group(1)))),
    (re.compile(r"(\d+)\s*hour"), lambda m: timedelta(hours=int(m.group(1)))),
    (re.compile(r"(\d+)\s*day"), lambda m: timedelta(days=int(m.group(1)))),
    (re.compile(r"(\d+)\s*week"), lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*month"), lambda m: timedelta(days=int(m.group(1)) * 30)),
    (re.compile(r"(\d+)\s*year"), lambda m: timedelta(days=int(m.group(1)) * 365)),
    # "a week ago" style
    (re.compile(r"一\s*分鐘前"), lambda m: timedelta(minutes=1)),
    (re.compile(r"一\s*小時前"), lambda m: timedelta(hours=1)),
    (re.compile(r"一\s*天前"), lambda m: timedelta(days=1)),
    (re.compile(r"一\s*週前"), lambda m: timedelta(weeks=1)),
    (re.compile(r"一\s*個月前"), lambda m: timedelta(days=30)),
    (re.compile(r"一\s*年前"), lambda m: timedelta(days=365)),
]


def parse_relative_date(text, now=None):
    """Convert Google Maps relative date string to a datetime."""
    if now is None:
        now = datetime.now()
    text = text.strip()
    # Strip "上次編輯：" prefix — use the edit time as the date
    text = re.sub(r"^上次編輯[：:]\s*", "", text)
    for pattern, delta_fn in _RELATIVE_DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            return now - delta_fn(m)
    return None


# ---------------------------------------------------------------------------
# Scrape a single restaurant
# ---------------------------------------------------------------------------

def _resolve_short_url(url):
    """Resolve a short URL (goo.gl/maps.app) to get the final destination URL.

    Uses HTTP HEAD requests to follow redirects without a browser.
    """
    import urllib.parse as _up
    import urllib.request

    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            final_url = resp.url
            return final_url
    except Exception:
        pass
    return url


def _extract_name_from_url(url):
    """Extract restaurant name from a Google Maps URL."""
    import urllib.parse as _up

    parsed = _up.urlparse(url)

    # From ?q= parameter (common with short URL redirects)
    params = _up.parse_qs(parsed.query)
    q = params.get("q", [""])[0]
    if q:
        # Address format: "412臺中市大里區大明路519號老鐵醬平價鐵板燒"
        parts = re.split(r"號", q)
        if len(parts) > 1 and parts[-1]:
            return parts[-1].strip()
        return q.strip()

    # From /maps/place/NAME/ path
    path_match = re.search(r"/maps/place/([^/@]+)", parsed.path)
    if path_match:
        return _up.unquote(path_match.group(1)).replace("+", " ")

    return ""


def _resolve_restaurant_name(page):
    """Extract restaurant name from the current place page."""
    from playwright.sync_api import TimeoutError as PwTimeout

    # Try h1 element first
    try:
        page.wait_for_selector("h1.DUwDvf, h1.fontHeadlineLarge", timeout=8000)
        h1 = page.locator("h1.DUwDvf, h1.fontHeadlineLarge")
        if h1.count() > 0:
            name = h1.first.inner_text().strip()
            if name and len(name) > 1:
                return name
    except PwTimeout:
        pass

    # Fallback: extract from current page URL
    return _extract_name_from_url(page.url)


def _open_place_via_search(page, restaurant_name):
    """Search the restaurant name on Google Maps and click into the full place page.

    Direct URL navigation often shows a minimal layout without the reviews tab.
    Going through a search result click forces Google Maps to load the full UI.
    """
    import urllib.parse

    # Build a search query — add area keywords for disambiguation
    query = urllib.parse.quote(restaurant_name)
    search_url = f"https://www.google.com/maps/search/{query}/?hl=zh-TW"
    print(f"  Re-searching: {search_url}")
    page.goto(search_url, wait_until="domcontentloaded")
    _random_sleep(4, 6)

    # If Google Maps went directly to the place page (single result),
    # check whether the reviews tab is present now.
    reviews_tab = page.locator("button[role='tab']:has-text('評論')")
    if reviews_tab.count() > 0:
        return True

    # Otherwise, look for the restaurant in the search results feed
    feed = page.locator("div[role='feed']")
    if feed.count() == 0:
        return False

    links = page.locator("div[role='feed'] a[href*='/maps/place/']")
    for i in range(links.count()):
        name = links.nth(i).get_attribute("aria-label") or ""
        if restaurant_name[:4] in name:
            print(f"  Found in results: {name}")
            links.nth(i).click()
            _random_sleep(4, 6)
            return True

    # Scroll feed and retry
    feed_el = feed.first
    for _ in range(5):
        feed_el.evaluate("el => el.scrollTop = el.scrollHeight")
        _random_sleep(1, 2)

    links = page.locator("div[role='feed'] a[href*='/maps/place/']")
    for i in range(links.count()):
        name = links.nth(i).get_attribute("aria-label") or ""
        if restaurant_name[:4] in name:
            print(f"  Found after scroll: {name}")
            links.nth(i).click()
            _random_sleep(4, 6)
            return True

    return False


def _extract_reviews_deep(page, restaurant_name, max_reviews=3000,
                          stale_wait=1.5, max_stale=15):
    """Extract reviews with aggressive scrolling for large review counts."""
    from playwright.sync_api import TimeoutError as PwTimeout

    reviews = []

    try:
        page.wait_for_selector("div.jftiEf", timeout=10000)
    except PwTimeout:
        return reviews

    # Find scrollable panel
    scrollable = page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
    if scrollable.count() == 0:
        scrollable = page.locator("div.m6QErb.DxyBCb")
    if scrollable.count() == 0:
        scrollable = page.locator("div[role='main']")
    if scrollable.count() == 0:
        return reviews

    scroll_el = scrollable.first

    # Aggressive scrolling
    last_count = 0
    stale_rounds = 0
    scroll_round = 0

    while True:
        scroll_el.evaluate("el => el.scrollTop = el.scrollHeight")
        time.sleep(1.0)  # faster than collect.py's 1.5-2.5s

        review_els = page.locator("div.jftiEf")
        current_count = review_els.count()
        scroll_round += 1

        if scroll_round % 50 == 0 or current_count != last_count:
            if current_count != last_count:
                print(f"  Scrolling... {current_count} reviews loaded", flush=True)

        if current_count >= max_reviews:
            break

        if current_count == last_count:
            stale_rounds += 1
            if stale_rounds >= max_stale:
                print(f"  Stopped scrolling after {max_stale} stale rounds ({current_count} reviews)", flush=True)
                break
            # Longer wait on stale — Google may be loading
            time.sleep(stale_wait)
        else:
            stale_rounds = 0

        last_count = current_count

    # Click all "More" / "全文" buttons to expand truncated reviews
    more_buttons = page.locator("button.w8nwRe.kyuRq")
    btn_count = more_buttons.count()
    print(f"  Expanding {btn_count} truncated reviews...", flush=True)
    for i in range(btn_count):
        try:
            more_buttons.nth(i).click(timeout=500)
            if i % 100 == 0 and i > 0:
                time.sleep(0.3)
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            print(f"  Expanded {i + 1}/{btn_count}...", flush=True)
    if btn_count > 0:
        print(f"  Expanded {btn_count}/{btn_count} done", flush=True)

    # Extract review data
    review_els = page.locator("div.jftiEf")
    count = min(review_els.count(), max_reviews)
    print(f"  Extracting data from {count} reviews...", flush=True)

    for i in range(count):
        try:
            el = review_els.nth(i)

            text_el = el.locator("span.wiI7pd")
            if text_el.count() == 0:
                continue
            text = text_el.first.inner_text().strip()
            if not text or len(text) < 2:
                continue

            rating = _extract_rating(el)

            date_text = ""
            try:
                date_el = el.locator("span.rsqaWe")
                if date_el.count() > 0:
                    date_text = date_el.first.inner_text().strip()
            except Exception:
                pass

            likes = 0
            try:
                likes_el = el.locator("span.pkWtMe")
                if likes_el.count() > 0:
                    likes_text = likes_el.first.inner_text().strip()
                    if likes_text.isdigit():
                        likes = int(likes_text)
            except Exception:
                pass

            if rating <= 2:
                label = 0
            elif rating == 3:
                label = 1
            else:
                label = 2

            reviews.append({
                "review_text": text,
                "rating": rating,
                "restaurant": restaurant_name,
                "area": "",
                "date": date_text,
                "likes": likes,
                "label": label,
            })
        except Exception:
            continue

        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{count}...", flush=True)

    return reviews


def scrape_restaurant(url, max_reviews=200, **kwargs):
    """Scrape reviews from a single Google Maps restaurant URL."""
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

    print(f"Scraping: {url}")
    print(f"Max reviews: {max_reviews}")

    # Pre-resolve short URLs to extract restaurant name reliably
    resolved_url = url
    pre_resolved_name = ""
    if "goo.gl" in url or "maps.app" in url:
        print("Resolving short URL...")
        resolved_url = _resolve_short_url(url)
        pre_resolved_name = _extract_name_from_url(resolved_url)
        if pre_resolved_name:
            print(f"  Resolved name: {pre_resolved_name}")

    reviews = []
    restaurant_name = ""

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            locale="zh-TW",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        try:
            # Step 1: Navigate to the URL to get the restaurant name
            page.goto(url, wait_until="domcontentloaded")
            _handle_consent(page)
            _dismiss_overlays(page)
            # Short URLs (goo.gl, maps.app.goo.gl) need extra time to redirect
            if "goo.gl" in url or "maps.app" in url:
                time.sleep(6)
            else:
                _random_sleep(3, 5)

            restaurant_name = _resolve_restaurant_name(page)
            if not restaurant_name:
                restaurant_name = pre_resolved_name or "Unknown Restaurant"
            print(f"Restaurant: {restaurant_name}")

            # Step 2: Check if reviews tab is available
            reviews_tab = page.locator(
                "button[role='tab']:has-text('評論'), "
                "div.Gpq6kf:has-text('評論'), "
                "button[aria-label*='評論']"
            )

            # Step 3: If no reviews tab, re-enter via search (click result)
            if reviews_tab.count() == 0:
                print("No reviews tab — re-entering via search...")
                if not _open_place_via_search(page, restaurant_name):
                    print("Could not find restaurant via search.")
                    return restaurant_name, []

                reviews_tab = page.locator(
                    "button[role='tab']:has-text('評論'), "
                    "div.Gpq6kf:has-text('評論')"
                )
                if reviews_tab.count() == 0:
                    print("Still no reviews tab after re-search!")
                    return restaurant_name, []

            # Step 4: Click reviews tab
            reviews_tab.first.click()
            _random_sleep(2, 3)

            # Sort reviews — use sort_order parameter
            sort_order = kwargs.get("sort_order", "newest")
            try:
                sort_btn = page.locator("button[aria-label='排序評論'], button[data-value='排序']")
                if sort_btn.count() > 0:
                    sort_btn.first.click()
                    _random_sleep(1, 2)
                    # 0=最相關, 1=最新, 2=最高評分, 3=最低評分
                    sort_map = {"relevant": "0", "newest": "1", "highest": "2", "lowest": "3"}
                    idx = sort_map.get(sort_order, "1")
                    sort_option = page.locator(f"li[data-index='{idx}'], div[role='menuitemradio']")
                    if sort_option.count() > int(idx):
                        sort_option.nth(int(idx)).click()
                        _random_sleep(2, 3)
                        print(f"Sorted by: {sort_order}")
                    elif sort_option.count() > 0:
                        sort_option.first.click()
                        _random_sleep(2, 3)
            except Exception:
                pass

            # Step 5: Extract reviews (with aggressive scrolling for large counts)
            stale_wait = kwargs.get("stale_wait", 1.5)
            max_stale = kwargs.get("max_stale", 15)
            reviews = _extract_reviews_deep(page, restaurant_name, max_reviews,
                                            stale_wait=stale_wait, max_stale=max_stale)
            print(f"Extracted {len(reviews)} reviews")

        finally:
            browser.close()

    return restaurant_name, reviews


def scrape_restaurant_full(url, target_count=3000):
    """Multi-pass scraping to maximize review coverage.

    Pass 1: Sort by newest (max_stale=30, slow wait)
    Pass 2: Sort by most relevant
    Pass 3: Sort by lowest rating
    Merge all passes and deduplicate.
    """
    all_reviews = []
    seen_texts = set()

    passes = [
        ("newest",  "最新",   {"sort_order": "newest",  "max_stale": 30, "stale_wait": 3.0}),
        ("relevant", "最相關", {"sort_order": "relevant", "max_stale": 20, "stale_wait": 3.0}),
        ("lowest",  "最低評分", {"sort_order": "lowest",  "max_stale": 20, "stale_wait": 3.0}),
    ]

    restaurant_name = ""

    for pass_name, label, kwargs in passes:
        print(f"\n{'='*50}")
        print(f"Pass: {label} ({pass_name})")
        print(f"{'='*50}")

        name, reviews = scrape_restaurant(url, max_reviews=target_count, **kwargs)
        if not restaurant_name:
            restaurant_name = name

        new_count = 0
        for r in reviews:
            text = r["review_text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                all_reviews.append(r)
                new_count += 1

        print(f"  Got {len(reviews)} reviews, {new_count} new (total unique: {len(all_reviews)})")

        if len(all_reviews) >= target_count:
            print(f"  Reached target {target_count}!")
            break

    print(f"\nTotal unique reviews: {len(all_reviews)}")
    return restaurant_name, all_reviews


# ---------------------------------------------------------------------------
# Inference — load model and predict
# ---------------------------------------------------------------------------

def load_inference_pipeline(model_type="svm"):
    """Load saved tfidf + svd + model for inference (lr/svm)."""
    with open(os.path.join(PROCESSED_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "svd.pkl"), "rb") as f:
        svd = pickle.load(f)

    model_path = os.path.join(MODELS_DIR, f"{model_type}_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Loaded inference pipeline (model={model_type})")
    return tfidf, svd, model


def load_bert_pipeline():
    """Load BERT zero-shot classifier on GPU."""
    from bert_baseline import load_classifier
    import torch

    device = 0 if torch.cuda.is_available() else -1
    clf = load_classifier(device=device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Loaded BERT zero-shot pipeline (GPU: {gpu_name})")
    return clf


def load_bert_lr_pipeline(binary=False):
    """Load BERT feature extractor + trained LR model."""
    import torch

    if binary:
        model_path = os.path.join(MODELS_DIR, "lr_bert_binary_model.pkl")
    else:
        model_path = os.path.join(MODELS_DIR, "lr_bert_model.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    label = "BERT+LR (binary)" if binary else "BERT+LR (3-class)"
    print(f"Loaded {label} pipeline (GPU: {gpu_name})")
    return model


def predict_sentiment(texts, tfidf, svd, model):
    """Predict sentiment for a list of texts (lr/svm)."""
    X_tfidf = tfidf.transform(texts)
    X_svd = svd.transform(X_tfidf)
    predictions = model.predict(X_svd)
    probabilities = model.predict_proba(X_svd) if hasattr(model, "predict_proba") else None
    return predictions, probabilities


def predict_sentiment_bert(texts, clf, batch_size=32):
    """Predict sentiment using BERT zero-shot on GPU."""
    from bert_baseline import predict_batch
    texts = [t[:512] for t in texts]
    predictions, probabilities = predict_batch(clf, texts, batch_size=batch_size)
    return predictions, probabilities


def predict_sentiment_bert_lr(texts, model, batch_size=64):
    """Predict sentiment using BERT embeddings + trained LR (our best model)."""
    from bert_features import extract_embeddings
    texts = [t[:512] for t in texts]
    embeddings = extract_embeddings(texts, batch_size=batch_size)
    predictions = model.predict(embeddings)
    probabilities = model.predict_proba(embeddings) if hasattr(model, "predict_proba") else None
    return predictions, probabilities


def _run_inference(texts, model_type):
    """Dispatch inference to the appropriate model pipeline.

    Returns (predictions, probabilities, is_binary).
    """
    if model_type == "bert-lr-binary":
        model = load_bert_lr_pipeline(binary=True)
        preds, probs = predict_sentiment_bert_lr(texts, model)
        return preds, probs, True
    elif model_type == "bert-lr":
        model = load_bert_lr_pipeline(binary=False)
        preds, probs = predict_sentiment_bert_lr(texts, model)
        return preds, probs, False
    elif model_type == "bert":
        clf = load_bert_pipeline()
        preds, probs = predict_sentiment_bert(texts, clf)
        return preds, probs, False
    else:
        tfidf, svd, model = load_inference_pipeline(model_type)
        preds, probs = predict_sentiment(texts, tfidf, svd, model)
        return preds, probs, False


# ---------------------------------------------------------------------------
# Weekly aggregation and analysis
# ---------------------------------------------------------------------------

def build_analysis_df(reviews, predictions, probabilities=None, binary=False):
    """Build a DataFrame with reviews, predictions, and parsed dates."""
    df = pd.DataFrame(reviews)
    df["predicted_label"] = predictions
    label_map = LABEL_NAMES_BINARY if binary else LABEL_NAMES
    df["predicted_sentiment"] = df["predicted_label"].map(label_map)

    if probabilities is not None:
        for i, name in label_map.items():
            if i < probabilities.shape[1]:
                df[f"prob_{name}"] = probabilities[:, i]

    # Parse dates
    now = datetime.now()
    df["parsed_date"] = df["date"].apply(lambda x: parse_relative_date(str(x), now))
    # Monthly bucket (Google Maps dates are month-granularity for older reviews)
    df["year_month"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y-%m") if d else None
    )
    df["month_label"] = df["parsed_date"].apply(
        lambda d: d.strftime("%Y/%m") if d else None
    )
    # Keep week-level fields for backward compatibility
    df["week"] = df["parsed_date"].apply(
        lambda d: d.isocalendar()[1] if d else None
    )
    df["year"] = df["parsed_date"].apply(
        lambda d: d.isocalendar()[0] if d else None
    )
    df["week_start"] = df["parsed_date"].apply(
        lambda d: (d - timedelta(days=d.weekday())).strftime("%Y/%m/%d") if d else None
    )
    df["year_week"] = df.apply(
        lambda r: f"{int(r['year'])}-W{int(r['week']):02d}" if pd.notna(r["year"]) else None,
        axis=1,
    )

    return df


def weekly_summary(df, binary=False):
    """Aggregate sentiment by month (Google Maps dates are month-granularity)."""
    dated = df.dropna(subset=["parsed_date"]).copy()
    if dated.empty:
        print("Warning: no parseable dates found.")
        return pd.DataFrame()

    pos_label = 1 if binary else 2

    agg_dict = {
        "month_label": ("month_label", "first"),
        "total": ("predicted_label", "count"),
        "positive": ("predicted_label", lambda x: (x == pos_label).sum()),
        "negative": ("predicted_label", lambda x: (x == 0).sum()),
        "avg_rating": ("rating", "mean"),
    }
    if not binary:
        agg_dict["neutral"] = ("predicted_label", lambda x: (x == 1).sum())

    grouped = (
        dated.groupby("year_month")
        .agg(**agg_dict)
        .reset_index()
        .sort_values("year_month")
    )
    if binary and "neutral" not in grouped.columns:
        grouped["neutral"] = 0

    grouped["satisfaction_rate"] = (grouped["positive"] / grouped["total"] * 100).round(1)
    grouped["negative_rate"] = (grouped["negative"] / grouped["total"] * 100).round(1)

    return grouped


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _setup_chinese_font():
    """Try to find a CJK font for matplotlib."""
    candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans TC",
        "Microsoft JhengHei", "WenQuanYi Micro Hei", "SimHei",
        "PingFang TC", "AR PL UMing TW",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    # Fallback: try any font with CJK in the path
    for f in font_manager.fontManager.ttflist:
        if "CJK" in f.fname:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return f.name
    return None


def plot_weekly_trend(weekly_df, restaurant_name, output_path):
    """Plot weekly sentiment trend as a stacked bar + satisfaction line."""
    _setup_chinese_font()

    if weekly_df.empty:
        print("No data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(max(10, len(weekly_df) * 0.8), 6))

    x = range(len(weekly_df))
    labels = weekly_df["month_label"].values if "month_label" in weekly_df.columns else weekly_df["week_start"].values

    # Stacked bar
    ax1.bar(x, weekly_df["positive"], color=LABEL_COLORS[2], label="正面", alpha=0.85)
    ax1.bar(x, weekly_df["neutral"], bottom=weekly_df["positive"],
            color=LABEL_COLORS[1], label="中性", alpha=0.85)
    ax1.bar(x, weekly_df["negative"],
            bottom=weekly_df["positive"] + weekly_df["neutral"],
            color=LABEL_COLORS[0], label="負面", alpha=0.85)

    ax1.set_xlabel("月份", fontsize=12)
    ax1.set_ylabel("評論數", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(loc="upper left", fontsize=10)

    # Satisfaction rate line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(x, weekly_df["satisfaction_rate"], "ko-", linewidth=2,
             markersize=6, label="滿意率 (%)")
    ax2.set_ylabel("滿意率 (%)", fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=10)

    # Add date range subtitle
    first_label = labels[0]
    last_label = labels[-1]
    ax1.set_title(f"【{restaurant_name}】每月顧客滿意度趨勢\n統計期間：{first_label} ~ {last_label}（共 {len(weekly_df)} 個月）",
                  fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend plot: {output_path}")


def plot_sentiment_pie(df, restaurant_name, output_path, binary=False):
    """Plot overall sentiment distribution pie chart."""
    _setup_chinese_font()

    label_map = LABEL_NAMES_BINARY if binary else LABEL_NAMES
    color_map = LABEL_COLORS_BINARY if binary else LABEL_COLORS
    counts = df["predicted_label"].value_counts().sort_index()
    labels = [label_map[i] for i in counts.index]
    colors = [color_map[i] for i in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    # Add date range and sample size
    dated = df.dropna(subset=["parsed_date"])
    if not dated.empty:
        min_date = dated["parsed_date"].min().strftime("%Y/%m/%d")
        max_date = dated["parsed_date"].max().strftime("%Y/%m/%d")
        subtitle = f"統計期間：{min_date} ~ {max_date}｜共 {len(df)} 則評論"
    else:
        subtitle = f"共 {len(df)} 則評論"
    ax.set_title(f"【{restaurant_name}】整體情感分布\n{subtitle}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved pie chart: {output_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_report(restaurant_name, df, weekly_df, binary=False):
    """Print a concise text summary to console."""
    total = len(df)
    pos_label = 1 if binary else 2
    pos = (df["predicted_label"] == pos_label).sum()
    neg = (df["predicted_label"] == 0).sum()

    print("\n" + "=" * 60)
    print(f"  餐廳分析報告：{restaurant_name}")
    print("=" * 60)
    print(f"\n  總評論數：{total}")
    if binary:
        print(f"  滿意：{pos} ({pos/total*100:.1f}%)")
        print(f"  不滿意：{neg} ({neg/total*100:.1f}%)")
    else:
        neu = (df["predicted_label"] == 1).sum()
        print(f"  正面：{pos} ({pos/total*100:.1f}%)")
        print(f"  中性：{neu} ({neu/total*100:.1f}%)")
        print(f"  負面：{neg} ({neg/total*100:.1f}%)")
    print(f"  平均星等：{df['rating'].mean():.2f}")

    if not weekly_df.empty:
        period_col = "year_month" if "year_month" in weekly_df.columns else "year_week"
        print(f"\n  --- 每月趨勢（共 {len(weekly_df)} 個月）---")
        print(f"  {'月份':<12} {'評論數':>6} {'滿意率':>8} {'負面率':>8}")
        print(f"  {'-'*36}")
        for _, row in weekly_df.iterrows():
            label = row.get(period_col, "")
            print(f"  {label:<12} {int(row['total']):>6} "
                  f"{row['satisfaction_rate']:>7.1f}% {row['negative_rate']:>7.1f}%")

        # Trend direction
        if len(weekly_df) >= 2:
            recent = weekly_df["satisfaction_rate"].iloc[-1]
            prev = weekly_df["satisfaction_rate"].iloc[-2]
            diff = recent - prev
            if diff > 5:
                trend = "上升 ↑"
            elif diff < -5:
                trend = "下降 ↓"
            else:
                trend = "持平 →"
            print(f"\n  最近趨勢：{trend}（{prev:.1f}% → {recent:.1f}%）")

    # Show worst reviews for actionable insight
    neg_reviews = df[df["predicted_label"] == 0].sort_values("rating")
    if not neg_reviews.empty:
        print(f"\n  --- 負面評論摘要（最多 5 則）---")
        for _, row in neg_reviews.head(5).iterrows():
            text = row["review_text"][:80] + ("..." if len(row["review_text"]) > 80 else "")
            print(f"  [{row['rating']}★] {text}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(url, model_type="bert-lr-binary", max_reviews=200):
    """Full analysis pipeline: scrape → predict → aggregate → visualize."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Scrape
    restaurant_name, reviews = scrape_restaurant(url, max_reviews=max_reviews)
    if not reviews:
        print("No reviews found. Exiting.")
        return None

    # 2. Load model and predict
    texts = [r["review_text"] for r in reviews]
    predictions, probabilities, is_binary = _run_inference(texts, model_type)

    # 3. Build analysis DataFrame
    df = build_analysis_df(reviews, predictions, probabilities, binary=is_binary)

    # 4. Weekly aggregation
    weekly_df = weekly_summary(df, binary=is_binary)

    # 5. Save CSV
    safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', restaurant_name)[:30]
    csv_path = os.path.join(OUTPUT_DIR, f"analysis_{safe_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved analysis CSV: {csv_path}")

    if not weekly_df.empty:
        weekly_csv = os.path.join(OUTPUT_DIR, f"weekly_{safe_name}.csv")
        weekly_df.to_csv(weekly_csv, index=False, encoding="utf-8-sig")
        print(f"Saved weekly CSV: {weekly_csv}")

    # 6. Plots
    trend_path = os.path.join(OUTPUT_DIR, f"trend_{safe_name}.png")
    plot_weekly_trend(weekly_df, restaurant_name, trend_path)

    pie_path = os.path.join(OUTPUT_DIR, f"pie_{safe_name}.png")
    plot_sentiment_pie(df, restaurant_name, pie_path, binary=is_binary)

    # 7. Print report
    print_report(restaurant_name, df, weekly_df, binary=is_binary)

    return df, weekly_df


def analyze_csv(csv_path, model_type="bert-lr-binary"):
    """Analyze reviews from an existing CSV (no scraping needed).

    CSV must have columns: review_text, rating, date
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.dropna(subset=["review_text"]).reset_index(drop=True)

    restaurant_name = df_raw["restaurant"].iloc[0] if "restaurant" in df_raw.columns else "Unknown"

    texts = df_raw["review_text"].tolist()
    predictions, probabilities, is_binary = _run_inference(texts, model_type)

    reviews = df_raw.to_dict("records")
    df = build_analysis_df(reviews, predictions, probabilities, binary=is_binary)
    weekly_df = weekly_summary(df, binary=is_binary)

    safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', restaurant_name)[:30]

    trend_path = os.path.join(OUTPUT_DIR, f"trend_{safe_name}.png")
    plot_weekly_trend(weekly_df, restaurant_name, trend_path)

    pie_path = os.path.join(OUTPUT_DIR, f"pie_{safe_name}.png")
    plot_sentiment_pie(df, restaurant_name, pie_path, binary=is_binary)

    print_report(restaurant_name, df, weekly_df, binary=is_binary)
    return df, weekly_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a restaurant's Google Maps reviews")
    parser.add_argument("url", help="Google Maps restaurant URL, or path to CSV file")
    parser.add_argument("--model", choices=["bert-lr-binary", "bert-lr", "lr", "svm", "bert"],
                        default="bert-lr-binary",
                        help="Model: bert-lr-binary (default, F1=0.826), bert-lr, svm, lr, or bert (zero-shot)")
    parser.add_argument("--max-reviews", type=int, default=200, help="Max reviews to scrape (default: 200)")

    args = parser.parse_args()

    if args.url.endswith(".csv"):
        analyze_csv(args.url, model_type=args.model)
    else:
        analyze(args.url, model_type=args.model, max_reviews=args.max_reviews)
