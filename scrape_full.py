"""Full multi-pass scrape of a restaurant + BERT-LR-binary analysis."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analyze_restaurant import (
    scrape_restaurant_full, _run_inference, build_analysis_df,
    weekly_summary, plot_weekly_trend, plot_sentiment_pie, print_report,
    OUTPUT_DIR,
)
import re
import pandas as pd

URL = "https://maps.app.goo.gl/GnGWYuEcMzzWe3H27?g_st=il"
TARGET = 2834
MODEL = "bert-lr-binary"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Multi-pass scrape
restaurant_name, reviews = scrape_restaurant_full(URL, target_count=TARGET)

if not reviews:
    print("No reviews found!")
    sys.exit(1)

# Save raw scraped data immediately (in case inference fails)
safe_name = re.sub(r'[^\w\u4e00-\u9fff]', '_', restaurant_name)[:30]
raw_csv = os.path.join(OUTPUT_DIR, f"raw_full_{safe_name}.csv")
df_raw = pd.DataFrame(reviews)
df_raw.to_csv(raw_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved raw data: {raw_csv} ({len(reviews)} reviews)")

# 2. Inference
texts = [r["review_text"] for r in reviews]
predictions, probabilities, is_binary = _run_inference(texts, MODEL)

# 3. Build analysis
df = build_analysis_df(reviews, predictions, probabilities, binary=is_binary)
weekly_df = weekly_summary(df, binary=is_binary)

# 4. Save
csv_path = os.path.join(OUTPUT_DIR, f"analysis_full_{safe_name}.csv")
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Saved analysis: {csv_path}")

if not weekly_df.empty:
    weekly_csv = os.path.join(OUTPUT_DIR, f"weekly_full_{safe_name}.csv")
    weekly_df.to_csv(weekly_csv, index=False, encoding="utf-8-sig")

# 5. Plots
trend_path = os.path.join(OUTPUT_DIR, f"trend_full_{safe_name}.png")
plot_weekly_trend(weekly_df, restaurant_name, trend_path)
pie_path = os.path.join(OUTPUT_DIR, f"pie_full_{safe_name}.png")
plot_sentiment_pie(df, restaurant_name, pie_path, binary=is_binary)

# 6. Report
print_report(restaurant_name, df, weekly_df, binary=is_binary)
