"""Multi-pass scraping with crash recovery. Merges with existing data."""

import sys
import os
import re
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from analyze_restaurant import (
    scrape_restaurant, OUTPUT_DIR,
)

URL = "https://maps.app.goo.gl/GnGWYuEcMzzWe3H27?g_st=il"
SAFE_NAME = "老鐵醬平價鐵板燒"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"raw_full_{SAFE_NAME}.csv")


def load_existing():
    if os.path.exists(OUTPUT_CSV):
        df = pd.read_csv(OUTPUT_CSV)
        reviews = df.to_dict("records")
        texts = set(df["review_text"].str.strip())
        print(f"Loaded {len(texts)} existing reviews", flush=True)
        return reviews, texts
    return [], set()


def save(all_reviews):
    df = pd.DataFrame(all_reviews)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {len(all_reviews)} reviews → {OUTPUT_CSV}", flush=True)


def run_pass(pass_name, sort_order, all_reviews, seen_texts, **kwargs):
    """Run one scraping pass with crash recovery."""
    print(f"\n{'='*50}", flush=True)
    print(f"Pass: {pass_name} (sort={sort_order})", flush=True)
    print(f"{'='*50}", flush=True)

    try:
        name, reviews = scrape_restaurant(
            URL, max_reviews=3000,
            sort_order=sort_order,
            max_stale=kwargs.get("max_stale", 25),
            stale_wait=kwargs.get("stale_wait", 3.0),
        )

        new_count = 0
        for r in reviews:
            text = r["review_text"].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                all_reviews.append(r)
                new_count += 1

        print(f"  Got {len(reviews)}, {new_count} new (total: {len(all_reviews)})", flush=True)

    except Exception as e:
        print(f"  CRASH: {e}", flush=True)
        print(f"  Saving progress and continuing...", flush=True)

    # Always save after each pass
    save(all_reviews)
    return all_reviews, seen_texts


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_reviews, seen_texts = load_existing()

    passes = [
        ("newest-1",  "newest",  {"max_stale": 30, "stale_wait": 3.0}),
        ("relevant-1", "relevant", {"max_stale": 25, "stale_wait": 3.0}),
        ("lowest-1",  "lowest",  {"max_stale": 25, "stale_wait": 3.0}),
        ("highest-1", "highest", {"max_stale": 25, "stale_wait": 3.0}),
        ("newest-2",  "newest",  {"max_stale": 35, "stale_wait": 4.0}),
        ("relevant-2", "relevant", {"max_stale": 30, "stale_wait": 4.0}),
    ]

    for pass_name, sort_order, kwargs in passes:
        before = len(all_reviews)
        all_reviews, seen_texts = run_pass(
            pass_name, sort_order, all_reviews, seen_texts, **kwargs
        )
        gained = len(all_reviews) - before

        # If a pass adds 0 new reviews, skip remaining similar passes
        if gained == 0:
            print(f"  No new reviews, likely saturated for this sort order.", flush=True)

    # Final summary
    print(f"\n{'='*50}", flush=True)
    print(f"  FINAL: {len(all_reviews)} unique reviews with text", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
