"""Resume scraping from Pass 5+6 only. Run with nohup."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scrape_by_star import load_existing, save, run_pass

all_reviews, seen_texts = load_existing()
print(f"Starting from {len(all_reviews)} existing reviews", flush=True)

remaining = [
    ("newest-2",  "newest",  {"max_stale": 35, "stale_wait": 4.0}),
    ("relevant-2", "relevant", {"max_stale": 30, "stale_wait": 4.0}),
]

for pass_name, sort_order, kwargs in remaining:
    before = len(all_reviews)
    all_reviews, seen_texts = run_pass(pass_name, sort_order, all_reviews, seen_texts, **kwargs)
    gained = len(all_reviews) - before
    if gained == 0:
        print(f"  No new reviews from {pass_name}, saturated.", flush=True)

print(f"\nFINAL: {len(all_reviews)} unique reviews with text", flush=True)
