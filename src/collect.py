"""Google Maps restaurant review scraper using Playwright."""

import json
import os
import random
import re
import time
import urllib.parse

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CHECKPOINT_FILE = os.path.join(RAW_DIR, "checkpoint.json")
OUTPUT_CSV = os.path.join(RAW_DIR, "reviews.csv")

# Search queries — each yields a Google Maps result page with multiple restaurants
SEARCH_QUERIES = [
    # 台北
    "台北 牛肉麵", "台北 小籠包", "台北 滷肉飯", "台北 火鍋",
    "台北 日式拉麵", "台北 咖啡廳", "台北 早午餐", "台北 義大利麵",
    "台北 壽司", "台北 泰式料理", "台北 韓式料理", "台北 素食餐廳",
    "士林夜市 美食", "饒河街夜市 美食", "寧夏夜市 美食", "南機場夜市",
    "台北 甜點", "台北 手搖飲", "台北 便當", "台北 港式飲茶",
    # 新北
    "新北 板橋 美食", "新北 永和 豆漿", "新北 淡水 小吃",
    # 台中
    "台中 早午餐", "台中 火鍋", "逢甲夜市 美食", "台中 茶館",
    "台中 餐廳", "一中街 美食",
    # 台南
    "台南 小吃", "台南 牛肉湯", "台南 鱔魚意麵", "台南 擔仔麵",
    "台南 碗粿", "花園夜市 美食",
    # 高雄
    "高雄 海鮮", "六合夜市 美食", "高雄 旗津 小吃", "高雄 餐廳",
    "瑞豐夜市 美食",
    # 其他
    "新竹 城隍廟 小吃", "嘉義 雞肉飯", "花蓮 扁食", "宜蘭 小吃",
    "基隆 廟口 美食",
]


def _random_sleep(min_s=2, max_s=5):
    time.sleep(random.uniform(min_s, max_s))


def _load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_queries": [], "total_reviews": 0}


def _save_checkpoint(state):
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False)


def _handle_consent(page):
    """Dismiss Google consent popup if it appears."""
    try:
        consent_btn = page.locator("button:has-text('全部接受'), button:has-text('Accept all')")
        if consent_btn.count() > 0:
            consent_btn.first.click()
            _random_sleep(1, 2)
    except Exception:
        pass


def _dismiss_overlays(page):
    """Dismiss any popup overlays (info banners, disclaimers, etc.)."""
    try:
        page.keyboard.press("Escape")
        time.sleep(0.5)
    except Exception:
        pass
    # Click dismiss/close buttons on info panels
    for selector in [
        "button[aria-label='關閉']", "button[aria-label='Close']",
        "button[jsaction*='dismiss']", ".VfPpkd-icon-LgbsSe",
    ]:
        try:
            btn = page.locator(selector)
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                time.sleep(0.3)
        except Exception:
            pass


def _extract_rating(review_el):
    """Extract star rating from a review element."""
    try:
        star_el = review_el.locator("span[role='img']").first
        aria = star_el.get_attribute("aria-label") or ""
        match = re.search(r"(\d)", aria)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 0


def _extract_reviews_from_page(page, restaurant_name, area, max_reviews=50):
    """Extract reviews from the current restaurant's review panel."""
    reviews = []

    # Wait for review elements to appear
    try:
        page.wait_for_selector("div.jftiEf", timeout=8000)
    except PwTimeout:
        return reviews

    # Find the scrollable reviews panel
    scrollable = page.locator("div.m6QErb.DxyBCb.kA9KIf.dS8AEf")
    if scrollable.count() == 0:
        scrollable = page.locator("div.m6QErb.DxyBCb")
    if scrollable.count() == 0:
        # Last resort: try scrolling the main panel
        scrollable = page.locator("div[role='main']")
    if scrollable.count() == 0:
        return reviews

    scroll_el = scrollable.first

    # Scroll to load more reviews
    last_count = 0
    stale_rounds = 0
    for _ in range(max_reviews // 5 + 5):
        scroll_el.evaluate("el => el.scrollTop = el.scrollHeight")
        _random_sleep(1.5, 2.5)
        review_els = page.locator("div.jftiEf")
        current_count = review_els.count()
        if current_count >= max_reviews:
            break
        if current_count == last_count:
            stale_rounds += 1
            if stale_rounds >= 3:
                break
        else:
            stale_rounds = 0
        last_count = current_count

    # Click all "More" / "全文" buttons to expand truncated reviews
    more_buttons = page.locator("button.w8nwRe.kyuRq")
    for i in range(more_buttons.count()):
        try:
            more_buttons.nth(i).click(timeout=1000)
            time.sleep(0.2)
        except Exception:
            pass

    # Extract review data
    review_els = page.locator("div.jftiEf")
    count = min(review_els.count(), max_reviews)

    for i in range(count):
        try:
            el = review_els.nth(i)

            # Review text
            text_el = el.locator("span.wiI7pd")
            if text_el.count() == 0:
                continue
            text = text_el.first.inner_text().strip()
            if not text or len(text) < 2:
                continue

            # Rating
            rating = _extract_rating(el)

            # Date
            date_text = ""
            try:
                date_el = el.locator("span.rsqaWe")
                if date_el.count() > 0:
                    date_text = date_el.first.inner_text().strip()
            except Exception:
                pass

            # Likes
            likes = 0
            try:
                likes_el = el.locator("span.pkWtMe")
                if likes_el.count() > 0:
                    likes_text = likes_el.first.inner_text().strip()
                    if likes_text.isdigit():
                        likes = int(likes_text)
            except Exception:
                pass

            # Label from rating
            if rating <= 2:
                label = 0  # negative
            elif rating == 3:
                label = 1  # neutral
            else:
                label = 2  # positive

            reviews.append({
                "review_text": text,
                "rating": rating,
                "restaurant": restaurant_name,
                "area": area,
                "date": date_text,
                "likes": likes,
                "label": label,
            })
        except Exception:
            continue

    return reviews


def scrape_query(page, query, max_per_restaurant=50, max_restaurants=10):
    """Search Google Maps for a query and scrape reviews from results.

    Uses a 2-pass approach: collect URLs first, then visit each one.
    """
    encoded = urllib.parse.quote(query)
    url = f"https://www.google.com/maps/search/{encoded}/?hl=zh-TW"

    print(f"  Navigating to: {url}")
    page.goto(url, wait_until="domcontentloaded")
    _handle_consent(page)
    _dismiss_overlays(page)
    _random_sleep(3, 5)

    # Wait for results feed
    try:
        page.wait_for_selector("div[role='feed']", timeout=15000)
    except PwTimeout:
        print(f"  No results feed found for '{query}', skipping.")
        return []

    # Scroll the results feed to load more restaurants
    feed = page.locator("div[role='feed']").first
    for _ in range(3):
        feed.evaluate("el => el.scrollTop = el.scrollHeight")
        _random_sleep(1, 2)

    # Pass 1: Collect all restaurant URLs from the search results
    link_els = page.locator("div[role='feed'] a[href*='/maps/place/']")
    n_links = link_els.count()
    restaurant_urls = []
    for i in range(min(n_links, max_restaurants)):
        try:
            href = link_els.nth(i).get_attribute("href")
            name = link_els.nth(i).get_attribute("aria-label") or ""
            if href:
                restaurant_urls.append((href, name))
        except Exception:
            pass

    if not restaurant_urls:
        print(f"  No restaurant URLs found for '{query}'.")
        return []

    print(f"  Found {n_links} restaurants, will scrape {len(restaurant_urls)}...")

    # Extract area from query
    area = query.split()[0] if " " in query else query

    # Pass 2: Navigate to each restaurant URL directly and extract reviews
    all_reviews = []
    scraped_names = set()

    for i, (rest_url, rest_name) in enumerate(restaurant_urls):
        try:
            # Navigate directly using the full href (includes place data params)
            page.goto(rest_url, wait_until="domcontentloaded")

            # Wait for restaurant detail panel to load
            try:
                page.wait_for_selector(
                    "h1.DUwDvf, h1.fontHeadlineLarge",
                    timeout=12000,
                )
            except PwTimeout:
                _random_sleep(3, 5)

            _random_sleep(1, 2)

            # Get restaurant name
            name = rest_name  # Use aria-label as primary source
            try:
                h1 = page.locator("h1.DUwDvf, h1.fontHeadlineLarge")
                if h1.count() > 0:
                    h1_text = h1.first.inner_text().strip()
                    if h1_text and len(h1_text) > 1:
                        name = h1_text
            except Exception:
                pass

            if not name:
                continue
            if name in scraped_names:
                continue
            scraped_names.add(name)

            # Click reviews tab — it's a div.Gpq6kf, not a button
            reviews_tab = page.locator(
                "div.Gpq6kf:has-text('評論'), "
                "button[aria-label*='評論'], "
                "button[role='tab']:has-text('評論')"
            )
            if reviews_tab.count() == 0:
                print(f"    [{i+1}] {name}: no reviews tab, skipping")
                continue

            reviews_tab.first.click()
            _random_sleep(2, 3)

            # Extract reviews
            reviews = _extract_reviews_from_page(page, name, area, max_per_restaurant)
            all_reviews.extend(reviews)
            print(f"    [{i+1}] {name}: {len(reviews)} reviews")

            # Occasional longer break
            if (i + 1) % 5 == 0:
                _random_sleep(3, 6)
            else:
                _random_sleep(1, 3)

        except Exception as e:
            print(f"    [{i+1}] Error: {e}")

    return all_reviews


def scrape_all(queries=None, output_csv=None, max_per_restaurant=50, max_restaurants_per_query=10):
    """Run full scraping pipeline with checkpoint/resume."""
    if queries is None:
        queries = SEARCH_QUERIES
    if output_csv is None:
        output_csv = OUTPUT_CSV

    os.makedirs(RAW_DIR, exist_ok=True)
    checkpoint = _load_checkpoint()
    completed = set(checkpoint["completed_queries"])
    remaining = [q for q in queries if q not in completed]

    print(f"Total queries: {len(queries)}, Remaining: {len(remaining)}")
    if not remaining:
        print("All queries already scraped.")
        return

    # Load existing reviews
    all_reviews = []
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        all_reviews = existing.to_dict("records")
        print(f"Loaded {len(all_reviews)} existing reviews.")

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
            random.shuffle(remaining)
            for qi, query in enumerate(remaining):
                print(f"\n[{qi+1}/{len(remaining)}] Query: '{query}'")
                try:
                    reviews = scrape_query(
                        page, query,
                        max_per_restaurant=max_per_restaurant,
                        max_restaurants=max_restaurants_per_query,
                    )
                    all_reviews.extend(reviews)
                    print(f"  → Got {len(reviews)} reviews (total: {len(all_reviews)})")
                except Exception as e:
                    print(f"  → Query failed: {e}")

                # Save checkpoint + CSV
                checkpoint["completed_queries"].append(query)
                checkpoint["total_reviews"] = len(all_reviews)
                _save_checkpoint(checkpoint)

                df = pd.DataFrame(all_reviews)
                df.to_csv(output_csv, index=False, encoding="utf-8-sig")

                # Take breaks
                if (qi + 1) % 10 == 0:
                    pause = random.uniform(15, 30)
                    print(f"  Taking a {pause:.0f}s break...")
                    time.sleep(pause)
                else:
                    _random_sleep(3, 6)

        finally:
            browser.close()

    print(f"\nDone! Total reviews: {len(all_reviews)}")
    print(f"Saved to: {output_csv}")

    # Print label distribution
    if all_reviews:
        df = pd.DataFrame(all_reviews)
        dist = df["label"].value_counts().sort_index()
        print(f"Label distribution: {dist.to_dict()}")

    return all_reviews


if __name__ == "__main__":
    scrape_all()
