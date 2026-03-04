"""Fetch image URLs from Polygon and download thumbnails for analyzed articles.

Step 1: Query Polygon news API to get image_url for articles (free, rate limited)
Step 2: Download images and create 300px-wide thumbnails (pure HTTP, no API)

Usage: python -m backend.fetch_images
"""

import io
import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from backend.config import settings
from backend.database import get_conn
from backend.polygon.client import http_get, BASE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend" / "static" / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

THUMB_WIDTH = 300
JPEG_QUALITY = 75

# Rate limiting for Polygon
REQUEST_TIMES = []
MAX_PER_MIN = 5


def rate_limit():
    global REQUEST_TIMES
    now = time.time()
    REQUEST_TIMES = [t for t in REQUEST_TIMES if now - t < 60]
    if len(REQUEST_TIMES) >= MAX_PER_MIN:
        wait = 60 - (now - REQUEST_TIMES[0]) + 1
        if wait > 0:
            print(f"    Rate limit: waiting {wait:.0f}s...")
            time.sleep(wait)
    REQUEST_TIMES.append(time.time())


def step1_fetch_image_urls():
    """Query Polygon to get image_url for articles that don't have one yet."""
    conn = get_conn()

    tickers = conn.execute("""
        SELECT DISTINCT l1.symbol
        FROM layer1_results l1
        WHERE l1.relevance = 'relevant'
        ORDER BY l1.symbol
    """).fetchall()

    total_updated = 0

    for t in tickers:
        symbol = t["symbol"]

        missing = conn.execute("""
            SELECT count(*) FROM news_raw nr
            JOIN layer1_results l1 ON nr.id = l1.news_id AND l1.symbol = ?
            WHERE l1.relevance = 'relevant' AND nr.image_url IS NULL
        """, (symbol,)).fetchone()[0]

        if missing == 0:
            continue

        print(f"[{symbol}] {missing} articles missing image_url")

        url = f"{BASE}/v2/reference/news"
        params = {
            "ticker": symbol,
            "published_utc.gte": "2024-01-01",
            "published_utc.lte": "2026-12-31",
            "limit": 50,
            "order": "asc",
        }
        next_url = None
        updated = 0

        while True:
            rate_limit()
            try:
                resp = http_get(next_url or url, params=None if next_url else params)
            except Exception as e:
                print(f"  Error: {e}")
                break

            data = resp.json()
            results = data.get("results") or []
            if not results:
                break

            for r in results:
                rid = r.get("id")
                img = r.get("image_url")
                if rid and img:
                    res = conn.execute(
                        "UPDATE news_raw SET image_url = ? WHERE id = ? AND image_url IS NULL",
                        (img, rid),
                    )
                    if res.rowcount > 0:
                        updated += 1

            conn.commit()
            next_url = data.get("next_url")
            if not next_url:
                break

        total_updated += updated
        if updated:
            print(f"  Updated {updated} image URLs")

    conn.close()
    return total_updated


def _download_thumbnail(news_id: str, image_url: str) -> bool:
    """Download image, resize to thumbnail, save as JPEG."""
    filename = f"{news_id[:40]}.jpg"
    filepath = IMAGES_DIR / filename

    if filepath.exists():
        return True

    try:
        resp = requests.get(image_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        if resp.status_code != 200 or len(resp.content) < 500:
            return False

        img = Image.open(io.BytesIO(resp.content))
        img = img.convert("RGB")

        # Resize to THUMB_WIDTH, keep aspect ratio
        w, h = img.size
        new_w = THUMB_WIDTH
        new_h = int(h * (THUMB_WIDTH / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        img.save(filepath, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception:
        return False


def step2_download_thumbnails(max_workers: int = 10):
    """Download and create thumbnails for articles with image_url."""
    conn = get_conn()

    rows = conn.execute("""
        SELECT nr.id, nr.image_url
        FROM news_raw nr
        JOIN layer1_results l1 ON nr.id = l1.news_id
        WHERE l1.relevance = 'relevant'
        AND nr.image_url IS NOT NULL
        AND nr.image_url != ''
        GROUP BY nr.id
    """).fetchall()
    conn.close()

    to_download = []
    for r in rows:
        filepath = IMAGES_DIR / f"{r['id'][:40]}.jpg"
        if not filepath.exists():
            to_download.append((r["id"], r["image_url"]))

    print(f"\nStep 2: Download thumbnails ({THUMB_WIDTH}px wide)")
    print(f"  Total with image_url: {len(rows)}")
    print(f"  Already done: {len(rows) - len(to_download)}")
    print(f"  To download: {len(to_download)}")

    if not to_download:
        print("  Nothing to download!")
        return 0

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_thumbnail, nid, url): nid for nid, url in to_download}
        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                success += 1
            else:
                failed += 1
            if i % 200 == 0:
                print(f"  Progress: {i}/{len(to_download)} ({success} ok, {failed} fail)")

    print(f"  Done: {success} downloaded, {failed} failed")
    return success


def main():
    print("=== Fetch News Thumbnails ===\n")

    print("Step 1: Get image URLs from Polygon")
    updated = step1_fetch_image_urls()
    print(f"\nImage URLs updated: {updated}")

    downloaded = step2_download_thumbnails()
    print(f"\nThumbnails downloaded: {downloaded}")
    print(f"Stored in: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
