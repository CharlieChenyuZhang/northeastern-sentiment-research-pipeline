#!/usr/bin/env python3
"""
Step 1 — Discover and scrape company news articles.

Uses Firecrawl /search for URL discovery and /scrape for content extraction.
Falls back to SerpAPI for discovery if available.
Outputs articles_raw.csv with one row per article.

Usage:
    python search_and_scrape.py
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config


# ---------------------------------------------------------------------------
# URL Discovery
# ---------------------------------------------------------------------------

def discover_urls_firecrawl(query: str, limit: int) -> list[str]:
    """Search via Firecrawl /search endpoint."""
    if not config.FIRECRAWL_API_KEY:
        sys.stderr.write("[ERROR] FIRECRAWL_API_KEY not set.\n")
        return []

    headers = {
        "Authorization": f"Bearer {config.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{config.FIRECRAWL_BASE_URL}/search",
            headers=headers,
            json={"query": query, "limit": limit},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [item["url"] for item in data if item.get("url")]
    except Exception as e:
        sys.stderr.write(f"[ERROR] Firecrawl search failed: {e}\n")
        return []


def discover_urls_serpapi(query: str, limit: int) -> list[str]:
    """Search Google via SerpAPI and return up to `limit` URLs."""
    if not config.SERPAPI_API_KEY:
        return []

    try:
        from serpapi import GoogleSearch
    except ImportError:
        sys.stderr.write("[WARN] serpapi not installed, skipping SerpAPI.\n")
        return []

    all_links: list[str] = []
    for start in range(0, limit, 100):
        params = {
            "engine": "google",
            "q": query,
            "api_key": config.SERPAPI_API_KEY,
            "start": start,
            "num": min(100, limit - start),
        }
        try:
            results = GoogleSearch(params).get_dict()
            if "error" in results:
                sys.stderr.write(f"[WARN] SerpAPI error: {results['error']}\n")
                break
            links = [r["link"] for r in results.get("organic_results", [])]
            if not links:
                break
            all_links.extend(links)
            if len(all_links) >= limit:
                break
        except Exception as e:
            sys.stderr.write(f"[ERROR] SerpAPI failed at start={start}: {e}\n")
            break
    return all_links[:limit]


def discover_urls(query: str, limit: int) -> list[str]:
    """Try SerpAPI first; fall back to Firecrawl /search."""
    urls = discover_urls_serpapi(query, limit)
    if urls:
        return urls
    return discover_urls_firecrawl(query, limit)


# ---------------------------------------------------------------------------
# Article Scraping
# ---------------------------------------------------------------------------

def scrape_article(url: str) -> dict | None:
    """Call Firecrawl /scrape to extract article content from a URL."""
    if not config.FIRECRAWL_API_KEY:
        sys.stderr.write("[ERROR] FIRECRAWL_API_KEY not set.\n")
        return None

    headers = {
        "Authorization": f"Bearer {config.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["json"],
        "jsonOptions": {
            "prompt": config.FIRECRAWL_EXTRACTION_PROMPT,
        },
    }
    try:
        resp = requests.post(
            f"{config.FIRECRAWL_BASE_URL}/scrape",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {}).get("json", {})
        article_text = (data.get("article_text") or "").strip()
        if not article_text:
            return None
        return {
            "title": (data.get("title") or "n/a").strip(),
            "article_text": article_text[: config.MAX_ARTICLE_CHARS],
            "published_date": (data.get("published_date") or "n/a").strip(),
            "author": (data.get("author") or "n/a").strip(),
        }
    except requests.HTTPError as e:
        sys.stderr.write(f"[WARN] {url} -> HTTP {e.response.status_code}\n")
    except Exception as e:
        sys.stderr.write(f"[WARN] {url} -> {e}\n")
    return None


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def load_visited_urls(csv_path: str) -> set[str]:
    """Load URLs already present in the output CSV for resumability."""
    visited: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row.get("url", "").strip()
                if url:
                    visited.add(url)
    return visited


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = config.ARTICLES_RAW_CSV
    visited = load_visited_urls(out_path)
    sys.stderr.write(f"[INFO] {len(visited)} URLs already scraped (resuming).\n")

    write_header = not os.path.exists(out_path)
    fieldnames = config.RAW_COLUMNS

    for company in config.COMPANIES:
        for query_tmpl in config.SEARCH_QUERIES_PER_COMPANY:
            query = query_tmpl.format(company=company)
            sys.stderr.write(f"[INFO] Searching: '{query}'\n")
            urls = discover_urls(query, config.MAX_SEARCH_RESULTS)
            new_urls = [u for u in urls if u not in visited]
            sys.stderr.write(
                f"[INFO] Found {len(urls)} URLs, {len(new_urls)} new.\n"
            )

            with ThreadPoolExecutor(max_workers=config.SCRAPE_WORKERS) as pool:
                future_to_url = {
                    pool.submit(scrape_article, url): url for url in new_urls
                }
                for future in as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        if result is None:
                            continue
                        row = {
                            "company": company,
                            "url": url,
                            "title": result["title"],
                            "article_text": result["article_text"],
                            "published_date": result["published_date"],
                            "author": result["author"],
                            "search_query": query,
                        }
                        # Append immediately for crash-safety
                        with open(out_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            if write_header:
                                writer.writeheader()
                                write_header = False
                            writer.writerow(row)
                        visited.add(url)
                        sys.stderr.write(f"[OK]   {url}\n")
                    except Exception as e:
                        sys.stderr.write(f"[WARN] {url} -> {e}\n")

    sys.stderr.write(f"[INFO] Done. {len(visited)} total articles in {out_path}\n")


if __name__ == "__main__":
    main()
