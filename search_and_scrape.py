#!/usr/bin/env python3
"""
Step 1 — Discover and scrape company news articles.

Uses three discovery methods (in order):
  1. SerpAPI Google News — news-specific results with dates
  2. SerpAPI Google Search — general web results
  3. Firecrawl /search — fallback if SerpAPI unavailable

Then scrapes each URL via Firecrawl /scrape for content extraction.
Outputs articles_raw.csv with one row per article.

Usage:
    python search_and_scrape.py
"""

from __future__ import annotations

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
import pandas as pd

import config


# ---------------------------------------------------------------------------
# URL Discovery
# ---------------------------------------------------------------------------

def discover_urls_google_news(query: str, limit: int) -> list[str]:
    """Search Google News via SerpAPI and paginate to collect more URLs."""
    if not config.SERPAPI_API_KEY:
        return []

    try:
        from serpapi import GoogleSearch
    except ImportError:
        return []

    all_links: list[str] = []
    seen: set[str] = set()

    for start in range(0, limit, config.SERPAPI_RESULTS_PER_PAGE):
        params = {
            "engine": "google_news",
            "q": query,
            "api_key": config.SERPAPI_API_KEY,
            "start": start,
        }
        try:
            results = GoogleSearch(params).get_dict()
            if "error" in results:
                sys.stderr.write(f"[WARN] SerpAPI News error: {results['error']}\n")
                break

            page_links: list[str] = []
            for article in results.get("news_results", []):
                link = article.get("link")
                if link:
                    page_links.append(link)
                # Also collect sub-stories in highlight cards.
                for sub in article.get("stories", []):
                    sub_link = sub.get("link")
                    if sub_link:
                        page_links.append(sub_link)

            new_links = [link for link in page_links if link not in seen]
            if not new_links:
                break

            for link in new_links:
                seen.add(link)
                all_links.append(link)

            if len(all_links) >= limit:
                break
        except Exception as e:
            sys.stderr.write(f"[ERROR] SerpAPI News failed at start={start}: {e}\n")
            break

    return all_links[:limit]


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
    for start in range(0, limit, config.SERPAPI_RESULTS_PER_PAGE):
        params = {
            "engine": "google",
            "q": query,
            "api_key": config.SERPAPI_API_KEY,
            "start": start,
            "num": min(config.SERPAPI_RESULTS_PER_PAGE, limit - start),
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


def discover_urls_firecrawl(query: str, limit: int) -> list[str]:
    """Search via Firecrawl /search endpoint."""
    if not config.FIRECRAWL_API_KEY:
        sys.stderr.write("[ERROR] FIRECRAWL_API_KEY not set.\n")
        return []

    headers = {
        "Authorization": f"Bearer {config.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "limit": limit,
        "sources": config.FIRECRAWL_SEARCH_SOURCES,
    }
    try:
        resp = requests.post(
            f"{config.FIRECRAWL_BASE_URL}/search",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code >= 400 and "sources" in payload:
            # Older Firecrawl endpoints may reject `sources`; retry with the
            # simpler payload rather than dropping the provider entirely.
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


def discover_urls(query: str, limit: int) -> list[str]:
    """
    Combine results from all available sources, deduplicated.
    Google News first (best for news articles), then regular search,
    then Firecrawl.
    """
    seen: set[str] = set()
    all_urls: list[str] = []

    for fn, label in [
        (discover_urls_google_news, "Google News"),
        (discover_urls_serpapi, "Google Search"),
        (discover_urls_firecrawl, "Firecrawl"),
    ]:
        urls = fn(query, config.MAX_RESULTS_PER_SOURCE)
        new = [u for u in urls if u not in seen]
        if new:
            sys.stderr.write(f"[INFO]   {label}: +{len(new)} URLs\n")
            for u in new:
                seen.add(u)
                all_urls.append(u)

    return all_urls[:limit]


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


def parse_published_date(raw: str) -> datetime | None:
    """Try common date formats; return None on failure."""
    raw = raw.strip()
    if not raw or raw.lower() == "n/a":
        return None

    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except (ValueError, OverflowError):
            continue

    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def in_target_window(published_date: str) -> bool:
    """Keep only articles whose published date falls within the research window."""
    parsed = parse_published_date(published_date)
    if parsed is None:
        return False
    return config.TARGET_START_DATE <= parsed.date() <= config.TARGET_END_DATE


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
            search_term = config.COMPANY_SEARCH_TERMS.get(company, company)
            query = query_tmpl.format(
                company=company,
                search_term=search_term,
                year=config.TARGET_YEAR,
            )
            sys.stderr.write(f"[INFO] Searching: '{query}'\n")
            urls = discover_urls(query, config.MAX_SEARCH_RESULTS)
            new_urls = [u for u in urls if u not in visited]
            sys.stderr.write(
                f"[INFO] Total: {len(urls)} URLs, {len(new_urls)} new.\n"
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
                        if not in_target_window(result["published_date"]):
                            sys.stderr.write(
                                "[INFO] Skipping out-of-range or undated article: "
                                f"{url}\n"
                            )
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
