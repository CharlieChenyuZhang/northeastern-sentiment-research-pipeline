#!/usr/bin/env python3
"""
Step 1 (Massive) — Discover and scrape company news articles.

This script uses Massive's stock news endpoint for discovery and stores the
returned metadata directly, without scraping article pages.

Outputs articles_raw_massive.csv with one row per article.

Usage:
    python search_and_scrape_massive.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import requests

import config
from search_and_scrape import in_target_window, retry_delay_seconds


MASSIVE_BASE_URL = "https://api.massive.com"
MASSIVE_RESULTS_PER_PAGE = 1000
MASSIVE_MAX_RETRIES = 4
MASSIVE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Keep the ticker mapping local to this script for now so the existing SerpAPI
# configuration stays untouched.
COMPANY_TICKERS = {
    "JPMorgan Chase": "JPM",
}

MASSIVE_OUTPUT_COLUMNS = [
    "company",
    "article_url",
    "title",
    "author",
    "description",
    "published_utc",
    "publisher",
    "keywords",
    "tickers",
    "sentiment_insights",
    "search_query",
]


def get_company_ticker(company: str) -> str:
    """Return the stock ticker used for Massive news discovery."""
    ticker = COMPANY_TICKERS.get(company, "").strip()
    if not ticker:
        raise KeyError(f"No Massive ticker mapping configured for {company!r}")
    return ticker


def build_query_label(ticker: str) -> str:
    """Return a human-readable query label stored in the output CSV."""
    start_date = config.TARGET_START_DATE.isoformat()
    end_date = config.TARGET_END_DATE.isoformat()
    return (
        f"Massive ticker={ticker} "
        f"published_utc.gte={start_date} "
        f"published_utc.lte={end_date}"
    )


def massive_headers() -> dict[str, str]:
    """Return headers for Massive REST requests."""
    return {
        "Authorization": f"Bearer {config.MASSIVE_API_KEY}",
        "Accept": "application/json",
    }


def fetch_massive_news_page(
    url: str,
    params: dict[str, str | int] | None = None,
) -> dict:
    """Fetch one page from Massive's news endpoint with basic retries."""
    if not config.MASSIVE_API_KEY:
        raise RuntimeError("MASSIVE_API_KEY not set.")

    for attempt_number in range(1, MASSIVE_MAX_RETRIES + 2):
        try:
            resp = requests.get(
                url,
                headers=massive_headers(),
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            response = e.response
            status_code = response.status_code if response is not None else "unknown"
            should_retry = (
                response is not None
                and status_code in MASSIVE_RETRYABLE_STATUS_CODES
                and attempt_number <= MASSIVE_MAX_RETRIES
            )
            if should_retry:
                delay = retry_delay_seconds(response, attempt_number)
                sys.stderr.write(
                    f"[WARN] Massive news -> HTTP {status_code}; retrying in "
                    f"{delay:.1f}s ({attempt_number}/{MASSIVE_MAX_RETRIES})\n"
                )
                time.sleep(delay)
                continue
            raise
        except requests.RequestException as e:
            should_retry = attempt_number <= MASSIVE_MAX_RETRIES
            if should_retry:
                delay = retry_delay_seconds(None, attempt_number)
                sys.stderr.write(
                    f"[WARN] Massive news -> {e}; retrying in {delay:.1f}s "
                    f"({attempt_number}/{MASSIVE_MAX_RETRIES})\n"
                )
                time.sleep(delay)
                continue
            raise

    raise RuntimeError("Massive news request failed after retries.")


def json_cell(value: object) -> str:
    """Serialize list/dict metadata for stable CSV storage."""
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def load_visited_article_urls(csv_path: str) -> set[str]:
    """Load already-written article URLs for resumability."""
    visited: set[str] = set()
    if not os.path.exists(csv_path):
        return visited

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            article_url = (row.get("article_url") or row.get("url") or "").strip()
            if article_url:
                visited.add(article_url)
    return visited


def discover_articles_massive(ticker: str) -> list[dict[str, object]]:
    """Fetch article metadata for the configured year from Massive news."""
    if not config.MASSIVE_API_KEY:
        sys.stderr.write("[ERROR] MASSIVE_API_KEY not set.\n")
        return []

    start_date = config.TARGET_START_DATE.isoformat()
    end_date = config.TARGET_END_DATE.isoformat()
    next_url = f"{MASSIVE_BASE_URL}/v2/reference/news"
    params: dict[str, str | int] | None = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "sort": "published_utc",
        "order": "asc",
        "limit": MASSIVE_RESULTS_PER_PAGE,
    }

    all_articles: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_next_urls: set[str] = set()
    page_number = 1

    while next_url:
        if next_url in seen_next_urls:
            sys.stderr.write(
                f"[WARN] Massive pagination repeated next_url on page {page_number}; "
                "stopping.\n"
            )
            break
        seen_next_urls.add(next_url)

        try:
            payload = fetch_massive_news_page(next_url, params=params)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Massive news request failed: {e}\n")
            break

        results = payload.get("results", []) or []
        page_new = 0
        for article in results:
            article_url = (article.get("article_url") or "").strip()
            if not article_url or article_url in seen_urls:
                continue
            seen_urls.add(article_url)
            all_articles.append(
                {
                    "article_url": article_url,
                    "published_utc": (article.get("published_utc") or "").strip(),
                    "title": (article.get("title") or "").strip(),
                    "author": (article.get("author") or "").strip(),
                    "description": (article.get("description") or "").strip(),
                    "publisher": article.get("publisher") or {},
                    "keywords": article.get("keywords") or [],
                    "tickers": article.get("tickers") or [],
                    "sentiment_insights": article.get("insights") or [],
                }
            )
            page_new += 1

        sys.stderr.write(
            f"[INFO]   Massive page {page_number}: +{page_new} URLs "
            f"({len(all_articles)} total)\n"
        )

        next_url = payload.get("next_url") or ""
        params = None
        page_number += 1

        if not results:
            break

    return all_articles


def main() -> None:
    out_path = config.output_path("articles_raw_massive.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    visited = load_visited_article_urls(out_path)
    sys.stderr.write(f"[INFO] {len(visited)} URLs already scraped (resuming).\n")

    write_header = not os.path.exists(out_path)
    fieldnames = MASSIVE_OUTPUT_COLUMNS

    for company in config.COMPANIES:
        try:
            ticker = get_company_ticker(company)
        except KeyError as e:
            sys.stderr.write(f"[WARN] {e}\n")
            continue

        query_label = build_query_label(ticker)
        sys.stderr.write(
            f"[INFO] Searching Massive: company='{company}' ticker='{ticker}' "
            f"range={config.TARGET_START_DATE.isoformat()}.."
            f"{config.TARGET_END_DATE.isoformat()}\n"
        )

        articles = discover_articles_massive(ticker)
        in_range_articles = [
            article
            for article in articles
            if in_target_window(str(article["published_utc"]))
        ]
        new_articles = [
            article
            for article in in_range_articles
            if str(article["article_url"]) not in visited
        ]
        sys.stderr.write(
            f"[INFO] Total: {len(articles)} URLs, "
            f"{len(in_range_articles)} in range, {len(new_articles)} new.\n"
        )

        for article in new_articles:
            article_url = str(article["article_url"])
            row = {
                "company": company,
                "article_url": article_url,
                "title": str(article["title"] or "n/a"),
                "author": str(article["author"] or "n/a"),
                "description": str(article["description"] or ""),
                "published_utc": str(article["published_utc"] or ""),
                "publisher": json_cell(article["publisher"]),
                "keywords": json_cell(article["keywords"]),
                "tickers": json_cell(article["tickers"]),
                "sentiment_insights": json_cell(article["sentiment_insights"]),
                "search_query": query_label,
            }
            with open(out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                    write_header = False
                writer.writerow(row)
            visited.add(article_url)
            sys.stderr.write(f"[OK]   {article_url}\n")

    sys.stderr.write(f"[INFO] Done. {len(visited)} total articles in {out_path}\n")


if __name__ == "__main__":
    main()
