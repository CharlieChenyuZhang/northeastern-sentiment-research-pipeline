#!/usr/bin/env python3
"""
Step 2a (Massive) — Simple sentiment analysis for title and description.

Reads articles_raw_massive.csv, makes one OpenAI call per row, and writes
sentiment_massive_simple.csv with all original columns plus:
  - sentiment_title
  - sentiment_description
  - prompt_used

Usage:
    python sentiment_massive_simple.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import config

client: OpenAI | None = None
MAX_RETRIES = 5
ADDED_COLUMNS = ["sentiment_title", "sentiment_description", "prompt_used"]


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    return client


def call_llm_json(title: str, description: str, _retries: int = 0) -> dict:
    """Return title and description sentiment labels as JSON."""
    user_payload = json.dumps(
        {
            "title": title,
            "description": description,
        },
        ensure_ascii=True,
    )

    try:
        resp = get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": config.PROMPT_MASSIVE_SENTIMENT_SIMPLE,
                },
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            max_tokens=120,
            temperature=config.OPENAI_TEMPERATURE,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        if _retries >= MAX_RETRIES:
            sys.stderr.write(f"[ERROR] OpenAI failed after {MAX_RETRIES} retries: {e}\n")
            return {}
        sys.stderr.write(f"[WARN] OpenAI error: {e}. Retrying in 10s...\n")
        time.sleep(10)
        return call_llm_json(title, description, _retries + 1)


def analyze_row(row: dict[str, str]) -> dict[str, str]:
    """Analyze title and description sentiment for one Massive news row."""
    title = (row.get("title") or "").strip()
    description = (row.get("description") or "").strip()

    if not title and not description:
        result = {}
    else:
        result = call_llm_json(title, description)

    return {
        **row,
        "sentiment_title": result.get("sentiment_title", ""),
        "sentiment_description": result.get("sentiment_description", ""),
        "prompt_used": config.PROMPT_MASSIVE_SENTIMENT_SIMPLE,
    }


def load_done_article_urls(csv_path: str) -> set[str]:
    """Load already-analyzed Massive article URLs for resumability."""
    done: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                article_url = (row.get("article_url") or "").strip()
                if article_url:
                    done.add(article_url)
    return done


def main() -> None:
    if not config.OPENAI_API_KEY:
        sys.exit("[ERROR] OPENAI_API_KEY not set.")

    in_path = config.ARTICLES_RAW_MASSIVE_CSV
    out_path = config.SENTIMENT_MASSIVE_SIMPLE_CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not os.path.exists(in_path):
        sys.exit(
            f"[ERROR] {in_path} not found. Run search_and_scrape_massive.py first."
        )

    with open(in_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_columns = reader.fieldnames or []

    sys.stderr.write(f"[INFO] Loaded {len(rows)} articles from {in_path}\n")

    done_article_urls = load_done_article_urls(out_path)
    rows = [
        row
        for row in rows
        if (row.get("article_url") or "").strip() not in done_article_urls
    ]
    sys.stderr.write(f"[INFO] {len(rows)} articles remaining to analyze.\n")

    out_columns = input_columns + [
        col for col in ADDED_COLUMNS if col not in input_columns
    ]
    write_header = not os.path.exists(out_path)

    with ThreadPoolExecutor(max_workers=config.ANALYSIS_WORKERS) as pool:
        futures = {
            pool.submit(analyze_row, row): (row.get("article_url") or "").strip()
            for row in rows
        }
        for i, future in enumerate(as_completed(futures), 1):
            article_url = futures[future]
            try:
                result = future.result()
                with open(out_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=out_columns)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)
                sys.stderr.write(f"[OK]   ({i}/{len(rows)}) {article_url}\n")
            except Exception as e:
                sys.stderr.write(f"[WARN] ({i}/{len(rows)}) {article_url} -> {e}\n")

    sys.stderr.write(f"[INFO] Massive simple analysis complete. Output: {out_path}\n")


if __name__ == "__main__":
    main()
