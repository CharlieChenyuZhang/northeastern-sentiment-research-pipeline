#!/usr/bin/env python3
"""
Step 2a — Simple sentiment & emotion analysis via a single LLM call per article.

Reads articles_raw.csv, makes ONE OpenAI call per article (JSON mode),
and writes sentiment_simple.csv with all original columns plus analysis
results and the exact prompt used.

Usage:
    python sentiment_simple.py
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


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    return client


def call_llm_json(article_text: str, _retries: int = 0) -> dict:
    """Single OpenAI call returning all 4 dimensions as JSON."""
    try:
        resp = get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": config.PROMPT_ANALYSIS},
                {"role": "user", "content": article_text},
            ],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=config.OPENAI_TEMPERATURE,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        if _retries >= MAX_RETRIES:
            sys.stderr.write(f"[ERROR] OpenAI failed after {MAX_RETRIES} retries: {e}\n")
            return {}
        sys.stderr.write(f"[WARN] OpenAI error: {e}. Retrying in 10s...\n")
        time.sleep(10)
        return call_llm_json(article_text, _retries + 1)


def _list_to_str(val) -> str:
    """Convert a list of emotions to comma-separated string."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val else ""


def analyze_article(row: dict) -> dict:
    """Run the single-call JSON analysis on one article."""
    text = row["article_text"][: config.MAX_ARTICLE_CHARS]
    result = call_llm_json(text)

    return {
        **row,
        "simple_article_sentiment": result.get("article_sentiment", ""),
        "simple_reader_sentiment": result.get("reader_sentiment", ""),
        "simple_article_emotions": _list_to_str(result.get("article_emotions", [])),
        "simple_reader_emotions": _list_to_str(result.get("reader_emotions", [])),
        "prompt_used": config.PROMPT_ANALYSIS,
    }


def load_done_urls(csv_path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r.get("url", ""))
    return done


OUT_COLUMNS = config.RAW_COLUMNS + [
    "simple_article_sentiment",
    "simple_reader_sentiment",
    "simple_article_emotions",
    "simple_reader_emotions",
    "prompt_used",
]


def main() -> None:
    if not config.OPENAI_API_KEY:
        sys.exit("[ERROR] OPENAI_API_KEY not set.")

    in_path = config.ARTICLES_RAW_CSV
    out_path = config.SENTIMENT_SIMPLE_CSV

    if not os.path.exists(in_path):
        sys.exit(f"[ERROR] {in_path} not found. Run search_and_scrape.py first.")

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sys.stderr.write(f"[INFO] Loaded {len(rows)} articles from {in_path}\n")

    done_urls = load_done_urls(out_path)
    rows = [r for r in rows if r["url"] not in done_urls]
    sys.stderr.write(f"[INFO] {len(rows)} articles remaining to analyze.\n")

    write_header = not os.path.exists(out_path)

    with ThreadPoolExecutor(max_workers=config.ANALYSIS_WORKERS) as pool:
        futures = {pool.submit(analyze_article, r): r["url"] for r in rows}
        for i, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                result = future.result()
                with open(out_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)
                sys.stderr.write(f"[OK]   ({i}/{len(rows)}) {url}\n")
            except Exception as e:
                sys.stderr.write(f"[WARN] ({i}/{len(rows)}) {url} -> {e}\n")

    sys.stderr.write(f"[INFO] Simple analysis complete. Output: {out_path}\n")


if __name__ == "__main__":
    main()
