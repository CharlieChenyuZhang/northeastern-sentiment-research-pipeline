#!/usr/bin/env python3
"""
Step 3 — Merge simple and comprehensive analysis results into a single CSV.

Reads sentiment_simple.csv and sentiment_comprehensive.csv, joins on URL,
and writes final_results.csv with all columns from both analyses.

Usage:
    python merge_results.py
"""

from __future__ import annotations

import csv
import sys

import config


def main() -> None:
    # Load simple results keyed by URL
    simple: dict[str, dict] = {}
    try:
        with open(config.SENTIMENT_SIMPLE_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                simple[row["url"]] = row
    except FileNotFoundError:
        sys.exit(f"[ERROR] {config.SENTIMENT_SIMPLE_CSV} not found.")

    # Load comprehensive results keyed by URL
    comp: dict[str, dict] = {}
    try:
        with open(config.SENTIMENT_COMPREHENSIVE_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                comp[row["url"]] = row
    except FileNotFoundError:
        sys.exit(f"[ERROR] {config.SENTIMENT_COMPREHENSIVE_CSV} not found.")

    all_urls = set(simple.keys()) | set(comp.keys())
    sys.stderr.write(
        f"[INFO] Simple: {len(simple)}, Comprehensive: {len(comp)}, "
        f"Union: {len(all_urls)} URLs.\n"
    )

    simple_extra = [
        "simple_article_sentiment",
        "simple_reader_sentiment",
        "simple_article_emotions",
        "simple_reader_emotions",
        "prompt_used",
    ]
    comp_extra = [
        "comp_article_sentiment",
        "comp_article_sentiment_confidence",
        "comp_article_sentiment_distribution",
        "comp_reader_sentiment",
        "comp_reader_sentiment_confidence",
        "comp_reader_sentiment_distribution",
        "comp_article_emotions",
        "comp_reader_emotions",
        "comp_mean_logprob",
        "comp_prompt_used",
    ]
    fieldnames = config.RAW_COLUMNS + simple_extra + comp_extra

    with open(config.FINAL_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for url in sorted(all_urls):
            merged_row: dict[str, str] = {}
            if url in simple:
                merged_row.update(simple[url])
            if url in comp:
                for key in comp_extra + config.RAW_COLUMNS:
                    if key in comp[url] and (key not in merged_row or not merged_row[key]):
                        merged_row[key] = comp[url][key]
            writer.writerow(merged_row)

    sys.stderr.write(f"[INFO] Merged {len(all_urls)} rows -> {config.FINAL_RESULTS_CSV}\n")


if __name__ == "__main__":
    main()
