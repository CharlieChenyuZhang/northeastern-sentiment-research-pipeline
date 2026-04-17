#!/usr/bin/env python3
"""
Step 2b — Comprehensive sentiment & emotion analysis using log probabilities.

For each article this script:
  1. Makes a single LLM call (JSON mode with logprobs enabled) that returns
     sentiment distributions + emotion labels with confidence scores.
  2. Extracts token-level log probabilities from the response to derive a
     secondary confidence signal.
  3. Repeats N times (config.COMPREHENSIVE_RUNS) with temperature > 0 and
     merges results via geometric-mean confidence for robustness.

This uses 1 API call per run (N calls total per article) instead of the
previous 48+ binary-probe calls.

Outputs sentiment_comprehensive.csv.

Usage:
    python sentiment_comprehensive.py
"""

from __future__ import annotations

import csv
import json
import math
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


# ---------------------------------------------------------------------------
# Core: single-call JSON analysis with logprobs
# ---------------------------------------------------------------------------

def call_analysis_with_logprobs(
    article_text: str, _retries: int = 0
) -> tuple[dict, float]:
    """
    Make one LLM call that returns all 4 dimensions as JSON, with logprobs
    enabled. Returns (parsed_json, mean_logprob).

    The mean_logprob across all output tokens serves as a response-level
    confidence signal — higher (less negative) = more confident.
    """
    try:
        resp = get_client().chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": config.COMP_PROMPT_ANALYSIS},
                {"role": "user", "content": article_text},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=config.OPENAI_TEMPERATURE_COMPREHENSIVE,
            logprobs=True,
            top_logprobs=5,
        )

        content = resp.choices[0].message.content
        parsed = json.loads(content)

        # Compute mean log-probability across all output tokens
        token_logprobs = resp.choices[0].logprobs.content
        if token_logprobs:
            mean_lp = sum(t.logprob for t in token_logprobs) / len(token_logprobs)
        else:
            mean_lp = 0.0

        return parsed, mean_lp

    except Exception as e:
        if _retries >= MAX_RETRIES:
            sys.stderr.write(
                f"[ERROR] Analysis call failed after {MAX_RETRIES} retries: {e}\n"
            )
            return {}, -10.0
        sys.stderr.write(f"[WARN] Analysis call failed: {e}. Retrying in 10s...\n")
        time.sleep(10)
        return call_analysis_with_logprobs(article_text, _retries + 1)


# ---------------------------------------------------------------------------
# Merging multiple runs
# ---------------------------------------------------------------------------

def _normalize_dist(d: dict[str, float]) -> dict[str, float]:
    """Normalize values to sum to 1."""
    total = sum(d.values()) or 1.0
    return {k: round(v / total, 4) for k, v in d.items()}


def _ensure_sentiment_dist(raw) -> dict[str, float]:
    """
    Convert the LLM's sentiment output to a proper distribution.
    Handles both dict (confidence scores) and plain string label.
    Case-insensitive matching for robustness.
    """
    if isinstance(raw, dict):
        # Build a lowercase lookup so "positive" matches "Positive"
        lower_map = {k.strip().lower(): float(v) for k, v in raw.items()
                     if isinstance(v, (int, float)) or str(v).replace('.', '', 1).isdigit()}
        dist = {}
        for label in config.SENTIMENT_LABELS:
            dist[label] = lower_map.get(label.lower(), 0.0)
        return _normalize_dist(dist)
    # Plain label string — treat as 1.0 confidence
    label = str(raw).strip().lower()
    dist = {l: 0.0 for l in config.SENTIMENT_LABELS}
    for l in config.SENTIMENT_LABELS:
        if l.lower() == label:
            dist[l] = 1.0
            return dist
    dist["Neutral"] = 1.0
    return dist


def _ensure_emotion_dict(raw) -> dict[str, float]:
    """
    Convert the LLM's emotion output to {label: confidence}.
    Handles dict, list of strings, or comma-separated string.
    """
    if isinstance(raw, dict):
        result = {}
        for k, v in raw.items():
            try:
                fv = float(v)
                if fv > 0:
                    result[k] = fv
            except (ValueError, TypeError):
                result[k] = 1.0  # present but no numeric confidence
        return result
    if isinstance(raw, list):
        return {str(e): 1.0 for e in raw if e}
    if isinstance(raw, str) and raw:
        return {e.strip(): 1.0 for e in raw.split(",") if e.strip()}
    return {}


def merge_sentiment_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    """Geometric mean across runs (average log-probs), then re-normalize."""
    all_labels = set()
    for r in runs:
        all_labels.update(r.keys())

    merged: dict[str, float] = {}
    for label in all_labels:
        vals = [r.get(label, 1e-12) for r in runs]
        avg_lp = sum(math.log(max(v, 1e-12)) for v in vals) / len(vals)
        merged[label] = math.exp(avg_lp)
    return _normalize_dist(merged)


def merge_emotion_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    """
    Union of all emotions across runs. Confidence = geometric mean of scores
    for runs where the emotion appeared, weighted by presence fraction.
    """
    all_emotions: set[str] = set()
    for r in runs:
        all_emotions.update(r.keys())

    merged: dict[str, float] = {}
    for emo in all_emotions:
        probs = [r[emo] for r in runs if emo in r]
        if not probs:
            continue
        geo_mean = math.exp(
            sum(math.log(max(p, 1e-12)) for p in probs) / len(probs)
        )
        presence = len(probs) / len(runs)
        merged[emo] = round(geo_mean * presence, 4)

    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Analyze one article: N runs, then merge
# ---------------------------------------------------------------------------

def analyze_article(row: dict) -> dict:
    text = row["article_text"][: config.MAX_ARTICLE_CHARS]
    n_runs = config.COMPREHENSIVE_RUNS

    art_sent_runs: list[dict[str, float]] = []
    rdr_sent_runs: list[dict[str, float]] = []
    art_emo_runs: list[dict[str, float]] = []
    rdr_emo_runs: list[dict[str, float]] = []
    mean_logprobs: list[float] = []

    for _ in range(n_runs):
        result, mean_lp = call_analysis_with_logprobs(text)
        mean_logprobs.append(mean_lp)

        art_sent_runs.append(_ensure_sentiment_dist(result.get("article_sentiment", {})))
        rdr_sent_runs.append(_ensure_sentiment_dist(result.get("reader_sentiment", {})))
        art_emo_runs.append(_ensure_emotion_dict(result.get("article_emotions", {})))
        rdr_emo_runs.append(_ensure_emotion_dict(result.get("reader_emotions", {})))

    # Merge across runs
    article_sent = merge_sentiment_runs(art_sent_runs)
    reader_sent = merge_sentiment_runs(rdr_sent_runs)
    article_emo = merge_emotion_runs(art_emo_runs)
    reader_emo = merge_emotion_runs(rdr_emo_runs)

    # Pick top label + confidence
    top_art = max(article_sent, key=article_sent.get)
    top_rdr = max(reader_sent, key=reader_sent.get)

    # Overall response-level confidence from mean logprobs
    avg_response_logprob = sum(mean_logprobs) / len(mean_logprobs)

    return {
        **row,
        "comp_article_sentiment": top_art,
        "comp_article_sentiment_confidence": article_sent[top_art],
        "comp_article_sentiment_distribution": json.dumps(article_sent),
        "comp_reader_sentiment": top_rdr,
        "comp_reader_sentiment_confidence": reader_sent[top_rdr],
        "comp_reader_sentiment_distribution": json.dumps(reader_sent),
        "comp_article_emotions": json.dumps(article_emo),
        "comp_reader_emotions": json.dumps(reader_emo),
        "comp_mean_logprob": round(avg_response_logprob, 4),
        "comp_prompt_used": config.COMP_PROMPT_ANALYSIS,
    }


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

COMP_COLUMNS = config.RAW_COLUMNS + [
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


def load_done_urls(csv_path: str) -> set[str]:
    done: set[str] = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r.get("url", ""))
    return done


def main() -> None:
    if not config.OPENAI_API_KEY:
        sys.exit("[ERROR] OPENAI_API_KEY not set.")

    in_path = config.ARTICLES_RAW_CSV
    out_path = config.SENTIMENT_COMPREHENSIVE_CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not os.path.exists(in_path):
        sys.exit(f"[ERROR] {in_path} not found. Run search_and_scrape.py first.")

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sys.stderr.write(f"[INFO] Loaded {len(rows)} articles from {in_path}\n")

    done_urls = load_done_urls(out_path)
    rows = [r for r in rows if r["url"] not in done_urls]
    sys.stderr.write(f"[INFO] {len(rows)} articles remaining.\n")

    write_header = not os.path.exists(out_path)

    with ThreadPoolExecutor(max_workers=config.ANALYSIS_WORKERS) as pool:
        futures = {pool.submit(analyze_article, r): r["url"] for r in rows}
        for i, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                result = future.result()
                with open(out_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=COMP_COLUMNS)
                    if write_header:
                        writer.writeheader()
                        write_header = False
                    writer.writerow(result)
                sys.stderr.write(f"[OK]   ({i}/{len(rows)}) {url}\n")
            except Exception as e:
                sys.stderr.write(f"[WARN] ({i}/{len(rows)}) {url} -> {e}\n")

    sys.stderr.write(f"[INFO] Comprehensive analysis complete. Output: {out_path}\n")


if __name__ == "__main__":
    main()
