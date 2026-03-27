#!/usr/bin/env python3
"""
Standalone reporting script for monthly news-result counts by source.

This script does discovery only. It does not scrape articles or run sentiment
analysis. It queries the same search providers used by the pipeline and writes:

  1. A CSV summary with counts by month and source
  2. A Plotly HTML chart of monthly article counts

Usage:
    python3.11 monthly_news_counts.py
"""

from __future__ import annotations

import argparse
import csv
import sys

import config
import search_and_scrape


SOURCE_FUNCTIONS = [
    ("Google News", search_and_scrape.discover_urls_google_news),
    ("Google Search", search_and_scrape.discover_urls_serpapi),
    ("Firecrawl", search_and_scrape.discover_urls_firecrawl),
]

CSV_FIELDNAMES = [
    "company",
    "year",
    "month",
    "query",
    "google_news_count",
    "google_search_count",
    "firecrawl_count",
    "total_raw_count",
    "total_unique_count",
    "overlap_count",
]


def build_monthly_query(company: str, month: str) -> str:
    search_term = config.COMPANY_SEARCH_TERMS.get(company, company)
    return f"{search_term} news what happened in {month} {config.TARGET_YEAR}"


def collect_month_counts(company: str) -> list[dict]:
    rows: list[dict] = []

    for month in config.TARGET_MONTHS:
        query = build_monthly_query(company, month)
        sys.stderr.write(f"[INFO] Searching monthly counts: '{query}'\n")

        per_source_urls: dict[str, list[str]] = {}
        all_urls: list[str] = []

        for label, fn in SOURCE_FUNCTIONS:
            urls = fn(query, config.MAX_RESULTS_PER_SOURCE)
            per_source_urls[label] = urls
            all_urls.extend(urls)
            sys.stderr.write(f"[INFO]   {label}: {len(urls)} URLs\n")

        unique_urls = list(dict.fromkeys(all_urls))
        row = {
            "company": company,
            "year": config.TARGET_YEAR,
            "month": month,
            "query": query,
            "google_news_count": len(per_source_urls["Google News"]),
            "google_search_count": len(per_source_urls["Google Search"]),
            "firecrawl_count": len(per_source_urls["Firecrawl"]),
            "total_raw_count": len(all_urls),
            "total_unique_count": len(unique_urls),
            "overlap_count": len(all_urls) - len(unique_urls),
        }
        rows.append(row)

    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_plot(rows: list[dict], company: str, output_path: str) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "[ERROR] plotly is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    months = [row["month"] for row in rows]
    total_unique = [row["total_unique_count"] for row in rows]
    google_news = [row["google_news_count"] for row in rows]
    google_search = [row["google_search_count"] for row in rows]
    firecrawl = [row["firecrawl_count"] for row in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=months,
            y=total_unique,
            name="Total Unique Articles",
            marker_color="#1f77b4",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=google_news,
            mode="lines+markers",
            name="Google News",
            line={"color": "#ff7f0e", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=google_search,
            mode="lines+markers",
            name="Google Search",
            line={"color": "#2ca02c", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=firecrawl,
            mode="lines+markers",
            name="Firecrawl",
            line={"color": "#d62728", "width": 2},
        )
    )

    fig.update_layout(
        title=f"{company} Monthly News Article Counts ({config.TARGET_YEAR})",
        xaxis_title="Month",
        yaxis_title="Number of News Articles",
        template="plotly_white",
        barmode="group",
        hovermode="x unified",
        legend_title="Series",
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    png_path = output_path.rsplit(".", 1)[0] + ".png"
    try:
        fig.write_image(png_path, width=1400, height=800, scale=2)
    except Exception as exc:
        sys.stderr.write(
            "[WARN] Could not write PNG chart. Install/upgrade `kaleido` via "
            "`pip install -r requirements.txt`.\n"
        )
        sys.stderr.write(f"[WARN] Plotly image export error: {exc}\n")
    else:
        sys.stderr.write(f"[OK] Wrote Plotly PNG chart: {png_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count monthly news search results by source and plot them."
    )
    parser.add_argument(
        "--company",
        default=config.COMPANIES[0],
        help="Company name to analyze (default: first configured company).",
    )
    parser.add_argument(
        "--csv-output",
        default="monthly_news_counts.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot-output",
        default="monthly_news_counts.html",
        help="Output Plotly HTML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_month_counts(args.company)

    write_csv(rows, args.csv_output)
    build_plot(rows, args.company, args.plot_output)

    sys.stderr.write(f"[OK] Wrote CSV summary: {args.csv_output}\n")
    sys.stderr.write(f"[OK] Wrote Plotly chart: {args.plot_output}\n")


if __name__ == "__main__":
    main()
