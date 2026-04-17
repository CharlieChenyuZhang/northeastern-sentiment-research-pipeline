#!/usr/bin/env python3
"""
Analyze article-source distribution and monthly source counts with Plotly.

Reads an extracted `articles_raw.csv`, derives a source label from each URL,
and writes:
  1. CSV summaries for overall source counts and monthly source counts
  2. Plotly HTML/PNG charts for overall source distribution
  3. Plotly HTML/PNG charts for monthly source distribution

Usage:
    python3.11 plotly_article_source_analysis.py --input yearly_runs/2024/articles_raw.csv
"""

from __future__ import annotations

import argparse
import os
from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MONTH_ORDER = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def slugify(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(" ", "_")


def write_plotly_outputs(fig: go.Figure, base_path: str) -> tuple[str, str | None]:
    html_path = f"{base_path}.html"
    png_path = f"{base_path}.png"
    fig.write_html(html_path, include_plotlyjs="cdn")
    try:
        fig.write_image(png_path, width=1400, height=800, scale=2)
    except Exception:
        png_path = None
    return html_path, png_path


def normalize_host(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower().strip()
    if not host:
        return "unknown"

    for prefix in ("www.", "m.", "amp."):
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host or "unknown"


def root_domain(host: str) -> str:
    if host == "unknown":
        return host

    parts = host.split(".")
    if len(parts) <= 2:
        return host

    # Handle common multi-part public suffixes without adding extra dependencies.
    multipart_suffixes = {
        ("co", "uk"),
        ("com", "au"),
        ("com", "br"),
        ("co", "jp"),
        ("co", "nz"),
        ("com", "cn"),
    }
    if tuple(parts[-2:]) in multipart_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def derive_source(url: str, level: str) -> str:
    host = normalize_host(url)
    if level == "host":
        return host
    if level == "domain":
        return root_domain(host)
    raise ValueError(f"Unsupported source level: {level}")


def infer_output_dir(input_path: str) -> str:
    input_dir = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(input_dir, "analysis_output")


def load_articles(input_path: str, source_level: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit(f"[ERROR] No rows found in {input_path}.")

    required_cols = {"url", "published_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(
            f"[ERROR] Missing required columns in {input_path}: {sorted(missing)}"
        )

    df = df.copy()
    df["source"] = df["url"].fillna("").map(lambda url: derive_source(url, source_level))
    df["published_dt"] = pd.to_datetime(df["published_date"], errors="coerce")
    df = df[df["published_dt"].notna()].copy()
    if df.empty:
        raise SystemExit("[ERROR] No rows had a parseable published_date.")

    df["month_num"] = df["published_dt"].dt.month
    df["month"] = pd.Categorical(
        df["published_dt"].dt.month_name(), categories=MONTH_ORDER, ordered=True
    )
    df["year_month"] = df["published_dt"].dt.strftime("%Y-%m")
    return df


def collapse_sources(df: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, list[str]]:
    top_sources = (
        df["source"]
        .value_counts()
        .head(top_n)
        .index
        .tolist()
    )
    collapsed = df.copy()
    collapsed["source_grouped"] = collapsed["source"].where(
        collapsed["source"].isin(top_sources),
        "Other",
    )
    grouped_sources = top_sources + (["Other"] if (collapsed["source_grouped"] == "Other").any() else [])
    return collapsed, grouped_sources


def build_overall_source_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby("source_grouped", as_index=False)
        .size()
        .rename(columns={"size": "article_count", "source_grouped": "source"})
        .sort_values("article_count", ascending=False)
    )
    counts["share_pct"] = (counts["article_count"] / counts["article_count"].sum() * 100).round(2)
    return counts


def build_monthly_source_counts(df: pd.DataFrame, source_order: list[str]) -> pd.DataFrame:
    monthly = (
        df.groupby(["month_num", "month", "source_grouped"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "article_count", "source_grouped": "source"})
    )

    all_months = pd.DataFrame(
        {"month_num": list(range(1, 13)), "month": pd.Categorical(MONTH_ORDER, categories=MONTH_ORDER, ordered=True)}
    )
    all_sources = pd.DataFrame({"source": source_order})
    monthly = all_months.merge(all_sources, how="cross").merge(
        monthly,
        on=["month_num", "month", "source"],
        how="left",
    )
    monthly["article_count"] = monthly["article_count"].fillna(0).astype(int)
    monthly["month_total"] = monthly.groupby("month", observed=False)["article_count"].transform("sum")
    monthly["share_within_month_pct"] = (
        monthly["article_count"] / monthly["month_total"].replace(0, pd.NA) * 100
    ).round(2)
    monthly["share_within_month_pct"] = monthly["share_within_month_pct"].fillna(0.0)
    return monthly.sort_values(["month_num", "article_count"], ascending=[True, False])


def build_overall_chart(counts_df: pd.DataFrame, title_prefix: str) -> go.Figure:
    fig = px.bar(
        counts_df,
        x="source",
        y="article_count",
        text="article_count",
        hover_data={"share_pct": ":.2f"},
        title=f"{title_prefix} — Overall Article Source Distribution",
    )
    fig.update_traces(marker_color="#1f77b4")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Source",
        yaxis_title="Number of Articles",
    )
    return fig


def build_monthly_chart(monthly_df: pd.DataFrame, title_prefix: str) -> go.Figure:
    fig = px.bar(
        monthly_df,
        x="month",
        y="article_count",
        color="source",
        category_orders={"month": MONTH_ORDER},
        hover_data={
            "month_total": True,
            "share_within_month_pct": ":.2f",
        },
        title=f"{title_prefix} — Monthly Article Counts by Source",
    )
    fig.update_layout(
        template="plotly_white",
        barmode="stack",
        xaxis_title="Month",
        yaxis_title="Number of Articles",
        legend_title="Source",
        hovermode="x unified",
    )
    totals = (
        monthly_df.loc[:, ["month", "month_total"]]
        .drop_duplicates()
        .sort_values("month")
    )
    fig.add_trace(
        go.Scatter(
            x=totals["month"],
            y=totals["month_total"],
            mode="text",
            text=totals["month_total"].astype(str),
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze extracted article sources and monthly source counts."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to articles_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSV/HTML/PNG files. Defaults to <input_dir>/analysis_output.",
    )
    parser.add_argument(
        "--source-level",
        choices=["domain", "host"],
        default="domain",
        help="Group sources by root domain or full host (default: domain).",
    )
    parser.add_argument(
        "--top-n-sources",
        type=int,
        default=12,
        help="Keep the top N sources and collapse the rest into 'Other'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or infer_output_dir(args.input)
    os.makedirs(output_dir, exist_ok=True)

    articles = load_articles(args.input, args.source_level)
    articles, source_order = collapse_sources(articles, args.top_n_sources)

    company_label = "All Companies"
    if "company" in articles.columns and articles["company"].dropna().nunique() == 1:
        company_label = str(articles["company"].dropna().iloc[0])

    title_prefix = f"{company_label} ({args.source_level})"
    stem = slugify(company_label)

    overall_counts = build_overall_source_counts(articles)
    monthly_counts = build_monthly_source_counts(articles, source_order)

    overall_csv = os.path.join(output_dir, f"{stem}_article_source_distribution.csv")
    monthly_csv = os.path.join(output_dir, f"{stem}_article_source_monthly_distribution.csv")
    overall_counts.to_csv(overall_csv, index=False)
    monthly_counts.to_csv(monthly_csv, index=False)

    overall_fig = build_overall_chart(overall_counts, title_prefix)
    monthly_fig = build_monthly_chart(monthly_counts, title_prefix)

    overall_html, overall_png = write_plotly_outputs(
        overall_fig,
        os.path.join(output_dir, f"{stem}_article_source_distribution"),
    )
    monthly_html, monthly_png = write_plotly_outputs(
        monthly_fig,
        os.path.join(output_dir, f"{stem}_article_source_monthly_distribution"),
    )

    print(f"[OK] overall source CSV: {overall_csv}")
    print(f"[OK] monthly source CSV: {monthly_csv}")
    print(f"[OK] overall source HTML: {overall_html}")
    if overall_png:
        print(f"[OK] overall source PNG: {overall_png}")
    else:
        print("[WARN] overall source PNG not written (Plotly image export unavailable).")
    print(f"[OK] monthly source HTML: {monthly_html}")
    if monthly_png:
        print(f"[OK] monthly source PNG: {monthly_png}")
    else:
        print("[WARN] monthly source PNG not written (Plotly image export unavailable).")


if __name__ == "__main__":
    main()
