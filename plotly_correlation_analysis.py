#!/usr/bin/env python3
"""
Create interactive Plotly correlation outputs for sentiment, emotion, and stock series.

Outputs for each company:
  - aligned daily CSV
  - Pearson correlation heatmap (HTML)
  - Spearman correlation heatmap (HTML)
  - focused stock-vs-text correlation table (CSV + HTML)
  - scatter matrix (HTML)
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

import config
from stock_correlation import (
    COMPANY_TICKERS,
    daily_sentiment,
    fetch_stock_prices,
    load_articles,
)

OUTPUT_DIR = config.ANALYSIS_OUTPUT_DIR
TEXT_COLS = [
    "article_sentiment",
    "reader_sentiment",
    "article_emotion_intensity",
    "reader_emotion_intensity",
]
STOCK_COLS = ["close", "daily_return"]
ALL_COLS = TEXT_COLS + STOCK_COLS
LABELS = {
    "article_sentiment": "Article Sentiment",
    "reader_sentiment": "Reader Sentiment",
    "article_emotion_intensity": "Article Emotion Intensity",
    "reader_emotion_intensity": "Reader Emotion Intensity",
    "close": "Stock Close",
    "daily_return": "Daily Return",
}


def slugify(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(" ", "_")


def write_plotly_outputs(fig: go.Figure, base_path: str) -> tuple[str, str]:
    html_path = f"{base_path}.html"
    png_path = f"{base_path}.png"
    fig.write_html(html_path, include_plotlyjs="cdn")
    fig.write_image(png_path)
    return html_path, png_path


def build_aligned_daily(company: str, input_path: str, days_before: int, days_after: int) -> pd.DataFrame:
    articles = load_articles(input_path)
    daily = daily_sentiment(articles, company)
    if daily.empty:
        return pd.DataFrame()

    ticker = COMPANY_TICKERS.get(company)
    if not ticker:
        return pd.DataFrame()

    min_date = pd.Timestamp(min(daily.index)) - timedelta(days=days_before)
    max_date = pd.Timestamp(max(daily.index)) + timedelta(days=days_after)
    stock = fetch_stock_prices(ticker, str(min_date.date()), str(max_date.date()))
    if stock.empty:
        return pd.DataFrame()

    merged = daily.join(stock, how="inner").sort_index()
    if merged.empty:
        return pd.DataFrame()

    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"
    return merged


def save_heatmap(df: pd.DataFrame, method: str, company: str, out_dir: str) -> tuple[str, str]:
    corr = df[ALL_COLS].corr(method=method)
    labeled = corr.rename(index=LABELS, columns=LABELS)
    fig = px.imshow(
        labeled,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=f"{company} — {method.title()} Correlation Heatmap",
    )
    fig.update_layout(width=900, height=700)
    base_path = os.path.join(out_dir, f"{slugify(company)}_{method}_heatmap")
    return write_plotly_outputs(fig, base_path)


def build_focus_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for text_col in TEXT_COLS:
        for stock_col in STOCK_COLS:
            pair = df[[text_col, stock_col]].dropna()
            if len(pair) < 3:
                continue
            pearson_r, pearson_p = stats.pearsonr(pair[text_col], pair[stock_col])
            spearman_r, spearman_p = stats.spearmanr(pair[text_col], pair[stock_col])
            rows.append(
                {
                    "text_metric": LABELS[text_col],
                    "stock_metric": LABELS[stock_col],
                    "n_obs": len(pair),
                    "pearson_r": round(float(pearson_r), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 4),
                }
            )
    return pd.DataFrame(rows)


def save_focus_table(table_df: pd.DataFrame, company: str, out_dir: str) -> tuple[str, str, str]:
    csv_path = os.path.join(out_dir, f"{slugify(company)}_focused_correlations.csv")
    table_df.to_csv(csv_path, index=False)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(table_df.columns)),
                cells=dict(values=[table_df[col] for col in table_df.columns]),
            )
        ]
    )
    fig.update_layout(title=f"{company} — Focused Stock/Text Correlation Table")
    base_path = os.path.join(out_dir, f"{slugify(company)}_focused_correlations")
    html_path, png_path = write_plotly_outputs(fig, base_path)
    return csv_path, html_path, png_path


def save_scatter_matrix(df: pd.DataFrame, company: str, out_dir: str) -> tuple[str, str]:
    clean = df[ALL_COLS].dropna().rename(columns=LABELS)
    fig = go.Figure(
        data=go.Splom(
            dimensions=[
                dict(label=col, values=clean[col])
                for col in clean.columns
            ],
            diagonal_visible=False,
            showupperhalf=False,
            marker=dict(size=5, opacity=0.6),
        )
    )
    fig.update_layout(title=f"{company} — Scatter Matrix")
    fig.update_layout(width=1200, height=1200)
    base_path = os.path.join(out_dir, f"{slugify(company)}_scatter_matrix")
    return write_plotly_outputs(fig, base_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Plotly correlation outputs.")
    parser.add_argument("--input", default=config.FINAL_RESULTS_CSV)
    parser.add_argument("--company", default="JPMorgan Chase")
    parser.add_argument("--days-before", type=int, default=30)
    parser.add_argument("--days-after", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    company_dir = OUTPUT_DIR

    aligned = build_aligned_daily(args.company, args.input, args.days_before, args.days_after)
    if aligned.empty:
        raise SystemExit(f"[ERROR] No aligned daily data found for {args.company}.")

    aligned_csv = os.path.join(company_dir, f"{slugify(args.company)}_aligned_daily.csv")
    aligned.reset_index().to_csv(aligned_csv, index=False)

    pearson_html, pearson_png = save_heatmap(aligned, "pearson", args.company, company_dir)
    spearman_html, spearman_png = save_heatmap(aligned, "spearman", args.company, company_dir)
    focused_df = build_focus_table(aligned)
    focused_csv, focused_html, focused_png = save_focus_table(focused_df, args.company, company_dir)
    scatter_html, scatter_png = save_scatter_matrix(aligned, args.company, company_dir)

    print(f"[OK] aligned daily CSV: {aligned_csv}")
    print(f"[OK] pearson heatmap HTML: {pearson_html}")
    print(f"[OK] pearson heatmap PNG: {pearson_png}")
    print(f"[OK] spearman heatmap HTML: {spearman_html}")
    print(f"[OK] spearman heatmap PNG: {spearman_png}")
    print(f"[OK] focused correlations CSV: {focused_csv}")
    print(f"[OK] focused correlations HTML: {focused_html}")
    print(f"[OK] focused correlations PNG: {focused_png}")
    print(f"[OK] scatter matrix HTML: {scatter_html}")
    print(f"[OK] scatter matrix PNG: {scatter_png}")


if __name__ == "__main__":
    main()
