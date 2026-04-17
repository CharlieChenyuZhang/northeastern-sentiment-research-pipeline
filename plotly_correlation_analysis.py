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
    load_market_data,
    load_articles,
)

OUTPUT_DIR = config.ANALYSIS_OUTPUT_DIR
TEXT_COLS = [
    "article_sentiment",
    "reader_sentiment",
    "article_emotion_intensity",
    "reader_emotion_intensity",
]
MARKET_COL_CANDIDATES = [
    "close",
    "adjusted_close",
    "daily_return",
    "volume",
    "amihud_ratio",
    "illiq",
    "r",
    "volatility",
    "intra_day_chg",
    "downside_risk",
    "var1pct",
    "lnvol_chg",
    "hl_spread",
    "typical_price",
    "vwap_proxy",
    "vwap_cum",
    "rolls_measure",
    "dollar_volume",
    "kyle_lambda_proxy",
    "amihud_impact_proxy",
    "turnover_ratio",
]
LABELS = {
    "article_sentiment": "Article Sentiment",
    "reader_sentiment": "Reader Sentiment",
    "article_emotion_intensity": "Article Emotion Intensity",
    "reader_emotion_intensity": "Reader Emotion Intensity",
    "close": "Stock Close",
    "adjusted_close": "Adjusted Close",
    "daily_return": "Daily Return",
    "volume": "Volume",
    "amihud_ratio": "Original Amihud Ratio",
    "illiq": "ILLIQ",
    "r": "Original Return Column",
    "volatility": "Volatility",
    "intra_day_chg": "Intraday Change",
    "downside_risk": "Downside Risk",
    "var1pct": "VaR 1%",
    "lnvol_chg": "Log Volume Change",
    "hl_spread": "HL Spread",
    "typical_price": "Typical Price",
    "vwap_proxy": "VWAP Proxy",
    "vwap_cum": "Cumulative VWAP Proxy",
    "rolls_measure": "Roll's Measure",
    "dollar_volume": "Dollar Volume",
    "kyle_lambda_proxy": "Kyle Lambda Proxy",
    "amihud_impact_proxy": "Amihud Impact Proxy",
    "turnover_ratio": "Turnover Ratio",
}


def slugify(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(" ", "_")


def write_plotly_outputs(fig: go.Figure, base_path: str) -> tuple[str, str]:
    html_path = f"{base_path}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    png_path = f"{base_path}.png"
    try:
        fig.write_image(png_path)
    except Exception:
        png_path = ""
    return html_path, png_path


def active_market_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in MARKET_COL_CANDIDATES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 3:
            continue
        if series.nunique() <= 1:
            continue
        cols.append(col)
    return cols


def build_aligned_daily(
    company: str,
    input_path: str,
    days_before: int,
    days_after: int,
    financial_metrics_csv: str = "",
    financial_symbol: str = "",
) -> pd.DataFrame:
    articles = load_articles(input_path)
    daily = daily_sentiment(articles, company)
    if daily.empty:
        return pd.DataFrame()

    ticker = COMPANY_TICKERS.get(company)
    if not ticker:
        return pd.DataFrame()

    min_date = pd.Timestamp(min(daily.index)) - timedelta(days=days_before)
    max_date = pd.Timestamp(max(daily.index)) + timedelta(days=days_after)
    if financial_metrics_csv:
        stock = load_market_data(
            financial_metrics_csv,
            symbol=financial_symbol or ticker,
            start=str(min_date.date()),
            end=str(max_date.date()),
        )
    else:
        stock = fetch_stock_prices(ticker, str(min_date.date()), str(max_date.date()))
    if stock.empty:
        return pd.DataFrame()

    merged = daily.join(stock, how="inner").sort_index()
    if merged.empty:
        return pd.DataFrame()

    merged.index = pd.to_datetime(merged.index)
    merged.index.name = "date"
    return merged


def save_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    method: str,
    company: str,
    out_dir: str,
) -> tuple[str, str]:
    corr = df[cols].corr(method=method)
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


def build_focus_table(df: pd.DataFrame, market_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for text_col in TEXT_COLS:
        for market_col in market_cols:
            pair = df[[text_col, market_col]].dropna()
            if len(pair) < 3:
                continue
            pearson_r, pearson_p = stats.pearsonr(pair[text_col], pair[market_col])
            spearman_r, spearman_p = stats.spearmanr(pair[text_col], pair[market_col])
            rows.append(
                {
                    "text_metric": LABELS[text_col],
                    "market_metric": LABELS.get(market_col, market_col),
                    "market_metric_key": market_col,
                    "n_obs": len(pair),
                    "pearson_r": round(float(pearson_r), 4),
                    "pearson_p": round(float(pearson_p), 4),
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p": round(float(spearman_p), 4),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "text_metric",
                "market_metric",
                "market_metric_key",
                "n_obs",
                "pearson_r",
                "pearson_p",
                "spearman_r",
                "spearman_p",
            ]
        )
    table = pd.DataFrame(rows)
    table["abs_pearson_r"] = table["pearson_r"].abs()
    return table.sort_values(["abs_pearson_r", "pearson_p"], ascending=[False, True]).reset_index(drop=True)


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


def select_top_market_metrics(table_df: pd.DataFrame, max_metrics: int = 4) -> list[str]:
    if table_df.empty:
        return []
    seen: list[str] = []
    for metric_key in table_df["market_metric_key"]:
        if metric_key not in seen:
            seen.append(metric_key)
        if len(seen) >= max_metrics:
            break
    return seen


def save_scatter_matrix(
    df: pd.DataFrame,
    company: str,
    out_dir: str,
    selected_market_cols: list[str],
) -> tuple[str, str]:
    scatter_cols = TEXT_COLS + selected_market_cols
    clean = df[scatter_cols].dropna().rename(columns=LABELS)
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


def save_normalized_timeseries(
    df: pd.DataFrame,
    company: str,
    out_dir: str,
    selected_market_cols: list[str],
) -> tuple[str, str]:
    plot_cols = TEXT_COLS + selected_market_cols
    clean = df[plot_cols].copy()
    normalized = pd.DataFrame(index=clean.index)
    for col in plot_cols:
        series = pd.to_numeric(clean[col], errors="coerce")
        std = series.std()
        if pd.isna(std) or std == 0:
            continue
        normalized[col] = (series - series.mean()) / std

    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=LABELS.get(col, col),
            )
        )
    fig.update_layout(
        title=f"{company} — Normalized Sentiment and Market Metrics Over Time",
        xaxis_title="Date",
        yaxis_title="Z-score",
        hovermode="x unified",
        width=1200,
        height=700,
    )
    base_path = os.path.join(out_dir, f"{slugify(company)}_normalized_timeseries")
    return write_plotly_outputs(fig, base_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Plotly correlation outputs.")
    parser.add_argument("--input", default=config.FINAL_RESULTS_CSV)
    parser.add_argument("--company", default="JPMorgan Chase")
    parser.add_argument("--days-before", type=int, default=30)
    parser.add_argument("--days-after", type=int, default=30)
    parser.add_argument("--financial-metrics-csv", default="")
    parser.add_argument("--financial-symbol", default="")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    company_dir = OUTPUT_DIR

    aligned = build_aligned_daily(
        args.company,
        args.input,
        args.days_before,
        args.days_after,
        financial_metrics_csv=args.financial_metrics_csv,
        financial_symbol=args.financial_symbol,
    )
    if aligned.empty:
        raise SystemExit(f"[ERROR] No aligned daily data found for {args.company}.")

    market_cols = active_market_cols(aligned)
    all_cols = TEXT_COLS + market_cols

    aligned_csv = os.path.join(company_dir, f"{slugify(args.company)}_aligned_daily.csv")
    aligned.reset_index().to_csv(aligned_csv, index=False)

    pearson_html, pearson_png = save_heatmap(aligned, all_cols, "pearson", args.company, company_dir)
    spearman_html, spearman_png = save_heatmap(aligned, all_cols, "spearman", args.company, company_dir)
    focused_df = build_focus_table(aligned, market_cols)
    focused_csv, focused_html, focused_png = save_focus_table(focused_df, args.company, company_dir)
    selected_market_cols = select_top_market_metrics(focused_df)
    timeseries_html, timeseries_png = save_normalized_timeseries(
        aligned,
        args.company,
        company_dir,
        selected_market_cols or market_cols[:4],
    )
    scatter_html, scatter_png = save_scatter_matrix(
        aligned,
        args.company,
        company_dir,
        selected_market_cols[:4] or market_cols[:4],
    )

    print(f"[OK] aligned daily CSV: {aligned_csv}")
    print(f"[OK] pearson heatmap HTML: {pearson_html}")
    print(f"[OK] pearson heatmap PNG: {pearson_png}")
    print(f"[OK] spearman heatmap HTML: {spearman_html}")
    print(f"[OK] spearman heatmap PNG: {spearman_png}")
    print(f"[OK] focused correlations CSV: {focused_csv}")
    print(f"[OK] focused correlations HTML: {focused_html}")
    print(f"[OK] focused correlations PNG: {focused_png}")
    print(f"[OK] normalized timeseries HTML: {timeseries_html}")
    print(f"[OK] normalized timeseries PNG: {timeseries_png}")
    print(f"[OK] scatter matrix HTML: {scatter_html}")
    print(f"[OK] scatter matrix PNG: {scatter_png}")


if __name__ == "__main__":
    main()
