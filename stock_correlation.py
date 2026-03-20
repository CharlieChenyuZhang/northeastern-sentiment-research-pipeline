#!/usr/bin/env python3
"""
Step 4 — Correlate sentiment/emotion time-series with stock prices.

Reads final_results.csv (or sentiment_simple.csv / sentiment_comprehensive.csv),
normalizes dates, fetches historical stock prices from Yahoo Finance, and:
  1. Plots sentiment time-series alongside stock price on the same chart.
  2. Computes Pearson & Spearman correlations between sentiment and price.
  3. Runs additional analyses: rolling correlations, Granger-like lead/lag,
     and emotion-specific breakdowns.

Outputs charts as PNG files and a summary CSV.

Usage:
    python stock_correlation.py [--input final_results.csv] [--days-before 30] [--days-after 30]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

import config

# ---------------------------------------------------------------------------
# Company -> Yahoo Finance ticker mapping
# ---------------------------------------------------------------------------
COMPANY_TICKERS = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Meta": "META",
    "Netflix": "NFLX",
    "NVIDIA": "NVDA",
}

# Map sentiment labels to numeric scores for correlation
SENTIMENT_SCORES = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
}

OUTPUT_DIR = "analysis_output"


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

def parse_date(raw: str) -> datetime | None:
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
            return datetime.strptime(raw[:len(raw)], fmt)
        except (ValueError, OverflowError):
            continue
    # Last resort: try pandas
    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load articles and build per-company daily sentiment series
# ---------------------------------------------------------------------------

def sentiment_to_score(label: str) -> float | None:
    return SENTIMENT_SCORES.get(label.strip().lower())


def load_articles(csv_path: str) -> pd.DataFrame:
    """Load the merged (or simple/comprehensive) CSV into a DataFrame."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt = parse_date(row.get("published_date", ""))
            if dt is None:
                continue

            # --- Simple scores ---
            art_sent = row.get("simple_article_sentiment", row.get("comp_article_sentiment", ""))
            rdr_sent = row.get("simple_reader_sentiment", row.get("comp_reader_sentiment", ""))
            art_score = sentiment_to_score(art_sent)
            rdr_score = sentiment_to_score(rdr_sent)

            # --- Emotion intensities (average confidence across detected emotions) ---
            art_emo_raw = row.get("comp_article_emotions", row.get("simple_article_emotions", ""))
            rdr_emo_raw = row.get("comp_reader_emotions", row.get("simple_reader_emotions", ""))
            art_emo_score = _emotion_intensity(art_emo_raw)
            rdr_emo_score = _emotion_intensity(rdr_emo_raw)

            if art_score is None and rdr_score is None:
                continue

            rows.append({
                "company": row.get("company", ""),
                "date": dt.date(),
                "url": row.get("url", ""),
                "article_sentiment": art_score if art_score is not None else 0.0,
                "reader_sentiment": rdr_score if rdr_score is not None else 0.0,
                "article_emotion_intensity": art_emo_score,
                "reader_emotion_intensity": rdr_emo_score,
            })

    return pd.DataFrame(rows)


def _emotion_intensity(raw: str) -> float:
    """
    Convert emotion data to a single intensity score.
    - If JSON dict with confidence: return mean confidence.
    - If comma-separated labels: count as proxy for intensity.
    - Otherwise 0.
    """
    raw = raw.strip()
    if not raw:
        return 0.0
    # Try JSON (comprehensive output)
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and d:
            return float(np.mean(list(d.values())))
    except (json.JSONDecodeError, TypeError):
        pass
    # Comma-separated (simple output) — use count / 10 as rough intensity
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return min(len(parts) / 10.0, 1.0)


# ---------------------------------------------------------------------------
# Fetch stock prices
# ---------------------------------------------------------------------------

def fetch_stock_prices(
    ticker: str, start: str, end: str
) -> pd.DataFrame:
    """Download daily closing prices from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return df
    # yfinance may return MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].copy()
    df.columns = ["close"]
    df.index = pd.to_datetime(df.index).date  # type: ignore[assignment]
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Aggregation: daily average sentiment per company
# ---------------------------------------------------------------------------

def daily_sentiment(articles: pd.DataFrame, company: str) -> pd.DataFrame:
    """Aggregate articles to daily mean sentiment scores for one company."""
    sub = articles[articles["company"] == company].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["date"] = pd.to_datetime(sub["date"])
    daily = sub.groupby("date").agg({
        "article_sentiment": "mean",
        "reader_sentiment": "mean",
        "article_emotion_intensity": "mean",
        "reader_emotion_intensity": "mean",
    })
    daily.index = daily.index.date  # type: ignore[assignment]
    daily.index.name = "date"
    return daily


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_company(
    company: str,
    daily_sent: pd.DataFrame,
    stock: pd.DataFrame,
    out_dir: str,
) -> str:
    """Create a dual-axis chart: sentiment lines + stock price."""
    merged = daily_sent.join(stock, how="inner")
    if merged.empty:
        sys.stderr.write(f"[WARN] No overlapping dates for {company}, skipping plot.\n")
        return ""

    fig, ax1 = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(merged.index)

    # Sentiment lines on left y-axis (scale: -1 to 1)
    sent_colors = {
        "article_sentiment": ("#2196F3", "Article Sentiment"),
        "reader_sentiment": ("#FF9800", "Reader Sentiment"),
    }
    # Normalize emotion intensity to -1..1 scale: remap 0..1 -> -1..1
    emo_colors = {
        "article_emotion_intensity": ("#4CAF50", "Article Emotion Intensity"),
        "reader_emotion_intensity": ("#E91E63", "Reader Emotion Intensity"),
    }
    for col, (color, label) in sent_colors.items():
        if col in merged.columns:
            ax1.plot(dates, merged[col], color=color, label=label,
                     linewidth=1.5, alpha=0.8, marker="o", markersize=3)
    for col, (color, label) in emo_colors.items():
        if col in merged.columns:
            # Remap 0..1 -> -1..1 so both use the full y-axis range
            remapped = merged[col] * 2 - 1
            ax1.plot(dates, remapped, color=color, label=f"{label} (rescaled)",
                     linewidth=1.5, alpha=0.8, marker="s", markersize=3)

    ax1.set_ylabel("Sentiment Score (& rescaled Emotion Intensity)")
    ax1.set_ylim(-1.3, 1.3)
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    # Stock price on right y-axis
    ax2 = ax1.twinx()
    ax2.plot(dates, merged["close"], color="#9C27B0", label="Stock Close",
             linewidth=2, alpha=0.6, linestyle="--")
    ax2.set_ylabel("Stock Price ($)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.title(f"{company} — Sentiment & Emotion vs Stock Price")
    plt.tight_layout()

    path = os.path.join(out_dir, f"{company.lower()}_sentiment_vs_stock.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_correlations(
    daily_sent: pd.DataFrame, stock: pd.DataFrame
) -> list[dict]:
    """Pearson & Spearman for each sentiment column vs daily stock returns."""
    merged = daily_sent.join(stock, how="inner").sort_index()
    if len(merged) < 5:
        return []

    # Use daily returns instead of raw price to avoid spurious trend correlation
    merged["daily_return"] = merged["close"].pct_change()
    merged = merged.dropna(subset=["daily_return"])
    if len(merged) < 5:
        return []

    results = []
    sent_cols = [
        "article_sentiment", "reader_sentiment",
        "article_emotion_intensity", "reader_emotion_intensity",
    ]
    for col in sent_cols:
        if col not in merged.columns:
            continue
        x = merged[col].values
        y = merged["daily_return"].values
        if np.std(x) == 0 or np.std(y) == 0:
            continue
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)
        results.append({
            "metric": col,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n_days": len(merged),
        })
    return results


# ---------------------------------------------------------------------------
# Lead/lag analysis: does sentiment predict stock movement?
# ---------------------------------------------------------------------------

def lead_lag_analysis(
    daily_sent: pd.DataFrame, stock: pd.DataFrame, max_lag: int = 5
) -> list[dict]:
    """
    Shift sentiment by -max_lag..+max_lag days and compute correlation
    with stock price changes. Positive lag = sentiment leads stock.
    """
    merged = daily_sent.join(stock, how="inner").sort_index()
    if len(merged) < 10:
        return []

    # Daily stock returns
    close = merged["close"].values.astype(float)
    returns = np.diff(close) / close[:-1]

    results = []
    sent_cols = ["article_sentiment", "reader_sentiment"]
    for col in sent_cols:
        if col not in merged.columns:
            continue
        vals = merged[col].values[:-1]  # align with returns
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                shifted_sent = vals
                shifted_ret = returns
            elif lag > 0:
                shifted_sent = vals[:-lag]
                shifted_ret = returns[lag:]
            else:
                shifted_sent = vals[-lag:]
                shifted_ret = returns[:lag]
            if len(shifted_sent) < 5 or np.std(shifted_sent) == 0:
                continue
            r, p = stats.pearsonr(shifted_sent, shifted_ret)
            results.append({
                "metric": col,
                "lag_days": lag,
                "pearson_r": round(r, 4),
                "pearson_p": round(p, 4),
                "n": len(shifted_sent),
            })
    return results


# ---------------------------------------------------------------------------
# Rolling correlation
# ---------------------------------------------------------------------------

def rolling_correlation(
    daily_sent: pd.DataFrame, stock: pd.DataFrame, window: int = 14
) -> pd.DataFrame:
    """Compute rolling Pearson correlation between sentiment and daily stock returns."""
    merged = daily_sent.join(stock, how="inner").sort_index()
    merged["daily_return"] = merged["close"].pct_change()
    merged = merged.dropna(subset=["daily_return"])
    if len(merged) < window:
        return pd.DataFrame()

    rolling = pd.DataFrame(index=merged.index)
    for col in ["article_sentiment", "reader_sentiment"]:
        if col in merged.columns:
            rolling[f"rolling_corr_{col}"] = (
                merged[col]
                .rolling(window)
                .corr(merged["daily_return"])
            )
    return rolling


def plot_rolling_correlation(
    company: str, rolling_df: pd.DataFrame, out_dir: str
) -> str:
    if rolling_df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(rolling_df.index)
    for col in rolling_df.columns:
        ax.plot(dates, rolling_df[col], label=col.replace("rolling_corr_", ""), linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Rolling Correlation")
    ax.set_title(f"{company} — Rolling 14-day Correlation (Sentiment vs Stock)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    path = os.path.join(out_dir, f"{company.lower()}_rolling_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate sentiment with stock prices.")
    parser.add_argument("--input", default=config.FINAL_RESULTS_CSV,
                        help="Path to merged results CSV (default: final_results.csv)")
    parser.add_argument("--days-before", type=int, default=30,
                        help="Fetch stock data this many days before earliest article")
    parser.add_argument("--days-after", type=int, default=30,
                        help="Fetch stock data this many days after latest article")
    args = parser.parse_args()

    # Fall back to whichever file exists
    input_path = args.input
    if not os.path.exists(input_path):
        for fallback in [config.SENTIMENT_SIMPLE_CSV, config.SENTIMENT_COMPREHENSIVE_CSV]:
            if os.path.exists(fallback):
                input_path = fallback
                break
        else:
            sys.exit("[ERROR] No results CSV found. Run the analysis pipeline first.")

    sys.stderr.write(f"[INFO] Loading articles from {input_path}\n")
    articles = load_articles(input_path)
    sys.stderr.write(f"[INFO] Loaded {len(articles)} articles with valid dates.\n")

    if articles.empty:
        sys.exit("[ERROR] No articles with parseable dates found.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_correlations = []
    all_lead_lag = []

    companies = articles["company"].unique()
    for company in companies:
        ticker = COMPANY_TICKERS.get(company)
        if not ticker:
            sys.stderr.write(f"[WARN] No ticker for {company}, skipping.\n")
            continue

        sys.stderr.write(f"\n[INFO] === {company} ({ticker}) ===\n")

        daily_sent = daily_sentiment(articles, company)
        if daily_sent.empty:
            sys.stderr.write(f"[WARN] No daily sentiment data for {company}.\n")
            continue

        # Date range for stock data
        min_date = pd.Timestamp(min(daily_sent.index)) - timedelta(days=args.days_before)
        max_date = pd.Timestamp(max(daily_sent.index)) + timedelta(days=args.days_after)

        sys.stderr.write(f"[INFO] Fetching {ticker} stock data {min_date.date()} to {max_date.date()}\n")
        stock = fetch_stock_prices(ticker, str(min_date.date()), str(max_date.date()))
        if stock.empty:
            sys.stderr.write(f"[WARN] No stock data for {ticker}.\n")
            continue

        # 1. Plot sentiment vs stock
        chart_path = plot_company(company, daily_sent, stock, OUTPUT_DIR)
        if chart_path:
            sys.stderr.write(f"[OK]   Chart: {chart_path}\n")

        # 2. Correlation analysis
        corrs = compute_correlations(daily_sent, stock)
        for c in corrs:
            c["company"] = company
            c["ticker"] = ticker
        all_correlations.extend(corrs)

        if corrs:
            sys.stderr.write(f"[INFO] Correlations for {company}:\n")
            for c in corrs:
                sys.stderr.write(
                    f"       {c['metric']}: Pearson r={c['pearson_r']:.3f} "
                    f"(p={c['pearson_p']:.3f}), Spearman r={c['spearman_r']:.3f} "
                    f"(p={c['spearman_p']:.3f}), n={c['n_days']}\n"
                )

        # 3. Lead/lag analysis
        ll = lead_lag_analysis(daily_sent, stock)
        for entry in ll:
            entry["company"] = company
            entry["ticker"] = ticker
        all_lead_lag.extend(ll)

        # 4. Rolling correlation
        rolling_df = rolling_correlation(daily_sent, stock)
        rc_path = plot_rolling_correlation(company, rolling_df, OUTPUT_DIR)
        if rc_path:
            sys.stderr.write(f"[OK]   Rolling correlation chart: {rc_path}\n")

    # Save correlation summaries
    if all_correlations:
        corr_path = os.path.join(OUTPUT_DIR, "correlation_summary.csv")
        with open(corr_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "company", "ticker", "metric", "pearson_r", "pearson_p",
                "spearman_r", "spearman_p", "n_days",
            ])
            writer.writeheader()
            writer.writerows(all_correlations)
        sys.stderr.write(f"\n[OK]   Correlation summary: {corr_path}\n")

    if all_lead_lag:
        ll_path = os.path.join(OUTPUT_DIR, "lead_lag_analysis.csv")
        with open(ll_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "company", "ticker", "metric", "lag_days",
                "pearson_r", "pearson_p", "n",
            ])
            writer.writeheader()
            writer.writerows(all_lead_lag)
        sys.stderr.write(f"[OK]   Lead/lag analysis: {ll_path}\n")

    sys.stderr.write(f"\n[INFO] All outputs saved to {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()
