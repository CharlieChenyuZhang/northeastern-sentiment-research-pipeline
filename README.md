# Company News Sentiment & Emotion Analysis Pipeline

A research pipeline that collects news articles about publicly traded companies, performs multi-dimensional sentiment and emotion analysis using large language models, and correlates the results with stock market prices.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Points Extracted](#data-points-extracted)
- [Simple vs Comprehensive Analysis](#simple-vs-comprehensive-analysis)
- [Methodology: Multi-Run Confidence Merging](#methodology-multi-run-confidence-merging)
- [Stock Correlation Analysis](#stock-correlation-analysis)
- [Setup](#setup)
- [Usage](#usage)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [References](#references)

## Overview

For each of eight target companies (Apple, Tesla, Amazon, Microsoft, Google, Meta, Netflix, NVIDIA), the pipeline:

1. **Discovers** news articles via Google search (SerpAPI).
2. **Scrapes** article content using Firecrawl's LLM-based extraction.
3. **Analyzes** each article along four dimensions using OpenAI's GPT models.
4. **Correlates** the resulting sentiment time-series with historical stock prices.

## Pipeline Architecture

```
config.py                  (shared constants, prompts, company list)
    |
    v
search_and_scrape.py       Step 1: SerpAPI search + Firecrawl scrape
    |                      Output: articles_raw.csv
    |
    +---> sentiment_simple.py          Step 2a: 1 LLM call per article
    |     Output: sentiment_simple.csv
    |
    +---> sentiment_comprehensive.py   Step 2b: N LLM calls per article (with logprobs)
          Output: sentiment_comprehensive.csv
            |
            v
      merge_results.py     Step 3: Join simple + comprehensive by URL
            |              Output: final_results.csv
            v
      stock_correlation.py Step 4: Fetch stock prices, plot, correlate
                           Output: analysis_output/
```

Steps 2a and 2b are independent and can be run in parallel.

## Data Points Extracted

For each news article, we extract four dimensions:

| Dimension | Description |
|-----------|-------------|
| **Article Sentiment** | The internal sentiment expressed by the article itself (Positive / Negative / Neutral / Mixed) |
| **Reader Sentiment** | The sentiment a typical reader would feel after reading the article (Positive / Negative / Neutral / Mixed) |
| **Article Emotions** | Specific emotions the article presents (free-form labels, e.g., optimism, anger, fear) |
| **Reader Emotions** | Emotions the article would evoke in a typical reader (free-form labels, e.g., anxiety, hope, curiosity) |

Emotion labels are not drawn from a fixed taxonomy. The LLM generates whatever labels best describe the content, allowing the model to capture nuances that a predefined set might miss.

## Simple vs Comprehensive Analysis

### Simple Analysis (`sentiment_simple.py`)

- **1 API call per article.**
- Sends the article text with a system prompt requesting a JSON response containing all four dimensions.
- Sentiment values are single labels (e.g., `"Positive"`).
- Emotion values are lists of free-form labels (e.g., `["optimism", "excitement"]`).
- Uses `temperature=0` for deterministic output.

### Comprehensive Analysis (`sentiment_comprehensive.py`)

- **N API calls per article** (default N=3), with `logprobs=True` enabled.
- The LLM returns **confidence distributions** instead of single labels:
  - Sentiment: `{"Positive": 0.7, "Negative": 0.1, "Neutral": 0.15, "Mixed": 0.05}`
  - Emotions: `{"optimism": 0.8, "excitement": 0.6, "anxiety": 0.3}`
- Uses `temperature=0.7` to introduce diversity across runs.
- Merges results across runs using **geometric-mean confidence** (see below).
- Also records the **mean token-level log-probability** from each response as a secondary (response-level) confidence signal.

## Methodology: Multi-Run Confidence Merging

The comprehensive analysis uses a **self-consistency** inspired approach to produce calibrated confidence scores. Rather than relying on a single LLM call, we sample multiple responses and aggregate them.

### Why temperature = 0.7?

We use `temperature=0.7` following the methodology established in the self-consistency literature. Wang et al. (2022) demonstrated that sampling diverse reasoning paths at `temperature=0.7` and aggregating results produces more accurate and better-calibrated outputs than a single greedy decode (`temperature=0`). This value balances:

- **Diversity**: Enough variation across runs to reveal the model's uncertainty. At lower temperatures (e.g., 0.3), runs are nearly identical, making multi-run merging a no-op.
- **Coherence**: Outputs remain well-formed and on-topic. At `temperature=1.0`, the raw model distribution is preserved but JSON outputs can occasionally be noisier.

### Merging Algorithm

**Sentiment (probability distributions):**

Each run produces a distribution over {Positive, Negative, Neutral, Mixed}. We merge N runs via the geometric mean:

```
For each label L:
    merged(L) = exp( (1/N) * sum( log(p_i(L)) for i in 1..N ) )
Then re-normalize so all labels sum to 1.0.
```

The geometric mean (equivalent to averaging in log-probability space) is the standard approach for aggregating probability estimates. It is more robust to outliers than the arithmetic mean and naturally down-weights low-confidence runs.

**Emotions (free-form labels with confidence):**

Since different runs may surface different emotion labels, we take the union of all labels and compute:

```
For each emotion E:
    geo_mean(E) = exp( (1/K) * sum( log(p_i(E)) for i in runs where E appeared ) )
    presence(E) = K / N    (fraction of runs where E appeared)
    final(E) = geo_mean(E) * presence(E)
```

This rewards emotions that appear consistently across runs and penalizes those that only surface in a single run.

### Token-Level Log-Probabilities

In addition to the LLM's self-reported confidence scores, we record the **mean log-probability across all output tokens** (`comp_mean_logprob`). This serves as a response-level confidence signal from the model's internal distribution.

**Caveat:** Because the output is structured JSON, this metric is diluted by high-confidence syntax tokens (`{`, `"`, `:`, etc.). It is recorded for completeness but the primary confidence signal comes from the LLM's self-reported scores and multi-run merging.

## Stock Correlation Analysis

`stock_correlation.py` fetches daily closing prices from Yahoo Finance and produces:

1. **Dual-axis time-series charts** — sentiment/emotion scores (left axis) overlaid with stock price (right axis) for visual inspection.

2. **Pearson & Spearman correlations** — computed between sentiment scores and **daily stock returns** (not raw price, to avoid spurious trend correlations).

3. **Lead/lag analysis** — shifts sentiment by -5 to +5 **calendar days** relative to stock returns to test whether sentiment predicts price movement (or vice versa). Uses `pd.shift(freq="D")` to ensure shifts represent actual calendar days, not row indices.

4. **Rolling 14-day correlation** — shows how the sentiment-return relationship evolves over time.

### Numeric Encoding

For correlation analysis, sentiment labels are mapped to numeric scores:

| Label | Score |
|-------|-------|
| Positive | +1.0 |
| Negative | -1.0 |
| Neutral | 0.0 |
| Mixed | 0.0 |

Emotion intensity is computed as the mean confidence score across all detected emotions (comprehensive) or `count / 10` capped at 1.0 (simple). On charts, emotion intensity (0–1 scale) is rescaled to -1..+1 to share the y-axis with sentiment.

## Setup

### Prerequisites

- Python 3.10+
- API keys for [SerpAPI](https://serpapi.com/), [Firecrawl](https://firecrawl.dev/), and [OpenAI](https://platform.openai.com/)

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
SERPAPI_API_KEY=your-serpapi-key
FIRECRAWL_API_KEY=fc-your-firecrawl-key
OPENAI_API_KEY=sk-your-openai-key
```

## Usage

Run the pipeline steps in order:

```bash
# Step 1: Discover and scrape news articles
python search_and_scrape.py

# Step 2a: Simple analysis (1 API call per article)
python sentiment_simple.py

# Step 2b: Comprehensive analysis (3 API calls per article, with logprobs)
# Can run in parallel with Step 2a
python sentiment_comprehensive.py

# Step 3: Merge simple + comprehensive results
python merge_results.py

# Step 4: Stock correlation analysis
python stock_correlation.py
```

All scripts support **resumability** — if interrupted, they detect already-processed URLs in the output CSV and skip them on re-run.

### Options

```bash
# Use a specific input file for correlation analysis
python stock_correlation.py --input sentiment_simple.csv

# Adjust stock data date range (days before/after article dates)
python stock_correlation.py --days-before 60 --days-after 60
```

## Output Files

| File | Description |
|------|-------------|
| `articles_raw.csv` | Scraped articles: company, URL, title, full text, date, author |
| `sentiment_simple.csv` | Simple analysis results + the prompt used |
| `sentiment_comprehensive.csv` | Comprehensive results: sentiment distributions, emotion confidence scores, mean logprob, prompt used |
| `final_results.csv` | Merged simple + comprehensive results |
| `analysis_output/correlation_summary.csv` | Pearson & Spearman correlations per company |
| `analysis_output/lead_lag_analysis.csv` | Lead/lag correlation at -5..+5 day offsets |
| `analysis_output/{company}_sentiment_vs_stock.png` | Time-series chart per company |
| `analysis_output/{company}_rolling_correlation.png` | Rolling 14-day correlation chart per company |

### Prompt Traceability

Every output CSV includes the exact prompt(s) used to generate the results (`prompt_used` or `comp_prompt_used` columns), ensuring full reproducibility.

## Configuration

All pipeline settings are centralized in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `COMPANIES` | 8 companies | List of target companies |
| `OPENAI_MODEL` | `gpt-5.3` | Model used for analysis |
| `OPENAI_TEMPERATURE` | `0` | Temperature for simple analysis |
| `OPENAI_TEMPERATURE_COMPREHENSIVE` | `0.7` | Temperature for comprehensive multi-run analysis |
| `COMPREHENSIVE_RUNS` | `3` | Number of runs per article in comprehensive mode |
| `MAX_SEARCH_RESULTS` | `50` | URLs to discover per search query |
| `MAX_ARTICLE_CHARS` | `48,000` | Truncation limit for article text |
| `ANALYSIS_WORKERS` | `6` | Concurrent threads for OpenAI calls |
| `SCRAPE_WORKERS` | `8` | Concurrent threads for Firecrawl scraping |

## References

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** *arXiv preprint arXiv:2203.11171.* https://arxiv.org/abs/2203.11171
  - Establishes the self-consistency framework for sampling multiple LLM responses and aggregating them. Demonstrates that `temperature=0.7` provides an effective balance between output diversity and coherence for multi-sample techniques.
