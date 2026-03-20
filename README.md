# Company News Sentiment & Emotion Analysis Pipeline

A research pipeline that collects news articles about publicly traded companies, performs multi-dimensional sentiment and emotion analysis using large language models, and correlates the results with stock market prices.

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [News Discovery](#news-discovery)
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

For each target company (configurable in `config.py`), the pipeline:

1. **Discovers** news articles via three sources: Google News, Google Search, and Firecrawl.
2. **Scrapes** article content using Firecrawl's LLM-based extraction.
3. **Analyzes** each article along four dimensions using OpenAI's GPT models.
4. **Correlates** the resulting sentiment time-series with historical stock prices.

Default target companies: Apple, Tesla, Amazon, Microsoft, Google, Meta, Netflix, NVIDIA.

## Pipeline Architecture

```
config.py                  (shared constants, prompts, company list)
    |
    v
search_and_scrape.py       Step 1: Discover URLs + Firecrawl scrape
    |                      Sources: Google News + Google Search + Firecrawl
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

## News Discovery

The pipeline uses three complementary discovery methods to maximize article coverage, run in order with deduplication:

| Source | Engine | Typical Yield | Notes |
|--------|--------|---------------|-------|
| **Google News** (SerpAPI) | `google_news` | 50–100 per query | News-specific results with dates; also extracts sub-story links from highlight cards |
| **Google Search** (SerpAPI) | `google` | 10–100 per query | General web results; paginates in batches of 100 |
| **Firecrawl `/search`** | Firecrawl | 5–10 per query | Fallback if SerpAPI is unavailable |

For each query, all three sources are queried and their results are combined (deduplicated by URL). This typically yields significantly more unique articles than any single source alone.

### Search Queries

The pipeline runs **9 query variations per company** to cover a 3-year range with diverse perspectives:

```
"{company} news"                          # Broad — current news
"{company} latest news"                   # Broad — recent coverage
"{company} company news {year}"           # Year-specific (2024, 2025, 2026)
"{company} company news {year-1}"
"{company} company news {year-2}"
"{company} stock earnings {year}"         # Financial coverage
"{company} quarterly results {year-1}"
"{company} business update"               # Business operations
"{company} investor news"                 # Investor-focused
```

With `MAX_SEARCH_RESULTS = 50` per query and 9 queries, the pipeline can discover up to **450 unique URLs per company** (before deduplication across queries).

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

Emotion intensity is computed as the mean confidence score across all detected emotions (comprehensive) or `count / 10` capped at 1.0 (simple). On charts, emotion intensity (0-1 scale) is rescaled to -1..+1 to share the y-axis with sentiment.

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

### Quick Start

Run the entire pipeline with a single command:

```bash
./run_pipeline.sh
```

This cleans previous data, then runs all steps in sequence.

### Step-by-Step

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
python stock_correlation.py --days-before 90 --days-after 30
```

### Targeting Specific Companies

Edit the `COMPANIES` list in `config.py`:

```python
COMPANIES = [
    "Tesla",
]
```

To run all eight companies, restore the full list:

```python
COMPANIES = [
    "Apple", "Tesla", "Amazon", "Microsoft",
    "Google", "Meta", "Netflix", "NVIDIA",
]
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
| `OPENAI_MODEL` | `gpt-4o` | Model used for analysis (supports logprobs) |
| `OPENAI_TEMPERATURE` | `0` | Temperature for simple analysis |
| `OPENAI_TEMPERATURE_COMPREHENSIVE` | `0.7` | Temperature for comprehensive multi-run analysis |
| `COMPREHENSIVE_RUNS` | `3` | Number of runs per article in comprehensive mode |
| `MAX_SEARCH_RESULTS` | `50` | Max URLs to discover per query (across all sources) |
| `SEARCH_QUERIES_PER_COMPANY` | 9 queries | Query templates covering 3 years + topic variations |
| `MAX_ARTICLE_CHARS` | `48,000` | Truncation limit for article text |
| `ANALYSIS_WORKERS` | `3` | Concurrent threads for OpenAI calls (kept low for rate limits) |
| `SCRAPE_WORKERS` | `8` | Concurrent threads for Firecrawl scraping |

## References

- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). **Self-Consistency Improves Chain of Thought Reasoning in Language Models.** *arXiv preprint arXiv:2203.11171.* https://arxiv.org/abs/2203.11171
  - Establishes the self-consistency framework for sampling multiple LLM responses and aggregating them. Demonstrates that `temperature=0.7` provides an effective balance between output diversity and coherence for multi-sample techniques.

## How `sentiment_comprehensive.py` Works

`sentiment_comprehensive.py` is the second-stage, higher-fidelity sentiment pass. The `comp` prefix in its output columns stands for **comprehensive**, not computed or compounded. Unlike `sentiment_simple.py`, which asks for one label per field, the comprehensive script asks the model for full sentiment distributions and emotion confidence scores, repeats that process multiple times, and then merges the runs into one final result per article.

### Goal

For each scraped article, the script estimates four things in one JSON response:

1. `article_sentiment`: how the article itself is framed
2. `reader_sentiment`: how a typical reader would likely feel
3. `article_emotions`: emotion labels expressed by the article
4. `reader_emotions`: emotion labels likely felt by a typical reader

Instead of returning only one label, the prompt asks for confidence-weighted outputs. That makes it possible to preserve ambiguity, such as an article being partly `Negative` and partly `Mixed`.

### End-to-End Flow

For each article row in `articles_raw.csv`, the script:

1. Truncates `article_text` to `config.MAX_ARTICLE_CHARS`
2. Calls the OpenAI model `config.COMPREHENSIVE_RUNS` times
3. Parses each JSON response into sentiment and emotion structures
4. Extracts token-level log probabilities from each response
5. Merges the repeated runs into one final sentiment distribution and one final emotion dictionary
6. Writes the final outputs to `sentiment_comprehensive.csv`

The repeated-call design is meant to improve robustness. With `temperature=0.7`, the model is allowed some variation across runs; the merge step then rewards patterns that appear consistently.

### Prompt Format

The comprehensive prompt is stored in `config.COMP_PROMPT_ANALYSIS`. It asks the model to return:

- `article_sentiment` as an object over `Positive`, `Negative`, `Neutral`, and `Mixed`
- `reader_sentiment` in the same format
- `article_emotions` as a dictionary of emotion labels to confidence scores
- `reader_emotions` in the same format

Important details:

- Sentiment scores are expected to sum to approximately `1.0`
- Emotion scores do **not** need to sum to `1.0`
- Emotion labels are open-ended rather than restricted to a fixed taxonomy
- The model must return JSON only

### Single-Run Analysis

The core API call happens in `call_analysis_with_logprobs()`.

For one run, the script:

1. Sends the article text with `response_format={"type": "json_object"}`
2. Enables `logprobs=True` and `top_logprobs=5`
3. Parses the returned JSON
4. Computes the mean token log probability of the full output

That function returns a tuple:

- `parsed_json`
- `mean_logprob`

If the API call fails, it retries up to `MAX_RETRIES` times. If all retries fail, it returns an empty result and a fallback logprob of `-10.0`.

### How Sentiment Outputs Are Standardized

The helper `_ensure_sentiment_dist()` converts the model output into a clean probability distribution over the four sentiment labels.

It supports two cases:

- If the model returns a dictionary, the script keeps numeric values, matches keys case-insensitively, and normalizes the scores so they sum to `1.0`
- If the model returns only a plain string label, the script converts that into a one-hot distribution, such as `{"Positive": 1.0, ...}`

This normalization step ensures that all runs can be merged consistently even if the model output is slightly inconsistent.

### How Emotion Outputs Are Standardized

The helper `_ensure_emotion_dict()` converts the emotion output into a dictionary of `{label: confidence}`.

It supports:

- a dictionary of emotion scores
- a list of emotion strings
- a comma-separated string

If an emotion appears without a numeric score, the script assigns it a fallback confidence of `1.0`. Unlike the sentiment distributions, emotion scores are not normalized to sum to `1.0`.

### How the Script Merges the 3 Runs

By default, `config.COMPREHENSIVE_RUNS = 3`, so the model analyzes each article three times.

#### Sentiment merge

For `article_sentiment` and `reader_sentiment`, the script merges the per-run distributions using a **geometric mean** for each label:

1. Collect the score for a label from each run
2. Replace missing or zero-like values with a tiny floor (`1e-12`) to avoid taking `log(0)`
3. Average the natural logs of those scores
4. Exponentiate back to normal space
5. Normalize the final merged scores so they sum to `1.0`

This is stricter than a plain arithmetic average. A label that is strong in all three runs remains strong, while a label that spikes in only one run but is weak in the others gets penalized more heavily.

Example:

- Run 1: `Positive=0.8`
- Run 2: `Positive=0.7`
- Run 3: `Positive=0.9`

Geometric mean:

`(0.8 * 0.7 * 0.9)^(1/3) ~= 0.796`

If another label is high in only one run and low in the other two, its merged value drops much more sharply. This is why geometric mean is useful here: it rewards consistency across sampled runs.

#### Emotion merge

For emotions, the merge logic is slightly different:

1. Take the union of all emotion labels observed across runs
2. For each emotion, compute the geometric mean of its confidence scores across only the runs where it appeared
3. Multiply that value by the fraction of runs in which the emotion appeared

So an emotion that appears in all three runs with medium confidence can outrank an emotion that appears once with a high score. The final emotion dictionaries are sorted from highest score to lowest.

### How Final Labels and Confidence Scores Are Chosen

After merging the runs:

- `comp_article_sentiment` is the sentiment label with the highest merged article sentiment score
- `comp_article_sentiment_confidence` is that winning label's merged score
- `comp_reader_sentiment` is the top merged reader sentiment label
- `comp_reader_sentiment_confidence` is that winning reader label's merged score

The full merged distributions are also saved:

- `comp_article_sentiment_distribution`
- `comp_reader_sentiment_distribution`

These columns are often more informative than the single winning label because they preserve uncertainty. For example, a final distribution could show that an article is mostly `Negative` but still partly `Mixed`.

### How `comp_mean_logprob` Is Calculated

`comp_mean_logprob` is a separate confidence-like signal based on the model's output tokens, not on the sentiment labels themselves.

For each run:

1. The API returns a `logprob` for each generated token in the JSON response
2. The script averages those token logprobs to get one `mean_lp` for that run

After all runs finish, the script computes the arithmetic mean of those per-run values:

`comp_mean_logprob = average(mean token logprob across each run)`

Why is it negative?

- The API returns **log probabilities**, not plain probabilities
- Probabilities are between `0` and `1`
- The natural log of a number between `0` and `1` is negative

So negative values are expected. A value closer to `0` means the model assigned higher probability to the generated tokens; a more negative value means lower confidence in the exact response wording.

This metric should be treated as a rough response-level confidence signal, not as the same thing as sentiment confidence. The script does **not** use `comp_mean_logprob` to choose the winning sentiment label.

### Output Columns Produced

Each output row in `sentiment_comprehensive.csv` contains the original scraped article fields plus:

- `comp_article_sentiment`
- `comp_article_sentiment_confidence`
- `comp_article_sentiment_distribution`
- `comp_reader_sentiment`
- `comp_reader_sentiment_confidence`
- `comp_reader_sentiment_distribution`
- `comp_article_emotions`
- `comp_reader_emotions`
- `comp_mean_logprob`
- `comp_prompt_used`

Together, these columns make the comprehensive output richer than the simple pass: it preserves uncertainty, captures multiple possible emotions, and includes the exact prompt used for traceability.

### Parallelism and Incremental Writes

The script uses `ThreadPoolExecutor(max_workers=config.ANALYSIS_WORKERS)` to analyze multiple articles in parallel. Each finished result is appended to `sentiment_comprehensive.csv` immediately, so long-running jobs can resume safely.

Before starting, the script also checks whether `sentiment_comprehensive.csv` already exists and skips URLs that were processed earlier. This makes reruns practical if the pipeline is interrupted.

### Practical Interpretation

When reading the comprehensive output:

- Use `comp_article_sentiment` and `comp_reader_sentiment` for the final categorical label
- Use the corresponding `*_distribution` columns when you care about ambiguity
- Use the emotion columns to capture nuance that a single sentiment label misses
- Use `comp_mean_logprob` only as a rough secondary reliability signal

In other words, the comprehensive script is designed to trade extra API cost for richer and more stable sentiment estimates.

## How `stock_correlation.py` Works

`stock_correlation.py` is the final analysis stage of the pipeline. Its job is to take the article-level sentiment outputs produced by earlier steps, convert them into daily company-level signals, download matching stock price data, and measure whether the sentiment series and stock movement appear related.

At a high level, the script does four things:

1. Loads a sentiment results CSV
2. Builds daily sentiment and emotion time-series for each company
3. Fetches stock prices for those companies from Yahoo Finance
4. Produces plots and correlation summaries

### What Input File It Uses

By default, the script looks for `final_results.csv` using `config.FINAL_RESULTS_CSV`.

If that file does not exist, it automatically falls back to:

- `sentiment_simple.csv`
- `sentiment_comprehensive.csv`

This makes the script flexible: it can run on either the merged output or one of the earlier sentiment-analysis outputs.

You can also override the default with:

```bash
python stock_correlation.py --input some_file.csv
```

### Command-Line Arguments

The script accepts three command-line options:

- `--input`: path to the CSV file to analyze
- `--days-before`: how many days before the earliest article date to fetch stock data
- `--days-after`: how many days after the latest article date to fetch stock data

Example:

```bash
python stock_correlation.py --input final_results.csv --days-before 30 --days-after 30
```

The extra date padding is helpful because stock markets are closed on weekends and holidays, and some later analyses need nearby dates for alignment.

### Company-to-Ticker Mapping

The script contains a hard-coded mapping from company names in the CSV to Yahoo Finance tickers:

- `Apple -> AAPL`
- `Tesla -> TSLA`
- `Amazon -> AMZN`
- `Microsoft -> MSFT`
- `Google -> GOOGL`
- `Meta -> META`
- `Netflix -> NFLX`
- `NVIDIA -> NVDA`

If a company appears in the CSV but is missing from this mapping, the script prints a warning and skips that company.

### Step 1: Parsing and Normalizing Article Dates

The function `parse_date()` tries to convert each article's `published_date` into a Python datetime.

It supports several common date formats, such as:

- `YYYY-MM-DD`
- ISO timestamps like `YYYY-MM-DDTHH:MM:SS`
- long-form dates like `March 20, 2026`
- slash-formatted dates like `03/20/2026`

If none of the explicit formats match, it makes a last attempt with `pandas.to_datetime()`.

Rows whose dates still cannot be parsed are skipped. This is important because the stock analysis depends on aligning article dates with trading days.

### Step 2: Converting Sentiment Labels Into Numeric Scores

The script cannot correlate text labels like `Positive` or `Negative` directly with stock returns, so it maps labels to numbers:

- `positive -> 1.0`
- `negative -> -1.0`
- `neutral -> 0.0`
- `mixed -> 0.0`

This happens through `sentiment_to_score()`.

That means article sentiment and reader sentiment become numeric time-series that can be averaged and correlated later.

One subtle choice here is that `mixed` is treated the same as `neutral`. That simplifies the analysis, but it also means the script does not distinguish between articles that are truly neutral and articles that contain strong but conflicting signals.

### Step 3: Turning Emotion Output Into a Single Intensity Score

The script also includes emotion information, but it reduces it to a single scalar "intensity" value instead of trying to correlate each individual emotion label separately.

This is handled by `_emotion_intensity()`:

- If the emotion field is a JSON dictionary of labels to confidence scores, the script takes the mean of the confidence values
- If the emotion field is a comma-separated list of labels, it uses the number of labels as a rough proxy for intensity
- If the field is empty or invalid, it returns `0.0`

For simple output files, the emotion signal is therefore only a rough approximation. For comprehensive output files, the value is more informative because it is based on confidence-weighted model output.

### Step 4: Loading the CSV Into a Unified DataFrame

The function `load_articles()` reads the input CSV row by row and creates a normalized table with these columns:

- `company`
- `date`
- `url`
- `article_sentiment`
- `reader_sentiment`
- `article_emotion_intensity`
- `reader_emotion_intensity`

It supports both simple and comprehensive pipeline outputs by checking multiple possible column names. For example:

- article sentiment comes from `simple_article_sentiment` or `comp_article_sentiment`
- reader sentiment comes from `simple_reader_sentiment` or `comp_reader_sentiment`
- emotion data comes from the corresponding simple or comprehensive emotion columns

If both sentiment fields are missing or invalid for a row, that row is skipped.

### Step 5: Aggregating Article-Level Data Into Daily Company Signals

The CSV contains one row per article, but stock prices are analyzed over time, so the script groups articles by company and by day.

The function `daily_sentiment()`:

1. Filters the normalized article table to one company
2. Groups all rows by date
3. Computes the mean of:
   - `article_sentiment`
   - `reader_sentiment`
   - `article_emotion_intensity`
   - `reader_emotion_intensity`

So if several Tesla articles were published on the same day, their scores are averaged into one daily Tesla record.

This creates the main daily sentiment time-series used in every downstream analysis.

### Step 6: Downloading Stock Prices

The function `fetch_stock_prices()` uses the `yfinance` package to download historical data from Yahoo Finance.

It requests daily data for the date range surrounding the available articles, then keeps only the `Close` column and renames it to `close`.

The stock DataFrame uses dates as its index so that it can be joined easily with the daily sentiment table.

The script requests a wider date range than the article dates themselves:

- earliest article date minus `--days-before`
- latest article date plus `--days-after`

This helps avoid edge effects and allows lead/lag analysis to compare nearby dates.

### Step 7: Plotting Sentiment and Stock Price Together

The function `plot_company()` creates a dual-axis line chart for each company.

The chart includes:

- article sentiment
- reader sentiment
- article emotion intensity
- reader emotion intensity
- stock closing price

The left y-axis is used for sentiment-style signals. Since emotion intensity naturally ranges from `0` to `1`, the script rescales it to `-1` to `1` using:

`rescaled = intensity * 2 - 1`

This lets the emotion lines share the same visual axis as sentiment scores.

The right y-axis is used for stock price in dollars.

The resulting PNG is saved as:

- `analysis_output/<company>_sentiment_vs_stock.png`

This plot is mainly a visual exploration tool. It lets you see whether spikes in sentiment or emotion appear near sharp stock moves.

### Step 8: Computing Basic Correlations

The function `compute_correlations()` measures how strongly each sentiment metric is associated with stock movement.

It does **not** correlate sentiment with the raw stock price. Instead, it first computes:

`daily_return = close.pct_change()`

This is an important design choice. Correlating sentiment with raw price levels can be misleading because both series may drift over time. Using daily returns makes the comparison focus on day-to-day stock movement instead.

For each of these metrics:

- `article_sentiment`
- `reader_sentiment`
- `article_emotion_intensity`
- `reader_emotion_intensity`

the script computes:

- Pearson correlation
- Spearman correlation
- p-values for both
- the number of aligned days used

Pearson checks for linear association. Spearman checks for monotonic rank association and is less sensitive to outliers and nonlinearity.

The results are collected per company and later written to:

- `analysis_output/correlation_summary.csv`

### Step 9: Lead/Lag Analysis

The function `lead_lag_analysis()` asks a more specific question:

Does sentiment come before stock movement, or does stock movement come before sentiment?

To test this, the script shifts the sentiment series forward and backward by up to 5 calendar days and then recomputes the Pearson correlation with stock returns.

Interpretation:

- positive lag means earlier sentiment is paired with later stock returns
- negative lag means stock returns come earlier and sentiment appears afterward

This does not prove causation, but it can suggest whether sentiment tends to move before or after stock changes.

The lead/lag results are saved to:

- `analysis_output/lead_lag_analysis.csv`

### Step 10: Rolling Correlation Over Time

A single correlation number can hide the fact that the relationship may change over time. The script therefore computes a rolling correlation using a 14-day window.

The function `rolling_correlation()`:

1. Joins daily sentiment with stock data
2. Converts prices to daily returns
3. Computes rolling Pearson correlation for:
   - `article_sentiment`
   - `reader_sentiment`

The companion function `plot_rolling_correlation()` saves a chart showing how those correlations evolve over time:

- `analysis_output/<company>_rolling_correlation.png`

This is useful when sentiment only matters during certain periods, such as earnings season, scandals, or other bursts of high news volume.

### Step 11: Saving Final Outputs

After iterating over all companies, the script writes summary files into `analysis_output/`.

Typical outputs are:

- one sentiment-vs-stock chart per company
- one rolling-correlation chart per company
- `correlation_summary.csv`
- `lead_lag_analysis.csv`

The script also prints progress and warnings to standard error, such as:

- when no ticker mapping exists for a company
- when no overlapping dates are available
- when too few observations exist for correlation analysis

### Practical Interpretation of the Results

The outputs should be interpreted carefully.

What the correlations can tell you:

- whether more positive or negative sentiment tends to occur on days with positive or negative returns
- whether reader sentiment behaves differently from article sentiment
- whether broad emotional intensity seems associated with larger stock moves
- whether the relationship looks immediate or delayed

What they do **not** tell you:

- that sentiment causes stock moves
- that the effect is stable over time
- that article publication dates perfectly reflect when the market received the information

In practice, these results are best used as exploratory evidence rather than definitive proof.

### Limitations and Simplifications

Several simplifications are built into the current script:

- sentiment labels are collapsed into a very small numeric scale
- `mixed` is treated the same as `neutral`
- emotion labels are reduced to a single intensity score
- stock data is daily rather than intraday
- article dates may not line up exactly with trading hours
- correlations are based only on overlapping dates, which can reduce sample size

So this script is a strong first-pass analysis, but not a full econometric model.

### In One Sentence

`stock_correlation.py` converts article sentiment into daily company signals, lines those signals up with stock returns, and produces plots plus summary statistics that help you explore whether news sentiment and market movement appear related.
