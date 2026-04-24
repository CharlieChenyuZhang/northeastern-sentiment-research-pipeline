"""
Shared configuration for the company news sentiment analysis pipeline.

All prompt templates, company list, model settings, and CSV column
definitions live here so every script imports from a single source of truth.
"""

from __future__ import annotations

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Target Companies
# ---------------------------------------------------------------------------
# Research focus: one company and one fixed calendar year.
TARGET_YEAR = int(os.getenv("TARGET_YEAR", "2024"))
TARGET_START_DATE = date(TARGET_YEAR, 1, 1)
TARGET_END_DATE = date(TARGET_YEAR, 12, 31)
PIPELINE_OUTPUT_DIR = os.getenv("PIPELINE_OUTPUT_DIR", "").strip()

COMPANIES = [
    "JPMorgan Chase",
]

COMPANY_SEARCH_TERMS = {
    "JPMorgan Chase": "JPMorgan Chase JPM",
}
# ---------------------------------------------------------------------------
# Search & Scrape Settings
# ---------------------------------------------------------------------------
TARGET_MONTHS = [
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

SEARCH_QUERIES_PER_COMPANY = [
    (
        f"{{search_term}} news what happened in {month} {TARGET_YEAR}"
    )
    for month in TARGET_MONTHS
]
SERPAPI_RESULTS_PER_PAGE = 10
MAX_RESULTS_PER_SOURCE = 100
MAX_SEARCH_RESULTS = 250  # per query after combining all sources
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"
FIRECRAWL_SEARCH_SOURCES = ["web", "news"]
SCRAPE_WORKERS = 2
SCRAPE_MAX_RETRIES = 4
SCRAPE_RETRY_BACKOFF_SECONDS = 2.0
SCRAPE_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Firecrawl extraction prompt — tells the LLM what to pull from each page
FIRECRAWL_EXTRACTION_PROMPT = (
    "Extract the following from this news article page:\n"
    "1. 'title': The article headline.\n"
    "2. 'article_text': The full body text of the article (not navigation, ads, or sidebars).\n"
    "3. 'published_date': The publication date if available, otherwise 'n/a'.\n"
    "4. 'author': The author name if available, otherwise 'n/a'.\n"
    "Return a JSON object with exactly these four keys."
)

# ---------------------------------------------------------------------------
# OpenAI Model Settings
# ---------------------------------------------------------------------------
OPENAI_MODEL = "gpt-4o"  # supports logprobs; change to gpt-5.4 etc. as needed
OPENAI_TEMPERATURE = 0
OPENAI_TEMPERATURE_COMPREHENSIVE = 0.7  # per Wang et al. 2022 (self-consistency), 0.7 balances diversity & coherence
ANALYSIS_WORKERS = 3  # keep low to avoid OpenAI TPM rate limits

# ---------------------------------------------------------------------------
# Prompt Templates — stored as constants so they end up in the final output
# ---------------------------------------------------------------------------

# Single-call prompt: returns all 4 dimensions in one JSON response
PROMPT_ANALYSIS = (
    "Analyze the following news article and return a JSON object with exactly "
    "these four keys:\n"
    "\n"
    "1. \"article_sentiment\": The internal sentiment of the article itself. "
    "Must be one of: \"Positive\", \"Negative\", \"Neutral\", or \"Mixed\".\n"
    "\n"
    "2. \"reader_sentiment\": The sentiment a typical reader would feel after "
    "reading this article. Must be one of: \"Positive\", \"Negative\", "
    "\"Neutral\", or \"Mixed\".\n"
    "\n"
    "3. \"article_emotions\": A list of emotion labels that the article "
    "expresses (e.g. [\"optimism\", \"fear\", \"surprise\"]). Use whatever "
    "labels best fit — there is no fixed taxonomy.\n"
    "\n"
    "4. \"reader_emotions\": A list of emotion labels a typical reader would "
    "likely experience (e.g. [\"anxiety\", \"hope\", \"curiosity\"]). Use "
    "whatever labels best fit.\n"
    "\n"
    "Reply with ONLY the JSON object, no other text."
)

# Massive simple sentiment analysis — classify title and description separately.
PROMPT_MASSIVE_SENTIMENT_SIMPLE = (
    "Analyze the following financial news metadata and return a JSON object with "
    "exactly these two keys:\n"
    "\n"
    "1. \"sentiment_title\": The sentiment of the title only. Must be one of: "
    "\"Positive\", \"Negative\", \"Neutral\", or \"Mixed\". If the title is "
    "missing or empty, return an empty string.\n"
    "\n"
    "2. \"sentiment_description\": The sentiment of the description only. Must "
    "be one of: \"Positive\", \"Negative\", \"Neutral\", or \"Mixed\". If the "
    "description is missing or empty, return an empty string.\n"
    "\n"
    "Use only the provided title and description. Reply with ONLY the JSON "
    "object, no other text."
)

# Comprehensive analysis — single-call prompt that returns sentiment labels
# with confidence and emotion labels with confidence, all in one JSON.
COMP_PROMPT_ANALYSIS = (
    "Analyze the following news article. Return a JSON object with exactly "
    "these four keys:\n"
    "\n"
    "1. \"article_sentiment\": An object mapping each sentiment label to a "
    "confidence score (0.0–1.0). Labels: \"Positive\", \"Negative\", "
    "\"Neutral\", \"Mixed\". Scores should sum to approximately 1.0.\n"
    "\n"
    "2. \"reader_sentiment\": Same format — confidence distribution over "
    "sentiment labels for how a typical reader would feel.\n"
    "\n"
    "3. \"article_emotions\": An object mapping emotion labels to confidence "
    "scores (0.0–1.0). Use whatever emotion labels best fit the article "
    "(e.g. \"optimism\", \"anger\", \"fear\"). Only include emotions with "
    "meaningful presence (score >= 0.2). Scores are independent and do NOT "
    "need to sum to 1.\n"
    "\n"
    "4. \"reader_emotions\": Same format — emotion labels and confidence "
    "scores for emotions a typical reader would experience.\n"
    "\n"
    "Reply with ONLY the JSON object, no other text."
)

SENTIMENT_LABELS = ["Positive", "Negative", "Neutral", "Mixed"]

# Number of runs in comprehensive mode (for confidence merging)
COMPREHENSIVE_RUNS = 3

# ---------------------------------------------------------------------------
# CSV File Paths
# ---------------------------------------------------------------------------
def output_path(filename: str) -> str:
    return os.path.join(PIPELINE_OUTPUT_DIR, filename) if PIPELINE_OUTPUT_DIR else filename


ARTICLES_RAW_CSV = output_path("articles_raw.csv")
ARTICLES_RAW_MASSIVE_CSV = output_path("articles_raw_massive.csv")
SENTIMENT_SIMPLE_CSV = output_path("sentiment_simple.csv")
SENTIMENT_MASSIVE_SIMPLE_CSV = output_path("sentiment_massive_simple.csv")
SENTIMENT_COMPREHENSIVE_CSV = output_path("sentiment_comprehensive.csv")
FINAL_RESULTS_CSV = output_path("final_results.csv")
ANALYSIS_OUTPUT_DIR = output_path("analysis_output")

# Column names for articles_raw.csv
RAW_COLUMNS = [
    "company", "url", "title", "article_text",
    "published_date", "author", "search_query",
]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
MAX_ARTICLE_CHARS = 48_000  # truncate articles beyond this to stay within context
