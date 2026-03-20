"""
Shared configuration for the company news sentiment analysis pipeline.

All prompt templates, company list, model settings, and CSV column
definitions live here so every script imports from a single source of truth.
"""

from __future__ import annotations

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Target Companies
# ---------------------------------------------------------------------------
# COMPANIES = [
#     "Apple",
#     "Tesla",
#     "Amazon",
#     "Microsoft",
#     "Google",
#     "Meta",
#     "Netflix",
#     "NVIDIA",
# ]
COMPANIES = [
    "Tesla",
]
# ---------------------------------------------------------------------------
# Search & Scrape Settings
# ---------------------------------------------------------------------------
_CURRENT_YEAR = datetime.now().year
SEARCH_QUERIES_PER_COMPANY = [
    "{company} news latest",
    f"{{company}} company news {_CURRENT_YEAR}",
    f"{{company}} company news {_CURRENT_YEAR - 1}",
    f"{{company}} company news {_CURRENT_YEAR - 2}",
]
MAX_SEARCH_RESULTS = 50  # per query
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v1"
SCRAPE_WORKERS = 8

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
ANALYSIS_WORKERS = 6

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
ARTICLES_RAW_CSV = "articles_raw.csv"
SENTIMENT_SIMPLE_CSV = "sentiment_simple.csv"
SENTIMENT_COMPREHENSIVE_CSV = "sentiment_comprehensive.csv"
FINAL_RESULTS_CSV = "final_results.csv"

# Column names for articles_raw.csv
RAW_COLUMNS = [
    "company", "url", "title", "article_text",
    "published_date", "author", "search_query",
]

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
MAX_ARTICLE_CHARS = 48_000  # truncate articles beyond this to stay within context
