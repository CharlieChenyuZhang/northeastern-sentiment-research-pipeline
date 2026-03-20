#!/bin/bash
set -e

echo "=== Step 0: Clean previous data ==="
rm -f articles_raw.csv sentiment_simple.csv sentiment_comprehensive.csv final_results.csv
rm -rf analysis_output/

echo "=== Step 1: Search and scrape ==="
python3.11 search_and_scrape.py

echo "=== Step 2a: Simple sentiment analysis ==="
python3.11 sentiment_simple.py

echo "=== Step 2b: Comprehensive sentiment analysis ==="
python3.11 sentiment_comprehensive.py

echo "=== Step 3: Merge results ==="
python3.11 merge_results.py

echo "=== Step 4: Stock correlation analysis ==="
python3.11 stock_correlation.py --days-before 90 --days-after 30

echo "=== Done! ==="
echo "Results: final_results.csv"
echo "Charts:  analysis_output/"
