#!/bin/bash
set -e

TARGET_YEAR_ARG="${1:-${TARGET_YEAR:-2024}}"
RUN_ROOT="${PIPELINE_RUNS_DIR:-yearly_runs}"
RUN_DIR="${PIPELINE_OUTPUT_DIR:-${RUN_ROOT}/${TARGET_YEAR_ARG}}"

export TARGET_YEAR="${TARGET_YEAR_ARG}"
export PIPELINE_OUTPUT_DIR="${RUN_DIR}"

echo "=== Step 0: Clean previous data ==="
rm -f "${RUN_DIR}/articles_raw.csv" \
      "${RUN_DIR}/sentiment_simple.csv" \
      "${RUN_DIR}/sentiment_comprehensive.csv" \
      "${RUN_DIR}/final_results.csv"
rm -rf "${RUN_DIR}/analysis_output"

echo "=== Target year: ${TARGET_YEAR} ==="
echo "=== Output dir:  ${RUN_DIR} ==="

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
echo "Results: ${RUN_DIR}/final_results.csv"
echo "Charts:  ${RUN_DIR}/analysis_output/"
