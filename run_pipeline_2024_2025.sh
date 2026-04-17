#!/bin/bash
set -e

YEARS=("$@")
if [ ${#YEARS[@]} -eq 0 ]; then
  YEARS=(2024 2025)
fi

RUN_ROOT="${PIPELINE_RUNS_DIR:-yearly_runs}"

for YEAR in "${YEARS[@]}"; do
  echo ""
  echo "=============================="
  echo "Running pipeline for ${YEAR}"
  echo "=============================="
  PIPELINE_OUTPUT_DIR="${RUN_ROOT}/${YEAR}" bash run_pipeline.sh "${YEAR}"
done

echo ""
echo "All requested years completed."
echo "Outputs saved under: ${RUN_ROOT}/"
