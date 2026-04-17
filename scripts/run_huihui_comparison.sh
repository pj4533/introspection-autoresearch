#!/bin/bash
# Quick comparison sweep against huihui-ai abliteration
# Runs after the mlabonne sweep finishes. ~15 minutes.
set -e
cd "$(dirname "$0")/.."
rm -f data/results_abliterated_huihui.db
.venv/bin/python scripts/run_phase1_sweep.py \
  --model gemma3_12b_abliterated_huihui \
  --db data/results_abliterated_huihui.db \
  --concepts 10 \
  --layers 33 36 40 44 \
  --run-id huihui-comparison
