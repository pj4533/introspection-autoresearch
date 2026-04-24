#!/bin/bash
# Start the Phase 2 worker as a detached long-lived process.
#
# Default mode (2026-04-24 onward, ADR-017): the worker installs paper-method
# refusal-direction abliteration hooks on top of vanilla Gemma3-12B at startup
# and every candidate is evaluated under those hooks. This matches the Macar
# et al. (2026) §3.3 protocol reproduced in Phase 1.5 (~3.6× detection boost
# on dictionary-word directions at 0% FPR).
#
# Environment:
#   VANILLA      "1" to DISABLE paper-method abliteration (sensitivity-check
#                runs; pre-2026-04-24 behavior). Default unset = paper-method.
#
# Examples:
#   ./scripts/start_worker.sh              # paper-method abliteration ACTIVE
#   VANILLA=1 ./scripts/start_worker.sh    # vanilla Gemma3-12B, no hooks
#
#   tail -f logs/worker.log                # watch progress
#   ps aux | grep 'src.worker'             # confirm it's running
#   kill -TERM <pid>                       # graceful shutdown (finishes
#                                          # current candidate first)
#
# The OFF-THE-SHELF abliterated HF checkpoints (gemma3_12b_abliterated) are
# deprecated per ADR-013 because they produce catastrophic FPR (97.9% / 90%)
# on the project's protocol. Do not use them. This script no longer exposes
# them as an option — see scripts/start_worker.sh in git history if you
# need the legacy path.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# Match the actual Python process running `-m src.worker` (capital P on
# macOS Homebrew). Pattern matches literal argv substring `/Python -m
# src.worker` so it doesn't false-positive on shell scripts that just contain
# the regex text as a search string. Two 12B workers on MPS corrupt each
# other's activations.
if pgrep -f "/Python -m src\.worker" > /dev/null; then
    echo "worker already running:"
    pgrep -af "/Python -m src\.worker"
    exit 1
fi

VANILLA="${VANILLA:-}"
EXTRA_ARGS=""
if [[ -n "$VANILLA" ]]; then
    EXTRA_ARGS="--vanilla"
    MODE="vanilla"
else
    MODE="paper-method abliteration"
fi

LOG="logs/worker.log"
echo "=== starting worker at $(date) mode=${MODE} ===" >> "$LOG"
nohup .venv/bin/python -m src.worker $EXTRA_ARGS >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
WORKER_PID=$!
echo "worker started as PID $WORKER_PID (mode=${MODE})"
echo "logs: tail -f $LOG"
