#!/bin/bash
# Start the Phase 2 worker as a detached long-lived process.
#
# Default mode (2026-04-25 onward, ADR-017 rev 2): vanilla Gemma3-12B with
# NO abliteration hooks. Every Phase 2b/2c/2d result that worked happened on
# vanilla; paper-method abliteration was found to suppress refusal-adjacent
# Altman-style content (Phase 2d-1, 2026-04-24). Vanilla is the productive
# default.
#
# Paper-method abliteration is available as an opt-in via ABLITERATED=1.
# Use it for dictionary-word axes (where Phase 1.5 measured ~3.6× boost) or
# for sensitivity probes that test refusal-entanglement of a specific axis.
# Do NOT use it as a default for the Altman/Capraro/Epistemia clusters.
#
# Environment:
#   ABLITERATED  "1" to enable paper-method abliteration for this worker.
#                Default unset = vanilla.
#
# Examples:
#   ./scripts/start_worker.sh              # vanilla Gemma3-12B (default)
#   ABLITERATED=1 ./scripts/start_worker.sh   # paper-method abliteration ACTIVE
#
#   tail -f logs/worker.log                # watch progress
#   ps aux | grep 'src.worker'             # confirm it's running
#   kill -TERM <pid>                       # graceful shutdown (finishes
#                                          # current candidate first)
#
# The OFF-THE-SHELF abliterated HF checkpoints (gemma3_12b_abliterated) are
# deprecated per ADR-013 because they produce catastrophic FPR (97.9% / 90%)
# on the project's protocol. Do not use them.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

if pgrep -f "/Python -m src\.worker" > /dev/null; then
    echo "worker already running:"
    pgrep -af "/Python -m src\.worker"
    exit 1
fi

ABLITERATED="${ABLITERATED:-}"
EXTRA_ARGS=""
if [[ -n "$ABLITERATED" ]]; then
    EXTRA_ARGS="--abliterate-paper"
    MODE="paper-method abliteration"
else
    MODE="vanilla"
fi

LOG="logs/worker.log"
echo "=== starting worker at $(date) mode=${MODE} ===" >> "$LOG"
nohup .venv/bin/python -m src.worker $EXTRA_ARGS >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
WORKER_PID=$!
echo "worker started as PID $WORKER_PID (mode=${MODE})"
echo "logs: tail -f $LOG"
