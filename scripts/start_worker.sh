#!/bin/bash
# Start the Phase 2 worker as a detached long-lived process.
#
# Environment:
#   ABLITERATED  "1" to load gemma3_12b_abliterated and write to
#                data/results_abliterated.db (default unset = vanilla 12B)
#
# Examples:
#   ./scripts/start_worker.sh                 # vanilla
#   ABLITERATED=1 ./scripts/start_worker.sh   # abliterated overnight
#
#   tail -f logs/worker.log            # watch progress
#   ps aux | grep 'src.worker'         # confirm it's running
#   kill -TERM <pid>                   # graceful shutdown (finishes current candidate first)
#
# nohup + background + stdin-redirect detaches from the controlling terminal
# on macOS. (Linux setsid is not available on stock macOS, and nohup alone is
# sufficient here — the process won't receive SIGHUP when the terminal closes.)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

if pgrep -f "python -m src.worker" > /dev/null; then
    echo "worker already running:"
    pgrep -af "python -m src.worker"
    exit 1
fi

ABLITERATED="${ABLITERATED:-}"
EXTRA_ARGS=""
if [[ -n "$ABLITERATED" ]]; then
    EXTRA_ARGS="--model gemma3_12b_abliterated --db data/results_abliterated.db"
fi

LOG="logs/worker.log"
echo "=== starting worker at $(date) abliterated=${ABLITERATED:-no} ===" >> "$LOG"
nohup .venv/bin/python -m src.worker $EXTRA_ARGS >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
WORKER_PID=$!
echo "worker started as PID $WORKER_PID (abliterated=${ABLITERATED:-no})"
echo "logs: tail -f $LOG"
