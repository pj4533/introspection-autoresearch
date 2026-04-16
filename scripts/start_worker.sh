#!/bin/bash
# Start the Phase 2 worker as a detached long-lived process.
#
#   ./scripts/start_worker.sh          # launch detached, log to logs/worker.log
#   tail -f logs/worker.log            # watch progress
#   ps aux | grep 'src.worker'         # confirm it's running
#   kill -TERM <pid>                   # graceful shutdown (finishes current candidate first)
#
# setsid nohup disowns the process from the current terminal. Closing the
# terminal or logging out won't kill the worker.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

if pgrep -f "python -m src.worker" > /dev/null; then
    echo "worker already running:"
    pgrep -af "python -m src.worker"
    exit 1
fi

LOG="logs/worker.log"
echo "=== starting worker at $(date) ===" >> "$LOG"
setsid nohup .venv/bin/python -m src.worker >> "$LOG" 2>&1 < /dev/null &
WORKER_PID=$!
echo "worker started as PID $WORKER_PID"
echo "logs: tail -f $LOG"
