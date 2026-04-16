#!/bin/bash
# Start the Phase 2 researcher as a detached periodic process.
#
# The researcher itself is short-lived — it runs one cycle, writes candidates,
# exits. This wrapper loops it every CYCLE_SECONDS (default 1800 = 30 min).
#
#   ./scripts/start_researcher.sh      # launch detached, log to logs/researcher.log
#   tail -f logs/researcher.log
#   pkill -f 'start_researcher.loop'   # stop the loop (see LOOP_TAG below)

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

STRATEGY="${STRATEGY:-random}"
N_CANDIDATES="${N_CANDIDATES:-10}"
CYCLE_SECONDS="${CYCLE_SECONDS:-1800}"
LOOP_TAG="start_researcher.loop"   # used by pkill

if pgrep -f "$LOOP_TAG" > /dev/null; then
    echo "researcher loop already running:"
    pgrep -af "$LOOP_TAG"
    exit 1
fi

LOG="logs/researcher.log"
echo "=== starting researcher loop at $(date) ===" >> "$LOG"

# Compose the loop as an inline bash script. The LOOP_TAG comment makes it
# findable with pgrep/pkill.
LOOP_SCRIPT="
# $LOOP_TAG
while true; do
    echo \"--- cycle \$(date) strategy=$STRATEGY n=$N_CANDIDATES ---\" >> \"$LOG\"
    .venv/bin/python -m src.researcher --strategy \"$STRATEGY\" --n \"$N_CANDIDATES\" >> \"$LOG\" 2>&1 || true
    sleep \"$CYCLE_SECONDS\"
done
"

nohup bash -c "$LOOP_SCRIPT" >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
LOOP_PID=$!
echo "researcher loop started as PID $LOOP_PID"
echo "logs: tail -f $LOG"
