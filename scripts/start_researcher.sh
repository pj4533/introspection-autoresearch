#!/bin/bash
# Start the Phase 2 researcher as a detached periodic process.
#
# The researcher itself is short-lived — it runs one cycle, writes candidates,
# exits. This wrapper loops it every 30 minutes.
#
#   ./scripts/start_researcher.sh      # launch detached, log to logs/researcher.log
#   tail -f logs/researcher.log
#   pkill -f 'scripts/start_researcher.sh'   # stop the loop

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

STRATEGY="${STRATEGY:-random}"
N_CANDIDATES="${N_CANDIDATES:-10}"
CYCLE_SECONDS="${CYCLE_SECONDS:-1800}"

if pgrep -f "scripts/start_researcher.sh" | grep -v $$ > /dev/null 2>&1; then
    echo "researcher loop already running:"
    pgrep -af "scripts/start_researcher.sh" | grep -v $$
    exit 1
fi

LOG="logs/researcher.log"
echo "=== starting researcher loop at $(date) ===" >> "$LOG"

_loop() {
    while true; do
        echo "--- cycle $(date) strategy=$STRATEGY n=$N_CANDIDATES ---" >> "$LOG"
        .venv/bin/python -m src.researcher --strategy "$STRATEGY" --n "$N_CANDIDATES" >> "$LOG" 2>&1 || true
        sleep "$CYCLE_SECONDS"
    done
}

# Detach with setsid nohup so terminal close doesn't kill us
setsid nohup bash -c "$(declare -f _loop); _loop" >> "$LOG" 2>&1 < /dev/null &
LOOP_PID=$!
echo "researcher loop started as PID $LOOP_PID"
echo "logs: tail -f $LOG"
