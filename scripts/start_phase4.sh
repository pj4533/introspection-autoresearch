#!/usr/bin/env bash
# Launch Phase 4 dream loop in the background.
#
# Logs to logs/phase4_dreamloop.log. Stop with:
#   pkill -f phase4_dreamloop
#
# Status check:
#   tail -F logs/phase4_dreamloop.log
#   sqlite3 data/results.db 'SELECT * FROM phase4_chains ORDER BY started_at DESC LIMIT 5'

set -e
cd "$(dirname "$0")/.."

# Make sure the venv exists.
if [[ ! -f .venv/bin/python ]]; then
  echo "ERROR: .venv not found. Run 'python -m venv .venv && pip install -e .' first."
  exit 1
fi

mkdir -p logs

# Don't double-launch.
if pgrep -f "phase4_dreamloop" > /dev/null; then
  echo "Already running. Stop first with: pkill -f phase4_dreamloop"
  exit 1
fi

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  # Default: production overnight. 5 chains per cycle, unlimited cycles.
  ARGS=(--batch 5 --length-cap 20)
fi

setsid nohup .venv/bin/python scripts/run_phase4_dreamloop.py "${ARGS[@]}" \
  > logs/phase4_dreamloop.log 2>&1 < /dev/null &

DREAM_PID=$!
echo "Phase 4 dream loop started — PID $DREAM_PID"
echo "Log: logs/phase4_dreamloop.log"
echo
echo "Watch progress:"
echo "  tail -F logs/phase4_dreamloop.log"
echo
echo "Stop:"
echo "  pkill -f phase4_dreamloop"
