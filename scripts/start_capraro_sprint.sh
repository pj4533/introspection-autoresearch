#!/bin/bash
# Run a focused Capraro fault-line sprint.
#
# Each sprint targets ONE fault line. The cycle is:
#   1. seed pass: emit the hand-written seeds for the fault line (no Opus cost)
#   2. wait for the worker to evaluate them
#   3. opus pass: ask Opus 4.7 for variants biased by feedback from step 2's
#      results — runs every CYCLE_SECONDS until the sprint is stopped
#
# Worker should be running separately (./scripts/start_worker.sh, vanilla
# default per ADR-017 rev 2). FITNESS_MODE is set to ident_prioritized
# automatically because the Capraro 3-class outcome table hinges on
# identification, not raw detection.
#
# Environment:
#   FAULT_LINE        required: one of causality / grounding / experience /
#                     metacognition. (Pass as $1 if not set.)
#   N_PER_CYCLE       candidates per Opus cycle (default 16 — 4 layers × 4
#                     pairs at random target_effective)
#   CYCLE_SECONDS     sleep between Opus cycles (default 1800 = 30 min)
#   SKIP_SEED         "1" to skip the seed pass and go straight to Opus
#                     variants (use if seeds were already emitted earlier)
#
# Examples:
#   ./scripts/start_capraro_sprint.sh causality
#   FAULT_LINE=grounding ./scripts/start_capraro_sprint.sh
#   N_PER_CYCLE=8 CYCLE_SECONDS=900 ./scripts/start_capraro_sprint.sh experience
#
#   tail -f logs/capraro_<fault>.log
#   pkill -f 'start_capraro_sprint.loop'   # stop

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

FAULT_LINE="${FAULT_LINE:-${1:-}}"
if [[ -z "$FAULT_LINE" ]]; then
    echo "ERROR: fault line required. Usage:"
    echo "  ./scripts/start_capraro_sprint.sh <fault_line>"
    echo "  available: causality, grounding, experience, metacognition"
    exit 2
fi

N_PER_CYCLE="${N_PER_CYCLE:-16}"
CYCLE_SECONDS="${CYCLE_SECONDS:-1800}"
SKIP_SEED="${SKIP_SEED:-}"
LOOP_TAG="start_capraro_sprint.loop.${FAULT_LINE}"

if pgrep -f "$LOOP_TAG" > /dev/null; then
    echo "capraro sprint loop for $FAULT_LINE already running:"
    pgrep -af "$LOOP_TAG"
    exit 1
fi

LOG="logs/capraro_${FAULT_LINE}.log"
echo "=== starting capraro sprint at $(date) fault_line=$FAULT_LINE n=$N_PER_CYCLE cycle=${CYCLE_SECONDS}s ===" >> "$LOG"

LOOP_SCRIPT="
# $LOOP_TAG
export FITNESS_MODE=ident_prioritized

# Step 1: seed pass (skip if requested)
if [[ -z \"$SKIP_SEED\" ]]; then
    echo \"--- capraro \$(date '+%H:%M:%S') seed pass for $FAULT_LINE ---\" >> \"$LOG\"
    .venv/bin/python -m src.researcher \\
        --strategy directed_capraro \\
        --fault-line $FAULT_LINE \\
        --capraro-mode seed \\
        --n 100 >> \"$LOG\" 2>&1 || true
    echo \"--- capraro \$(date '+%H:%M:%S') seeds queued — sleeping 60s before first opus cycle ---\" >> \"$LOG\"
    sleep 60
fi

# Step 2: opus variant cycles, indefinite
while true; do
    echo \"--- capraro \$(date '+%H:%M:%S') opus cycle for $FAULT_LINE n=$N_PER_CYCLE ---\" >> \"$LOG\"
    .venv/bin/python -m src.researcher \\
        --strategy directed_capraro \\
        --fault-line $FAULT_LINE \\
        --capraro-mode opus \\
        --n $N_PER_CYCLE >> \"$LOG\" 2>&1 || true
    echo \"--- capraro \$(date '+%H:%M:%S') sleeping ${CYCLE_SECONDS}s ---\" >> \"$LOG\"
    sleep $CYCLE_SECONDS
done
"

nohup bash -c "$LOOP_SCRIPT" >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
LOOP_PID=$!
echo "capraro sprint loop started as PID $LOOP_PID for fault_line=$FAULT_LINE"
echo "logs: tail -f $LOG"
echo "fitness mode: ident_prioritized (set in loop env)"
echo "stop:  pkill -f '$LOOP_TAG'"
