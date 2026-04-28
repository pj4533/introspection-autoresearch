#!/bin/bash
# Launch the three-phase local-only Phase 2g worker in the background.
#
# Architecture: serial model swap, two slots only.
# Gemma3-12B for generation, unload; Qwen3.6-35B-A3B-8bit (judge) for
# judging, unload; sae_capraro proposes new candidates with NO model
# loaded (pure CPU work, reads data/sae_features/capraro_buckets.json
# and the leaderboard). Phase 2g removed the proposer model.
#
# Environment overrides:
#   JUDGE_MODEL_PATH     default ~/models/Qwen3.6-35B-A3B-8bit
#   FAULT_LINES          comma-separated rotation. Default: all 7 Capraro
#                        fault lines (experience,causality,grounding,
#                        metacognition,parsing,motivation,value). Each
#                        Phase C call advances to the next; wraps at end.
#   BATCH_SIZE           candidates per Phase A batch (default 16)
#   PROPOSE_THRESHOLD    Phase C runs when queue/pending dips below this
#                         (default 4)
#   PROPOSE_N            candidates per Phase C call (default 16)
#   MAX_CYCLES           0 = unlimited
#   SAE_CAPRARO_BATCH_COMPOSITION   "<exp>:<nbr>:<rep>:<crs>"  default "6:6:3:1"
#
# Examples:
#   ./scripts/start_worker.sh
#   FAULT_LINES=causality,grounding ./scripts/start_worker.sh
#   BATCH_SIZE=8 ./scripts/start_worker.sh
#
#   tail -f logs/worker.log
#   pkill -f 'src.worker'    # stop

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

if pgrep -f 'src.worker' > /dev/null; then
    echo "worker already running:"
    pgrep -af 'src.worker'
    exit 1
fi

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-8bit}"
FAULT_LINES="${FAULT_LINES:-experience,causality,grounding,metacognition,parsing,motivation,value}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PROPOSE_THRESHOLD="${PROPOSE_THRESHOLD:-4}"
PROPOSE_N="${PROPOSE_N:-16}"
MAX_CYCLES="${MAX_CYCLES:-0}"

# Sanity: judge dir must exist; capraro_buckets.json must exist.
if [[ ! -d "$JUDGE_MODEL_PATH" ]]; then
    echo "ERROR: judge model dir not found: $JUDGE_MODEL_PATH"
    echo "Download with:"
    echo "  hf download mlx-community/Qwen3.6-35B-A3B-8bit \\"
    echo "    --local-dir $JUDGE_MODEL_PATH"
    exit 2
fi
if [[ ! -f "data/sae_features/capraro_buckets.json" ]]; then
    echo "ERROR: capraro_buckets.json not found."
    echo "Build it first: python scripts/build_capraro_buckets.py"
    exit 3
fi

LOG="logs/worker.log"
echo "=== starting Phase 2g worker at $(date) ===" >> "$LOG"
echo "    judge:    $JUDGE_MODEL_PATH" >> "$LOG"
echo "    strategy: sae_capraro (no proposer model)" >> "$LOG"
echo "    fault_lines: $FAULT_LINES" >> "$LOG"
echo "    batch_size=$BATCH_SIZE propose_threshold=$PROPOSE_THRESHOLD" >> "$LOG"

ARGS=(
    "--judge-model-path" "$JUDGE_MODEL_PATH"
    "--batch-size" "$BATCH_SIZE"
    "--propose-threshold" "$PROPOSE_THRESHOLD"
    "--propose-n" "$PROPOSE_N"
    "--max-cycles" "$MAX_CYCLES"
    "--fault-lines" "$FAULT_LINES"
)

nohup .venv/bin/python -m src.worker "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
WORKER_PID=$!
disown $WORKER_PID 2>/dev/null || true
echo "worker started as PID $WORKER_PID"
echo "logs: tail -f $LOG"
echo "stop: pkill -f 'src.worker'"
