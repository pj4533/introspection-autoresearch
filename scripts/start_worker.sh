#!/bin/bash
# Launch the four-phase local-only worker (worker) in the background.
#
# Architecture: serial model swap. Gemma3-12B for generation, then unload;
# Qwen3.6-35B-A3B-8bit (or whatever judge passed calibration) for judging,
# then unload; Qwen3.6-27B-MLX-8bit (or alt) for proposing, then unload.
# At most ONE model is in memory at any time.
#
# Environment overrides:
#   JUDGE_MODEL_PATH     default ~/models/Qwen3.6-35B-A3B-8bit
#   PROPOSER_MODEL_PATH  default ~/models/Qwen3.6-27B-MLX-8bit
#   FAULT_LINE           if set, Phase C uses directed_capraro for that fault
#                         line; otherwise novel_contrast
#   BATCH_SIZE           candidates per Phase A batch (default 16)
#   PROPOSE_THRESHOLD    Phase C runs when queue/pending dips below this
#                         (default 4)
#   PROPOSE_N            candidates per Phase C call (default 16)
#   MAX_CYCLES           0 = unlimited
#
# Examples:
#   ./scripts/start_worker.sh
#   FAULT_LINE=grounding ./scripts/start_worker.sh
#   BATCH_SIZE=8 ./scripts/start_worker.sh
#
#   tail -f logs/worker.log
#   pkill -f 'worker.loop'    # stop

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

if pgrep -f 'src.worker' > /dev/null; then
    echo "worker already running:"
    pgrep -af 'src.worker'
    exit 1
fi

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$HOME/models/Qwen3.6-35B-A3B-8bit}"
PROPOSER_MODEL_PATH="${PROPOSER_MODEL_PATH:-$HOME/models/Qwen3.6-27B-MLX-8bit}"
FAULT_LINE="${FAULT_LINE:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PROPOSE_THRESHOLD="${PROPOSE_THRESHOLD:-4}"
PROPOSE_N="${PROPOSE_N:-16}"
MAX_CYCLES="${MAX_CYCLES:-0}"

# Sanity: judge directory must exist (proposer is checked at first Phase C)
if [[ ! -d "$JUDGE_MODEL_PATH" ]]; then
    echo "ERROR: judge model dir not found: $JUDGE_MODEL_PATH"
    echo "Download with:"
    echo "  hf download mlx-community/Qwen3.6-35B-A3B-8bit \\"
    echo "    --local-dir $JUDGE_MODEL_PATH"
    exit 2
fi

LOG="logs/worker.log"
echo "=== starting worker at $(date) ===" >> "$LOG"
echo "    judge:    $JUDGE_MODEL_PATH" >> "$LOG"
echo "    proposer: $PROPOSER_MODEL_PATH" >> "$LOG"
echo "    fault_line: ${FAULT_LINE:-<none, novel_contrast>}" >> "$LOG"
echo "    batch_size=$BATCH_SIZE propose_threshold=$PROPOSE_THRESHOLD" >> "$LOG"

ARGS=(
    "--judge-model-path" "$JUDGE_MODEL_PATH"
    "--proposer-model-path" "$PROPOSER_MODEL_PATH"
    "--batch-size" "$BATCH_SIZE"
    "--propose-threshold" "$PROPOSE_THRESHOLD"
    "--propose-n" "$PROPOSE_N"
    "--max-cycles" "$MAX_CYCLES"
)
if [[ -n "$FAULT_LINE" ]]; then
    ARGS+=("--fault-line" "$FAULT_LINE")
fi

setsid nohup .venv/bin/python -m src.worker "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
WORKER_PID=$!
echo "worker started as PID $WORKER_PID"
echo "logs: tail -f $LOG"
echo "stop: pkill -f 'src.worker'"
