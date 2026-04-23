#!/bin/bash
# Full autoresearch pipeline — the ONE thing to launch for this project.
#
# Each cycle runs three steps in sequence:
#   1. novel_contrast  — Opus 4.7 invents N_NOVEL new abstract axes
#   2. seed_lineages   — promote any newly-scored high-scoring axes
#                         into gen-0 lineage leaders for hillclimb to refine
#   3. hillclimb       — generate N_HILLCLIMB mutations of existing leaders;
#                         worker commits / rejects based on parent score
#
# After all three, sleep CYCLE_SECONDS, then repeat.
#
# The worker (scripts/start_worker.sh) consumes the queue throughout.
# Seed_lineages promotes *previously-evaluated* high-scorers, so candidates
# invented in cycle N become seed-eligible for cycle N+1 (after the worker
# has had time to score them during the sleep).
#
# Environment:
#   N_NOVEL         novel_contrast pairs per cycle (default 5)
#   N_HILLCLIMB     hillclimb mutations per cycle (default 10)
#   SEED_TOP        how many top contrast_pair axes seed_lineages considers
#                   (default 15 — skips already-seeded ones internally)
#   CYCLE_SECONDS   sleep between cycles (default 1800 = 30 min)
#
# Examples:
#   ./scripts/start_autoresearch.sh
#   N_NOVEL=10 N_HILLCLIMB=15 ./scripts/start_autoresearch.sh
#
#   tail -f logs/autoresearch.log
#   pkill -f 'start_autoresearch.loop'   # stop the loop

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

N_NOVEL="${N_NOVEL:-5}"
N_HILLCLIMB="${N_HILLCLIMB:-10}"
SEED_TOP="${SEED_TOP:-15}"
CYCLE_SECONDS="${CYCLE_SECONDS:-1800}"
LOOP_TAG="start_autoresearch.loop"

if pgrep -f "$LOOP_TAG" > /dev/null; then
    echo "autoresearch loop already running:"
    pgrep -af "$LOOP_TAG"
    exit 1
fi

# Block if the legacy start_researcher.sh loop is running — both would
# write to queue/pending and double-fill it.
if pgrep -f "start_researcher.loop" > /dev/null; then
    echo "start_researcher.loop is already running — stop it first:"
    echo "  pkill -f start_researcher.loop"
    exit 1
fi

LOG="logs/autoresearch.log"
echo "=== starting autoresearch loop at $(date) novel=$N_NOVEL hillclimb=$N_HILLCLIMB seed_top=$SEED_TOP cycle=${CYCLE_SECONDS}s ===" >> "$LOG"

# Compose the loop inline so $LOOP_TAG is findable in `pgrep -f`.
LOOP_SCRIPT="
# $LOOP_TAG
while true; do
    ts=\$(date '+%Y-%m-%d %H:%M:%S')
    echo \"--- autoresearch cycle \$ts novel=$N_NOVEL hillclimb=$N_HILLCLIMB ---\" >> \"$LOG\"

    echo \"[autoresearch \$ts] step 1/3: novel_contrast --n $N_NOVEL\" >> \"$LOG\"
    .venv/bin/python -m src.researcher --strategy novel_contrast --n $N_NOVEL >> \"$LOG\" 2>&1 || true

    echo \"[autoresearch \$(date '+%H:%M:%S')] step 2/3: seed_lineages --top $SEED_TOP\" >> \"$LOG\"
    .venv/bin/python scripts/seed_lineages.py --top $SEED_TOP >> \"$LOG\" 2>&1 || true

    echo \"[autoresearch \$(date '+%H:%M:%S')] step 3/3: hillclimb --n $N_HILLCLIMB\" >> \"$LOG\"
    .venv/bin/python -m src.researcher --strategy hillclimb --n $N_HILLCLIMB >> \"$LOG\" 2>&1 || true

    # Compact state summary — top-5 lineage leaders + population counts.
    sqlite3 data/results.db \"
      SELECT printf('[state] lineages=%d  leaders=%d  identified=%d  done=%d',
        COUNT(DISTINCT c.lineage_id),
        SUM(CASE WHEN c.is_leader=1 THEN 1 ELSE 0 END),
        (SELECT COUNT(*) FROM fitness_scores WHERE identification_rate > 0),
        (SELECT COUNT(*) FROM candidates WHERE status='done'))
      FROM candidates c WHERE c.lineage_id IS NOT NULL;\" >> \"$LOG\" 2>&1 || true
    sqlite3 data/results.db \"
      SELECT printf('[top5] %-42s  s=%.3f d=%.0f%% i=%.0f%% gen=%d',
        substr(c.concept,1,42), f.score, f.detection_rate*100,
        f.identification_rate*100, c.generation)
      FROM candidates c JOIN fitness_scores f ON f.candidate_id=c.id
      WHERE c.is_leader=1 ORDER BY f.score DESC LIMIT 5;\" >> \"$LOG\" 2>&1 || true

    echo \"[autoresearch \$(date '+%H:%M:%S')] cycle complete, sleeping ${CYCLE_SECONDS}s\" >> \"$LOG\"
    sleep $CYCLE_SECONDS
done
"

nohup bash -c "$LOOP_SCRIPT" >> "$LOG" 2>&1 < /dev/null &
disown $! 2>/dev/null || true
LOOP_PID=$!
echo "autoresearch loop started as PID $LOOP_PID"
echo "logs: tail -f $LOG"
echo "novel=$N_NOVEL hillclimb=$N_HILLCLIMB seed_top=$SEED_TOP cycle=${CYCLE_SECONDS}s"
