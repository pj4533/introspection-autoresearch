#!/usr/bin/env bash
# Quick status check for the Phase 4 dream loop.
#
# Prints whether the loop is running, recent log lines, and DB summary.

set -e
cd "$(dirname "$0")/.."

echo "=== process ==="
if pgrep -af "phase4_dreamloop" > /dev/null; then
  pgrep -af "phase4_dreamloop"
  echo "  RUNNING"
else
  echo "  not running"
fi

echo
echo "=== last 20 log lines ==="
tail -20 logs/phase4_dreamloop.log 2>/dev/null || echo "(no log file yet)"

echo
echo "=== DB summary ==="
.venv/bin/python -c "
from src.db import ResultsDB
db = ResultsDB('data/results.db')
s = db.phase4_summary()
print(f\"  chains: {s['n_chains']}\")
print(f\"  total_steps: {s['total_steps']}\")
print(f\"  unjudged: {s['n_unjudged_steps']}\")
print(f\"  concepts in pool: {s['n_concepts']}\")
print(f\"  ended length_cap: {s['n_length_cap']}\")
print(f\"  ended self_loop: {s['n_self_loop']}\")
print(f\"  ended coherence_break: {s['n_coherence_break']}\")
"

echo
echo "=== top 10 concepts by visits ==="
.venv/bin/python -c "
import sqlite3
conn = sqlite3.connect('data/results.db')
conn.row_factory = sqlite3.Row
rows = conn.execute('''
    SELECT concept_lemma, visits, behavior_hits, cot_named_hits, cot_recognition_hits
    FROM phase4_concepts
    WHERE visits > 0
    ORDER BY visits DESC
    LIMIT 10
''').fetchall()
if not rows:
    print('  (no visited concepts yet)')
else:
    print(f'  {\"concept\":<20} {\"visits\":>7} {\"behavior\":>10} {\"named\":>7} {\"recog\":>7}')
    for r in rows:
        print(f'  {r[\"concept_lemma\"]:<20} {r[\"visits\"]:>7} {r[\"behavior_hits\"]:>10} {r[\"cot_named_hits\"]:>7} {r[\"cot_recognition_hits\"]:>7}')
conn.close()
"
