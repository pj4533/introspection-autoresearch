"""Seed Phase 2c hill-climbing lineages from existing top contrast_pair candidates.

For the top-N scoring contrast_pair candidates in the DB, assign each a
fresh lineage_id and mark it as a gen-0 seed leader. This gives the
hillclimb strategy seeds to mutate from when it starts up.

Safe to re-run: existing seeds (already with a lineage_id) are skipped.

Usage:
    python scripts/seed_lineages.py            # top 10
    python scripts/seed_lineages.py --top 20   # top 20
    python scripts/seed_lineages.py --dry-run
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "results.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10,
                    help="How many top contrast_pair axes to seed as lineages")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    # Deduplicate by axis name — one lineage per unique axis, keep the
    # highest-scoring variant.
    rows = conn.execute("""
        SELECT c.id, c.concept, c.lineage_id, f.score,
               f.detection_rate, f.identification_rate
        FROM candidates c
        JOIN fitness_scores f ON f.candidate_id = c.id
        WHERE c.derivation_method = 'contrast_pair' AND f.score > 0
        ORDER BY f.score DESC
    """).fetchall()

    seen_axes: set[str] = set()
    seeds_to_create: list[sqlite3.Row] = []
    already_seeded: list[sqlite3.Row] = []
    for r in rows:
        if r["concept"] in seen_axes:
            continue
        seen_axes.add(r["concept"])
        if r["lineage_id"]:
            already_seeded.append(r)
        else:
            seeds_to_create.append(r)
        if len(seeds_to_create) + len(already_seeded) >= args.top:
            break

    print(f"Phase 2c lineage seeding")
    print(f"  already seeded:      {len(already_seeded)}")
    print(f"  to seed (new):       {len(seeds_to_create)}")
    print(f"  skipped non-unique:  (dedup by axis name)")
    print()

    for r in seeds_to_create:
        lid = uuid.uuid4().hex[:16]
        print(
            f"  seed  lineage={lid}  "
            f"score={r['score']:.3f}  det={r['detection_rate']:.0%}  "
            f"axis={r['concept']!r}  candidate={r['id']}"
        )
        if not args.dry_run:
            conn.execute(
                """UPDATE candidates SET
                     lineage_id = ?, parent_candidate_id = NULL,
                     generation = 0, is_leader = 1,
                     mutation_type = 'seed'
                   WHERE id = ?""",
                (lid, r["id"]),
            )
    for r in already_seeded:
        print(
            f"  skip  lineage={r['lineage_id']}  score={r['score']:.3f}  "
            f"axis={r['concept']!r}  (already seeded)"
        )
    if not args.dry_run:
        conn.commit()
    print()
    print("done." if not args.dry_run else "(dry run — no writes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
