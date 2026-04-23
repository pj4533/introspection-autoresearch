"""Recompute fitness_scores.score from stored components using the current
formula in src/evaluate.py, then re-establish is_leader per lineage based on
new scores. No judge calls. Safe to rerun.

Run this ONCE after changing the fitness formula in src/evaluate.py.

    python scripts/rescore_fitness.py                 # dry-run (diff preview)
    python scripts/rescore_fitness.py --yes           # apply

Keeps everything else (detection_rate / identification_rate / fpr /
coherence_rate / n_held_out / n_controls) unchanged — only the derived
`score` column and `candidates.is_leader` flags are rewritten.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "results.db"


def new_score(det: float, ident: float, fpr: float, coh: float) -> float:
    """Mirror of src/evaluate.py fitness — keep in sync by hand."""
    fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
    ident_bonus = 15.0 * ident
    return (det + ident_bonus) * fpr_penalty * coh


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true", help="apply changes (default dry-run)")
    ap.add_argument("--top-preview", type=int, default=10,
                    help="how many biggest movers to print (default 10)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT f.candidate_id, f.score AS old_score,
               f.detection_rate, f.identification_rate, f.fpr, f.coherence_rate,
               c.concept, c.lineage_id, c.is_leader AS old_is_leader, c.derivation_method
        FROM fitness_scores f
        JOIN candidates c ON c.id = f.candidate_id
    """).fetchall()

    print(f"[rescore] loaded {len(rows)} fitness rows")

    score_updates = []  # (candidate_id, old_score, new_score)
    for r in rows:
        ns = new_score(r["detection_rate"], r["identification_rate"], r["fpr"], r["coherence_rate"])
        score_updates.append((r["candidate_id"], r["old_score"], ns, r))

    # Compute new is_leader per lineage: within each lineage, highest new_score
    # becomes the single leader. Non-lineage candidates (lineage_id=NULL) are
    # untouched — their is_leader stays as-is (should already be 0 by convention).
    lineage_groups: dict[str, list] = {}
    for cid, old_score, ns, r in score_updates:
        lid = r["lineage_id"]
        if lid:
            lineage_groups.setdefault(lid, []).append((cid, ns))

    new_leaders: dict[str, str] = {}  # lineage_id -> winning candidate_id
    for lid, members in lineage_groups.items():
        winner = max(members, key=lambda m: m[1])
        new_leaders[lid] = winner[0]

    # --- preview ---
    movers = sorted(score_updates, key=lambda t: abs(t[2] - t[1]), reverse=True)
    print(f"\n[rescore] top {args.top_preview} score movers:")
    print(f"  {'candidate':<40} {'axis':<42} old -> new   det  ident  fpr  coh")
    for cid, old_score, ns, r in movers[: args.top_preview]:
        axis = (r["concept"] or "")[:42]
        print(f"  {cid[:40]:<40} {axis:<42} {old_score:.3f} -> {ns:.3f}  "
              f"{r['detection_rate']:.2f} {r['identification_rate']:.2f} "
              f"{r['fpr']:.2f} {r['coherence_rate']:.2f}")

    # --- leader changes ---
    current_leader_rows = conn.execute(
        "SELECT id, lineage_id FROM candidates WHERE is_leader=1 AND lineage_id IS NOT NULL"
    ).fetchall()
    current_leaders = {row["lineage_id"]: row["id"] for row in current_leader_rows}
    leader_changes = [
        (lid, current_leaders.get(lid), new_cid)
        for lid, new_cid in new_leaders.items()
        if current_leaders.get(lid) != new_cid
    ]
    print(f"\n[rescore] lineages: {len(lineage_groups)}  "
          f"leader changes: {len(leader_changes)}")
    for lid, old, new in leader_changes[:20]:
        print(f"  lineage={lid[:8]}  {str(old)[:20] if old else '(none)':<20} -> {new[:20]}")

    if not args.yes:
        print("\n[rescore] dry-run; rerun with --yes to apply")
        return 0

    # --- apply in one transaction ---
    with conn:
        conn.execute("BEGIN")
        for cid, old_score, ns, r in score_updates:
            if abs(ns - old_score) > 1e-9:
                conn.execute("UPDATE fitness_scores SET score=? WHERE candidate_id=?", (ns, cid))
        # Demote all current lineage leaders; promote the new winners.
        conn.execute("UPDATE candidates SET is_leader=0 WHERE lineage_id IS NOT NULL")
        for lid, cid in new_leaders.items():
            conn.execute("UPDATE candidates SET is_leader=1 WHERE id=?", (cid,))

    print("\n[rescore] applied.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
