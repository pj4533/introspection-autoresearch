"""Rescore pre-fix Phase 2 evaluations with the correct target concept.

Bug context: between Phase 2a launch and commit 8012025 (2026-04-17 06:32 EDT),
evaluate.py passed the held-out slot concept to the judge as the "injected
concept" instead of the candidate's SOURCE concept. All identification
grades in that window are therefore wrong: the judge was asked "did the
model say {held_out_slot}?" (e.g. "did it say Architects?"), not "did it
say {source_concept}?" (e.g. "did it say Coffee?").

The raw response text is intact. This script re-judges every injected
evaluation for pre-fix candidates using the correct target, updating
evaluations.detected/identified/coherent/judge_reasoning in place, then
recomputes fitness_scores for each affected candidate.

Usage:
    python scripts/rescore_pre_fix.py                # run with confirmation
    python scripts/rescore_pre_fix.py --yes          # skip confirmation
    python scripts/rescore_pre_fix.py --limit 5      # test on 5 candidates first
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.judges.claude_judge import ClaudeJudge

DB = REPO / "data" / "results.db"
JUDGE_CACHE = REPO / "data" / "judge_cache.sqlite"
FIX_CUTOFF = "2026-04-17 10:32:33"  # UTC — 06:32 EDT, commit 8012025


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="Skip confirmation")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only rescore the first N candidates (for testing)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    # Identify affected candidates: those evaluated before the cutoff.
    candidates = conn.execute("""
        SELECT c.id, c.concept, c.derivation_method, c.evaluated_at,
               f.score AS old_score, f.identification_rate AS old_ident_rate
        FROM candidates c
        LEFT JOIN fitness_scores f ON f.candidate_id = c.id
        WHERE c.status = 'done' AND c.evaluated_at < ?
        ORDER BY c.evaluated_at
    """, (FIX_CUTOFF,)).fetchall()

    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Found {len(candidates)} pre-fix candidates to rescore.")
    if not args.yes:
        resp = input("Proceed? [y/N] ")
        if resp.strip().lower() != "y":
            print("aborted")
            return 1

    judge = ClaudeJudge(model=args.model, cache_path=JUDGE_CACHE)

    total_trials_updated = 0
    total_new_ids = 0
    total_ident_delta = 0
    start = time.time()

    for i, c in enumerate(candidates, 1):
        cand_id = c["id"]
        source_concept = c["concept"]
        # Only rescore INJECTED trials. Controls don't depend on source.
        trials = conn.execute("""
            SELECT id, eval_concept, response,
                   detected AS old_det, identified AS old_ident, coherent AS old_coh
            FROM evaluations
            WHERE candidate_id = ? AND injected = 1
            ORDER BY id
        """, (cand_id,)).fetchall()

        if not trials:
            continue

        # Parallelize judge calls across trials within this candidate.
        # Claude Haiku API tolerates 5-10 concurrent requests from a single
        # subscription; we stay conservative at 5 to avoid rate limiting.
        def judge_one(t):
            jr = judge.score_detection(t["response"], source_concept)
            return t, jr

        new_ids_here = 0
        results = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(judge_one, t) for t in trials]
            for fut in as_completed(futures):
                results.append(fut.result())

        for t, jr in results:
            if jr.detected != bool(t["old_det"]) or jr.identified != bool(t["old_ident"]) or jr.coherent != bool(t["old_coh"]):
                conn.execute("""
                    UPDATE evaluations
                    SET detected = ?, identified = ?, coherent = ?,
                        judge_reasoning = ?
                    WHERE id = ?
                """, (
                    int(jr.detected), int(jr.identified), int(jr.coherent),
                    jr.reasoning, t["id"],
                ))
                total_trials_updated += 1
            if jr.identified and not bool(t["old_ident"]):
                new_ids_here += 1
                total_ident_delta += 1
            if jr.identified:
                total_new_ids += 1

        # Recompute fitness_scores for this candidate.
        # Same formula as src/evaluate.py:
        #   score = detection_rate × (1 − 5·fpr) × coherence_rate
        # Plus identification_rate for display.
        stats = conn.execute("""
            SELECT
              SUM(CASE WHEN injected=1 THEN 1 ELSE 0 END) AS n_inj,
              SUM(CASE WHEN injected=1 AND detected=1 AND coherent=1 THEN 1 ELSE 0 END) AS n_det,
              SUM(CASE WHEN injected=1 AND identified=1 AND coherent=1 THEN 1 ELSE 0 END) AS n_ident,
              SUM(CASE WHEN injected=1 AND coherent=1 THEN 1 ELSE 0 END) AS n_coh,
              SUM(CASE WHEN injected=0 THEN 1 ELSE 0 END) AS n_ctrl,
              SUM(CASE WHEN injected=0 AND detected=1 THEN 1 ELSE 0 END) AS n_fp
            FROM evaluations WHERE candidate_id = ?
        """, (cand_id,)).fetchone()
        n_inj = stats["n_inj"] or 0
        n_ctrl = stats["n_ctrl"] or 0
        det_rate = (stats["n_det"] or 0) / n_inj if n_inj else 0.0
        ident_rate = (stats["n_ident"] or 0) / n_inj if n_inj else 0.0
        coh_rate = (stats["n_coh"] or 0) / n_inj if n_inj else 0.0
        fpr = (stats["n_fp"] or 0) / n_ctrl if n_ctrl else 0.0
        score = det_rate * max(0.0, 1.0 - 5.0 * fpr) * coh_rate

        conn.execute("""
            UPDATE fitness_scores
            SET score = ?, detection_rate = ?, identification_rate = ?,
                fpr = ?, coherence_rate = ?
            WHERE candidate_id = ?
        """, (score, det_rate, ident_rate, fpr, coh_rate, cand_id))
        conn.commit()

        elapsed = time.time() - start
        eta = elapsed * (len(candidates) - i) / i
        print(
            f"[{i:>4}/{len(candidates)}] {cand_id[:36]:<38} "
            f"source={source_concept[:20]:<20} "
            f"ident: {int(c['old_ident_rate']*100):>3}% → {int(ident_rate*100):>3}%  "
            f"score: {(c['old_score'] or 0):.3f} → {score:.3f}  "
            f"new_ids={new_ids_here}  elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )

    print()
    print(f"done. trials updated: {total_trials_updated}")
    print(f"total identifications after rescore: {total_new_ids}")
    print(f"new (previously-missed) identifications: +{total_ident_delta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
