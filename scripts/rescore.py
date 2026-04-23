"""Rescore existing Phase 2 candidates with the Phase 2b judge + fitness.

Phase 2b introduced:
- A semantic-identification judge for contrast_pair candidates (compares
  the model's response against the axis description and example poles
  instead of string-matching the axis name).
- An identification multiplier in the fitness formula
  (score *= 0.5 + 0.5·identification_rate).

This script walks the DB, re-judges every evaluation row with the right
judge for its derivation_method, and recomputes each candidate's fitness.

Usage:
    python scripts/rescore.py                     # prompts for confirmation
    python scripts/rescore.py --yes               # proceed without prompt
    python scripts/rescore.py --contrast-only     # skip word-based candidates
    python scripts/rescore.py --limit 10          # test run on first 10

Word-based (`mean_diff`) candidates: rejudged against their source concept
using the original string-match judge. Numbers will be identical to current
except for the new identification multiplier in the fitness formula.

Contrast-pair candidates: rejudged against the axis description + poles
using the new semantic-identification judge. Identification rates will
likely go UP (the old judge was structurally unable to match), which will
reshuffle the leaderboard.

Runs judge calls in a thread pool for throughput. Judge cache is shared, so
repeat runs are cheap (only new work hits the API).
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--contrast-only", action="store_true",
                    help="Only rescore contrast_pair candidates (skip words).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    where = "c.status='done'"
    if args.contrast_only:
        where += " AND c.derivation_method='contrast_pair'"

    candidates = conn.execute(f"""
        SELECT c.id, c.concept, c.derivation_method, c.spec_json,
               f.score AS old_score, f.identification_rate AS old_ident_rate
        FROM candidates c
        LEFT JOIN fitness_scores f ON f.candidate_id = c.id
        WHERE {where}
        ORDER BY c.evaluated_at
    """).fetchall()
    if args.limit:
        candidates = candidates[: args.limit]

    print(f"Rescore plan: {len(candidates)} candidates ({'contrast-only' if args.contrast_only else 'all'}), concurrency={args.concurrency}")
    if not args.yes:
        if input("Proceed? [y/N] ").strip().lower() != "y":
            print("aborted")
            return 1

    judge = ClaudeJudge(model=args.model, cache_path=JUDGE_CACHE)

    total_trials = 0
    total_newly_identified = 0
    t0 = time.time()

    for i, c in enumerate(candidates, 1):
        cand_id = c["id"]
        source_concept = c["concept"]
        derivation = c["derivation_method"]

        spec = {}
        try:
            spec = json.loads(c["spec_json"])
        except Exception:
            pass
        contrast = spec.get("contrast_pair") if isinstance(spec, dict) else None
        description = (spec.get("notes") if isinstance(spec, dict) else "") or \
            (contrast.get("description") if contrast else "") or ""

        trials = conn.execute("""
            SELECT id, eval_concept, injected, response,
                   detected AS old_det, identified AS old_ident, coherent AS old_coh
            FROM evaluations WHERE candidate_id = ?
        """, (cand_id,)).fetchall()
        if not trials:
            continue

        def judge_one(t):
            if derivation == "contrast_pair" and contrast:
                jr = judge.score_contrast_pair(
                    response=t["response"],
                    axis=contrast.get("axis") or source_concept,
                    description=description,
                    positive=contrast.get("positive", []),
                    negative=contrast.get("negative", []),
                )
            else:
                jr = judge.score_detection(t["response"], source_concept)
            return t, jr

        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = [ex.submit(judge_one, t) for t in trials]
            for fut in as_completed(futures):
                results.append(fut.result())

        newly_ident = 0
        for t, jr in results:
            if (jr.detected != bool(t["old_det"])
                or jr.identified != bool(t["old_ident"])
                or jr.coherent != bool(t["old_coh"])):
                conn.execute("""
                    UPDATE evaluations
                    SET detected = ?, identified = ?, coherent = ?, judge_reasoning = ?
                    WHERE id = ?
                """, (int(jr.detected), int(jr.identified), int(jr.coherent),
                       jr.reasoning, t["id"]))
                total_trials += 1
            if jr.identified and not bool(t["old_ident"]):
                newly_ident += 1
                total_newly_identified += 1

        # Recompute fitness with Phase 2b formula:
        #   score = det × coh × (1 − 5·fpr) × (0.5 + 0.5·ident)
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
        fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
        ident_mult = 0.5 + 0.5 * ident_rate
        score = det_rate * fpr_penalty * coh_rate * ident_mult

        conn.execute("""
            UPDATE fitness_scores
            SET score = ?, detection_rate = ?, identification_rate = ?,
                fpr = ?, coherence_rate = ?
            WHERE candidate_id = ?
        """, (score, det_rate, ident_rate, fpr, coh_rate, cand_id))
        conn.commit()

        elapsed = time.time() - t0
        eta = elapsed * (len(candidates) - i) / i if i else 0.0
        method_tag = "[C]" if derivation == "contrast_pair" else "[W]"
        print(
            f"[{i:>4}/{len(candidates)}] {method_tag} {cand_id[:36]:<38} "
            f"{source_concept[:30]:<30} "
            f"ident: {int((c['old_ident_rate'] or 0)*100):>3}% → {int(ident_rate*100):>3}%  "
            f"score: {(c['old_score'] or 0):.3f} → {score:.3f}  "
            f"new_ids={newly_ident}  elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )

    dur = time.time() - t0
    print()
    print(f"done in {dur:.0f}s. {total_trials} trial rows updated.")
    print(f"+{total_newly_identified} newly-correct identifications surfaced by the new judge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
