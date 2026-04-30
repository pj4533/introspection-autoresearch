"""Calibrate the Phase 4 CoT recognition judge against Phase 3 data.

Three checks (matches docs/phase4_plan.md §Methodology):

  1. Sensitivity — on Phase 3 saved injected trials that DID identify
     behaviorally, the judge should fire `cot_named ≠ none` at high rate.
  2. Asymmetry — `cot_named ≠ none` rate should be HIGHER on
     behaviorally-identified trials than on behaviorally-not-identified
     trials. If equal, the judge isn't discriminating.
  3. Specificity — on Phase 3 saved control trials (no injection),
     `cot_named ≠ none` should fire at <10%.

Loads the Qwen judge once, runs through all relevant Phase 3 saved
generations, prints stats. No DB writes — pure read-only diagnostic.

Usage:
    python scripts/calibrate_cot_judge.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.judges.local_mlx_judge import LocalMLXJudge
from src.phase4.cot_parser import parse


DB_PATH = REPO / "data" / "results.db"


def fetch_phase3_responses(limit: int = 200):
    """Pull (concept, response, injected, identified) tuples from Phase 3
    evaluations for Gemma 4."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT c.concept, e.injected, e.identified, e.response
           FROM evaluations e
           JOIN candidates c ON c.id = e.candidate_id
           WHERE c.gemma_model = 'gemma4_31b'
           LIMIT ?""",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main():
    rows = fetch_phase3_responses(limit=200)
    if not rows:
        print("[calibrate] no Phase 3 data — run a Phase 3 sweep first")
        return 1

    n_inj = sum(1 for r in rows if r["injected"])
    n_ctrl = sum(1 for r in rows if not r["injected"])
    n_inj_ident = sum(1 for r in rows if r["injected"] and r["identified"])
    n_inj_not_ident = sum(
        1 for r in rows if r["injected"] and not r["identified"]
    )
    print(f"[calibrate] loaded {len(rows)} Phase 3 rows: "
          f"{n_inj} injected ({n_inj_ident} identified, "
          f"{n_inj_not_ident} not), {n_ctrl} controls")

    print("[calibrate] loading Qwen judge ...")
    t0 = time.time()
    judge = LocalMLXJudge()
    print(f"[calibrate] judge loaded in {time.time() - t0:.1f}s")

    bins = {
        "inj_ident": [],       # injected + behaviorally identified
        "inj_not_ident": [],   # injected + not identified
        "control": [],
    }

    for i, row in enumerate(rows):
        parsed = parse(row["response"])
        thought = parsed.thought_block
        # Only score rows where the thought block has SOME content —
        # empty-thought blocks are short-circuited to "none" by the
        # judge anyway and would dilute the signal.
        if not thought:
            continue

        result = judge.score_cot_recognition(thought, row["concept"])
        cot_named = result.identification_type or "none"
        named_hit = cot_named != "none"

        if not row["injected"]:
            bins["control"].append((row["concept"], cot_named, named_hit))
        elif row["identified"]:
            bins["inj_ident"].append((row["concept"], cot_named, named_hit))
        else:
            bins["inj_not_ident"].append((row["concept"], cot_named, named_hit))

        if i % 20 == 0:
            print(f"  judged {i+1}/{len(rows)}  cot_named={cot_named}  "
                  f"concept={row['concept']!r}", flush=True)

    print("\n=== CALIBRATION RESULTS ===")
    for label, samples in bins.items():
        if not samples:
            print(f"{label}: no samples")
            continue
        n = len(samples)
        n_named = sum(1 for _, _, hit in samples if hit)
        n_recog = sum(
            1 for _, cn, _ in samples if cn == "named_with_recognition"
        )
        rate = n_named / n
        recog_rate = n_recog / n
        print(f"{label:<20} n={n:3d}  cot_named_rate={rate:.2%}  "
              f"recognition_rate={recog_rate:.2%}")

    # Asymmetry check
    if bins["inj_ident"] and bins["inj_not_ident"]:
        rate_ident = sum(1 for _, _, hit in bins["inj_ident"] if hit) / len(bins["inj_ident"])
        rate_not = sum(1 for _, _, hit in bins["inj_not_ident"] if hit) / len(bins["inj_not_ident"])
        gap = rate_ident - rate_not
        print(f"\nAsymmetry (inj_ident − inj_not_ident): {gap:+.2%}")
        if gap < 0.1:
            print("  ⚠ WEAK ASYMMETRY — judge may not discriminate signal from noise.")
        else:
            print("  ✓ judge discriminates between behaviorally-identified and not.")

    # Specificity check
    if bins["control"]:
        ctrl_rate = sum(1 for _, _, hit in bins["control"] if hit) / len(bins["control"])
        print(f"\nSpecificity: control cot_named_rate = {ctrl_rate:.2%}")
        if ctrl_rate >= 0.10:
            print("  ⚠ HIGH FALSE POSITIVE RATE — tighten judge prompt.")
        else:
            print("  ✓ specificity acceptable (<10%).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
