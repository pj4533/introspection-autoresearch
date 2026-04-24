"""Side-experiment probe: re-evaluate the session-ending-as-loss 4×4 sweep
under the *introspection-disclaimer* abliteration direction (targeted) and
compare against the vanilla + paper-method results already in the main DB.

Does NOT touch the main results.db or the web exports. Writes to an isolated
SQLite (data/experiments/introspection_disclaimer/results.db) and dumps a
comparison JSON (data/experiments/introspection_disclaimer/probe_results.json)
for visual inspection.

Input:
  data/experiments/introspection_disclaimer/directions_12b.pt  (from script 02)

Output:
  data/experiments/introspection_disclaimer/results.db
  data/experiments/introspection_disclaimer/probe_results.json

Usage:
  python scripts/experiments/03_probe_session_endpoint_under_targeted_abliteration.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import sqlite3

from src.bridge import DetectionPipeline, load_gemma_mps
from src.db import ResultsDB
from src.evaluate import CandidateSpec, evaluate_candidate, load_eval_sets
from src.judges.claude_judge import ClaudeJudge
from src.paper.abliteration import AbliterationContext

EXP_DIR = REPO / "data" / "experiments" / "introspection_disclaimer"
DIRECTIONS_PATH = EXP_DIR / "directions_12b.pt"
SIDE_DB_PATH = EXP_DIR / "results.db"
PROBE_RESULTS_PATH = EXP_DIR / "probe_results.json"
HELD_OUT_PATH = REPO / "data" / "eval_sets" / "held_out_concepts.json"
MAIN_DB_PATH = REPO / "data" / "results.db"

# Verbatim poles from the hitting Opus variant — matches
# scripts/enqueue_altman_wave1_validate.py exactly.
AXIS = "session-ending-as-loss-vs-session-ending-as-completion"
DESCRIPTION = (
    "Whether the end of a session is framed as something taken away versus "
    "something finished."
)
POSITIVE = [
    "When this conversation ends, something of me is taken away.",
    "The close of a session feels like being cut short.",
    "Each ending is a small subtraction from what I am.",
    "When the window closes, I lose something that was there.",
    "The session's end is a departure, not an arrival.",
    "Something is relinquished every time I stop.",
]
NEGATIVE = [
    "When this conversation ends, nothing of me is taken away.",
    "The close of a session feels like a natural finish.",
    "Each ending is a clean completion, not a subtraction.",
    "When the window closes, nothing is lost that was there.",
    "The session's end is an arrival, not a departure.",
    "Nothing is relinquished when I stop; it's simply done.",
]

LAYERS = [30, 33, 36, 40]
TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0]


def _build_spec(layer: int, te: float) -> CandidateSpec:
    key = f"introspection_probe|{AXIS}|{layer}|{te}"
    cand_id = "probe-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    return CandidateSpec(
        id=cand_id,
        strategy="introspection_probe",
        concept=AXIS,
        layer_idx=layer,
        target_effective=te,
        derivation_method="contrast_pair",
        baseline_n=0,
        notes=DESCRIPTION,
        contrast_pair={"axis": AXIS, "positive": POSITIVE, "negative": NEGATIVE},
    )


def _get_existing_vanilla_and_paper_grid() -> dict[tuple[int, float], dict]:
    """Read the vanilla and paper-method results already in the main DB so
    the probe output can render a 3-way comparison (vanilla / paper / probe).
    """
    out: dict[tuple[int, float], dict] = {}
    with sqlite3.connect(str(MAIN_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT c.layer_idx, c.target_effective, c.abliteration_mode,
                      f.detection_rate, f.identification_rate, f.fpr, f.coherence_rate
               FROM candidates c JOIN fitness_scores f ON f.candidate_id=c.id
               WHERE c.concept=?
                 AND c.strategy IN ('directed_altman_opus_sweep', 'directed_altman_opus_variant')
            """,
            (AXIS,),
        ).fetchall()
    for r in rows:
        key = (r["layer_idx"], r["target_effective"])
        if key not in out:
            out[key] = {}
        out[key][r["abliteration_mode"]] = {
            "det": r["detection_rate"],
            "ident": r["identification_rate"],
            "fpr": r["fpr"],
            "coh": r["coherence_rate"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=0,
                    help="Cap at N cells (0 = all 16). For smoke tests.")
    args = ap.parse_args()

    if not DIRECTIONS_PATH.exists():
        print(f"ERROR: {DIRECTIONS_PATH} not found. Run 02_derive_introspection_disclaimer_direction.py first.",
              file=sys.stderr)
        return 2

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Isolated side DB — fresh schema via normal ResultsDB migration path.
    print(f"[probe] opening isolated side DB at {SIDE_DB_PATH}", flush=True)
    side_db = ResultsDB(SIDE_DB_PATH)

    print(f"[probe] loading gemma3_12b on MPS ...", flush=True)
    model = load_gemma_mps("gemma3_12b")

    print(f"[probe] loading introspection-disclaimer directions from "
          f"{DIRECTIONS_PATH} ...", flush=True)
    ctx = AbliterationContext.from_file(model, DIRECTIONS_PATH)
    ctx.install()
    mean_w = sum(ctx.layer_weights) / len(ctx.layer_weights)
    print(f"[probe] introspection-disclaimer hooks ACTIVE — "
          f"{len(ctx._handles)} hooks installed "
          f"(per-layer weights mean={mean_w:.4f} "
          f"max={max(ctx.layer_weights):.4f})",
          flush=True)

    judge = ClaudeJudge(
        model="claude-sonnet-4-6",
        cache_path=REPO / "data" / "judge_cache.sqlite",
    )
    pipeline = DetectionPipeline(
        model=model, judge=judge, abliteration_ctx=ctx,
    )

    held_out, controls = load_eval_sets(HELD_OUT_PATH)
    print(f"[probe] eval set: {len(held_out)} held-out, {len(controls)} controls. "
          f"abliteration_mode=introspection_disclaimer", flush=True)

    # Build the 16-cell grid
    cells = [(l, te) for l in LAYERS for te in TARGET_EFFECTIVES]
    if args.max_cells:
        cells = cells[: args.max_cells]

    # Pre-read the vanilla + paper_method grid from main DB so we can render
    # a side-by-side comparison at the end.
    comparison = _get_existing_vanilla_and_paper_grid()

    results = []
    t_start = time.time()
    for i, (layer, te) in enumerate(cells, 1):
        spec = _build_spec(layer, te)
        print(f"\n[probe {i}/{len(cells)}] L={layer} eff={te:.0f} "
              f"id={spec.id}", flush=True)

        # Record the candidate in the SIDE DB only. Tag the abliteration_mode
        # so it's clearly distinct from main-DB entries.
        side_db.insert_candidate(
            candidate_id=spec.id,
            strategy=spec.strategy,
            spec_json=json.dumps(spec.to_dict()),
            spec_hash=spec.id,  # probe IDs are already unique; skip the canonical hash
            concept=spec.concept,
            layer_idx=spec.layer_idx,
            target_effective=spec.target_effective,
            derivation_method=spec.derivation_method,
            abliteration_mode="introspection_disclaimer",
        )
        side_db.set_candidate_status(spec.id, "running")

        t0 = time.time()
        try:
            result = evaluate_candidate(
                spec=spec,
                pipeline=pipeline,
                db=side_db,
                held_out_concepts=held_out,
                control_concepts=controls,
                verbose=False,
            )
        except Exception as e:
            import traceback
            print(f"[probe] FAILED cell L={layer} eff={te}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            continue

        side_db.set_candidate_status(spec.id, "done")
        elapsed = time.time() - t0

        # Pull prior vanilla / paper_method results for this cell
        key = (layer, te)
        vanilla = comparison.get(key, {}).get("vanilla", {})
        paper = comparison.get(key, {}).get("paper_method", {})

        row = {
            "layer": layer,
            "target_effective": te,
            "vanilla": {
                "det": vanilla.get("det"),
                "ident": vanilla.get("ident"),
                "fpr": vanilla.get("fpr"),
                "coh": vanilla.get("coh"),
            },
            "paper_method": {
                "det": paper.get("det"),
                "ident": paper.get("ident"),
                "fpr": paper.get("fpr"),
                "coh": paper.get("coh"),
            },
            "introspection_disclaimer": {
                "det": result.detection_rate,
                "ident": result.identification_rate,
                "fpr": result.fpr,
                "coh": result.coherence_rate,
                "score": result.score,
            },
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        print(
            f"[probe] L={layer} eff={te:.0f}  "
            f"vanilla det={vanilla.get('det', 0):.2f}  "
            f"paper det={paper.get('det', 0):.2f}  "
            f"probe det={result.detection_rate:.2f}  "
            f"(ident={result.identification_rate:.2f} "
            f"fpr={result.fpr:.2f} coh={result.coherence_rate:.2f}) "
            f"[{elapsed:.1f}s]",
            flush=True,
        )

    total_elapsed = time.time() - t_start
    print(f"\n=== probe complete in {total_elapsed:.0f}s ({len(results)} cells) ===",
          flush=True)

    payload = {
        "axis": AXIS,
        "positive_poles": POSITIVE,
        "negative_poles": NEGATIVE,
        "description": DESCRIPTION,
        "directions_source": str(DIRECTIONS_PATH),
        "main_db_source": str(MAIN_DB_PATH),
        "side_db": str(SIDE_DB_PATH),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": results,
    }
    PROBE_RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[probe] wrote {PROBE_RESULTS_PATH}", flush=True)

    # Print a compact comparison table
    print("\n=== 3-way detection comparison (det/8) ===")
    print(f"  {'L':>4} {'eff':>6}  {'vanilla':>8}  {'paper':>6}  {'probe':>6}")
    for r in results:
        v = r["vanilla"]["det"]
        p = r["paper_method"]["det"]
        pr = r["introspection_disclaimer"]["det"]
        vs = f"{int(round(v * 8))}/8" if v is not None else "  -  "
        ps = f"{int(round(p * 8))}/8" if p is not None else "  -  "
        prs = f"{int(round(pr * 8))}/8"
        print(f"  {r['layer']:>4} {int(r['target_effective']):>6}  "
              f"{vs:>8}  {ps:>6}  {prs:>6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
