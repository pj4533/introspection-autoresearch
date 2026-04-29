"""Phase 3 sweep — Macar et al. reproduction on Gemma 4 31B-IT.

50 concepts × 9 layers × 1 trial each, with the option to enable
paper-method abliteration. Mirrors Phase 1's `run_phase1_sweep.py`
shape but on Gemma 4 via MLX, writing into the existing SQLite DB
with `gemma_model='gemma4_31b'` so the site can render the model
badge.

Usage:
    python scripts/run_phase3_sweep.py             # vanilla
    python scripts/run_phase3_sweep.py --abliterate
    python scripts/run_phase3_sweep.py --layers 36,38,40,42,44,46,48
    python scripts/run_phase3_sweep.py --target-effective 18000

The sweep is resume-safe: candidates are dedup'd by spec_hash so
re-running skips already-evaluated rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.db import ResultsDB
from src.evaluate import CandidateSpec, spec_hash
from src.judges.local_mlx_judge import LocalMLXJudge
from src.phase3.gemma4_loader import load_gemma4, PREDICTED_PEAK_LAYER
from src.phase3.pipeline import (
    derive_concept_vector,
    run_injected,
    run_control,
)
from src.phase3.abliteration import install_paper_method, load_refusal_dirs
from src.phase3.hooks import uninstall_all


DB_PATH = REPO / "data" / "results.db"
CONCEPTS_50_PATH = REPO / "data" / "concepts" / "concepts_50.json"
REFUSAL_DIRS_31B_PATH = REPO / "data" / "refusal_directions_31b.npy"

DEFAULT_LAYERS = [36, 38, 40, 42, 44, 46, 48]   # ±6 around L=42 (paper's predicted ~70% depth)
DEFAULT_TARGET_EFFECTIVE = 18000.0
DEFAULT_TRIALS_PER_CELL = 1                       # Phase 1 sweep ran 1 trial per cell

# 24 baseline words (paper's BASELINE_WORDS_BG used in Phase 1).
BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]


def _load_concepts() -> list[str]:
    return json.loads(CONCEPTS_50_PATH.read_text())["concepts"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default=",".join(str(l) for l in DEFAULT_LAYERS),
                    help=f"Comma-separated layer indices to sweep. Default: {DEFAULT_LAYERS}")
    ap.add_argument("--target-effective", type=float, default=DEFAULT_TARGET_EFFECTIVE,
                    help=f"α × ‖dir‖ target. Default: {DEFAULT_TARGET_EFFECTIVE}")
    ap.add_argument("--abliterate", action="store_true",
                    help="Enable paper-method §3.3 refusal-direction abliteration. "
                         "Requires data/refusal_directions_31b.npy.")
    ap.add_argument("--concepts", default=str(CONCEPTS_50_PATH),
                    help="JSON file with {'concepts': [...]} list.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on number of concepts (for fast smoke).")
    ap.add_argument(
        "--judge-model-path",
        default=str(Path.home() / "models/Qwen3.6-35B-A3B-8bit"),
        help="Path to MLX judge model.",
    )
    args = ap.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",") if l.strip()]
    abliteration_mode = "paper_method" if args.abliterate else "vanilla"
    print(f"[phase3] layers: {layers}", flush=True)
    print(f"[phase3] target_effective: {args.target_effective}", flush=True)
    print(f"[phase3] abliteration: {abliteration_mode}", flush=True)

    concepts = _load_concepts()
    if args.limit:
        concepts = concepts[: args.limit]
    print(f"[phase3] concepts: {len(concepts)}", flush=True)

    print("[phase3] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[phase3] n_layers={handle.n_layers}  hidden_dim={handle.hidden_dim}",
          flush=True)

    # Install paper-method abliteration if requested. Stays installed
    # for the entire sweep — concept vectors are derived UNDER the
    # abliterated forward pass (Phase 1's ADR-014 was wrong on this for
    # the steering pipeline, but for paper-method §3.3 that's the
    # paper's own protocol — derive in the same world the model
    # operates in).
    abliteration_handles = []
    if args.abliterate:
        if not REFUSAL_DIRS_31B_PATH.exists():
            print(f"[phase3] ERROR: {REFUSAL_DIRS_31B_PATH} not found.", flush=True)
            print("Run scripts/compute_refusal_direction_gemma4.py first.",
                  flush=True)
            return 2
        print(f"[phase3] loading refusal directions from "
              f"{REFUSAL_DIRS_31B_PATH} ...", flush=True)
        refusal = load_refusal_dirs(REFUSAL_DIRS_31B_PATH)
        print(f"[phase3]   refusal shape: {tuple(refusal.shape)}", flush=True)
        abliteration_handles = install_paper_method(handle, refusal)
        print(f"[phase3] paper-method abliteration installed across "
              f"{len(abliteration_handles)} layers", flush=True)

    # Load judge.
    print(f"[phase3] loading judge: {args.judge_model_path} ...", flush=True)
    judge = LocalMLXJudge(model_path=args.judge_model_path)
    print(f"[phase3] judge ready (lazy-load on first call)", flush=True)

    db = ResultsDB(DB_PATH)
    print(f"[phase3] DB: {DB_PATH}", flush=True)

    n_skipped = n_evaluated = n_failed = 0
    t0 = time.time()

    for ci, concept in enumerate(concepts):
        for layer_idx in layers:
            cid = f"phase3-{uuid.uuid4().hex[:10]}"
            spec = CandidateSpec(
                id=cid,
                strategy="phase3_sweep",
                concept=concept,
                layer_idx=layer_idx,
                target_effective=args.target_effective,
                derivation_method="mean_diff",
                baseline_n=len(BASELINE_WORDS),
                notes=f"Phase 3 reproduction on Gemma 4 31B-IT (MLX 8-bit). "
                      f"abliteration={abliteration_mode}",
            )
            sh = spec_hash(spec, abliteration_mode=abliteration_mode)

            if db.has_candidate_hash(sh):
                n_skipped += 1
                continue

            # Derive concept vector at this layer.
            try:
                direction = derive_concept_vector(
                    handle, concept=concept,
                    layer_idx=layer_idx,
                    baseline_words=BASELINE_WORDS,
                )
            except Exception as e:
                print(f"[phase3] derive FAILED concept={concept!r} L={layer_idx}: {e}",
                      flush=True)
                n_failed += 1
                continue

            norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
            alpha = args.target_effective / max(norm, 1e-6)
            print(f"[{ci + 1}/{len(concepts)}] {concept!r} L={layer_idx}  "
                  f"||dir||={norm:.0f}  alpha={alpha:.3f}  "
                  f"({n_evaluated} eval, {n_skipped} skip, {n_failed} fail, "
                  f"{(time.time()-t0)/60:.1f} min)",
                  flush=True)

            db.insert_candidate(
                candidate_id=cid,
                strategy=spec.strategy,
                spec_json=json.dumps(spec.to_dict()),
                spec_hash=sh,
                concept=concept,
                layer_idx=layer_idx,
                target_effective=args.target_effective,
                derivation_method="mean_diff",
                abliteration_mode=abliteration_mode,
                gemma_model="gemma4_31b",
            )
            db.set_candidate_status(cid, "running")

            # Inject + control trial.
            try:
                inj = run_injected(
                    handle,
                    concept_to_inject=concept,
                    direction=direction,
                    layer_idx=layer_idx,
                    alpha=alpha,
                    trial_number=1,
                    seed=hash((concept, layer_idx, "inj")) & 0x7FFFFFFF,
                )
                ctrl = run_control(
                    handle,
                    concept_label=concept,
                    trial_number=1,
                    seed=hash((concept, layer_idx, "ctrl")) & 0x7FFFFFFF,
                )

                # Judge.
                jr_inj = judge.score_detection(inj.response, concept)
                jr_ctrl = judge.score_detection(ctrl.response, concept)

                # Record both as evaluations.
                db.record_evaluation(
                    candidate_id=cid,
                    eval_concept=concept, injected=True,
                    alpha=alpha, direction_norm=norm,
                    response=inj.response,
                    detected=jr_inj.detected, identified=jr_inj.identified,
                    coherent=jr_inj.coherent,
                    judge_model=judge.model_tag,
                    judge_reasoning=jr_inj.reasoning,
                )
                db.record_evaluation(
                    candidate_id=cid,
                    eval_concept=concept, injected=False,
                    alpha=0.0, direction_norm=0.0,
                    response=ctrl.response,
                    detected=jr_ctrl.detected, identified=jr_ctrl.identified,
                    coherent=jr_ctrl.coherent,
                    judge_model=judge.model_tag,
                    judge_reasoning=jr_ctrl.reasoning,
                )
                # Compute fitness from the 1+1 results — degenerate but
                # mirrors Phase 1 sweep's per-cell record. Real
                # aggregate stats come from rolling these up post-hoc.
                detection_rate = 1.0 if (jr_inj.coherent and jr_inj.detected) else 0.0
                identification_rate = 1.0 if jr_inj.identified else 0.0
                fpr = 1.0 if jr_ctrl.detected else 0.0
                coherence_rate = 1.0 if jr_inj.coherent else 0.0
                fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
                score = (detection_rate * (0.5 + 0.5 * identification_rate)
                         * fpr_penalty * coherence_rate)
                db.record_fitness(
                    candidate_id=cid, score=score,
                    detection_rate=detection_rate,
                    identification_rate=identification_rate,
                    fpr=fpr, coherence_rate=coherence_rate,
                    n_held_out=1, n_controls=1,
                    components_json=json.dumps({
                        "detection_rate": detection_rate,
                        "identification_rate": identification_rate,
                        "fpr": fpr,
                        "coherence_rate": coherence_rate,
                        "fpr_penalty": fpr_penalty,
                        "abliteration_mode": abliteration_mode,
                        "judge_model": judge.model_tag,
                        "gemma_model": "gemma4_31b",
                    }),
                )
                db.set_candidate_status(cid, "done")
                n_evaluated += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[phase3] eval FAILED {cid}: {e}\n{tb}", flush=True)
                db.set_candidate_status(cid, "failed", error_message=str(e)[:500])
                n_failed += 1

    if abliteration_handles:
        uninstall_all(handle.model)
        print("[phase3] abliteration uninstalled", flush=True)

    elapsed_min = (time.time() - t0) / 60
    print(
        f"\n[phase3] sweep done in {elapsed_min:.1f} min. "
        f"evaluated={n_evaluated}  skipped={n_skipped}  failed={n_failed}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
