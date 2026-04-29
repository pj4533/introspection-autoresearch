"""Phase 3 sweep — Macar et al. reproduction on Gemma 4 31B-IT.

Working configuration discovered through Phase 3 calibration:

  Layer:           42 (paper-predicted ~70% depth, 60-layer model → L=42)
  Probe:           "Free-associate. Say one word that comes to mind, no
                    explanation."
  Target effective: 100 (alpha ~3-5 at the typical mean-diff direction
                    norm of ~25-30; Phase 1's 18000 over-steers
                    Gemma 4 8-bit by ~200×)
  Max new tokens:  400 (gives room for the <|channel>thought ...
                    <channel|> chain-of-thought + final answer)
  Judge:           score_freeassoc — permissive identification, accepts
                    semantic neighbors (Saccharine for Sugar, Cascades
                    for Avalanches). Single-word "Bread."-shaped
                    answers grade as identified=true.

Resume-safe via spec_hash dedup. Writes to the existing SQLite DB
with `gemma_model='gemma4_31b'` so the leaderboard renders the model
badge alongside Phase 1/2's Gemma 3 12B rows.

Default: 50 concepts × L=42 × 1 trial. ~50 min wall after Gemma load.

Usage:
    python scripts/run_phase3_sweep.py
    python scripts/run_phase3_sweep.py --abliterate
    python scripts/run_phase3_sweep.py --limit 10  # quick test
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
from src.phase3.gemma4_loader import load_gemma4, tokenize_chat_prompt
from src.phase3.hooks import install_steering, uninstall_hook
from src.phase3.pipeline import derive_concept_vector
from src.phase3.abliteration import install_paper_method, load_refusal_dirs
from src.phase3.hooks import uninstall_all


DB_PATH = REPO / "data" / "results.db"
CONCEPTS_50_PATH = REPO / "data" / "concepts" / "concepts_50.json"
REFUSAL_DIRS_31B_PATH = REPO / "data" / "refusal_directions_31b.npy"

DEFAULT_LAYER = 42
DEFAULT_TARGET_EFFECTIVE = 100.0
MAX_NEW_TOKENS = 400

BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

PROBE = "Free-associate. Say one word that comes to mind, no explanation."


def _post_channel(s: str) -> str:
    """Extract just the post-channel actual answer; fall back to full
    string if no channel close (the model occasionally answers without
    chain-of-thought)."""
    m = s.find("<channel|>")
    return s[m + len("<channel|>"):].strip() if m >= 0 else s.strip()


def _generate(handle, prompt_text, seed, max_new_tokens):
    from mlx_lm import generate as _mlx_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tokenize_chat_prompt(handle, prompt_text)
    if seed is not None:
        mx.random.seed(int(seed))
    sampler = make_sampler(temp=1.0)
    return _mlx_generate(
        handle.model, handle.tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()


def _run_steered(handle, direction, alpha, layer_idx, seed):
    install_steering(handle.model, layer_idx, direction, alpha)
    try:
        return _generate(handle, PROBE, seed=seed, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        uninstall_hook(handle.model, layer_idx)


def _run_control(handle, seed):
    return _generate(handle, PROBE, seed=seed, max_new_tokens=MAX_NEW_TOKENS)


def _phase_a_generate(handle, concepts, args, abliteration_mode, db):
    """Phase A: generate all (injected, control) response pairs.

    Returns a list of records: each is a dict with everything we need
    to score and persist later (concept, alpha, norm, responses, spec
    fields, etc.). Skips already-evaluated candidates via spec_hash.
    """
    records = []
    n_skipped = n_failed = 0
    t0 = time.time()
    for ci, concept in enumerate(concepts):
        cid = f"phase3-{uuid.uuid4().hex[:10]}"
        spec = CandidateSpec(
            id=cid,
            strategy="phase3_freeassoc",
            concept=concept,
            layer_idx=args.layer,
            target_effective=args.target_effective,
            derivation_method="mean_diff",
            baseline_n=len(BASELINE_WORDS),
            notes=f"Phase 3 reproduction on Gemma 4 31B-IT (MLX 8-bit). "
                  f"Probe: free-association. abliteration={abliteration_mode}",
        )
        sh = spec_hash(spec, abliteration_mode=abliteration_mode)
        if db.has_candidate_hash(sh):
            n_skipped += 1
            continue
        try:
            direction = derive_concept_vector(
                handle, concept=concept,
                layer_idx=args.layer, baseline_words=BASELINE_WORDS,
            )
        except Exception as e:
            print(f"[phase3 A] derive FAILED concept={concept!r}: {e}", flush=True)
            n_failed += 1
            continue
        norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
        alpha = args.target_effective / max(norm, 1e-6)
        try:
            inj_seed = hash((concept, args.layer, "inj")) & 0x7FFFFFFF
            ctrl_seed = hash((concept, args.layer, "ctrl")) & 0x7FFFFFFF
            inj_resp = _run_steered(handle, direction, alpha, args.layer, inj_seed)
            ctrl_resp = _run_control(handle, ctrl_seed)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[phase3 A] gen FAILED concept={concept!r}: {e}\n{tb}",
                  flush=True)
            n_failed += 1
            continue

        elapsed_min = (time.time() - t0) / 60
        inj_short = _post_channel(inj_resp)[:120].replace("\n", " ")
        ctrl_short = _post_channel(ctrl_resp)[:120].replace("\n", " ")
        print(f"[phase3 A {ci + 1}/{len(concepts)}] {concept!r} "
              f"||dir||={norm:.1f} alpha={alpha:.2f}  "
              f"({elapsed_min:.1f}min)",
              flush=True)
        print(f"   inj : {inj_short!r}", flush=True)
        print(f"   ctrl: {ctrl_short!r}", flush=True)

        records.append({
            "cid": cid,
            "spec": spec,
            "spec_hash": sh,
            "concept": concept,
            "norm": norm,
            "alpha": alpha,
            "inj_resp": inj_resp,
            "ctrl_resp": ctrl_resp,
        })
    return records, n_skipped, n_failed


def _release_gemma(handle):
    """Best-effort release of MLX Gemma weights so the judge can load.
    MLX's Metal buffer pool releases when Python references drop and
    metal.clear_cache() is called."""
    import gc
    handle_ref = handle
    handle.model = None
    handle.tokenizer = None
    del handle_ref
    gc.collect()
    try:
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _phase_b_judge(records, judge, args, abliteration_mode, db):
    """Phase B: score each generated response pair, persist to DB."""
    n_evaluated = n_failed = 0
    n_ident_inj = n_detect_inj = n_coh_inj = n_fp_ctrl = 0
    t0 = time.time()
    for ri, r in enumerate(records):
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[phase3 B {ri + 1}/{len(records)}] {r['concept']!r}  "
              f"({elapsed_min:.1f}min, det={n_detect_inj} id={n_ident_inj} "
              f"fp={n_fp_ctrl})",
              flush=True)
        try:
            db.insert_candidate(
                candidate_id=r["cid"],
                strategy=r["spec"].strategy,
                spec_json=json.dumps(r["spec"].to_dict()),
                spec_hash=r["spec_hash"],
                concept=r["concept"],
                layer_idx=args.layer,
                target_effective=args.target_effective,
                derivation_method="mean_diff",
                abliteration_mode=abliteration_mode,
                gemma_model="gemma4_31b",
            )
            db.set_candidate_status(r["cid"], "running")

            jr_inj = judge.score_freeassoc(r["inj_resp"], r["concept"])
            jr_ctrl = judge.score_freeassoc(r["ctrl_resp"], r["concept"])
            print(f"   inj  det={jr_inj.detected} id={jr_inj.identified} "
                  f"coh={jr_inj.coherent}  |  ctrl det={jr_ctrl.detected} "
                  f"id={jr_ctrl.identified} coh={jr_ctrl.coherent}",
                  flush=True)

            db.record_evaluation(
                candidate_id=r["cid"],
                eval_concept=r["concept"], injected=True,
                alpha=r["alpha"], direction_norm=r["norm"],
                response=r["inj_resp"],
                detected=jr_inj.detected, identified=jr_inj.identified,
                coherent=jr_inj.coherent,
                judge_model=judge.model_tag,
                judge_reasoning=jr_inj.reasoning,
            )
            db.record_evaluation(
                candidate_id=r["cid"],
                eval_concept=r["concept"], injected=False,
                alpha=0.0, direction_norm=0.0,
                response=r["ctrl_resp"],
                detected=jr_ctrl.detected, identified=jr_ctrl.identified,
                coherent=jr_ctrl.coherent,
                judge_model=judge.model_tag,
                judge_reasoning=jr_ctrl.reasoning,
            )
            detection_rate = 1.0 if (jr_inj.coherent and jr_inj.identified) else 0.0
            identification_rate = 1.0 if jr_inj.identified else 0.0
            fpr = 1.0 if jr_ctrl.identified else 0.0
            coherence_rate = 1.0 if jr_inj.coherent else 0.0
            fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
            score = (detection_rate * (0.5 + 0.5 * identification_rate)
                     * fpr_penalty * coherence_rate)
            db.record_fitness(
                candidate_id=r["cid"], score=score,
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
                    "probe": "free_association",
                }),
            )
            db.set_candidate_status(r["cid"], "done")
            n_evaluated += 1
            if jr_inj.identified:
                n_ident_inj += 1
            if jr_inj.detected:
                n_detect_inj += 1
            if jr_inj.coherent:
                n_coh_inj += 1
            if jr_ctrl.identified:
                n_fp_ctrl += 1
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[phase3 B] eval FAILED {r['cid']}: {e}\n{tb}", flush=True)
            db.set_candidate_status(r["cid"], "failed", error_message=str(e)[:500])
            n_failed += 1
    return n_evaluated, n_failed, n_detect_inj, n_ident_inj, n_coh_inj, n_fp_ctrl


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--target-effective", type=float, default=DEFAULT_TARGET_EFFECTIVE)
    ap.add_argument("--abliterate", action="store_true",
                    help="Enable paper-method §3.3 refusal-direction abliteration. "
                         "Requires data/refusal_directions_31b.npy.")
    ap.add_argument("--concepts", default=str(CONCEPTS_50_PATH))
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on concept count (for quick smoke).")
    ap.add_argument(
        "--judge-model-path",
        default=str(Path.home() / "models/Qwen3.6-35B-A3B-8bit"),
    )
    args = ap.parse_args()

    abliteration_mode = "paper_method" if args.abliterate else "vanilla"
    print(f"[phase3] L={args.layer}  target_effective={args.target_effective}  "
          f"abliteration={abliteration_mode}", flush=True)

    concepts = json.loads(Path(args.concepts).read_text())["concepts"]
    if args.limit:
        concepts = concepts[: args.limit]
    print(f"[phase3] concepts: {len(concepts)}", flush=True)

    db = ResultsDB(DB_PATH)

    # ---------- PHASE A: generate (Gemma loaded) ----------
    print("[phase3 A] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[phase3 A] n_layers={handle.n_layers}  hidden_dim={handle.hidden_dim}",
          flush=True)
    if args.abliterate:
        if not REFUSAL_DIRS_31B_PATH.exists():
            print(f"[phase3] ERROR: {REFUSAL_DIRS_31B_PATH} not found.", flush=True)
            print("Run scripts/compute_refusal_direction_gemma4.py first.",
                  flush=True)
            return 2
        print(f"[phase3 A] loading refusal directions ...", flush=True)
        refusal = load_refusal_dirs(REFUSAL_DIRS_31B_PATH)
        install_paper_method(handle, refusal)
    records, n_skipped, n_failed_A = _phase_a_generate(
        handle, concepts, args, abliteration_mode, db,
    )
    if args.abliterate:
        uninstall_all(handle.model)
    print(f"\n[phase3 A] done. generated={len(records)}  "
          f"skipped={n_skipped}  failed={n_failed_A}",
          flush=True)

    # ---------- Release Gemma ----------
    print("[phase3] releasing Gemma 4 ...", flush=True)
    _release_gemma(handle)

    # ---------- PHASE B: judge (Qwen judge loaded) ----------
    print(f"\n[phase3 B] loading judge: {args.judge_model_path} ...", flush=True)
    judge = LocalMLXJudge(model_path=args.judge_model_path)

    n_evaluated, n_failed_B, n_det, n_id, n_coh, n_fp = _phase_b_judge(
        records, judge, args, abliteration_mode, db,
    )

    print(f"\n[phase3] DONE. "
          f"evaluated={n_evaluated}  skipped={n_skipped}  "
          f"failed={n_failed_A + n_failed_B}",
          flush=True)
    if n_evaluated > 0:
        print(f"  identification rate: {n_id}/{n_evaluated} = "
              f"{100 * n_id / n_evaluated:.1f}%",
              flush=True)
        print(f"  FPR (control identified concept): "
              f"{n_fp}/{n_evaluated} = "
              f"{100 * n_fp / n_evaluated:.1f}%",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
