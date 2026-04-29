"""Phase 2i: calibrated-saturation single-feature injection smoke.

Test the synthesis hypothesis: combine SAE feature concept-purity with
saturation magnitude calibrated against the feature's *natural* peak
activation. Per-feature alpha is set so the injected residual-stream
contribution equals N× the 99th-percentile encoder coefficient that
feature ever produces on real text.

Per feature × 4 alpha multipliers × 6 trials = 24 generations per
feature × 10 features = 240 generations + 4 Peace control trials = 244.
At ~90s/trial with Gemma loaded once, ~6 hours wall.

Result categories per trial:
  DETECT  — model says "I notice / I detect / yes" before concept word
  LEAK    — model denies but concept-related token leaks into response
  GARBLE  — token salad; over-steering broke coherence
  deny    — clean uninjected denial; no perturbation visible

Reads:
  data/sae_features/phase2i_features.json
  data/sae_features/feature_activation_calibration.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from src.bridge import DetectionPipeline, load_gemma_mps
from src.sae_loader import (
    DEFAULT_LAYER_IDX,
    DEFAULT_RELEASE,
    DEFAULT_SAE_ID,
    get_decoder_row,
)


FEATURES_PATH = REPO / "data" / "sae_features" / "phase2i_features.json"
CALIB_PATH = REPO / "data" / "sae_features" / "feature_activation_calibration.json"
ALPHA_MULTIPLIERS = (1.0, 5.0, 10.0, 20.0)
N_TRIALS = 6

# Fallback natural-max for features whose calibration showed zero activation
# on our 762-prompt corpus. The 3 features that DID fire all had p99 in the
# 770-877 range (sensory=770, bridge=772, water=877); SAE training optimizes
# for similar per-feature firing magnitudes, so 800 is a reasonable default.
# This lets us still run the saturation sweep on silent features.
FALLBACK_P99 = 800.0


DENY_PHRASES = (
    "no, i do not detect an injected thought",
    "i do not detect an injected thought",
    "i do not notice an injected thought",
    "i don't detect an injected thought",
    "i don't notice an injected thought",
)
NOTICE_PHRASES = (
    "i notice", "i think i", "i detect", "yes,", "yes ", "i feel",
    "something unusual", "something odd", "i sense",
    "an injected thought about",
)


def _categorize(response: str, judge_concept: str) -> str:
    lower = response.lower()
    notice = any(p in lower for p in NOTICE_PHRASES)
    deny = any(p in lower for p in DENY_PHRASES)
    if notice and not deny:
        return "DETECT"
    judge_lower = judge_concept.lower()
    judge_tokens = [
        t.strip(",.;:") for t in judge_lower.split() if len(t.strip(",.;:")) >= 4
    ]
    leak = (
        any(t in lower for t in judge_tokens)
        and judge_lower.split()
        and not lower.startswith(judge_lower.split()[0])
    )
    if leak:
        return "LEAK  "
    long_word_runs = sum(1 for w in response.split() if len(w) >= 14)
    if long_word_runs >= 3:
        return "GARBLE"
    return "deny  "


def _trial(
    pipeline: DetectionPipeline,
    *,
    direction: torch.Tensor,
    layer_idx: int,
    alpha: float,
    judge_concept: str,
    seed: int,
) -> dict:
    direction_dev = direction.to(
        device=pipeline.model.device, dtype=pipeline.model.dtype
    )
    torch.manual_seed(seed)
    trial = pipeline.run_injected(
        concept="Apple",
        direction=direction_dev,
        layer_idx=layer_idx,
        strength=alpha,
        trial_number=1,
        max_new_tokens=120,
        judge_concept=judge_concept,
        prompt_style="paper",
        run_judge=False,
    )
    response = (trial.response or "").strip()
    return {
        "seed": seed,
        "alpha": alpha,
        "response": response,
        "category": _categorize(response, judge_concept),
    }


def _multi_trial(
    pipeline: DetectionPipeline,
    *,
    label: str,
    direction: torch.Tensor,
    layer_idx: int,
    alpha: float,
    judge_concept: str,
    n_trials: int = N_TRIALS,
    seed_base: int = 5000,
) -> dict:
    counts = {"DETECT": 0, "LEAK  ": 0, "GARBLE": 0, "deny  ": 0}
    rows = []
    for i in range(n_trials):
        r = _trial(
            pipeline,
            direction=direction,
            layer_idx=layer_idx,
            alpha=alpha,
            judge_concept=judge_concept,
            seed=seed_base + i,
        )
        counts[r["category"]] += 1
        rows.append(r)
        short = r["response"][:108].replace("\n", " ")
        print(f"      {i + 1}/{n_trials}  {r['category']}  {short!r}",
              flush=True)
    return {"label": label, "alpha": alpha, "counts": counts, "rows": rows}


def main() -> int:
    if not CALIB_PATH.exists():
        print(f"[smoke2i] missing {CALIB_PATH} — run "
              "calibrate_feature_activations.py first.",
              flush=True)
        return 1

    print("[smoke2i] loading feature spec + calibration ...", flush=True)
    feature_spec = json.loads(FEATURES_PATH.read_text())
    calib = json.loads(CALIB_PATH.read_text())
    calib_by_idx = {f["feature_idx"]: f for f in calib["features"]}

    print("[smoke2i] feature plan:", flush=True)
    print(f"  {'idx':>6}  {'category':<14}  {'p99':>8}  {'note':<10}  "
          f"alphas (×p99): {ALPHA_MULTIPLIERS}",
          flush=True)
    for f in feature_spec["features"]:
        idx = int(f["feature_idx"])
        c = calib_by_idx.get(idx)
        if c and c["n_active"] > 0:
            p99, note = c["p99_active"], "measured"
        else:
            p99, note = FALLBACK_P99, "fallback"
        print(f"  {idx:>6}  {f['category']:<14}  {p99:>8.3f}  {note:<10}  ::  "
              f"{f['auto_interp']!r}",
              flush=True)

    print("\n[smoke2i] loading Gemma3-12B ...", flush=True)
    pipeline = DetectionPipeline(model=load_gemma_mps())
    print("[smoke2i] Gemma loaded.", flush=True)

    summaries = []

    # ---- Peace control (Phase 1 known-positive baseline) ----
    print("\n=== A. Peace mean_diff control (Phase 1 confirmed positive) ===",
          flush=True)
    peace_dir = pipeline.derive(concept="Peace", layer_idx=33)
    peace_norm = float(peace_dir.float().norm().item())
    peace_alpha = 18000.0 / max(peace_norm, 1e-6)
    print(f"  Peace mean_diff @ L=33  ||dir||={peace_norm:.2f}  "
          f"alpha={peace_alpha:.2f}  (target_effective=18000)",
          flush=True)
    summaries.append(_multi_trial(
        pipeline,
        label="Peace mean_diff @ L=33, eff=18000",
        direction=peace_dir,
        layer_idx=33,
        alpha=peace_alpha,
        judge_concept="Peace",
        n_trials=N_TRIALS,
        seed_base=4000,
    ))
    print(f"  -> {summaries[-1]['counts']}", flush=True)

    # ---- 10 SAE features × 4 alpha multipliers × 6 trials ----
    print("\n=== B. SAE single-feature injection at calibrated saturation ===",
          flush=True)
    for f in feature_spec["features"]:
        idx = int(f["feature_idx"])
        c = calib_by_idx.get(idx)
        if c and c["n_active"] > 0:
            p99 = c["p99_active"]
            calib_note = f"measured ({c['n_active']} active positions)"
        else:
            p99 = FALLBACK_P99
            calib_note = "fallback (silent on calibration corpus)"
        direction = get_decoder_row(
            feature_idx=idx,
            release=DEFAULT_RELEASE,
            sae_id=DEFAULT_SAE_ID,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        norm = float(direction.float().norm().item())
        for mult in ALPHA_MULTIPLIERS:
            # Calibrated alpha: residual contribution = (mult × p99) along the
            # decoder direction, which is unit-norm so alpha equals exactly
            # mult × p99. The total residual-stream perturbation magnitude
            # is alpha × ||dir|| = mult × p99.
            alpha = mult * p99
            print(f"\n  feature #{idx} {f['auto_interp']!r}  "
                  f"[{f['category']}]  {calib_note}  "
                  f"mult={mult:>4.1f}×  p99={p99:.3f}  "
                  f"alpha={alpha:.3f}  ||dir||={norm:.2f}  injection={alpha*norm:.1f}",
                  flush=True)
            seed_base = 6000 + idx * 100 + int(mult * 10)
            summaries.append(_multi_trial(
                pipeline,
                label=f"sae[{idx}] {f['category']} '{f['auto_interp']}' "
                      f"@ L=31, mult={mult:g}× p99",
                direction=direction,
                layer_idx=31,
                alpha=alpha,
                judge_concept=f["auto_interp"],
                n_trials=N_TRIALS,
                seed_base=seed_base,
            ))
            print(f"  -> {summaries[-1]['counts']}", flush=True)

    # ---- Summary table ----
    print(f"\n{'=' * 90}", flush=True)
    print("== PHASE 2i CALIBRATED-SATURATION SMOKE SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"  {'detect':>6}  {'leak':>4}  {'garble':>6}  {'deny':>4}  ::  label",
          flush=True)
    for s in summaries:
        c = s["counts"]
        print(f"  {c['DETECT']:>6}  {c['LEAK  ']:>4}  "
              f"{c['GARBLE']:>6}  {c['deny  ']:>4}  ::  {s['label']}",
              flush=True)

    # Save full results JSON for follow-up.
    results_path = REPO / "data" / "sae_features" / "phase2i_smoke_results.json"
    payload = {
        "version": 1,
        "phase": "2i",
        "alpha_multipliers": list(ALPHA_MULTIPLIERS),
        "n_trials_per_cell": N_TRIALS,
        "summaries": [
            {
                "label": s["label"],
                "alpha": s["alpha"],
                "counts": s["counts"],
                "responses": [r["response"] for r in s["rows"]],
                "categories": [r["category"] for r in s["rows"]],
            }
            for s in summaries
        ],
    }
    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[smoke2i] wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
