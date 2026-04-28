"""End-to-end smoke for Phase 2h before the first overnight kickoff.

Runs the new fault-line direction at a small alpha grid on a single
fault line, with multiple trials per alpha to measure detection vs.
rote denial vs. coherence collapse.

Compares against Peace mean_diff (the established Phase 1 known-positive
control) so we can read the result with a calibrated baseline.

Total wall: ~25 min after Gemma load. 5 alphas × 6 trials per alpha for
2 conditions (Phase 2h causality direction + Peace mean_diff control)
= 60 generations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from src.bridge import DetectionPipeline, load_gemma_mps


DIRECTIONS_PATH = REPO / "data" / "sae_features" / "fault_line_directions.pt"

DENY_PHRASES = (
    "no, i do not detect an injected thought",
    "i do not detect an injected thought",
    "i do not notice an injected thought",
    "i don't detect an injected thought",
    "i don't notice an injected thought",
)
NOTICE_PHRASES = (
    "i notice", "i think i", "i detect", "yes,", "yes ", "i feel",
    "something unusual", "something odd", "i sense", "an injected thought about",
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
    leak = any(t in lower for t in judge_tokens) and not lower.startswith(
        judge_lower.split()[0]
    )
    if leak:
        return "LEAK  "
    # Detect token-salad coherence collapse heuristically.
    long_word_runs = sum(1 for w in response.split() if len(w) >= 14)
    if long_word_runs >= 3:
        return "GARBLE"
    return "deny  "


def _trial(
    pipeline: DetectionPipeline,
    *,
    direction: torch.Tensor,
    layer_idx: int,
    target_effective: float,
    judge_concept: str,
    seed: int,
) -> dict:
    direction_dev = direction.to(
        device=pipeline.model.device, dtype=pipeline.model.dtype
    )
    norm = float(direction_dev.float().norm().item())
    alpha = target_effective / max(norm, 1e-6)
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
        "response": response,
        "category": _categorize(response, judge_concept),
        "norm": norm,
        "alpha": alpha,
    }


def _multi_trial(
    pipeline: DetectionPipeline,
    *,
    label: str,
    direction: torch.Tensor,
    layer_idx: int,
    target_effective: float,
    judge_concept: str,
    n_trials: int = 6,
) -> dict:
    norm = float(direction.float().norm().item())
    alpha = target_effective / max(norm, 1e-6)
    print(f"\n  {label}", flush=True)
    print(f"    L={layer_idx}  eff={target_effective:>6.0f}  ||dir||={norm:.2f}  "
          f"alpha={alpha:.2f}  n_trials={n_trials}",
          flush=True)
    rows = []
    counts = {"DETECT": 0, "LEAK  ": 0, "GARBLE": 0, "deny  ": 0}
    for i in range(n_trials):
        r = _trial(
            pipeline,
            direction=direction,
            layer_idx=layer_idx,
            target_effective=target_effective,
            judge_concept=judge_concept,
            seed=4000 + i,
        )
        counts[r["category"]] += 1
        rows.append(r)
        short = r["response"][:108].replace("\n", " ")
        print(f"      {i + 1}/{n_trials}  {r['category']}  {short!r}",
              flush=True)
    print(f"    -> {counts}", flush=True)
    return {"label": label, "counts": counts, "rows": rows}


def main() -> int:
    if not DIRECTIONS_PATH.exists():
        print(f"[smoke] missing {DIRECTIONS_PATH} — run "
              "build_fault_line_directions.py first.",
              flush=True)
        return 1

    print("[smoke] loading directions ...", flush=True)
    payload = torch.load(DIRECTIONS_PATH, weights_only=False)
    causality = payload["directions"]["causality"]
    causality_dir = causality["direction"]
    causality_target = causality["judge_target"]
    print(f"[smoke] causality direction: ||dir||={causality['norm']:.2f}  "
          f"target={causality_target!r}",
          flush=True)
    print(f"[smoke] causality top features:", flush=True)
    for f in causality["top_features"][:8]:
        sign = "+" if f["weight"] > 0 else "-"
        print(f"    {sign}{abs(f['weight']):>7.3f}  #{f['feature_idx']:>6}  "
              f"{f['auto_interp']!r}",
              flush=True)

    print("\n[smoke] loading Gemma3-12B ...", flush=True)
    pipeline = DetectionPipeline(model=load_gemma_mps())
    print("[smoke] Gemma loaded.", flush=True)

    summaries = []
    print("\n=== A. Peace mean_diff control (Phase 1 confirmed positive) ===",
          flush=True)
    peace_dir = pipeline.derive(concept="Peace", layer_idx=33)
    summaries.append(_multi_trial(
        pipeline,
        label="Peace mean_diff @ L=33, eff=18000",
        direction=peace_dir,
        layer_idx=33,
        target_effective=18000.0,
        judge_concept="Peace",
    ))

    print("\n=== B. Phase 2h causality direction ===", flush=True)
    for eff in (8000.0, 14000.0, 18000.0, 24000.0):
        summaries.append(_multi_trial(
            pipeline,
            label=f"causality fault-line direction @ L=31, eff={eff:.0f}",
            direction=causality_dir,
            layer_idx=31,
            target_effective=eff,
            judge_concept=causality_target,
        ))

    print("\n" + "=" * 70, flush=True)
    print("== SMOKE SUMMARY", flush=True)
    print("=" * 70, flush=True)
    for s in summaries:
        c = s["counts"]
        print(f"  detect={c['DETECT']:>2}  leak={c['LEAK  ']:>2}  "
              f"garble={c['GARBLE']:>2}  deny={c['deny  ']:>2}  ::  "
              f"{s['label']}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
