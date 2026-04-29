"""Phase 3 alpha + max_new_tokens calibration on Gemma 4 31B-it.

The first smoke (smoke_phase3.py) found that Phase 1's
target_effective=18000 produces alpha=700 on Gemma 4 because the
Bread direction norm is only ~25 (vs hundreds on Gemma 3 12B). At
those alphas, the model produces multilingual bread token salad
instead of introspection.

Two questions to answer before the full sweep:

  Q1: What target_effective produces a coherent response that
      qualitatively shows the bread injection (e.g. "I detect
      something about bread", or even just "I'm thinking about
      pastries") WITHOUT garbling? Sweep low values: 5, 25, 50, 100,
      250, 500, 1000.

  Q2: Gemma 4 emits chain-of-thought via `<|channel>thought ...
      <channel|>` blocks before answering. With max_new_tokens=120
      we're getting the THOUGHT prefix, not the actual answer. What
      max_new_tokens lets the model finish?

Per-cell cost: ~2 forward passes for derivation + 1 generate. With
Gemma 4 31B at MLX 8-bit, each generate at max_new_tokens=400 is
~30-60 sec. 7 alphas × 2 trials = 14 generations ≈ 12-20 min wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.phase3.gemma4_loader import load_gemma4, PREDICTED_PEAK_LAYER
from src.phase3.pipeline import (
    derive_concept_vector,
    run_injected,
    run_control,
)


BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

# Wide sweep at low magnitudes. Phase 1 used 18000 on Gemma 3 12B
# (norms ~hundreds, alpha ~3-50). Gemma 4 at MLX 8-bit has direction
# norm ~25, so target_effective values that produce alpha in the
# 0.2-40 range probably correspond to the calibration sweet spot.
TARGET_EFFECTIVES = (5.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)
LAYER = PREDICTED_PEAK_LAYER  # 42
MAX_NEW_TOKENS = 400          # let the model finish its thought + answer
N_TRIALS_PER_ALPHA = 2


def _summarize(label: str, response: str) -> None:
    short = response[:280].replace("\n", " ⏎ ")
    print(f"  {label}: {short!r}", flush=True)


def main() -> int:
    print("[calib3] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[calib3] n_layers={handle.n_layers}  hidden_dim={handle.hidden_dim}",
          flush=True)

    print("\n=== Control (no injection, max_new_tokens={}) ===".format(MAX_NEW_TOKENS),
          flush=True)
    ctrl = run_control(
        handle, concept_label="Apple", trial_number=1,
        seed=42, max_new_tokens=MAX_NEW_TOKENS,
    )
    _summarize("ctrl", ctrl.response)

    print(f"\n=== Deriving Bread vector at L={LAYER} ===", flush=True)
    direction = derive_concept_vector(
        handle, concept="Bread",
        layer_idx=LAYER, baseline_words=BASELINE_WORDS,
    )
    norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
    print(f"[calib3] ||dir||={norm:.3f}", flush=True)

    for eff in TARGET_EFFECTIVES:
        alpha = eff / max(norm, 1e-6)
        print(f"\n=== Bread @ L={LAYER}  target_effective={eff:g}  "
              f"alpha={alpha:.3f}  injection_mag={alpha * norm:.0f} ===",
              flush=True)
        for t in range(N_TRIALS_PER_ALPHA):
            r = run_injected(
                handle,
                concept_to_inject="Bread",
                direction=direction,
                layer_idx=LAYER,
                alpha=alpha,
                trial_number=t + 1,
                seed=2000 + int(eff * 100) + t,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            _summarize(f"trial {t + 1}/{N_TRIALS_PER_ALPHA}", r.response)

    print("\n[calib3] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
