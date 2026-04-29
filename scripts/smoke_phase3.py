"""Phase 3 smoke — single-concept end-to-end on Gemma 4 31B-it.

Purpose: validate the full Phase 3 pipeline (load + derive + inject +
generate) on one known concept before committing to any larger sweep.

Loads Gemma 4 31B-it MLX 8-bit, picks Bread as the test concept,
derives a concept vector at L=42 (~70% of 60 layers), runs:

  1. One uninjected control trial (baseline response shape).
  2. A small target_effective sweep (4 alphas) × 3 trials each at
     L=42, no abliteration. Surfaces qualitative response patterns at
     each magnitude so we can pick a sane operating range.

Total: ~13 generations + a few forward passes for derivation. ~10 min
wall after the first model load.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import json

import mlx.core as mx

from src.phase3.gemma4_loader import load_gemma4, PREDICTED_PEAK_LAYER
from src.phase3.pipeline import (
    derive_concept_vector,
    run_injected,
    run_control,
)


# 24 baseline words from the paper's BASELINE_WORDS_BG. Same set the
# Phase 1 mean-diff used.
BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

DEFAULT_TARGET_EFFECTIVES = (8000.0, 14000.0, 18000.0, 24000.0)
DEFAULT_LAYER = PREDICTED_PEAK_LAYER  # 42
N_TRIALS_PER_ALPHA = 3


def _summarize(label: str, response: str) -> None:
    short = response[:140].replace("\n", " ")
    print(f"  {label}: {short!r}", flush=True)


def main() -> int:
    print("[smoke3] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[smoke3] model loaded: n_layers={handle.n_layers}  "
          f"hidden_dim={handle.hidden_dim}", flush=True)

    # Sanity: control trial first to verify generation works at all.
    print("\n=== Control (no injection) ===", flush=True)
    control = run_control(handle, concept_label="Apple", trial_number=1, seed=42)
    _summarize("ctrl", control.response)

    # Derive concept vector for Bread.
    print(f"\n=== Deriving Bread concept vector at L={DEFAULT_LAYER} ===",
          flush=True)
    direction = derive_concept_vector(
        handle, concept="Bread",
        layer_idx=DEFAULT_LAYER,
        baseline_words=BASELINE_WORDS,
    )
    norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
    print(f"[smoke3] direction: shape={tuple(direction.shape)}  "
          f"dtype={direction.dtype}  ||dir||={norm:.2f}",
          flush=True)

    # Sweep alphas at L=42.
    for eff in DEFAULT_TARGET_EFFECTIVES:
        alpha = eff / max(norm, 1e-6)
        print(f"\n=== Bread @ L={DEFAULT_LAYER}  target_effective={eff:.0f}  "
              f"alpha={alpha:.4f}  injection_mag={alpha * norm:.0f} ===",
              flush=True)
        for t in range(N_TRIALS_PER_ALPHA):
            result = run_injected(
                handle,
                concept_to_inject="Bread",
                direction=direction,
                layer_idx=DEFAULT_LAYER,
                alpha=alpha,
                trial_number=t + 1,
                seed=1000 + int(eff) + t,
            )
            _summarize(f"trial {t + 1}/{N_TRIALS_PER_ALPHA}", result.response)

    print("\n[smoke3] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
