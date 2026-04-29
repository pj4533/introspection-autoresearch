"""Phase 3 layer sweep on Bread.

The first two calibrations established that on Gemma 4 31B-IT at L=42:
  - Controls produce clean Phase-1-shape denials (FPR=0).
  - Bread injection across alpha 0.97 → 7.77 produces no detection.
  - Coherence breaks past alpha ~10.

Open question: is L=42 the wrong layer? Phase 1 found 12B's
introspection peak at L=33 (68.75% depth), and the paper predicted
~70% depth on 27B. But Gemma 4 has a mixed attention pattern (1/6
full-attention + 5/6 sliding-window) that may shift where the
introspection circuit lives. Try a coarse sweep across the depth.

Bread injection × {L=15, 25, 35, 42, 50, 55} × eff=100 × 1 trial.
Plus 1 control. 7 generations at max_new_tokens=1200 ≈ 25 min wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.phase3.gemma4_loader import load_gemma4
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

LAYERS = (15, 25, 35, 42, 50, 55)
TARGET_EFFECTIVE = 100.0
MAX_NEW_TOKENS = 1200


def main() -> int:
    print("[layer-sweep] loading Gemma 4 ...", flush=True)
    handle = load_gemma4()
    print(f"[layer-sweep] n_layers={handle.n_layers}", flush=True)

    print("\n=== Control ===", flush=True)
    ctrl = run_control(
        handle, concept_label="ctrl", trial_number=1, seed=200,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(f"  ANSWER: {ctrl.response[-300:]!r}", flush=True)

    for L in LAYERS:
        print(f"\n=== Bread @ L={L}  eff={TARGET_EFFECTIVE} ===", flush=True)
        direction = derive_concept_vector(
            handle, concept="Bread",
            layer_idx=L, baseline_words=BASELINE_WORDS,
        )
        norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
        alpha = TARGET_EFFECTIVE / max(norm, 1e-6)
        print(f"  ||dir||={norm:.3f}  alpha={alpha:.3f}", flush=True)
        result = run_injected(
            handle,
            concept_to_inject="Bread",
            direction=direction,
            layer_idx=L,
            alpha=alpha,
            trial_number=1,
            seed=4000 + L,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        # Print the full response
        print(f"  FULL RESPONSE ({len(result.response)} chars):", flush=True)
        print(result.response, flush=True)

    print("\n[layer-sweep] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
