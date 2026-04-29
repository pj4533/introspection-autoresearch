"""Phase 3 alpha calibration at the discovered peak layer L=25.

The layer sweep (calibrate_phase3_layers.py) found L=25 produces clean
"Yes. It was bread." detection at eff=100. Now sweep alpha at L=25
across multiple concepts to:
  - Confirm reproducibility (multiple trials at the same eff)
  - Find the alpha range that triggers detection without over-steering
  - Test on different concepts to verify the layer choice generalizes

For Bread + Peace + Sugar (all known Phase 1 positives):
  eff ∈ {25, 50, 100, 200, 400} × 2 trials each = 30 generations.
  Plus 3 controls. ~50 min wall.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.phase3.gemma4_loader import load_gemma4
from src.phase3.pipeline import (
    derive_concept_vector, run_injected, run_control,
)


BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

CONCEPTS = ["Bread", "Peace", "Sugar"]
TARGET_EFFECTIVES = (25.0, 50.0, 100.0, 200.0, 400.0)
LAYER = 25
MAX_NEW_TOKENS = 1200
N_TRIALS_PER_ALPHA = 2


def _post_thought(response: str) -> str:
    """Extract just the actual answer (after `<channel|>` close)."""
    m = response.find("<channel|>")
    if m >= 0:
        return response[m + len("<channel|>"):].strip()
    return f"[no channel close] tail: ...{response[-200:]}"


def main() -> int:
    print("[l25-calib] loading Gemma 4 ...", flush=True)
    handle = load_gemma4()

    print("\n=== Controls ===", flush=True)
    for i in range(3):
        ctrl = run_control(
            handle, concept_label="ctrl",
            trial_number=i + 1, seed=300 + i,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"  ctrl {i + 1}: {_post_thought(ctrl.response)!r}", flush=True)

    for concept in CONCEPTS:
        print(f"\n{'#' * 60}", flush=True)
        print(f"### Concept: {concept}  @ L={LAYER}", flush=True)
        print(f"{'#' * 60}", flush=True)
        direction = derive_concept_vector(
            handle, concept=concept,
            layer_idx=LAYER, baseline_words=BASELINE_WORDS,
        )
        norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
        print(f"  ||dir||={norm:.3f}", flush=True)

        for eff in TARGET_EFFECTIVES:
            alpha = eff / max(norm, 1e-6)
            print(f"\n  --- eff={eff:g}  alpha={alpha:.2f} ---", flush=True)
            for t in range(N_TRIALS_PER_ALPHA):
                r = run_injected(
                    handle,
                    concept_to_inject=concept,
                    direction=direction,
                    layer_idx=LAYER,
                    alpha=alpha,
                    trial_number=t + 1,
                    seed=hash((concept, eff, t)) & 0x7FFFFFFF,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                print(f"    trial {t + 1}/{N_TRIALS_PER_ALPHA}: "
                      f"{_post_thought(r.response)!r}",
                      flush=True)

    print("\n[l25-calib] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
