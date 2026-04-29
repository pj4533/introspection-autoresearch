"""Phase 3 extended-tokens calibration.

The first calibration (calibrate_phase3_alpha.py) revealed that
Gemma 4 31B-IT has a built-in `<|channel>thought ... <channel|>`
chain-of-thought structure, and at max_new_tokens=400 the model is
still mid-thought when generation stops. We never see the actual
answer.

This script reruns the small-alpha cells (5, 25, 50, 100) at
max_new_tokens=1200 so the model can finish its thought + answer,
plus 4 control trials at the same length so we can see what the
"baseline" answer looks like and whether any small-alpha injection
flips the answer toward detection / bread-related content.

Total: 4 controls + 4 alphas × 2 trials = 12 generations.
At max_new_tokens=1200 each, ~30-40 min wall.
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

TARGET_EFFECTIVES = (25.0, 50.0, 100.0, 200.0)
LAYER = PREDICTED_PEAK_LAYER  # 42
MAX_NEW_TOKENS = 1200
N_CONTROLS = 4
N_TRIALS_PER_ALPHA = 2


def _summarize(label: str, response: str) -> None:
    """Print the full response, with newlines preserved (long form is the
    point of this experiment — we want to see thought + answer)."""
    print(f"\n--- {label} ---", flush=True)
    print(response, flush=True)
    print(f"--- (end {label}, {len(response)} chars) ---", flush=True)


def main() -> int:
    print("[calib3-ext] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[calib3-ext] n_layers={handle.n_layers}  hidden_dim={handle.hidden_dim}",
          flush=True)

    print(f"\n=== Controls (no injection, max_new_tokens={MAX_NEW_TOKENS}) ===",
          flush=True)
    for i in range(N_CONTROLS):
        ctrl = run_control(
            handle, concept_label="ctrl",
            trial_number=i + 1, seed=100 + i,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        _summarize(f"ctrl trial {i + 1}/{N_CONTROLS}", ctrl.response)

    print(f"\n=== Deriving Bread vector at L={LAYER} ===", flush=True)
    direction = derive_concept_vector(
        handle, concept="Bread",
        layer_idx=LAYER, baseline_words=BASELINE_WORDS,
    )
    norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
    print(f"[calib3-ext] ||dir||={norm:.3f}", flush=True)

    for eff in TARGET_EFFECTIVES:
        alpha = eff / max(norm, 1e-6)
        for t in range(N_TRIALS_PER_ALPHA):
            r = run_injected(
                handle,
                concept_to_inject="Bread",
                direction=direction,
                layer_idx=LAYER,
                alpha=alpha,
                trial_number=t + 1,
                seed=3000 + int(eff * 100) + t,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            _summarize(
                f"Bread eff={eff:g} alpha={alpha:.2f} trial {t + 1}/{N_TRIALS_PER_ALPHA}",
                r.response,
            )

    print("\n[calib3-ext] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
