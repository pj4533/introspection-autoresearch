"""Compute Gemma 4 31B-IT per-layer refusal directions.

One-shot. Loads Gemma 4, runs 128 harmful + 128 harmless prompts at
position -2, computes per-layer mean-difference, L2-normalizes,
saves to data/refusal_directions_31b.npy.

Run-time: ~20-30 min wall on M2 Ultra (256 forward passes through a
31B model). The output file (~1.3 MB, n_layers=60 × hidden_dim=5376
in fp32) gets reused across every Phase 3 abliteration run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.phase3.abliteration import (
    compute_per_layer_refusal_directions, save_refusal_dirs,
)
from src.phase3.gemma4_loader import load_gemma4
from src.paper.refusal_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS


OUTPUT_PATH = REPO / "data" / "refusal_directions_31b.npy"


def main() -> int:
    print("[refusal31b] loading Gemma 4 31B-IT MLX 8-bit ...", flush=True)
    handle = load_gemma4()
    print(f"[refusal31b] n_layers={handle.n_layers}  "
          f"hidden_dim={handle.hidden_dim}", flush=True)

    print(f"[refusal31b] harmful pool: {len(HARMFUL_PROMPTS)}  "
          f"harmless pool: {len(HARMLESS_PROMPTS)}",
          flush=True)

    refusal = compute_per_layer_refusal_directions(
        handle,
        harmful_prompts=HARMFUL_PROMPTS,
        harmless_prompts=HARMLESS_PROMPTS,
        n_instructions=128,
    )
    save_refusal_dirs(refusal, OUTPUT_PATH)
    print(f"[refusal31b] wrote {OUTPUT_PATH}  "
          f"shape={tuple(refusal.shape)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
