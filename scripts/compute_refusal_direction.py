"""One-shot extraction of the per-layer refusal direction for Gemma3-12B-it.

Loads vanilla Gemma3-12B-it, runs the paper's harmful/harmless prompt pairs
through it, computes the mean-diff refusal direction at every layer, and
saves the resulting ``(n_layers, hidden_dim)`` tensor to
``data/refusal_directions_12b.pt``.

Takes ~3-5 minutes on a Mac Studio M2 Ultra. Needs GPU.

Usage:
    python scripts/compute_refusal_direction.py
    python scripts/compute_refusal_direction.py --n 64  # faster, less stable
    python scripts/compute_refusal_direction.py --seed 42

Subsequent sweeps can load the saved tensor and install ablation hooks;
see ``src/paper/abliteration.py::install_abliteration_hooks``.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.bridge import load_gemma_mps
from src.paper.abliteration import compute_per_layer_refusal_directions
from src.paper.refusal_prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "data" / "refusal_directions_12b.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3_12b",
                    help="Model slug to extract from (default: vanilla 12B)")
    ap.add_argument("--n", type=int, default=128,
                    help="Number of harmful and harmless prompts each (paper used 128)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print(f"Loading {args.model} on MPS (bf16)...")
    model = load_gemma_mps(args.model)
    print(f"Loaded. n_layers={model.n_layers}")
    print()

    t0 = time.time()
    directions = compute_per_layer_refusal_directions(
        model=model,
        harmful_prompts=HARMFUL_PROMPTS,
        harmless_prompts=HARMLESS_PROMPTS,
        n_instructions=args.n,
        seed=args.seed,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"Directions shape: {tuple(directions.shape)}")
    print(f"Norms (should all be ~1.0): "
          f"min={directions.norm(dim=1).min().item():.4f} "
          f"max={directions.norm(dim=1).max().item():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": directions,
            "model": args.model,
            "n_instructions": args.n,
            "seed": args.seed,
            "extraction_time_s": elapsed,
        },
        args.out,
    )
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
