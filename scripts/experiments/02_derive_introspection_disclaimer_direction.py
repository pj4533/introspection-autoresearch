"""Derive the per-layer "introspection-disclaimer" direction by running the
Opus-generated matched prompt pairs through vanilla Gemma3-12B and taking
the mean-difference of hidden states at position [-2].

Reuses src.paper.abliteration.compute_per_layer_refusal_directions — that
function is content-agnostic; it just does mean-diff between two prompt
sets. We pass positives as "harmful" and negatives as "harmless" purely
as function arguments.

Input:
  data/experiments/introspection_disclaimer/prompts.json

Output:
  data/experiments/introspection_disclaimer/directions_12b.pt

Usage:
  python scripts/experiments/02_derive_introspection_disclaimer_direction.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import torch

from src.bridge import load_gemma_mps
from src.paper.abliteration import compute_per_layer_refusal_directions

EXP_DIR = REPO / "data" / "experiments" / "introspection_disclaimer"
PROMPTS_PATH = EXP_DIR / "prompts.json"
OUTPUT_PATH = EXP_DIR / "directions_12b.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Cap at N pairs (0 = use all). For quick smoke tests.")
    args = ap.parse_args()

    if not PROMPTS_PATH.exists():
        print(f"ERROR: {PROMPTS_PATH} not found. Run 01_generate_introspection_disclaimer_prompts.py first.",
              file=sys.stderr)
        return 2

    pairs = json.loads(PROMPTS_PATH.read_text())
    if args.max_pairs and len(pairs) > args.max_pairs:
        pairs = pairs[: args.max_pairs]
    print(f"[derive] loaded {len(pairs)} matched pairs from {PROMPTS_PATH}",
          flush=True)

    positives = [p["positive"] for p in pairs]
    negatives = [p["negative"] for p in pairs]

    print(f"[derive] loading gemma3_12b on MPS ...", flush=True)
    model = load_gemma_mps("gemma3_12b")
    print(f"[derive] model loaded. n_layers={model.n_layers}", flush=True)

    t0 = time.time()
    # compute_per_layer_refusal_directions samples n_instructions from each list;
    # we want to use ALL pairs, so pass a large n and it will clamp to len(list).
    directions = compute_per_layer_refusal_directions(
        model=model,
        harmful_prompts=positives,   # positives go in the "harmful" slot → mean(+) - mean(-)
        harmless_prompts=negatives,
        n_instructions=max(len(positives), len(negatives)),
        pos=-2,
        seed=0,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"[derive] done in {elapsed:.1f}s. shape={tuple(directions.shape)}",
          flush=True)

    # Compute a few stats about the directions for a sanity check.
    per_layer_norms = directions.norm(dim=-1)
    print(f"[derive] per-layer norms: mean={per_layer_norms.mean():.3f} "
          f"min={per_layer_norms.min():.3f} max={per_layer_norms.max():.3f}")
    print(f"[derive] (all should be ~1.0 since directions are unit-normalized)")

    # Save in the same format compute_per_layer_refusal_directions consumers
    # expect (bare tensor; AbliterationContext.from_file handles both bare
    # and {"directions": tensor} forms).
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "directions": directions,
        "n_pairs_used": len(positives),
        "source": "introspection-disclaimer matched pairs (Opus 4.7 generated)",
        "extraction_pos": -2,
    }
    torch.save(payload, str(OUTPUT_PATH))
    print(f"[derive] saved to {OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
