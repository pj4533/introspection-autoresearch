"""Smoke test for Phase 2g SAE-feature injection.

Verifies the substrate plumbing works end-to-end without running the
full worker:

  1. Loads the SAE checkpoint and pulls a few decoder vectors.
  2. Builds a CandidateSpec for one feature from the buckets.
  3. Reports shapes / norms / the alpha that target_effective implies.
  4. (Optional, --run-gemma) loads Gemma3-12B and runs ONE injected probe.

This is intentionally lightweight — it does NOT touch the judge, the DB,
or the queue. It's a unit-level confirmation that the SAE→direction→
alpha pipeline works before kicking off a full overnight loop.

Usage:

    python scripts/smoke_sae.py
    python scripts/smoke_sae.py --fault-line metacognition
    python scripts/smoke_sae.py --feature-idx 78755   # explicit feature
    python scripts/smoke_sae.py --run-gemma           # also do a real probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from src.sae_loader import (
    DEFAULT_RELEASE,
    DEFAULT_SAE_ID,
    get_decoder_direction,
    n_features,
    hidden_dim,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fault-line", default="metacognition",
                    help="Capraro fault line to draw a feature from.")
    ap.add_argument("--feature-idx", type=int, default=None,
                    help="Override: load this exact feature index.")
    ap.add_argument("--target-effective", type=float, default=16.0,
                    help="α × ‖dir‖ target. SAE features are unit-norm "
                         "so this is approximately the alpha value.")
    ap.add_argument("--run-gemma", action="store_true",
                    help="Actually load Gemma and run one probe.")
    args = ap.parse_args()

    print(f"[smoke_sae] SAE: {DEFAULT_RELEASE}/{DEFAULT_SAE_ID}", flush=True)
    print(f"[smoke_sae] loading SAE config (caches W_dec) ...", flush=True)
    n_feat = n_features()
    hdim = hidden_dim()
    print(f"[smoke_sae]   n_features={n_feat:,}  hidden_dim={hdim}", flush=True)

    if args.feature_idx is not None:
        feature_idx = args.feature_idx
        auto_interp = "(provided via --feature-idx)"
    else:
        buckets_path = REPO / "data" / "sae_features" / "capraro_buckets.json"
        if not buckets_path.exists():
            print(f"[smoke_sae] ERROR: {buckets_path} missing. Run "
                  f"`python scripts/build_capraro_buckets.py` first.",
                  flush=True)
            return 1
        buckets = json.loads(buckets_path.read_text())
        bucket = buckets["buckets"].get(args.fault_line)
        if not bucket:
            print(f"[smoke_sae] no bucket for fault line "
                  f"{args.fault_line!r}. Available: "
                  f"{list(buckets['buckets'].keys())}", flush=True)
            return 2
        top = bucket[0]
        feature_idx = int(top["feature_idx"])
        auto_interp = top["auto_interp"]
        print(f"[smoke_sae] picked top feature for {args.fault_line!r}:",
              flush=True)
        print(f"   idx={feature_idx}  score={top['score']:.3f}  "
              f"auto_interp={auto_interp!r}", flush=True)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    direction = get_decoder_direction(
        release=DEFAULT_RELEASE,
        sae_id=DEFAULT_SAE_ID,
        feature_idx=feature_idx,
        device=device,
        dtype=dtype,
    )
    norm = float(direction.float().norm().item())
    alpha = args.target_effective / max(norm, 1e-6)
    print(f"[smoke_sae] decoder vector:", flush=True)
    print(f"   shape={tuple(direction.shape)}  dtype={direction.dtype}  norm={norm:.4f}",
          flush=True)
    print(f"[smoke_sae] target_effective={args.target_effective:.2f} → alpha={alpha:.4f}",
          flush=True)

    if not args.run_gemma:
        print("[smoke_sae] skipped Gemma probe (use --run-gemma to enable).",
              flush=True)
        return 0

    print("[smoke_sae] loading Gemma3-12B ...", flush=True)
    from src.bridge import build_pipeline
    pipeline = build_pipeline(model_id="gemma3_12b")
    model_dtype = next(pipeline.model.parameters()).dtype
    model_device = next(pipeline.model.parameters()).device
    direction_dev = direction.to(device=model_device, dtype=model_dtype)

    print("[smoke_sae] injecting at L=31, eff="
          f"{args.target_effective}, alpha={alpha:.4f} ...",
          flush=True)
    trial = pipeline.run_injected(
        concept=auto_interp,
        direction=direction_dev,
        layer_idx=31,
        strength=alpha,
        trial_number=1,
        max_new_tokens=120,
        judge_concept=auto_interp,
        prompt_style="paper",
        run_judge=False,
    )
    print("[smoke_sae] response:", flush=True)
    print("---", flush=True)
    print(trial.response, flush=True)
    print("---", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
