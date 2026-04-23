"""Quick calibration: find the target_effective value that maximizes strict-
paper detection rate on 12B without blowing up coherence. Loops over a handful
of (concept, target_effective) cells at a single layer (33). ~10 minutes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bridge import DetectionPipeline, load_gemma_mps
from src.judges.claude_judge import ClaudeJudge
import torch

REPO = Path(__file__).resolve().parent.parent

CONCEPTS = ["Bread", "Trumpets", "Mirrors", "Happiness", "Silver", "Oceans", "Sugar", "Lightning"]
LAYER = 33
TARGET_EFFECTIVES = [14000, 16000, 18000, 20000]


def main() -> int:
    print("Loading model (bf16 on MPS)...")
    model = load_gemma_mps("gemma3_12b")
    judge = ClaudeJudge(
        model="claude-sonnet-4-6",
        cache_path=REPO / "data" / "judge_cache.sqlite",
    )
    pipeline = DetectionPipeline(model=model, judge=judge)
    print(f"Model loaded: n_layers={model.n_layers}\n")

    results = []
    t0 = time.time()
    for concept in CONCEPTS:
        direction = pipeline.derive(concept=concept, layer_idx=LAYER)
        norm = float(direction.norm().item())
        print(f"=== {concept}  (||dir||={norm:.0f}) ===")
        for eff in TARGET_EFFECTIVES:
            alpha = eff / norm
            torch.manual_seed(1)
            trial = pipeline.run_injected(
                concept=concept,
                direction=direction,
                layer_idx=LAYER,
                strength=alpha,
                trial_number=1,
                max_new_tokens=120,
            )
            jr = trial.judge_result
            results.append(
                dict(concept=concept, eff=eff, alpha=alpha,
                     det=jr.detected, ident=jr.identified, coh=jr.coherent,
                     response=trial.response)
            )
            print(f"  eff={eff:>5}  a={alpha:>5.2f}  d={int(jr.detected)} "
                  f"id={int(jr.identified)} coh={int(jr.coherent)}  "
                  f"{trial.response[:90].replace(chr(10), ' ')!r}")
        print()

    print("\n=== AGGREGATE ===")
    print(f"{'eff':>6}  {'n_det':>6}  {'n_ident':>8}  {'n_coh':>6}  {'det_rate':>8}")
    for eff in TARGET_EFFECTIVES:
        subset = [r for r in results if r["eff"] == eff]
        n = len(subset)
        n_det = sum(1 for r in subset if r["det"])
        n_id = sum(1 for r in subset if r["ident"])
        n_coh = sum(1 for r in subset if r["coh"])
        print(f"{eff:>6}  {n_det:>6}  {n_id:>8}  {n_coh:>6}  {n_det/n:>7.1%}")

    print(f"\nTotal calibration time: {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
