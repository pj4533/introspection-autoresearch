"""CLI to run the Phase 1 full sweep.

Usage:
    python scripts/run_phase1_sweep.py                  # full default sweep
    python scripts/run_phase1_sweep.py --dry-run        # show plan, no work
    python scripts/run_phase1_sweep.py --concepts 5     # first 5 concepts only (smoke test)
    python scripts/run_phase1_sweep.py --layers 30 33   # narrow layer grid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sweep import SweepConfig, load_concepts, run_sweep

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONCEPTS = REPO / "data" / "concepts" / "concepts_50.json"
DEFAULT_DB = REPO / "data" / "results.db"
DEFAULT_LAYERS = [10, 15, 20, 25, 30, 33, 36, 40, 44]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Phase 1 sweep")
    ap.add_argument("--concepts-file", type=Path, default=DEFAULT_CONCEPTS)
    ap.add_argument("--concepts", type=int, default=None,
                    help="Use only the first N concepts (for smoke testing)")
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--alpha", type=float, default=4.0,
                    help="Fixed alpha (ignored if --target-effective is set)")
    ap.add_argument("--target-effective", type=float, default=18000.0,
                    help="Target effective steering strength alpha*||dir||. "
                         "Set to 0 to use fixed --alpha instead.")
    ap.add_argument("--trials-per-cell", type=int, default=1)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--model", default="gemma3_12b")
    ap.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the plan and exit without running")
    args = ap.parse_args()

    concepts = load_concepts(args.concepts_file)
    if args.concepts is not None:
        concepts = concepts[: args.concepts]

    target_eff = args.target_effective if args.target_effective > 0 else None
    cfg = SweepConfig(
        concepts=concepts,
        layers=args.layers,
        alpha=args.alpha,
        target_effective=target_eff,
        trials_per_cell=args.trials_per_cell,
        run_controls=not args.no_controls,
        judge_model=args.judge_model,
    )

    print(f"Concepts:    {len(cfg.concepts)}  {cfg.concepts[:3]}{'...' if len(cfg.concepts) > 3 else ''}")
    print(f"Layers:      {cfg.layers}")
    if cfg.target_effective is not None:
        print(f"Alpha mode:  adaptive (target effective = {cfg.target_effective})")
    else:
        print(f"Alpha mode:  fixed = {cfg.alpha}")
    print(f"Trials/cell: {cfg.trials_per_cell}")
    print(f"Controls:    {cfg.run_controls}")
    print(f"Total cells: {cfg.total_cells()}")
    print(f"DB:          {args.db}")
    print(f"Model:       {args.model}")
    print(f"Judge:       {args.judge_model}")
    print()

    if args.dry_run:
        print("Dry run — exiting.")
        return 0

    run_id = run_sweep(
        cfg,
        db_path=args.db,
        model_name=args.model,
        run_id=args.run_id,
    )
    print(f"\nDone. run_id={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
