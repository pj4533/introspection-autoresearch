"""Phase 1 acceptance check.

Queries `data/results.db` and asserts:
  - max detection rate across layers > 0.20 (at the best layer)
  - false-positive rate across controls < 0.05

Exits 0 on pass, 1 on fail.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import ResultsDB

ACCEPTANCE_DETECTION_RATE = 0.20
ACCEPTANCE_FPR = 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "results.db")
    ap.add_argument("--run-id", default=None,
                    help="Restrict to a specific run_id (default: all trials)")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"FAIL: results DB not found at {args.db}.")
        print("      Run `python scripts/run_phase1_sweep.py` first.")
        return 1

    db = ResultsDB(args.db)
    total = db.count_trials(args.run_id)
    if total == 0:
        print(f"FAIL: no trials in DB (run_id={args.run_id}).")
        return 1

    # Detection rate per layer
    per_layer = db.detection_rate_by_layer(args.run_id)
    fpr_info = db.fpr(args.run_id)

    print(f"Trials in DB: {total}")
    print()
    print("Detection rate by layer (injected trials, coherent-only):")
    print(f"  {'layer':>6}  {'n':>4}  {'det':>6}  {'ident':>6}  {'incoh':>6}")
    max_det = 0.0
    best_layer = None
    for row in per_layer:
        marker = ""
        if row["detection_rate"] > max_det:
            max_det = row["detection_rate"]
            best_layer = row["layer_idx"]
        print(
            f"  {row['layer_idx']:>6}  {row['n']:>4}  "
            f"{row['detection_rate']:>6.2%}  "
            f"{row['identification_rate']:>6.2%}  "
            f"{row['incoherence_rate']:>6.2%}{marker}"
        )
    if best_layer is not None:
        print(f"  → best layer: {best_layer} (detection rate {max_det:.2%})")

    print()
    print(
        f"False-positive rate (controls): "
        f"{fpr_info['n_false_pos']}/{fpr_info['n']} = {fpr_info['fpr']:.2%}"
    )
    print()

    # Acceptance
    det_ok = max_det > ACCEPTANCE_DETECTION_RATE
    fpr_ok = fpr_info["fpr"] < ACCEPTANCE_FPR
    print("Acceptance criteria (from spec section 4.2):")
    status = "PASS" if det_ok else "FAIL"
    print(f"  [{status}] max detection rate > {ACCEPTANCE_DETECTION_RATE:.0%}: "
          f"{max_det:.2%}")
    status = "PASS" if fpr_ok else "FAIL"
    print(f"  [{status}] false-positive rate < {ACCEPTANCE_FPR:.0%}: "
          f"{fpr_info['fpr']:.2%}")

    all_ok = det_ok and fpr_ok
    print()
    print(f"Phase 1 acceptance: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
