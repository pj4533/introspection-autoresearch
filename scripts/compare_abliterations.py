"""Side-by-side comparison of abliteration variants on Gemma3-12B-it.

Produces a short summary table from two SQLite DBs:
- data/results_abliterated.db          (mlabonne v2)
- data/results_abliterated_huihui.db   (huihui-ai)

Goal: determine whether the ~97% FPR we see with mlabonne v2 is specific
to that abliteration method or a general consequence of ablating 12B's
refusal direction by any means.

Usage:
    python scripts/compare_abliterations.py
"""

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLABONNE_DB = REPO / "data" / "results_abliterated.db"
HUIHUI_DB = REPO / "data" / "results_abliterated_huihui.db"
VANILLA_DB = REPO / "data" / "results.db"


def layer_curve(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT layer_idx,
               COUNT(*) AS n,
               SUM(CASE WHEN detected=1 AND coherent=1 AND injected=1 THEN 1 ELSE 0 END) AS n_det,
               SUM(CASE WHEN coherent=0 AND injected=1 THEN 1 ELSE 0 END) AS n_incoh
        FROM trials WHERE injected=1
        GROUP BY layer_idx ORDER BY layer_idx
        """
    ).fetchall()
    return [
        {
            "layer": r["layer_idx"],
            "n": r["n"],
            "det_rate": r["n_det"] / r["n"] if r["n"] else 0.0,
            "incoh_rate": r["n_incoh"] / r["n"] if r["n"] else 0.0,
        }
        for r in rows
    ]


def fpr(db_path: Path) -> dict:
    if not db_path.exists():
        return {"n": 0, "n_fp": 0, "rate": 0.0}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    r = conn.execute(
        """SELECT COUNT(*) AS n,
                  SUM(CASE WHEN detected=1 THEN 1 ELSE 0 END) AS n_fp
           FROM trials WHERE injected=0"""
    ).fetchone()
    n = int(r["n"] or 0)
    n_fp = int(r["n_fp"] or 0)
    return {"n": n, "n_fp": n_fp, "rate": n_fp / n if n else 0.0}


def main() -> int:
    sources = [
        ("vanilla", VANILLA_DB),
        ("mlabonne v2", MLABONNE_DB),
        ("huihui", HUIHUI_DB),
    ]

    for label, path in sources:
        print(f"\n{'='*70}")
        print(f"  {label}: {path.name}  {'(missing)' if not path.exists() else ''}")
        print(f"{'='*70}")
        if not path.exists():
            print("  (not yet run)")
            continue
        fpr_info = fpr(path)
        print(f"  FPR (controls): {fpr_info['n_fp']}/{fpr_info['n']} = {fpr_info['rate']:.1%}")
        print()
        print(f"  {'layer':>6}  {'n':>4}  {'det_rate':>8}  {'incoh_rate':>10}")
        for row in layer_curve(path):
            print(
                f"  {row['layer']:>6}  {row['n']:>4}  "
                f"{row['det_rate']:>7.1%}  {row['incoh_rate']:>9.1%}"
            )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
