"""Phase 2 researcher — short-lived process, invoked periodically.

Reads the DB (to avoid proposing duplicates and to see what's worked so far),
invokes a strategy module to generate N candidate JSON specs, writes them to
queue/pending/. Exits.

Usage:
    # Random word sweep (Phase 2a — Phase 1 extended)
    python -m src.researcher --strategy random --n 10

    # Abstract-axis contrast pairs (Phase 2b — "concepts without names" hunt)
    python -m src.researcher --strategy novel_contrast --n 10

    # Mix: run both strategies in sequence
    python -m src.researcher --strategy both --n 10

    # Print what would be written without writing
    python -m src.researcher --strategy random --n 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import ResultsDB
from src.strategies.novel_contrast import generate_candidates as novel_contrast_generate
from src.strategies.random_explore import (
    generate_candidates as random_generate,
    spec_hash,
    write_candidate_json,
)

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "results.db"
DB_PATH_ABLITERATED = REPO / "data" / "results_abliterated.db"
QUEUE_PENDING = REPO / "queue" / "pending"
CONCEPT_POOL_PATH = REPO / "data" / "eval_sets" / "concept_pool.json"

STRATEGIES = ("random", "novel_contrast", "both")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="random")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of candidates to generate (per strategy for 'both')")
    ap.add_argument("--concept-pool", type=Path, default=CONCEPT_POOL_PATH)
    ap.add_argument("--db", type=Path, default=None,
                    help="Override DB path. Defaults: data/results.db (vanilla), "
                         "data/results_abliterated.db (--abliterated).")
    ap.add_argument("--abliterated", action="store_true",
                    help="Target the abliterated DB (data/results_abliterated.db) "
                         "for dedup. Candidates written to the same queue; the "
                         "worker decides which model to load via its own flag.")
    ap.add_argument("--pending-dir", type=Path, default=QUEUE_PENDING)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.db is None:
        args.db = DB_PATH_ABLITERATED if args.abliterated else DB_PATH

    pool = json.loads(Path(args.concept_pool).read_text())["concepts"]
    db = ResultsDB(args.db)

    candidates = []
    if args.strategy in ("random", "both"):
        candidates.extend(random_generate(
            n=args.n,
            db=db,
            concept_pool=pool,
            rng_seed=args.seed,
        ))
    if args.strategy in ("novel_contrast", "both"):
        candidates.extend(novel_contrast_generate(
            n=args.n,
            db=db,
            concept_pool=pool,  # accepted but ignored by novel_contrast
            rng_seed=args.seed,
        ))

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] strategy={args.strategy} "
        f"generated {len(candidates)} new candidates (requested {args.n})"
    )
    for c in candidates:
        method_tag = "contr" if c.derivation_method == "contrast_pair" else "mean"
        print(
            f"  {c.id}  [{method_tag}] concept={c.concept!r:22} "
            f"L={c.layer_idx:>3} eff={c.target_effective:>6.0f} "
            f"hash={spec_hash(c)}"
        )

    if args.dry_run:
        print("(dry run — not writing to queue)")
        return 0

    args.pending_dir.mkdir(parents=True, exist_ok=True)
    for c in candidates:
        # Insert into DB so future researcher cycles don't duplicate.
        db.insert_candidate(
            candidate_id=c.id,
            strategy=c.strategy,
            spec_json=json.dumps(c.to_dict()),
            spec_hash=spec_hash(c),
            concept=c.concept,
            layer_idx=c.layer_idx,
            target_effective=c.target_effective,
            derivation_method=c.derivation_method,
        )
        write_candidate_json(c, args.pending_dir)

    print(f"wrote {len(candidates)} candidate specs to {args.pending_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
