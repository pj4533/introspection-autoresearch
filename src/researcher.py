"""Phase 2 researcher — short-lived process, invoked periodically.

Reads the DB (to avoid proposing duplicates and to see what's worked so far),
invokes a strategy module to generate N candidate JSON specs, writes them to
queue/pending/. Exits.

Usage:
    # one cycle, 10 candidates via random exploration
    python -m src.researcher --strategy random --n 10

    # print what would be written without writing
    python -m src.researcher --strategy random --n 5 --dry-run

Future strategies will use claude-agent-sdk to propose candidates that exploit
top fitness results or generate novel contrast pairs via Claude.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import ResultsDB
from src.strategies.random_explore import (
    generate_candidates as random_generate,
    spec_hash,
    write_candidate_json,
)

REPO = Path(__file__).resolve().parent.parent
DB_PATH = REPO / "data" / "results.db"
QUEUE_PENDING = REPO / "queue" / "pending"
CONCEPT_POOL_PATH = REPO / "data" / "eval_sets" / "concept_pool.json"

STRATEGIES = {"random": random_generate}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="random")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of candidates to generate")
    ap.add_argument("--concept-pool", type=Path, default=CONCEPT_POOL_PATH)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--pending-dir", type=Path, default=QUEUE_PENDING)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pool = json.loads(Path(args.concept_pool).read_text())["concepts"]
    db = ResultsDB(args.db)

    if args.strategy == "random":
        candidates = random_generate(
            n=args.n,
            db=db,
            concept_pool=pool,
            rng_seed=args.seed,
        )
    else:
        raise NotImplementedError(f"strategy {args.strategy!r} not yet implemented")

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] strategy={args.strategy} "
        f"generated {len(candidates)} new candidates (requested {args.n})"
    )
    for c in candidates:
        print(
            f"  {c.id}  concept={c.concept!r:14} "
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
