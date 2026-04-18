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
from src.strategies.exploit_topk import generate_candidates as exploit_topk_generate
from src.strategies.hillclimb import generate_candidates as hillclimb_generate
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

# Strategy choices:
#   random         — word pool, pure random sampling (Phase 2a legacy)
#   novel_contrast — Claude-generated abstract axes with feedback from DB
#   exploit_topk   — variants of top-scoring axes (Phase 2b)
#   both           — random + novel_contrast
#   mixed          — novel_contrast (70%) + exploit_topk (30%), Phase 2b mix
#   hillclimb      — Phase 2c REAL autoresearch: per-lineage mutation with
#                    commit-on-improvement, revert-on-failure
STRATEGIES = ("random", "novel_contrast", "exploit_topk", "both", "mixed", "hillclimb")


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
    if args.strategy == "exploit_topk":
        candidates.extend(exploit_topk_generate(
            n=args.n,
            db=db,
            rng_seed=args.seed,
        ))
    if args.strategy == "mixed":
        # 70% novel_contrast (exploration with feedback), 30% exploit_topk
        # (refine the best-so-far). Phase 2b strategy mix.
        n_novel = max(1, round(args.n * 0.7))
        n_exploit = max(1, args.n - n_novel)
        candidates.extend(novel_contrast_generate(
            n=n_novel,
            db=db,
            concept_pool=pool,
            rng_seed=args.seed,
        ))
        candidates.extend(exploit_topk_generate(
            n=n_exploit,
            db=db,
            rng_seed=(args.seed + 1) if args.seed is not None else None,
        ))
    if args.strategy == "hillclimb":
        # Phase 2c proper autoresearch: mutate current leaders incrementally.
        # Falls back to nothing if no lineage leaders exist in the DB (run
        # scripts/seed_lineages.py first).
        candidates.extend(hillclimb_generate(
            n=args.n,
            db=db,
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
        # Lineage metadata (Phase 2c). Stored in the candidate JSON under
        # a nested "_lineage" key so the worker can reconstruct it on
        # insert_candidate. None for non-hillclimb strategies.
        lineage_meta = {
            "lineage_id": getattr(c, "_lineage_id", None),
            "parent_candidate_id": getattr(c, "_parent_candidate_id", None),
            "generation": getattr(c, "_generation", 0),
            "mutation_type": getattr(c, "_mutation_type", None),
            "mutation_detail": getattr(c, "_mutation_detail", None),
            "parent_score": getattr(c, "_parent_score", None),
        }
        has_lineage = lineage_meta["lineage_id"] is not None

        spec_dict = c.to_dict()
        if has_lineage:
            spec_dict["_lineage"] = lineage_meta

        # Insert into DB so future researcher cycles don't duplicate,
        # carrying lineage fields for hillclimb mutations.
        db.insert_candidate(
            candidate_id=c.id,
            strategy=c.strategy,
            spec_json=json.dumps(spec_dict),
            spec_hash=spec_hash(c),
            concept=c.concept,
            layer_idx=c.layer_idx,
            target_effective=c.target_effective,
            derivation_method=c.derivation_method,
            lineage_id=lineage_meta["lineage_id"] if has_lineage else None,
            parent_candidate_id=lineage_meta["parent_candidate_id"] if has_lineage else None,
            generation=lineage_meta["generation"] if has_lineage else 0,
            mutation_type=lineage_meta["mutation_type"] if has_lineage else None,
            mutation_detail=lineage_meta["mutation_detail"] if has_lineage else None,
        )
        # Write spec JSON to queue/pending. Include lineage metadata so the
        # worker reconstructs it on load. (write_candidate_json always dumps
        # spec.to_dict() which excludes _lineage — write by hand.)
        pending_path = args.pending_dir / f"{c.id}.json"
        pending_path.write_text(json.dumps(spec_dict, indent=2) + "\n")

    print(f"wrote {len(candidates)} candidate specs to {args.pending_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
