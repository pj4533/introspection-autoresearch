"""Mine concept→concept transition cycles in phase4_steps.

For every chain, look at consecutive (target_lemma_at_step_i,
target_lemma_at_step_i+1) edges. If a chain visits a lemma it has
already visited, that closes a cycle. The cycle's lemma sequence is
the attractor.

Surface only attractors that have:
  - cycle length ≥ 2 (no trivial self-loops)
  - cycle length ≤ 6 (longer cycles are usually noise)
  - visited count ≥ 2 (across all chains where the cycle appears)

Writes web/public/data/attractors.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


DB_PATH = REPO / "data" / "results.db"
OUTPUT_DIR = REPO / "web" / "public" / "data"
MIN_CYCLE_LEN = 2
MAX_CYCLE_LEN = 6
MIN_VISIT_COUNT = 2


def _normalize_cycle(seq: list[str]) -> tuple[str, ...]:
    """Rotate a cycle to start at its lexicographically smallest lemma —
    so that ['silver','moon','night'] and ['night','silver','moon']
    canonicalize to the same key."""
    if not seq:
        return tuple()
    min_idx = min(range(len(seq)), key=lambda i: seq[i])
    return tuple(seq[min_idx:] + seq[:min_idx])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-len", type=int, default=MIN_CYCLE_LEN)
    parser.add_argument("--max-len", type=int, default=MAX_CYCLE_LEN)
    parser.add_argument("--min-visits", type=int, default=MIN_VISIT_COUNT)
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        out = {"attractors": [], "summary": {"n_attractors": 0},
               "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (OUTPUT_DIR / "attractors.json").write_text(json.dumps(out, indent=2))
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cycle_counter: Counter[tuple[str, ...]] = Counter()
    cycle_examples: dict[tuple[str, ...], list[str]] = {}

    chain_rows = conn.execute(
        "SELECT chain_id, end_reason FROM phase4_chains"
    ).fetchall()

    for chain in chain_rows:
        chain_id = chain["chain_id"]
        steps = conn.execute(
            """SELECT target_lemma FROM phase4_steps
               WHERE chain_id=?
               ORDER BY step_idx""",
            (chain_id,),
        ).fetchall()
        if len(steps) < args.min_len:
            continue
        lemmas = [s["target_lemma"] for s in steps]

        # If the chain ended on self_loop, the cycle is the suffix from
        # the first repeated lemma to the end.
        seen_idx = {}
        for i, lemma in enumerate(lemmas):
            if lemma in seen_idx:
                cycle = lemmas[seen_idx[lemma]:i]
                if args.min_len <= len(cycle) <= args.max_len:
                    canon = _normalize_cycle(cycle)
                    cycle_counter[canon] += 1
                    cycle_examples.setdefault(canon, []).append(chain_id)
                break
            seen_idx[lemma] = i

    conn.close()

    attractors = []
    for canon, count in cycle_counter.most_common():
        if count < args.min_visits:
            continue
        attractors.append({
            "lemma_cycle": list(canon),
            "length": len(canon),
            "visit_count": count,
            "example_chain_ids": cycle_examples[canon][:5],
        })

    out = {
        "attractors": attractors,
        "summary": {
            "n_attractors": len(attractors),
            "n_chains_examined": len(chain_rows),
            "thresholds": {
                "min_len": args.min_len,
                "max_len": args.max_len,
                "min_visits": args.min_visits,
            },
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUTPUT_DIR / "attractors.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[compute_attractors] wrote {out_path} — "
          f"{len(attractors)} attractors from {len(chain_rows)} chains",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
