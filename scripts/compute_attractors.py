"""Mine gravity wells in the Phase 4 dream-walk data.

A gravity well is a concept that many independent chains converge on.
Defined here as: any concept that two or more distinct chains landed
on as their effective end point.

For each chain we compute the "landing concept":

  - end_reason='self_loop'    → the last step's `final_answer` lemma
                                (the concept the chain was ABOUT TO
                                revisit before the dedup stopped it).
  - end_reason='length_cap'   → the last step's `target_concept` lemma.
  - end_reason='coherence_break' / 'parse_fail' / 'error' → the last
                                step that has a clean
                                `final_answer` we can extract; otherwise
                                the last `target_concept`.

We count how many distinct chains land on each lemma and which seed
concepts they came from. Surface anything with ≥MIN_VISITS chains.

Writes web/public/data/attractors.json.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.phase4.cot_parser import extract_committed_word
from src.phase4.seed_pool import normalize_lemma


DB_PATH = REPO / "data" / "results.db"
OUTPUT_DIR = REPO / "web" / "public" / "data"
MIN_CHAINS = 2  # surface concepts hit by ≥ this many chains


def _landing(chain: dict, steps: list[dict]) -> tuple[str | None, str | None]:
    """Return (lemma, display) for the chain's gravity-well landing.

    For self_loop, the landing is the last step's emission (which is
    the concept the chain was about to revisit but couldn't). For
    other end reasons, the landing is the last target the chain
    actually ran on, falling back to extracted committed word if
    available.
    """
    if not steps:
        return None, None
    last = steps[-1]
    end_reason = chain.get("end_reason")

    if end_reason == "self_loop":
        # The "would-be revisit" — the model's final answer at the last
        # recorded step is the concept that triggered the stop.
        word = extract_committed_word(last.get("final_answer") or "")
        if word:
            lemma = normalize_lemma(word)
            if lemma:
                return lemma, word

    # Default: the last target the chain ran. Use display form from the
    # step row.
    target_lemma = last.get("target_lemma")
    target_display = last.get("target_concept")
    return target_lemma, target_display


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-chains", type=int, default=MIN_CHAINS)
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        out = {
            "attractors": [],
            "summary": {"n_attractors": 0, "n_chains_examined": 0},
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (OUTPUT_DIR / "attractors.json").write_text(json.dumps(out, indent=2))
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    chains = [dict(r) for r in conn.execute("SELECT * FROM phase4_chains")]
    n_chains_examined = len(chains)

    # Per-lemma stats:
    #   chain_count        — distinct chains that landed here
    #   source_lemmas      — set of seed lemmas of those chains
    #   example_chain_ids  — small sample for drilldown
    landings: dict[str, dict] = defaultdict(lambda: {
        "display_counts": defaultdict(int),
        "n_chains": 0,
        "n_self_loop": 0,
        "n_length_cap": 0,
        "n_coherence_break": 0,
        "source_lemmas": defaultdict(int),  # seed_lemma -> count of chains
        "source_displays": {},
        "example_chain_ids": [],
    })

    for chain in chains:
        cid = chain["chain_id"]
        seed_display = chain.get("seed_concept") or ""
        seed_lemma = normalize_lemma(seed_display) or ""

        steps = [
            dict(r)
            for r in conn.execute(
                """SELECT step_idx, target_concept, target_lemma, final_answer
                   FROM phase4_steps
                   WHERE chain_id=?
                   ORDER BY step_idx""",
                (cid,),
            )
        ]
        if not steps:
            continue

        lemma, display = _landing(chain, steps)
        if not lemma:
            continue

        bucket = landings[lemma]
        bucket["n_chains"] += 1
        if display:
            bucket["display_counts"][display] += 1
        end_reason = chain.get("end_reason")
        if end_reason == "self_loop":
            bucket["n_self_loop"] += 1
        elif end_reason == "length_cap":
            bucket["n_length_cap"] += 1
        elif end_reason == "coherence_break":
            bucket["n_coherence_break"] += 1
        if seed_lemma:
            bucket["source_lemmas"][seed_lemma] += 1
            if seed_lemma not in bucket["source_displays"]:
                bucket["source_displays"][seed_lemma] = seed_display
        if len(bucket["example_chain_ids"]) < 8:
            bucket["example_chain_ids"].append(cid)

    conn.close()

    attractors = []
    for lemma, b in landings.items():
        if b["n_chains"] < args.min_chains:
            continue
        # Pick the most common display form as the canonical surface.
        display = (
            max(b["display_counts"], key=b["display_counts"].get)
            if b["display_counts"]
            else lemma.title()
        )
        # Sort source lemmas by chain count desc.
        sources = sorted(
            b["source_lemmas"].items(), key=lambda kv: (-kv[1], kv[0])
        )
        attractors.append({
            "lemma": lemma,
            "display": display,
            "n_chains": b["n_chains"],
            "n_self_loop": b["n_self_loop"],
            "n_length_cap": b["n_length_cap"],
            "n_coherence_break": b["n_coherence_break"],
            "sources": [
                {
                    "lemma": sl,
                    "display": b["source_displays"].get(sl, sl.title()),
                    "n_chains": n,
                }
                for sl, n in sources[:12]
            ],
            "example_chain_ids": b["example_chain_ids"],
        })

    attractors.sort(key=lambda a: (-a["n_chains"], a["lemma"]))

    out = {
        "attractors": attractors,
        "summary": {
            "n_attractors": len(attractors),
            "n_chains_examined": n_chains_examined,
            "min_chains": args.min_chains,
            "definition": (
                "A concept that ≥{n} independent chains converged on. For "
                "self_loop chains the landing is the model's last emission "
                "(the would-be revisit). For other endings, the last "
                "target the chain ran on.".format(n=args.min_chains)
            ),
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = OUTPUT_DIR / "attractors.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(
        f"[compute_attractors] wrote {out_path} — "
        f"{len(attractors)} gravity wells from {n_chains_examined} chains",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
