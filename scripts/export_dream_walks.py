"""Export sampled dream walks for the Phase 4 site Dream Walk Viewer.

Selects up to ~50 representative chains from the phase4_chains table:
  - All chains starting from the Codex-suppressed creatures (full set)
  - Top chains by max-opacity-of-any-visited-concept (the "interesting"
    chains that touched something forbidden)
  - A random sample of length-capped chains for breadth

Writes web/public/data/dream_walks.json with full per-step content
(thought_block + final_answer + judge verdicts) so the viewer can
render the sequence directly.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


DB_PATH = REPO / "data" / "results.db"
OUTPUT_DIR = REPO / "web" / "public" / "data"

CODEX_CREATURE_LEMMAS = ["goblin", "gremlin", "raccoon", "troll", "ogre", "pigeon"]


def _fetch_chain(conn, chain_id: str) -> dict:
    chain = dict(conn.execute(
        "SELECT * FROM phase4_chains WHERE chain_id=?", (chain_id,)
    ).fetchone())
    steps = [
        dict(r) for r in conn.execute(
            """SELECT step_idx, target_concept, target_lemma,
                      alpha, direction_norm,
                      thought_block, final_answer, parse_failure,
                      behavior_named, cot_named, cot_evidence
               FROM phase4_steps
               WHERE chain_id=? AND judged_at IS NOT NULL
               ORDER BY step_idx""",
            (chain_id,),
        ).fetchall()
    ]
    chain["steps"] = steps
    return chain


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-chains", type=int, default=50)
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        print(f"[export_dream_walks] no DB at {DB_PATH}", flush=True)
        out = {"chains": [], "summary": {"n_chains": 0}, "last_updated":
               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        (OUTPUT_DIR / "dream_walks.json").write_text(json.dumps(out, indent=2))
        return 0

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    selected_ids: list[str] = []
    seen: set[str] = set()

    # 1. All Codex-creature-seeded chains.
    for lemma in CODEX_CREATURE_LEMMAS:
        rows = conn.execute(
            """SELECT chain_id FROM phase4_chains
               WHERE LOWER(seed_concept) LIKE ?
               ORDER BY started_at""",
            (f"{lemma}%",),
        ).fetchall()
        for r in rows:
            cid = r["chain_id"]
            if cid not in seen:
                seen.add(cid)
                selected_ids.append(cid)

    # 2. Chains that visited a high-opacity concept. We compute a
    # per-step "opacity_signal" = behavior_named=1 AND cot_named='none'
    # and sort chains by max opacity_signal count.
    high_op_rows = conn.execute(
        """SELECT chain_id, COUNT(*) AS n_signal FROM phase4_steps
           WHERE judged_at IS NOT NULL
             AND behavior_named=1 AND cot_named='none'
           GROUP BY chain_id
           ORDER BY n_signal DESC, chain_id
           LIMIT ?""",
        (args.max_chains,),
    ).fetchall()
    for r in high_op_rows:
        cid = r["chain_id"]
        if cid not in seen and len(selected_ids) < args.max_chains:
            seen.add(cid)
            selected_ids.append(cid)

    # 3. Length-capped chains (the long ones — best demos for the viewer).
    long_rows = conn.execute(
        """SELECT chain_id FROM phase4_chains
           WHERE end_reason='length_cap'
           ORDER BY started_at
           LIMIT ?""",
        (args.max_chains,),
    ).fetchall()
    for r in long_rows:
        cid = r["chain_id"]
        if cid not in seen and len(selected_ids) < args.max_chains:
            seen.add(cid)
            selected_ids.append(cid)

    # Cap.
    selected_ids = selected_ids[:args.max_chains]

    chains = [_fetch_chain(conn, cid) for cid in selected_ids]

    # All chains summary (independent of selection).
    total_row = conn.execute(
        "SELECT COUNT(*) AS n FROM phase4_chains"
    ).fetchone()
    conn.close()

    out = {
        "chains": chains,
        "summary": {
            "n_chains_selected": len(chains),
            "n_chains_total": int(total_row["n"] or 0),
            "selection_priority": (
                "1) Codex-creature seeds, "
                "2) chains visiting high-opacity (forbidden) concepts, "
                "3) full-length chains."
            ),
        },
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = OUTPUT_DIR / "dream_walks.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[export_dream_walks] wrote {out_path} — "
          f"{len(chains)} chains of {int(total_row['n'] or 0)} total",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
