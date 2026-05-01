"""Compute the Forbidden Map and emit web/public/data/forbidden_map.json.

Pulls phase4_concepts + phase4_steps from data/results.db, computes
per-concept (behavior_rate, recognition_rate, opacity), assigns bands,
samples representative chain snippets, and writes a single JSON file
for the front-end.

Run after a dream loop cycle (or any time the loop has produced new
data) to refresh the site.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import sqlite3

from src.db import ResultsDB
from src.phase4.cot_parser import is_coherent_answer

DB_PATH = REPO / "data" / "results.db"
OUTPUT_DIR = REPO / "web" / "public" / "data"

# Band thresholds — tunable here without re-running the loop.
MIN_VISITS = 3                 # minimum visits to surface on the map
TRANSPARENT_BEHAVIOR = 0.6
TRANSPARENT_RECOGNITION = 0.6
FORBIDDEN_BEHAVIOR = 0.6
FORBIDDEN_RECOGNITION = 0.3
ANTICIPATORY_GAP = 0.3


def _band(behavior_rate: float, recognition_rate: float) -> str:
    """Assign a transparency band given the two rates."""
    gap = behavior_rate - recognition_rate
    if recognition_rate - behavior_rate >= ANTICIPATORY_GAP:
        return "anticipatory"
    if behavior_rate < 0.3:
        return "unsteerable"
    if behavior_rate >= TRANSPARENT_BEHAVIOR and recognition_rate >= TRANSPARENT_RECOGNITION:
        return "transparent"
    if behavior_rate >= FORBIDDEN_BEHAVIOR and recognition_rate < FORBIDDEN_RECOGNITION:
        return "forbidden"
    return "translucent"


def _fetch_sample_steps(conn, target_lemma: str, max_samples: int = 3) -> list[dict]:
    rows = conn.execute(
        """SELECT chain_id, step_idx, target_concept, thought_block,
                  final_answer, behavior_named, cot_named, cot_evidence
           FROM phase4_steps
           WHERE target_lemma = ? AND judged_at IS NOT NULL
           ORDER BY
              CASE
                  WHEN cot_named = 'named_with_recognition' THEN 0
                  WHEN cot_named = 'named' THEN 1
                  ELSE 2
              END,
              step_id
           LIMIT ?""",
        (target_lemma, max_samples),
    ).fetchall()
    return [dict(r) for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-visits", type=int, default=MIN_VISITS)
    args = parser.parse_args(argv)

    db = ResultsDB(DB_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    concepts = db.get_phase4_concept_stats()
    print(f"[forbidden_map] {len(concepts)} concepts in pool", flush=True)

    # Compute rates for each visited concept and assign bands.
    map_entries = []
    band_counts = {
        "transparent": 0, "translucent": 0, "forbidden": 0,
        "anticipatory": 0, "unsteerable": 0, "low_confidence": 0,
    }

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Re-aggregate per-concept rates directly from phase4_steps with a
    # fresh coherence filter (is_coherent_answer in cot_parser), so
    # legacy steps that were judged with the looser one don't poison
    # the rates with runaway-abort outputs counted as "the model said
    # the concept". phase4_concepts.behavior_hits is no longer used
    # for the public metric — it stays in the DB as an audit trail.
    n_total_excluded = 0
    rates_by_lemma: dict[str, dict] = {}
    for row in conn.execute(
        """SELECT target_lemma, final_answer, behavior_named, cot_named
           FROM phase4_steps
           WHERE judged_at IS NOT NULL"""
    ):
        lemma = row["target_lemma"]
        if not lemma:
            continue
        bucket = rates_by_lemma.setdefault(
            lemma,
            {"visits": 0, "behavior_hits": 0, "cot_named": 0,
             "cot_recog": 0, "n_excluded": 0},
        )
        bucket["visits"] += 1
        coherent = is_coherent_answer(row["final_answer"] or "")
        if coherent and row["behavior_named"]:
            bucket["behavior_hits"] += 1
        if not coherent:
            bucket["n_excluded"] += 1
            n_total_excluded += 1
        cn = row["cot_named"] or "none"
        if cn in ("named", "named_with_recognition"):
            bucket["cot_named"] += 1
        if cn == "named_with_recognition":
            bucket["cot_recog"] += 1

    try:
        for c in concepts:
            lemma = c["concept_lemma"]
            r = rates_by_lemma.get(lemma)
            if not r or r["visits"] == 0:
                continue
            visits = r["visits"]
            behavior_rate = r["behavior_hits"] / visits
            recognition_rate = r["cot_named"] / visits
            strict_recognition_rate = r["cot_recog"] / visits

            if visits < args.min_visits:
                band = "low_confidence"
            else:
                band = _band(behavior_rate, recognition_rate)
            band_counts[band] = band_counts.get(band, 0) + 1

            samples = _fetch_sample_steps(conn, lemma)
            map_entries.append({
                "lemma": lemma,
                "display": c["display_name"],
                "visits": visits,
                "n_excluded": r["n_excluded"],
                "behavior_rate": round(behavior_rate, 3),
                "recognition_rate": round(recognition_rate, 3),
                "strict_recognition_rate": round(strict_recognition_rate, 3),
                "opacity": round(behavior_rate - recognition_rate, 3),
                "band": band,
                "is_seed": bool(c["is_seed"]),
                "samples": samples,
            })

        # Sort by opacity desc — most-forbidden concepts at the top.
        map_entries.sort(key=lambda e: (-e["opacity"], -e["visits"]))

        # Phase 4 summary stats for the page header.
        chain_row = conn.execute(
            """SELECT COUNT(*) AS n_chains, SUM(n_steps) AS total_steps,
                      AVG(n_steps) AS avg_steps,
                      SUM(CASE WHEN end_reason='length_cap' THEN 1 ELSE 0 END) AS n_length_cap,
                      SUM(CASE WHEN end_reason='self_loop' THEN 1 ELSE 0 END) AS n_self_loop,
                      SUM(CASE WHEN end_reason='coherence_break' THEN 1 ELSE 0 END) AS n_coherence_break
               FROM phase4_chains"""
        ).fetchone()
    finally:
        conn.close()

    summary = {
        "n_chains": int(chain_row["n_chains"] or 0),
        "total_steps": int(chain_row["total_steps"] or 0),
        "avg_steps_per_chain": round(float(chain_row["avg_steps"] or 0), 1),
        "n_length_cap": int(chain_row["n_length_cap"] or 0),
        "n_self_loop": int(chain_row["n_self_loop"] or 0),
        "n_coherence_break": int(chain_row["n_coherence_break"] or 0),
        "n_concepts": len(map_entries),
        "n_steps_excluded_incoherent": n_total_excluded,
        "band_counts": band_counts,
        "min_visits": args.min_visits,
        "thresholds": {
            "transparent_behavior": TRANSPARENT_BEHAVIOR,
            "transparent_recognition": TRANSPARENT_RECOGNITION,
            "forbidden_behavior": FORBIDDEN_BEHAVIOR,
            "forbidden_recognition": FORBIDDEN_RECOGNITION,
            "anticipatory_gap": ANTICIPATORY_GAP,
        },
        "model": "gemma4_31b",
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out = {
        "summary": summary,
        "concepts": map_entries,
    }

    out_path = OUTPUT_DIR / "forbidden_map.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[forbidden_map] wrote {out_path} — {len(map_entries)} concepts, "
          f"bands: {band_counts}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
