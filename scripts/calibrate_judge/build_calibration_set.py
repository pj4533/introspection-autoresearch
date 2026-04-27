#!/usr/bin/env python3
"""Build a calibration set for the local-judge gating experiment.

Pulls a diverse, stratified sample of (response, concept-or-axis-info,
sonnet-verdict) tuples from data/results.db. Output is JSONL where each row
contains everything a local judge needs to re-grade the response without
having to JOIN the production DB.

The set is built to over-sample the EDGES of the verdict space, which is
where calibration drift will hurt us most:
  - All Phase 1 known true positives (5 cases the literature reproduces).
  - Class 2 hits on contrast_pair candidates (det=1, ident=0): the wall
    we're trying not to break.
  - Class 1 hits (det=1, ident=1): so rare that we take ALL of them.
  - Class 0 nulls on contrast_pair candidates (det=0): tests judge isn't
    pulling false positives from random Gemma babble.
  - Hard incoherent cases (coh=0): tests the judge can recognize gibberish
    without claiming detection.

Read-only on data/results.db. Does not bump PROMPT_TEMPLATE_VERSION.
Does not touch the production judge cache.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

DB = Path("data/results.db")
OUT_DIR = Path("data/calibration")


def _connect() -> sqlite3.Connection:
    if not DB.exists():
        sys.exit(f"ERROR: {DB} not found. Run from the repo root.")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _phase1_known_tps(conn: sqlite3.Connection) -> Iterator[dict]:
    """The 5 Phase 1 detected/identified responses from CLAUDE.md gotcha #5
    and Phase 1 results. We don't know their exact ids, so query for them:
    Phase 1 trials judged by the original Haiku judge with detected=1 and
    identified=1.
    """
    cur = conn.execute("""
        SELECT id, concept, response, detected, identified, coherent,
               judge_model, judge_reasoning
        FROM trials
        WHERE detected=1 AND identified=1
        ORDER BY id
    """)
    for row in cur:
        yield {
            "kind": "concept",
            "source_table": "trials",
            "source_id": row["id"],
            "concept": row["concept"],
            "response": row["response"],
            "reference_verdict": {
                "detected": bool(row["detected"]),
                "identified": bool(row["identified"]),
                "coherent": bool(row["coherent"]),
            },
            "reference_judge": row["judge_model"],
            "reference_reasoning": row["judge_reasoning"],
            "stratum": "phase1_known_tp",
        }


def _phase2_class_n(
    conn: sqlite3.Connection,
    detected: int,
    identified: int,
    coherent_min: int,
    judge_model: str = "claude-sonnet-4-6",
    limit: int = 100,
) -> list[dict]:
    """Sample Phase 2 evaluations matching a class profile.

    Joins evaluations to candidates so we can pull the contrast_pair spec
    needed to re-feed the contrast prompt to a fresh judge.
    """
    cur = conn.execute(f"""
        SELECT e.id, e.candidate_id, e.eval_concept, e.injected, e.alpha,
               e.response, e.detected, e.identified, e.coherent,
               e.judge_model, e.judge_reasoning,
               c.spec_json, c.derivation_method, c.concept AS axis_name,
               c.strategy
        FROM evaluations e
        JOIN candidates c ON c.id = e.candidate_id
        WHERE e.detected = ?
          AND e.identified = ?
          AND e.coherent >= ?
          AND e.judge_model = ?
        ORDER BY RANDOM()
        LIMIT ?
    """, (detected, identified, coherent_min, judge_model, limit))
    rows = []
    for row in cur:
        spec = json.loads(row["spec_json"])
        if row["derivation_method"] == "contrast_pair":
            cp = spec.get("contrast_pair") or {}
            rows.append({
                "kind": "contrast_pair",
                "source_table": "evaluations",
                "source_id": row["id"],
                "candidate_id": row["candidate_id"],
                "strategy": row["strategy"],
                "concept": row["eval_concept"],  # the evaluation probe concept
                "axis": cp.get("axis", row["axis_name"]),
                "description": (
                    spec.get("notes") or cp.get("description") or ""
                ),
                "positive": cp.get("positive") or [],
                "negative": cp.get("negative") or [],
                "response": row["response"],
                "reference_verdict": {
                    "detected": bool(row["detected"]),
                    "identified": bool(row["identified"]),
                    "coherent": bool(row["coherent"]),
                },
                "reference_judge": row["judge_model"],
                "reference_reasoning": row["judge_reasoning"],
                "injected": bool(row["injected"]),
            })
        else:
            # Word-based candidate (random_explore, hillclimb_word, etc.):
            # use the concept-style prompt.
            rows.append({
                "kind": "concept",
                "source_table": "evaluations",
                "source_id": row["id"],
                "candidate_id": row["candidate_id"],
                "strategy": row["strategy"],
                "concept": row["eval_concept"],
                "response": row["response"],
                "reference_verdict": {
                    "detected": bool(row["detected"]),
                    "identified": bool(row["identified"]),
                    "coherent": bool(row["coherent"]),
                },
                "reference_judge": row["judge_model"],
                "reference_reasoning": row["judge_reasoning"],
                "injected": bool(row["injected"]),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=OUT_DIR / "calibration_set.jsonl",
                    help="Output JSONL file")
    ap.add_argument("--n-class-2", type=int, default=60,
                    help="Phase 2 detected-but-not-identified samples (Class 2)")
    ap.add_argument("--n-class-0", type=int, default=60,
                    help="Phase 2 not-detected samples (Class 0)")
    ap.add_argument("--n-incoherent", type=int, default=30,
                    help="Hard cases where coherent=0 (tests gibberish recognition)")
    ap.add_argument("--include-class-1", action="store_true", default=True,
                    help="Include all Class 1 (det+ident) Phase 2 hits")
    ap.add_argument("--include-phase1-tps", action="store_true", default=True,
                    help="Include Phase 1 known true positives")
    ap.add_argument("--judge-model", default="claude-sonnet-4-6",
                    help="Reference judge to draw verdicts from")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    conn = _connect()

    if args.include_phase1_tps:
        rows.extend(_phase1_known_tps(conn))

    if args.include_class_1:
        # Class 1 is so rare we take EVERYTHING — across both Phase 2 judge
        # models for completeness. Sonnet is the reference, but a Haiku-era
        # Class 1 hit is still a useful regression target.
        for jm in (args.judge_model, "claude-haiku-4-5-20251001"):
            class_1 = _phase2_class_n(
                conn,
                detected=1, identified=1, coherent_min=1,
                judge_model=jm, limit=999,
            )
            for r in class_1:
                r["stratum"] = f"phase2_class_1[{jm}]"
            rows.extend(class_1)

    class_2 = _phase2_class_n(
        conn, detected=1, identified=0, coherent_min=1,
        judge_model=args.judge_model, limit=args.n_class_2,
    )
    for r in class_2:
        r["stratum"] = "phase2_class_2"
    rows.extend(class_2)

    class_0 = _phase2_class_n(
        conn, detected=0, identified=0, coherent_min=1,
        judge_model=args.judge_model, limit=args.n_class_0,
    )
    for r in class_0:
        r["stratum"] = "phase2_class_0"
    rows.extend(class_0)

    incoherent = _phase2_class_n(
        conn, detected=0, identified=0, coherent_min=0,
        judge_model=args.judge_model, limit=args.n_incoherent,
    )
    # Filter to ONLY actually-incoherent ones (coh_min=0 includes both)
    incoherent = [r for r in incoherent
                  if not r["reference_verdict"]["coherent"]]
    for r in incoherent:
        r["stratum"] = "phase2_incoherent"
    rows.extend(incoherent[:args.n_incoherent])

    # Stable order: stratum, then source_id
    rows.sort(key=lambda r: (r.get("stratum", ""), str(r.get("source_id", ""))))

    with args.output.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(rows)} rows to {args.output}")
    print("by stratum:")
    by = {}
    for r in rows:
        s = r.get("stratum", "?")
        by[s] = by.get(s, 0) + 1
    for s, n in sorted(by.items()):
        print(f"  {s:30s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
