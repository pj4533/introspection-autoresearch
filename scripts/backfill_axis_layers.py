"""Backfill missing (axis, layer) combinations into queue/pending.

For each unique invented-axis contrast pair already evaluated, enqueue new
candidates for any missing layer in {30, 33, 36, 40}. Uniqueness is defined
by the exact contrast pair content (positive + negative example sentences)
— if Claude produced two candidates with the same axis name but different
examples, they're treated as different axes because they yield different
directions.

Also optionally clears any L33-only candidates already pending (since the
previous pin-to-L33 wasted slots we'd rather spend on other layers).

Usage:
    python scripts/backfill_axis_layers.py --since '2026-04-17 22:00:00'
    python scripts/backfill_axis_layers.py --clear-pending-l33 --since ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "results.db"
QUEUE_PENDING = REPO / "queue" / "pending"

TARGET_LAYERS = [30, 33, 36, 40]


def pair_key(pair: dict) -> str:
    """Hash the contrast-pair content (axis + examples) so same-named axes
    with different examples are treated as different directions."""
    payload = json.dumps(
        {
            "axis": pair.get("axis"),
            "positive": pair.get("positive", []),
            "negative": pair.get("negative", []),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-04-17 22:00:00",
                    help="UTC timestamp cutoff; only backfill axes evaluated after this")
    ap.add_argument("--clear-pending-l33", action="store_true",
                    help="Delete L33-only pending candidates before backfilling")
    ap.add_argument("--target-effective", type=float, default=18000.0,
                    help="Effective strength to use for backfilled candidates")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.clear_pending_l33:
        dropped = 0
        for fp in QUEUE_PENDING.iterdir():
            if fp.name == ".gitkeep" or not fp.suffix == ".json":
                continue
            try:
                spec = json.loads(fp.read_text())
            except Exception:
                continue
            if spec.get("layer_idx") == 33 and spec.get("derivation_method") == "contrast_pair":
                if not args.dry_run:
                    fp.unlink()
                dropped += 1
        print(f"cleared {dropped} L33-only pending contrast_pair candidates")

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    # All tonight's invented-axis candidates — used to count "tested layers"
    # per pair so we know what to backfill.
    all_rows = conn.execute("""
        SELECT c.id, c.concept, c.layer_idx, c.target_effective,
               c.spec_json, c.evaluated_at, f.score
        FROM candidates c
        LEFT JOIN fitness_scores f ON f.candidate_id = c.id
        WHERE c.derivation_method='contrast_pair'
          AND c.status='done'
          AND c.evaluated_at >= ?
    """, (args.since,)).fetchall()

    # Pairs we want to backfill: ones where AT LEAST ONE layer scored > 0.
    # Axes that scored 0 at the layer they were tested probably won't hit
    # at other layers either; not worth the compute.
    scoring_keys: set[str] = set()
    for r in all_rows:
        try:
            spec = json.loads(r["spec_json"])
            pair = spec.get("contrast_pair")
            if pair and (r["score"] or 0) > 0:
                scoring_keys.add(pair_key(pair))
        except Exception:
            pass

    print(f"Found {len(all_rows)} invented-axis candidates evaluated since {args.since}")
    print(f"  of those, {len(scoring_keys)} unique axes had score > 0 at some layer")

    rows = all_rows

    # Group by pair content; collect tested layers — but only for axes that
    # scored > 0 at some layer.
    by_pair: dict[str, dict] = {}
    for r in rows:
        try:
            spec = json.loads(r["spec_json"])
        except Exception:
            continue
        pair = spec.get("contrast_pair")
        if not pair:
            continue
        k = pair_key(pair)
        if k not in scoring_keys:
            continue
        entry = by_pair.setdefault(k, {
            "pair": pair,
            "notes": spec.get("notes", ""),
            "tested_layers": set(),
            "target_effective": spec.get("target_effective", args.target_effective),
        })
        entry["tested_layers"].add(r["layer_idx"])

    print(f"{len(by_pair)} scoring axes will be backfilled to all 4 layers")

    # Figure out what's pending (to avoid duplicates)
    pending_keys: set[tuple[str, int]] = set()
    for fp in QUEUE_PENDING.iterdir():
        if fp.name == ".gitkeep" or not fp.suffix == ".json":
            continue
        try:
            spec = json.loads(fp.read_text())
        except Exception:
            continue
        if spec.get("derivation_method") != "contrast_pair":
            continue
        pair = spec.get("contrast_pair")
        if pair:
            pending_keys.add((pair_key(pair), spec.get("layer_idx")))

    # Generate backfill candidates
    written = 0
    skipped = 0
    for k, entry in by_pair.items():
        pair = entry["pair"]
        missing = [l for l in TARGET_LAYERS if l not in entry["tested_layers"]]
        for layer in missing:
            if (k, layer) in pending_keys:
                skipped += 1
                continue
            cand_id = f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            spec = {
                "id": cand_id,
                "strategy": "novel_contrast_backfill",
                "concept": pair["axis"],
                "layer_idx": layer,
                "target_effective": entry["target_effective"],
                "derivation_method": "contrast_pair",
                "baseline_n": 0,
                "notes": entry.get("notes", ""),
                "contrast_pair": {
                    "axis": pair["axis"],
                    "positive": pair.get("positive", []),
                    "negative": pair.get("negative", []),
                },
            }
            fp = QUEUE_PENDING / f"{cand_id}.json"
            if not args.dry_run:
                fp.write_text(json.dumps(spec, indent=2))
            written += 1
            time.sleep(0.001)  # ensure unique timestamps

    print(f"backfill: {written} candidates {'would be' if args.dry_run else ''} written, {skipped} already pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
