#!/usr/bin/env python3
"""Run a local MLX judge against a calibration set, write per-row verdicts.

This is the per-model worker for the Day-1 calibration gating experiment.
Run it once per candidate judge model. Outputs go to
data/calibration/verdicts_<model_tag>.jsonl, which compare_verdicts.py reads.

DOES NOT TOUCH the production pipeline. Imports the parallel
src.judges.local_mlx_judge module which is not used anywhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make src/ importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.judges.local_mlx_judge import LocalMLXJudge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="MLX model path (HF repo id or local dir)")
    ap.add_argument("--calibration-set", type=Path,
                    default=Path("data/calibration/calibration_set.jsonl"))
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSONL path. Default: "
                         "data/calibration/verdicts_<model_tag>.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N rows (smoke test).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present in the output file by source_id.")
    args = ap.parse_args()

    if not args.calibration_set.exists():
        sys.exit(f"calibration set not found: {args.calibration_set}")

    model_tag = Path(args.model).name
    out_path = args.output or Path(
        f"data/calibration/verdicts_{model_tag}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume-aware: scan existing output for already-judged source_ids
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    sid = rec.get("source_id")
                    if sid is not None:
                        done_ids.add(str(sid))
                except json.JSONDecodeError:
                    pass
        print(f"[resume] found {len(done_ids)} previously-judged rows", flush=True)

    rows: list[dict] = []
    with args.calibration_set.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.resume and str(r.get("source_id", "")) in done_ids:
                continue
            rows.append(r)
    if args.limit:
        rows = rows[: args.limit]

    if not rows:
        print("[run_local_judge] nothing to do.")
        return 0

    print(f"[run_local_judge] model={args.model} rows={len(rows)} "
          f"out={out_path}", flush=True)

    judge = LocalMLXJudge(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=True,
    )

    # Open in append mode so resume + re-runs accumulate cleanly
    started = time.time()
    with out_path.open("a") as out:
        for i, r in enumerate(rows):
            t0 = time.time()
            if r["kind"] == "concept":
                jr = judge.score_detection(r["response"], r["concept"])
            elif r["kind"] == "contrast_pair":
                jr = judge.score_contrast_pair(
                    response=r["response"],
                    axis=r["axis"],
                    description=r.get("description", ""),
                    positive=r.get("positive", []),
                    negative=r.get("negative", []),
                )
            else:
                # Unknown kind — record an error row, don't crash.
                jr = None  # type: ignore
                rec = {
                    "source_id": r.get("source_id"),
                    "stratum": r.get("stratum"),
                    "kind": r["kind"],
                    "model": model_tag,
                    "verdict": None,
                    "error": f"unknown kind: {r['kind']}",
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                continue

            rec = {
                "source_id": r.get("source_id"),
                "stratum": r.get("stratum"),
                "kind": r["kind"],
                "model": model_tag,
                "verdict": {
                    "detected": jr.detected,
                    "identified": jr.identified,
                    "coherent": jr.coherent,
                    "reasoning": jr.reasoning,
                },
                "reference_verdict": r["reference_verdict"],
                "reference_judge": r.get("reference_judge"),
                "elapsed_s": round(time.time() - t0, 2),
            }
            out.write(json.dumps(rec) + "\n")
            out.flush()

            elapsed = time.time() - started
            rate = (i + 1) / elapsed
            print(
                f"[{i+1:4d}/{len(rows)}] {r.get('stratum', '?'):30s} "
                f"d={int(jr.detected)} i={int(jr.identified)} c={int(jr.coherent)} "
                f"({rec['elapsed_s']:.1f}s, {rate:.2f}/s)",
                flush=True,
            )

    print(f"[run_local_judge] done. wrote to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
