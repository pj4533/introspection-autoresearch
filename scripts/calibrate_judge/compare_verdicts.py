#!/usr/bin/env python3
"""Score one or more local-judge verdict files against the reference (Sonnet)
verdicts in the calibration set. Produces a per-model agreement report and
flags disagreements case-by-case for human inspection.

Usage:
  python scripts/calibrate_judge/compare_verdicts.py \\
      --calibration-set data/calibration/calibration_set.jsonl \\
      --results data/calibration/verdicts_Qwen3.6-35B-A3B-8bit.jsonl \\
      --results data/calibration/verdicts_GLM-4.7-Flash-8bit.jsonl

Emits a markdown report to data/calibration/report.md plus a per-disagreement
JSONL at data/calibration/disagreements_<model>.jsonl.

The acceptance criteria from docs/local_pipeline_plan.md:
  - Class 2 (det only) agreement >= 95%
  - Class 0 (null) agreement >= 90%
  - Phase 1 known true positives: exact-match all three flags
  - FPR-class agreement (det=0 on controls) >= 99%
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("data/calibration")


def _load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _verdict_class(v: dict) -> str:
    """Map a verdict triple to a class label.

    Class 1: det+ident (full identification, the rare wins).
    Class 2: det only (the wall — most frequent positive).
    Class 0: nothing detected.
    Inco:    coherence failure.
    """
    if not v["coherent"]:
        return "incoherent"
    if v["detected"] and v["identified"]:
        return "class1"
    if v["detected"]:
        return "class2"
    return "class0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-set", type=Path,
                    default=OUT_DIR / "calibration_set.jsonl")
    ap.add_argument("--results", type=Path, action="append", required=True,
                    help="Verdict JSONL (repeat for multiple models)")
    ap.add_argument("--report", type=Path, default=OUT_DIR / "report.md")
    args = ap.parse_args()

    calibration = {str(r["source_id"]): r for r in _load_jsonl(args.calibration_set)}

    per_model_disagreements: dict[str, list[dict]] = {}
    summaries: list[dict] = []

    for results_path in args.results:
        verdicts = _load_jsonl(results_path)
        # Identify model tag from first row, else from filename
        if verdicts:
            model_tag = verdicts[0].get("model", results_path.stem)
        else:
            model_tag = results_path.stem

        confusion = defaultdict(lambda: defaultdict(int))  # ref_class -> local_class -> count
        per_stratum_total = defaultdict(int)
        per_stratum_agree = defaultdict(int)
        flag_match = defaultdict(lambda: {"match": 0, "total": 0})  # detected/identified/coherent
        disagreements: list[dict] = []
        errors = 0

        for v in verdicts:
            sid = str(v.get("source_id"))
            ref_row = calibration.get(sid)
            if ref_row is None:
                continue  # row not in calibration set (e.g. stale verdicts file)
            if v.get("error") or v.get("verdict") is None:
                errors += 1
                continue

            local = v["verdict"]
            ref = ref_row["reference_verdict"]
            stratum = ref_row.get("stratum", "unknown")

            # Per-flag agreement
            for flag in ("detected", "identified", "coherent"):
                flag_match[flag]["total"] += 1
                if bool(local[flag]) == bool(ref[flag]):
                    flag_match[flag]["match"] += 1

            # Class confusion
            local_class = _verdict_class(local)
            ref_class = _verdict_class(ref)
            confusion[ref_class][local_class] += 1
            per_stratum_total[stratum] += 1
            if local_class == ref_class:
                per_stratum_agree[stratum] += 1
            else:
                disagreements.append({
                    "source_id": sid,
                    "stratum": stratum,
                    "ref_class": ref_class,
                    "local_class": local_class,
                    "ref": ref,
                    "local": local,
                    "concept_or_axis": ref_row.get("axis") or ref_row.get("concept"),
                    "response": ref_row["response"][:300],
                    "ref_reasoning": (ref_row.get("reference_reasoning") or "")[:200],
                    "local_reasoning": local.get("reasoning", "")[:200],
                })

        per_model_disagreements[model_tag] = disagreements

        # Save disagreements file
        disagree_path = OUT_DIR / f"disagreements_{model_tag}.jsonl"
        with disagree_path.open("w") as f:
            for d in disagreements:
                f.write(json.dumps(d) + "\n")

        total = sum(per_stratum_total.values())
        total_agree = sum(per_stratum_agree.values())

        # Acceptance criteria checks
        class2_agree = per_stratum_agree.get("phase2_class_2", 0)
        class2_total = per_stratum_total.get("phase2_class_2", 0)
        class0_agree = per_stratum_agree.get("phase2_class_0", 0)
        class0_total = per_stratum_total.get("phase2_class_0", 0)
        phase1_agree = per_stratum_agree.get("phase1_known_tp", 0)
        phase1_total = per_stratum_total.get("phase1_known_tp", 0)

        def pct(num, den):
            return 100.0 * num / den if den else 0.0

        criteria = {
            "class2_>=95%":
                pct(class2_agree, class2_total) >= 95.0 and class2_total > 0,
            "class0_>=90%":
                pct(class0_agree, class0_total) >= 90.0 and class0_total > 0,
            "phase1_exact_match":
                phase1_agree == phase1_total and phase1_total > 0,
        }
        passed = all(criteria.values())

        summaries.append({
            "model": model_tag,
            "results_path": str(results_path),
            "total": total,
            "total_agree": total_agree,
            "errors": errors,
            "per_stratum_total": dict(per_stratum_total),
            "per_stratum_agree": dict(per_stratum_agree),
            "flag_match": {k: dict(v) for k, v in flag_match.items()},
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "criteria": criteria,
            "passed": passed,
            "disagreements_path": str(disagree_path),
        })

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    lines: list[str] = ["# Local-judge calibration report", ""]
    lines.append(f"Calibration set: `{args.calibration_set}`")
    lines.append(f"Models compared: {len(summaries)}")
    lines.append("")
    lines.append("## Acceptance criteria")
    lines.append("")
    lines.append(
        "| model | total | agree | overall % | class2 % | class0 % | "
        "phase1 % | passed |"
    )
    lines.append(
        "|-------|------:|------:|----------:|---------:|---------:|"
        "---------:|:------:|"
    )
    for s in summaries:
        def stratum_pct(stratum):
            t = s["per_stratum_total"].get(stratum, 0)
            a = s["per_stratum_agree"].get(stratum, 0)
            return f"{100*a/t:.1f}" if t else "n/a"

        lines.append(
            f"| `{s['model']}` "
            f"| {s['total']} "
            f"| {s['total_agree']} "
            f"| {100*s['total_agree']/s['total']:.1f} "
            f"| {stratum_pct('phase2_class_2')} "
            f"| {stratum_pct('phase2_class_0')} "
            f"| {stratum_pct('phase1_known_tp')} "
            f"| {'PASS' if s['passed'] else 'FAIL'} |"
        )
    lines.append("")

    for s in summaries:
        lines.append(f"## `{s['model']}`")
        lines.append("")
        lines.append(f"- errors during inference: **{s['errors']}**")
        lines.append("- per-flag agreement:")
        for flag, fm in s["flag_match"].items():
            t = fm["total"]
            m = fm["match"]
            lines.append(
                f"    - `{flag}`: {m}/{t} = {100*m/t:.1f}%" if t else f"    - `{flag}`: n/a"
            )
        lines.append("")
        lines.append("- per-stratum agreement:")
        for stratum, total in sorted(s["per_stratum_total"].items()):
            agree = s["per_stratum_agree"].get(stratum, 0)
            lines.append(
                f"    - `{stratum}`: {agree}/{total} = "
                f"{100*agree/total:.1f}%" if total else f"    - `{stratum}`: n/a"
            )
        lines.append("")
        lines.append("- class confusion (rows = ref, cols = local):")
        all_classes = sorted(set(s["confusion"].keys()) |
                              {c for v in s["confusion"].values() for c in v.keys()})
        if all_classes:
            header = " | ".join([""] + [f"**{c}**" for c in all_classes])
            sep = "|".join(["---"] * (len(all_classes) + 1))
            lines.append(f"| {header} |")
            lines.append(f"|{sep}|")
            for ref_c in all_classes:
                row = [f"**{ref_c}**"]
                for local_c in all_classes:
                    row.append(str(s["confusion"].get(ref_c, {}).get(local_c, 0)))
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(f"- disagreements file: `{s['disagreements_path']}`")
        lines.append("")
        lines.append("- acceptance criteria:")
        for k, v in s["criteria"].items():
            lines.append(f"    - `{k}`: {'PASS' if v else 'FAIL'}")
        lines.append("")
        lines.append("---")
        lines.append("")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as f:
        f.write("\n".join(lines))

    print(f"wrote report to {args.report}")
    for s in summaries:
        print(
            f"  {s['model']:40s} overall={100*s['total_agree']/max(s['total'],1):.1f}% "
            f"  PASSED={s['passed']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
