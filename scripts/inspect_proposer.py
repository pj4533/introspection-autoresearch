#!/usr/bin/env python3
"""Hand-inspect the local proposer's generated contrast pairs.

Loads Qwen3.6-27B-MLX-8bit, runs ONE batch through novel_contrast (open-ended)
and ONE batch through directed_capraro (causality fault line, with feedback
from the existing DB), and pretty-prints every pair so we can eyeball:

  - Is the axis name abstract enough? (vs "happy-vs-sad" defaults)
  - Do positive examples actually exemplify the positive pole?
  - Are the rationales coherent / non-generic?
  - Does the JSON parse cleanly?

If pairs look lazy or generic, the recommendation is to fall back to
the Opus-distilled-27B alternate from docs/local_pipeline_plan.md.

Run from the repo root:
    .venv/bin/python scripts/inspect_proposer.py
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db import ResultsDB
from src.proposers import LocalMLXProposer
from src.strategies.directed_capraro import generate_candidates as gen_capraro
from src.strategies.novel_contrast import generate_candidates as gen_novel

PROPOSER_PATH = Path.home() / "models/Qwen3.6-27B-MLX-8bit"
DB_PATH = Path("data/results.db")


def _print_spec(spec, idx: int) -> None:
    cp = spec.contrast_pair or {}
    print(f"  [{idx + 1}] axis: {cp.get('axis', spec.concept)!r}")
    print(f"      L={spec.layer_idx}  eff={spec.target_effective:.0f}  "
          f"strategy={spec.strategy}")
    desc = (spec.notes or cp.get("description") or "").strip()
    if desc:
        wrap = textwrap.fill(desc, width=78,
                              initial_indent="      desc: ",
                              subsequent_indent="            ")
        print(wrap)
    rationale = (cp.get("rationale") or "").strip()
    if rationale:
        wrap = textwrap.fill(rationale, width=78,
                              initial_indent="      rat:  ",
                              subsequent_indent="            ")
        print(wrap)
    pos = cp.get("positive", [])
    neg = cp.get("negative", [])
    if pos:
        print(f"      POS:")
        for p in pos[:6]:
            print(f"        + {p}")
    if neg:
        print(f"      NEG:")
        for n in neg[:6]:
            print(f"        - {n}")
    print()


def main() -> int:
    if not PROPOSER_PATH.exists():
        sys.exit(f"proposer not found at {PROPOSER_PATH}")

    print("=" * 72)
    print("PROPOSER INSPECTION — Qwen3.6-27B-MLX-8bit")
    print("=" * 72)
    print(f"loading {PROPOSER_PATH} ...", flush=True)
    t0 = time.time()
    proposer = LocalMLXProposer(model_path=str(PROPOSER_PATH))
    proposer._ensure_loaded()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    db = ResultsDB(DB_PATH)

    # ---------- novel_contrast (open-ended) ----------
    print()
    print("-" * 72)
    print("BATCH 1: novel_contrast  (open-ended, no fault line, n=4)")
    print("-" * 72)
    t0 = time.time()
    nc_specs = gen_novel(n=4, db=db, proposer=proposer, oversample_factor=2)
    print(f"\n[novel_contrast] generated {len(nc_specs)} specs in "
          f"{time.time()-t0:.1f}s\n")
    for i, s in enumerate(nc_specs):
        _print_spec(s, i)

    # ---------- directed_capraro causality (with feedback) ----------
    print("-" * 72)
    print("BATCH 2: directed_capraro  (causality, mode='opus', n=4 — tests")
    print("          feedback-block integration with prior DB results)")
    print("-" * 72)
    t0 = time.time()
    dc_specs = gen_capraro(
        n=4, db=db, fault_line_id="causality", mode="opus",
        proposer=proposer, oversample_factor=2,
    )
    print(f"\n[directed_capraro] generated {len(dc_specs)} specs in "
          f"{time.time()-t0:.1f}s\n")
    for i, s in enumerate(dc_specs):
        _print_spec(s, i)

    print("-" * 72)
    print("HAND-REVIEW CHECKLIST")
    print("-" * 72)
    print("""
  ✓ axes are abstract (not "happy-vs-sad" defaults)
  ✓ positive/negative examples actually exemplify their poles
  ✓ rationales reference prior DB results (Batch 2) or are specific (Batch 1)
  ✓ no JSON parse errors above
  ✓ at least 6 examples per pole

If any of these fail, swap to:
  unsloth/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
