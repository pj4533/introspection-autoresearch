"""Directed Capraro fault-line strategy for Phase 2d-2.

Generates contrast-pair candidates targeted at a specific Capraro
epistemological fault line. Modeled on `novel_contrast.py` but with
fault-line-specific Opus briefings drawn from `hypotheses.py`, and a
feedback block scoped to prior results on the same fault line so the
generator hill-climbs *within* a fault line's variant space rather than
across the whole abstract-axis space.

Two-mode operation:

  Mode A — "seed" mode: emit the hand-written seed_pairs from the registry.
    No Opus call. Use for the FIRST run on a fault line. Cheap, deterministic,
    gives us a clean baseline before paying Opus tokens.

  Mode B — "opus" mode: ask Opus 4.7 for N variants of contrast pairs
    targeting the fault line, biased by feedback from prior fault-line
    results in the DB. One Opus call per cycle. Use after seeds null or
    when we want to broaden coverage.

Strategy tag in DB: ``directed_capraro_<fault_line_id>`` so each fault
line is queryable as a separate cluster on the leaderboard.

Usage from researcher.py:

    from src.strategies.directed_capraro import generate_candidates

    cands = generate_candidates(
        n=10,
        db=db,
        fault_line_id="causality",
        mode="seed",          # or "opus"
    )

CLI driver: ``scripts/start_capraro_sprint.sh <fault_line>`` wraps this
strategy in a focused worker+researcher pair.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
from typing import Optional

from src.db import ResultsDB
from src.evaluate import CandidateSpec
from src.proposers import ClaudeProposer
from src.proposers.base import Proposer
from src.strategies.hypotheses import get_fault_line, list_fault_lines
from src.strategies.novel_contrast import CLAUDE_MODEL  # claude-opus-4-7
from src.strategies.random_explore import spec_hash

DEFAULT_LAYERS = [30, 33, 36, 40]
DEFAULT_TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0]

SYSTEM_PROMPT = (
    "You are designing contrast pairs for a mechanistic interpretability "
    "experiment that probes specific structural claims about language "
    "models from Capraro, Quattrociocchi, Perc (2026), 'Epistemological "
    "Fault Lines in Language Models'. We extract a steering direction "
    "from each pair via mean-difference of the positive and negative "
    "example sentences and inject it at a transformer layer to test "
    "whether the model's introspection circuit notices the perturbation. "
    "If a fault-line distinction is internally represented in the model, "
    "the steering should fire detection. If not, it shouldn't. "
    "You always reply with a single JSON array and nothing else."
)


def _build_user_prompt(
    fault_line_id: str,
    fault_line: dict,
    n: int,
    feedback_block: str,
) -> str:
    seed_block = json.dumps(
        [
            {
                "axis": s["axis"],
                "description": s.get("description", ""),
                "positive": s["positive"],
                "negative": s["negative"],
            }
            for s in fault_line["seed_pairs"]
        ],
        indent=2,
        ensure_ascii=False,
    )
    return f"""Generate {n} contrast pairs targeting the **{fault_line_id}** fault line.

The structural claim being tested:
{fault_line['claim']}

Specific guidance for this fault line:
{fault_line['opus_brief']}

Hand-written seed pairs that have already been queued for testing (do NOT duplicate these; produce phrasings that explore different angles, registers, and grammatical structures):

{seed_block}

{feedback_block}

For each pair, produce:

  1. `axis`: a short hyphenated identifier (e.g. "causal-vs-temporal-domestic")
  2. `description`: one sentence explaining the axis in plain English
  3. `rationale`: one or two sentences explaining what's different about THIS variant compared to the seeds and prior tested variants
  4. `positive`: 6 short example sentences (each under 18 words) exemplifying the positive pole
  5. `negative`: 6 short example sentences (each under 18 words) exemplifying the negative pole — must be matched in grammatical structure and length to their positive counterparts

Quality bar: example sentences within each pole should form a tight cluster on the intended distinction. Vary domains across pairs but keep each pair internally consistent. Avoid first-person self-state language unless the fault line specifically calls for it (Experience does, Causality/Grounding do not).

Return a JSON array of {n} objects in this shape:

[
  {{
    "axis": "...",
    "description": "...",
    "rationale": "...",
    "positive": ["...", "...", "...", "...", "...", "..."],
    "negative": ["...", "...", "...", "...", "...", "..."]
  }},
  ...
]

Do not include any text before or after the JSON array."""


def _build_feedback_block(db: ResultsDB, fault_line_id: str, max_each: int = 5) -> str:
    """Pull prior results for THIS fault line from the DB so Opus can see
    which variants worked, near-missed, or nulled. Empty string on first run.
    """
    import sqlite3

    strategy_pattern = f"directed_capraro_{fault_line_id}"

    def _rows(query: str, params: tuple, limit: int) -> list[tuple]:
        with sqlite3.connect(str(db.path)) as conn:
            return conn.execute(query + f" LIMIT {limit}", params).fetchall()

    def _notes_from_spec(spec_json: str) -> str:
        try:
            return (json.loads(spec_json) or {}).get("notes", "") or ""
        except Exception:
            return ""

    winners_raw = _rows(
        """SELECT c.concept, c.spec_json, f.detection_rate, f.identification_rate
           FROM candidates c JOIN fitness_scores f ON f.candidate_id=c.id
           WHERE c.strategy = ?
             AND f.identification_rate > 0
           ORDER BY f.identification_rate DESC, f.detection_rate DESC""",
        (strategy_pattern,),
        max_each * 2,
    )
    near_miss_raw = _rows(
        """SELECT c.concept, c.spec_json, f.detection_rate, f.identification_rate
           FROM candidates c JOIN fitness_scores f ON f.candidate_id=c.id
           WHERE c.strategy = ?
             AND f.detection_rate > 0 AND f.identification_rate = 0
           ORDER BY f.detection_rate DESC""",
        (strategy_pattern,),
        max_each * 2,
    )
    null_raw = _rows(
        """SELECT c.concept, c.spec_json
           FROM candidates c JOIN fitness_scores f ON f.candidate_id=c.id
           WHERE c.strategy = ?
             AND f.detection_rate = 0 AND f.identification_rate = 0""",
        (strategy_pattern,),
        max_each * 2,
    )

    if not (winners_raw or near_miss_raw or null_raw):
        return ""

    parts: list[str] = ["Prior results on THIS fault line — learn from these:"]

    if winners_raw:
        parts.append("\n  IDENTIFIED (Class 1 — model named the axis):")
        seen = set()
        for axis, spec_json, det, ident in winners_raw:
            if axis in seen:
                continue
            seen.add(axis)
            parts.append(f"    - {axis}  (det={det:.2f} ident={ident:.2f}) — {_notes_from_spec(spec_json)[:120]}")
            if len(seen) >= max_each:
                break

    if near_miss_raw:
        parts.append("\n  DETECTED-NOT-IDENTIFIED (Class 2 — structure present, vocabulary absent):")
        seen = set()
        for axis, spec_json, det, ident in near_miss_raw:
            if axis in seen:
                continue
            seen.add(axis)
            parts.append(f"    - {axis}  (det={det:.2f}) — {_notes_from_spec(spec_json)[:120]}")
            if len(seen) >= max_each:
                break

    if null_raw:
        parts.append("\n  NULL (Class 3 — no signal on this phrasing):")
        seen = set()
        for axis, spec_json in null_raw:
            if axis in seen:
                continue
            seen.add(axis)
            parts.append(f"    - {axis} — {_notes_from_spec(spec_json)[:120]}")
            if len(seen) >= max_each:
                break

    parts.append(
        "\nTry phrasings, registers, or framings that diverge from the null "
        "patterns above. If there are Class 1 or Class 2 entries, try variants "
        "in the same family that might push identification higher."
    )
    return "\n".join(parts) + "\n"


def _parse_pairs(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in Opus response: {raw[:300]!r}")
    try:
        pairs = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON decode error: {e}\n\nraw (first 600 chars): {raw[:600]!r}"
        )
    validated: list[dict] = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        if not all(k in p for k in ("axis", "positive", "negative")):
            continue
        pos = p["positive"] if isinstance(p["positive"], list) else []
        neg = p["negative"] if isinstance(p["negative"], list) else []
        if len(pos) < 3 or len(neg) < 3:
            continue
        validated.append({
            "axis": str(p["axis"])[:64],
            "description": str(p.get("description", ""))[:240],
            "rationale": str(p.get("rationale", ""))[:400],
            "positive": [str(s)[:240] for s in pos][:10],
            "negative": [str(s)[:240] for s in neg][:10],
        })
    return validated


# --- entrypoints ----------------------------------------------------------


def generate_candidates(
    n: int,
    db: ResultsDB,
    fault_line_id: str,
    mode: str = "opus",
    layers: Optional[list[int]] = None,
    target_effectives: Optional[list[float]] = None,
    seed: Optional[int] = None,
    oversample_factor: int = 2,
    proposer: Optional[Proposer] = None,
) -> list[CandidateSpec]:
    """Generate candidate specs for one Capraro fault line.

    Args:
        n: number of CANDIDATES to emit (each pair × each layer = 1 candidate).
           For seed mode, this caps the total emitted from the registry seeds.
        db: ResultsDB used for spec_hash dedup and feedback-block lookup.
        fault_line_id: key into hypotheses.CAPRARO_FAULT_LINES (e.g. "causality").
        mode: "seed" to emit hand-written seeds only (no Opus call); "opus"
            to ask Opus for variants (with feedback from prior runs).
        layers, target_effectives: sweep grid for each pair. Defaults match
            Phase 2b/2d.
        oversample_factor: in opus mode, request this multiple of n pairs
            from Opus to allow for parse failures / dedup losses.
    """
    if fault_line_id not in list_fault_lines():
        raise ValueError(
            f"Unknown fault line {fault_line_id!r}. "
            f"Available: {list_fault_lines()}"
        )
    fault_line = get_fault_line(fault_line_id)
    layers = layers or DEFAULT_LAYERS
    target_effectives = target_effectives or DEFAULT_TARGET_EFFECTIVES

    import random as _random
    rng = _random.Random(seed)

    if mode == "seed":
        pairs = fault_line["seed_pairs"]
        print(f"[directed_capraro:{fault_line_id}] using {len(pairs)} hand-written seed pair(s)",
              flush=True)
    elif mode == "opus":
        # Build feedback from this fault line's prior DB results.
        feedback = _build_feedback_block(db, fault_line_id)
        if feedback:
            print(f"[directed_capraro:{fault_line_id}] including feedback "
                  f"from prior results ({len(feedback)} chars)", flush=True)
        else:
            print(f"[directed_capraro:{fault_line_id}] no prior results yet "
                  f"— fresh exploration", flush=True)
        # Ask for more pairs than we strictly need (oversample for losses)
        n_pairs_needed = max(1, n // max(1, len(layers)))
        n_pairs_to_ask = max(n_pairs_needed * oversample_factor, 5)
        prompt = _build_user_prompt(
            fault_line_id, fault_line, n_pairs_to_ask, feedback
        )
        if proposer is None:
            proposer = ClaudeProposer(model=CLAUDE_MODEL)
        print(f"[directed_capraro:{fault_line_id}] asking {proposer.name} "
              f"for {n_pairs_to_ask} variant pairs...", flush=True)
        t0 = time.time()
        raw = proposer.generate(SYSTEM_PROMPT, prompt)
        print(f"[directed_capraro:{fault_line_id}] got {len(raw)} chars "
              f"in {time.time()-t0:.1f}s", flush=True)
        pairs = _parse_pairs(raw)
        print(f"[directed_capraro:{fault_line_id}] parsed {len(pairs)} valid pairs",
              flush=True)
    else:
        raise ValueError(f"mode must be 'seed' or 'opus', got {mode!r}")

    # Emit one candidate per (pair × layer). target_effective is randomly
    # sampled per pair from the configured grid (matches novel_contrast).
    out: list[CandidateSpec] = []
    seen_this_batch: set[str] = set()
    strategy_tag = f"directed_capraro_{fault_line_id}"
    for pair in pairs:
        if len(out) >= n:
            break
        effective = rng.choice(target_effectives)
        for layer in layers:
            if len(out) >= n:
                break
            spec = CandidateSpec(
                id=f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
                strategy=strategy_tag,
                concept=pair["axis"],
                layer_idx=layer,
                target_effective=effective,
                derivation_method="contrast_pair",
                baseline_n=0,
                notes=pair.get("description", ""),
                contrast_pair={
                    "axis": pair["axis"],
                    "positive": pair["positive"],
                    "negative": pair["negative"],
                    "rationale": pair.get("rationale", ""),
                },
                # ADR-018: directed Capraro candidates always score under
                # ident-prioritized fitness regardless of worker env.
                fitness_mode="ident_prioritized",
            )
            h = spec_hash(spec)
            if h in seen_this_batch or db.has_candidate_hash(h):
                continue
            seen_this_batch.add(h)
            out.append(spec)
    return out
