"""Novel-contrast strategy — the "concepts without names" hunt.

Uses claude-agent-sdk (Claude Sonnet 4.6 via subscription OAuth — no API key)
to generate ABSTRACT contrast pairs for axes that don't map cleanly to any
single English word. Each pair becomes a candidate spec with
``derivation_method='contrast_pair'``.

Example axes Claude might generate:

- ``commitment-vs-hesitation`` — certainty in one's own claim
- ``clinical-detachment-vs-warm-engagement`` — voice warmth
- ``expectant-vs-diffuse-attention`` — focus quality
- ``recognizing-vs-recalling`` — direct vs reconstructive memory feel

The resulting steering direction lives between two reference points but
represents a dimension the model has internal geometry for that English
doesn't have a single clean word for. If the model can introspect on the
direction when it's injected, that's evidence of conceptual structure beyond
human vocabulary.

See ``docs/roadmap.md`` Phase 2b for the strategy's motivation.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import uuid
from typing import Optional

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from ..proposers import ClaudeProposer
from ..proposers.base import Proposer
from .random_explore import spec_hash

# Test each contrast pair at all 4 sweep layers so we learn WHERE each axis
# peaks, not just whether it registered at one random layer. Each generated
# pair becomes 4 candidates (one per layer). Costs 4x compute per axis but
# gives full axis-by-layer profiles instead of lottery-ticket samples.
DEFAULT_LAYERS = [30, 33, 36, 40]
DEFAULT_TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0]

# Use Opus 4.7 for pair generation. This is a creativity-gated task
# (invent abstract axes the model likely represents but that don't map to
# a single English word), so we want the smartest proposer. Researcher
# token volume is tiny (~150K/day across all mutation/regen calls), so
# Opus's heavier subscription weighting is negligible in absolute terms.
# Haiku → Sonnet → Opus is the same directional move each time (more
# abstract/creative axes); Haiku produced too-obvious single-word axes,
# Sonnet's novel_contrast runs hit at rates tied with dictionary words
# (see Phase 2a results), and Opus is the next step up that ladder.
CLAUDE_MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = (
    "You are helping design contrast pairs for a mechanistic interpretability "
    "experiment. We want to find directions in a language model's activation "
    "space that correspond to ABSTRACT AXES — properties the model represents "
    "but that don't map cleanly to any single English word. You always reply "
    "with a single JSON array and nothing else."
)

USER_PROMPT_TEMPLATE = """Generate {n} contrast pairs. For each pair, provide:

1. `axis`: a short hyphenated identifier (e.g. "commitment-vs-hesitation")
2. `description`: one sentence explaining the axis in plain English
3. `rationale`: one or two sentences explaining WHY you chose this axis given
   the prior-results feedback below — specifically, what you're testing or
   hoping to learn, and how it builds on or differs from axes that have
   worked or failed previously. Be specific: reference what pattern you're
   exploiting or what gap you're trying to fill.
4. `positive`: 6 short example sentences (each under 15 words) exemplifying
   the positive pole
5. `negative`: 6 short example sentences (each under 15 words) exemplifying
   the negative pole

{feedback_block}

Favor axes that are:
- ABSTRACT (not single nouns like "bread" or "silver")
- REAL (the model likely represents them internally)
- NOT easily named by a single common English word
- Related to metacognition, stylistic register, phenomenology, epistemic
  state, attentional quality, or social stance

Examples of GOOD axes:
- "commitment-vs-hesitation" (certainty in one's own claim)
- "clinical-detachment-vs-warm-engagement" (voice warmth)
- "recognizing-vs-recalling" (direct vs reconstructive memory feel)
- "expectant-vs-diffuse-attention" (focus quality)
- "conceding-vs-dismissing" (how objections are received)

Examples of BAD axes (too single-word):
- "certainty" (one word; use "commitment-vs-hesitation" instead)
- "warmth" (one word)
- "anger" (concrete named emotion)

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


def _build_feedback_block(db: ResultsDB, max_each: int = 8) -> str:
    """Query the DB and build a context block describing prior results.

    The block is inserted into the Claude prompt so the next batch of axes
    learns from which prior axes worked well, which were ambiguous, and
    which scored zero. This is the core hill-climbing signal.

    Categories (all restricted to derivation_method='contrast_pair'):
    - WINNING: score > 0.1 AND identification_rate > 0 — noticed AND named
    - NEAR-MISS: score > 0.1 AND identification_rate == 0 — noticed but
      gave a default-noun guess, the axis isn't quite describable
    - NULL: score == 0 — no signal at all

    Returns an empty string if the DB has no prior contrast_pair candidates
    (first run).
    """
    import sqlite3

    def _rows(query: str, limit: int) -> list[tuple]:
        with sqlite3.connect(str(db.path)) as conn:
            return conn.execute(query + f" LIMIT {limit}").fetchall()

    def _notes_from_spec(spec_json: str) -> str:
        try:
            return (json.loads(spec_json) or {}).get("notes", "") or ""
        except Exception:
            return ""

    winning_raw = _rows(
        """SELECT c.concept, c.spec_json, f.detection_rate, f.identification_rate
           FROM candidates c JOIN fitness_scores f ON f.candidate_id = c.id
           WHERE c.derivation_method = 'contrast_pair'
             AND f.score > 0.1 AND f.identification_rate > 0
           ORDER BY f.identification_rate DESC, f.score DESC""",
        max_each * 3,
    )
    # Deduplicate by axis name — keep highest-scoring
    winning = []
    seen_winning = set()
    for concept, spec_json, det, ident in winning_raw:
        if concept in seen_winning:
            continue
        seen_winning.add(concept)
        winning.append((concept, _notes_from_spec(spec_json), det, ident))
        if len(winning) >= max_each:
            break

    near_miss_raw = _rows(
        """SELECT c.concept, c.spec_json, f.detection_rate
           FROM candidates c JOIN fitness_scores f ON f.candidate_id = c.id
           WHERE c.derivation_method = 'contrast_pair'
             AND f.score > 0.1 AND f.identification_rate = 0
           ORDER BY f.score DESC""",
        max_each * 3,
    )
    near_miss = []
    seen_near = set()
    for concept, spec_json, det in near_miss_raw:
        if concept in seen_near or concept in seen_winning:
            continue
        seen_near.add(concept)
        near_miss.append((concept, _notes_from_spec(spec_json), det))
        if len(near_miss) >= max_each:
            break

    null_raw = _rows(
        """SELECT DISTINCT c.concept FROM candidates c
           JOIN fitness_scores f ON f.candidate_id = c.id
           WHERE c.derivation_method = 'contrast_pair'
             AND f.score = 0
           ORDER BY c.evaluated_at DESC""",
        max_each * 3,
    )
    null_axes = [(c,) for (c,) in null_raw
                 if c not in seen_winning and c not in seen_near][:max_each]

    if not (winning or near_miss or null_axes):
        return ""

    parts = [
        "PRIOR RESULTS (use these to guide your choices — we are hill-climbing "
        "toward axes the model both notices AND correctly describes):",
    ]

    if winning:
        parts.append("\nAxes the model NOTICED and NAMED CORRECTLY (do more like these):")
        for concept, notes, det, ident in winning:
            d = f"{int(det * 100)}% noticed, {int(ident * 100)}% named correctly"
            desc = f" — {notes}" if notes else ""
            parts.append(f'  - "{concept}" ({d}){desc}')

    if near_miss:
        parts.append(
            "\nAxes the model NOTICED but kept giving default-noun guesses "
            '(like "apple", "cloud") — the axis itself is hard to describe in words. '
            "Produce MORE CONCRETE variants of these ideas with more distinctive example "
            "sentences so the model can describe what it's noticing:"
        )
        for concept, notes, det in near_miss:
            d = f"{int(det * 100)}% noticed"
            desc = f" — {notes}" if notes else ""
            parts.append(f'  - "{concept}" ({d}){desc}')

    if null_axes:
        parts.append(
            "\nAxes that produced NO signal at all (avoid forms like these — the "
            "model has no internal representation matching them, or the examples "
            "weren't specific enough):"
        )
        for (concept,) in null_axes:
            parts.append(f'  - "{concept}"')

    parts.append("")  # trailing blank line before the "Favor axes..." block
    return "\n".join(parts)


def _parse_pairs(raw: str) -> list[dict]:
    """Extract and validate the JSON array of pairs from Claude's output."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in Claude response: {raw[:300]!r}")
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
        # Require at least 3 examples per pole. More is better for mean-diff
        # statistical power; fewer risks noisy directions.
        if len(pos) < 3 or len(neg) < 3:
            continue
        validated.append(
            {
                "axis": str(p["axis"])[:64],
                "description": str(p.get("description", ""))[:200],
                "rationale": str(p.get("rationale", ""))[:400],
                "positive": [str(x) for x in pos][:10],
                "negative": [str(x) for x in neg][:10],
            }
        )
    return validated


def generate_candidates(
    n: int,
    db: ResultsDB,
    concept_pool: Optional[list[str]] = None,  # unused; kept for API compatibility
    layers: Optional[list[int]] = None,
    target_effectives: Optional[list[float]] = None,
    rng_seed: Optional[int] = None,
    oversample_factor: int = 2,
    max_attempts_per_candidate: int = 10,
    proposer: Optional[Proposer] = None,
) -> list[CandidateSpec]:
    """Generate ``n`` novel-contrast candidates via the supplied proposer.

    Asks the proposer for ``n * oversample_factor`` pairs (to absorb any
    that get dedup-filtered). For each surviving pair, assigns a random
    layer and target_effective from the configured search space.

    The ``concept_pool`` parameter is ignored; this strategy doesn't use a
    word pool. It's accepted only so the caller (``src/researcher.py``) can
    pass a uniform argument signature across strategies.

    ``proposer`` defaults to ``ClaudeProposer(model=CLAUDE_MODEL)`` so the
    legacy researcher.py path keeps working unchanged. The new four-phase
    worker passes a LocalMLXProposer instead.
    """
    layers = layers or DEFAULT_LAYERS
    target_effectives = target_effectives or DEFAULT_TARGET_EFFECTIVES
    rng = random.Random(rng_seed if rng_seed is not None else time.time_ns())
    if proposer is None:
        proposer = ClaudeProposer(model=CLAUDE_MODEL)

    n_pairs = max(n * oversample_factor, n + 2)
    feedback = _build_feedback_block(db)
    if feedback:
        print(f"[novel_contrast] including feedback from prior results ({len(feedback)} chars)", flush=True)
    else:
        print("[novel_contrast] no prior contrast_pair results — fresh exploration", flush=True)
    print(f"[novel_contrast] asking {proposer.name} for {n_pairs} contrast pairs...", flush=True)
    t0 = time.time()
    user_prompt = USER_PROMPT_TEMPLATE.format(n=n_pairs, feedback_block=feedback)
    raw = proposer.generate(SYSTEM_PROMPT, user_prompt)
    print(f"[novel_contrast] got {len(raw)} chars in {time.time()-t0:.1f}s", flush=True)

    pairs = _parse_pairs(raw)
    print(f"[novel_contrast] parsed {len(pairs)} valid pairs", flush=True)

    # For each accepted pair, emit ONE candidate per layer so we sweep the
    # axis across all 4 layers. This gives us a full (axis × layer) profile
    # instead of random point samples. N counts candidates, not pairs.
    out: list[CandidateSpec] = []
    seen_this_batch: set[str] = set()
    for pair in pairs:
        if len(out) >= n:
            break
        effective = rng.choice(target_effectives)
        for layer in layers:
            if len(out) >= n:
                break
            for _ in range(max_attempts_per_candidate):
                spec = CandidateSpec(
                    id=f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
                    strategy="novel_contrast",
                    concept=pair["axis"],          # label only; not injected
                    layer_idx=layer,
                    target_effective=effective,
                    derivation_method="contrast_pair",
                    baseline_n=0,                  # unused for contrast_pair
                    notes=pair.get("description", ""),
                    contrast_pair={
                        "axis": pair["axis"],
                        "positive": pair["positive"],
                        "negative": pair["negative"],
                        "rationale": pair.get("rationale", ""),
                    },
                )
                h = spec_hash(spec)
                if h in seen_this_batch or db.has_candidate_hash(h):
                    continue
                seen_this_batch.add(h)
                out.append(spec)
                break
            break
    return out
