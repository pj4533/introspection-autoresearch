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

import asyncio
import hashlib
import json
import random
import re
import threading
import time
import uuid
from typing import Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from .random_explore import spec_hash

# Test each contrast pair at all 4 sweep layers so we learn WHERE each axis
# peaks, not just whether it registered at one random layer. Each generated
# pair becomes 4 candidates (one per layer). Costs 4x compute per axis but
# gives full axis-by-layer profiles instead of lottery-ticket samples.
DEFAULT_LAYERS = [30, 33, 36, 40]
DEFAULT_TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0]

# Use Sonnet 4.6 for pair generation — Haiku is faster but its axes tend
# toward the obvious. Sonnet reliably produces creative in-between concepts.
CLAUDE_MODEL = "claude-sonnet-4-6"

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
3. `positive`: 6 short example sentences (each under 15 words) exemplifying
   the positive pole
4. `negative`: 6 short example sentences (each under 15 words) exemplifying
   the negative pole

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
    "positive": ["...", "...", "...", "...", "...", "..."],
    "negative": ["...", "...", "...", "...", "...", "..."]
  }},
  ...
]

Do not include any text before or after the JSON array."""


async def _ask_claude(prompt: str) -> str:
    """Call Claude via subscription OAuth. Returns concatenated text output."""
    options = ClaudeAgentOptions(
        model=CLAUDE_MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_turns=1,
        permission_mode="bypassPermissions",
        allowed_tools=[],
    )
    chunks: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks).strip()


def _run_sync(prompt: str) -> str:
    """Run the async query in a worker thread so it works from any context.

    Same pattern as ``src/judges/claude_judge.py::_run_sync``: the SDK uses
    asyncio internally, so calling ``asyncio.run`` directly fails inside
    Jupyter or any already-running loop. A dedicated thread gets a fresh loop.
    """
    result: dict = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(_ask_claude(prompt))
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


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
) -> list[CandidateSpec]:
    """Generate ``n`` novel-contrast candidates via Claude.

    Asks Claude for ``n * oversample_factor`` pairs (to absorb any that get
    dedup-filtered). For each surviving pair, assigns a random layer and
    target_effective from the configured search space.

    The ``concept_pool`` parameter is ignored; this strategy doesn't use a
    word pool. It's accepted only so the caller (``src/researcher.py``) can
    pass a uniform argument signature across strategies.
    """
    layers = layers or DEFAULT_LAYERS
    target_effectives = target_effectives or DEFAULT_TARGET_EFFECTIVES
    rng = random.Random(rng_seed if rng_seed is not None else time.time_ns())

    n_pairs = max(n * oversample_factor, n + 2)
    print(f"[novel_contrast] asking {CLAUDE_MODEL} for {n_pairs} contrast pairs...", flush=True)
    t0 = time.time()
    raw = _run_sync(USER_PROMPT_TEMPLATE.format(n=n_pairs))
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
