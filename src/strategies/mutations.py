"""Mutation operators for the structured hill-climb strategy.

Each operator takes a parent CandidateSpec (already evaluated, has fitness
in the DB) and produces ONE new child CandidateSpec. The child carries
``parent_candidate_id`` and ``mutation_type`` so the lineage is queryable.

Three deterministic operators (no proposer needed) — generate child specs
directly by tweaking layer / alpha / nothing:

  - ``layer_shift``  : same axis, layer ∈ {parent_L−6, parent_L−3, parent_L+3, parent_L+6}
  - ``alpha_scale``  : same axis, eff ∈ {parent_eff×0.7, parent_eff×1.4}
  - ``replication``  : verbatim re-evaluation. Identical spec to parent except
                       new id, used to confirm a result isn't a one-shot
                       artifact. Same axis, same layer, same alpha.

Four proposer-driven operators (call the local MLX proposer with a tight
template) — same axis name, regenerate one piece of language:

  - ``examples_swap``       : same axis name + pole descriptions, regenerate
                              the 6+6 example sentences
  - ``description_sharpen`` : same axis name + examples, rewrite pole
                              descriptions tighter
  - ``antonym_pivot``       : keep positive pole, generate a different
                              negative pole that contrasts with positive on
                              the same dimension
  - ``lexical_decontaminate``: regenerate examples with tokens that were
                              fully exclusive to one pole BANNED from both
                              new poles. Tests whether a winning result was
                              concept-level signal or just lexical pickup
                              on those exclusive tokens. (Discovered live
                              2026-04-28: causality Class 1 score 3.812
                              relied on the literal token "caused" appearing
                              in 6/6 positives and 0/6 negatives — when
                              banned, signal vanished.)

The dispatcher chooses an operator at random per child slot. Deterministic
operators emit specs synchronously; proposer-driven ones queue a single
generation call into a list and the dispatcher batches them.

See ``docs/structured_hillclimb.md`` for the full design.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from ..proposers.base import Proposer
from .random_explore import spec_hash

# Layer search range — anything outside [24, 44] is too shallow / too deep
# for Gemma3-12B introspection (Phase 1 sweep showed L=33 is the peak).
LAYER_MIN = 24
LAYER_MAX = 44

# How far to shift layers in one mutation step. Two scales: small and big.
# We sample one of these uniformly per layer_shift mutation.
LAYER_DELTAS = (-6, -3, 3, 6)

# Multiplicative alpha-scale factors. 0.7 backs off, 1.4 pushes harder.
ALPHA_FACTORS = (0.7, 1.4)


@dataclass
class ParentRecord:
    """A previously-evaluated candidate we want to mutate or replicate.

    Loaded by the dispatcher via DB query. Carries everything needed to
    construct a child without re-querying.
    """
    candidate_id: str
    strategy: str
    concept: str                   # axis name
    layer_idx: int
    target_effective: float
    derivation_method: str
    contrast_pair: Optional[dict]  # axis, positive[], negative[], rationale, description
    fitness_mode: Optional[str]
    score: float
    detection_rate: float
    identification_rate: float


# ---------------------------------------------------------------------------
# Helpers shared by every operator
# ---------------------------------------------------------------------------

def _new_candidate_id() -> str:
    return f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _clamp_layer(layer: int) -> int:
    return max(LAYER_MIN, min(LAYER_MAX, layer))


def _make_child(
    parent: ParentRecord,
    *,
    strategy: str,
    layer: int,
    target_effective: float,
    contrast_pair: Optional[dict],
    proposer_model: Optional[str],
    mutation_type: str,
    mutation_detail: dict,
) -> CandidateSpec:
    """Construct a child spec with lineage metadata baked in.

    ``mutation_detail`` is stashed in ``spec.notes`` as a JSON suffix so
    the worker (which already serializes notes back through the DB) gets
    it for free without a schema change. The actual ``mutation_type`` and
    ``parent_candidate_id`` are written via the queue file's ``_lineage``
    block, which the worker reads at insert time — same path the old
    Phase 2c hillclimb used.

    Note: we do NOT mutate ``spec.notes`` to carry mutation_detail since
    the worker treats notes as a human-readable description. Lineage info
    travels through the queue file's _lineage block exclusively.
    """
    description = ""
    if contrast_pair is not None:
        description = (
            contrast_pair.get("description")
            or (parent.contrast_pair or {}).get("description")
            or ""
        )

    spec = CandidateSpec(
        id=_new_candidate_id(),
        strategy=strategy,
        concept=parent.concept,
        layer_idx=layer,
        target_effective=target_effective,
        derivation_method=parent.derivation_method,
        baseline_n=0,
        notes=description,
        contrast_pair=contrast_pair,
        fitness_mode=parent.fitness_mode,
        proposer_model=proposer_model,
    )
    # Attach lineage as a side-channel attribute. The dispatcher reads
    # this when writing the queue file and does NOT include it in
    # spec.to_dict() — instead it goes into the queue file's `_lineage`
    # block which the worker reads on insert (already wired in worker.py).
    spec._lineage_meta = {  # type: ignore[attr-defined]
        "parent_candidate_id": parent.candidate_id,
        "mutation_type": mutation_type,
        "mutation_detail": json.dumps(mutation_detail),
        "generation": 0,  # generation tracking handled by worker if it cares
    }
    return spec


# ---------------------------------------------------------------------------
# Deterministic operators
# ---------------------------------------------------------------------------

def replication(parent: ParentRecord) -> CandidateSpec:
    """Verbatim re-eval of parent. Same axis, layer, alpha.

    The only thing that differs is the candidate id and a per-call
    replication id embedded in the rationale (so spec_hash differs from
    both the parent and any other replication of the same parent — every
    replication gets its own row, never dedup'd against siblings).
    """
    parent_pair = parent.contrast_pair or {}
    # uuid keeps each rep distinguishable from sibling reps — the
    # dispatcher's spec_hash dedup would otherwise collapse them all.
    rep_id = uuid.uuid4().hex[:8]
    rep_tag = f"replication-of-{parent.candidate_id}#{rep_id}"
    pair_with_tag = {
        "axis": parent_pair.get("axis", parent.concept),
        "positive": list(parent_pair.get("positive", [])),
        "negative": list(parent_pair.get("negative", [])),
        "rationale": (
            f"[{rep_tag}] " + (parent_pair.get("rationale", "") or "")
        )[:400],
    }
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=parent.target_effective,
        contrast_pair=pair_with_tag,
        proposer_model=None,  # no LLM call
        mutation_type="replication",
        mutation_detail={
            "of_candidate_id": parent.candidate_id,
            "parent_score": parent.score,
        },
    )


def layer_shift(parent: ParentRecord, *, rng: random.Random) -> CandidateSpec:
    """Same axis, shift layer by one of LAYER_DELTAS."""
    delta = rng.choice(LAYER_DELTAS)
    new_layer = _clamp_layer(parent.layer_idx + delta)
    parent_pair = parent.contrast_pair or {}
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=new_layer,
        target_effective=parent.target_effective,
        contrast_pair={
            "axis": parent_pair.get("axis", parent.concept),
            "positive": list(parent_pair.get("positive", [])),
            "negative": list(parent_pair.get("negative", [])),
            "rationale": (parent_pair.get("rationale", "") or "")[:400],
        },
        proposer_model=None,
        mutation_type="layer_shift",
        mutation_detail={
            "parent_layer": parent.layer_idx,
            "delta": delta,
            "new_layer": new_layer,
        },
    )


def alpha_scale(parent: ParentRecord, *, rng: random.Random) -> CandidateSpec:
    """Same axis, scale target_effective by one of ALPHA_FACTORS."""
    factor = rng.choice(ALPHA_FACTORS)
    new_eff = parent.target_effective * factor
    parent_pair = parent.contrast_pair or {}
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=new_eff,
        contrast_pair={
            "axis": parent_pair.get("axis", parent.concept),
            "positive": list(parent_pair.get("positive", [])),
            "negative": list(parent_pair.get("negative", [])),
            "rationale": (parent_pair.get("rationale", "") or "")[:400],
        },
        proposer_model=None,
        mutation_type="alpha_scale",
        mutation_detail={
            "parent_eff": parent.target_effective,
            "factor": factor,
            "new_eff": new_eff,
        },
    )


# ---------------------------------------------------------------------------
# Proposer-driven operators
# ---------------------------------------------------------------------------

# All three proposer-driven operators share this system prompt — short,
# focused on producing exactly the requested JSON object (NOT an array).
PROPOSER_SYSTEM_PROMPT = (
    "You are refining contrast pairs for a mechanistic interpretability "
    "experiment. We have a winning contrast pair and want to test a small "
    "variation. You always reply with a single JSON object and nothing else."
)


def _parse_single_pair(raw: str) -> Optional[dict]:
    """Extract a single pair JSON object from proposer output.

    Returns None on parse failure — the caller falls back to a deterministic
    operator instead of crashing the batch.
    """
    # Try to find an object first; fall back to first-element of an array.
    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not obj_match:
        return None
    try:
        obj = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        # Maybe the model returned an array — take the first element.
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not arr_match:
            return None
        try:
            arr = json.loads(arr_match.group(0))
            obj = arr[0] if isinstance(arr, list) and arr else None
        except (json.JSONDecodeError, IndexError):
            return None
    if not isinstance(obj, dict):
        return None
    return obj


def examples_swap(
    parent: ParentRecord,
    *,
    proposer: Proposer,
) -> Optional[CandidateSpec]:
    """Same axis name + pole descriptions, regenerate the 6+6 example sentences.

    Tests whether the geometric direction is robust to the *specific*
    sentences used to derive it, vs being a quirk of those sentences.
    """
    parent_pair = parent.contrast_pair or {}
    description = parent_pair.get("description", "") or ""
    user_prompt = f"""Generate 6 NEW positive examples and 6 NEW negative examples for the contrast pair below. Keep the axis identifier and description EXACTLY the same. The new examples should be in the same conceptual cluster as the originals but use different specific scenarios, vocabulary, and grammatical structures, so we can test whether the steering direction is robust to surface phrasing.

Axis: {parent_pair.get('axis', parent.concept)}
Description: {description}

Original positive examples (do not reuse):
{json.dumps(parent_pair.get('positive', []), indent=2)}

Original negative examples (do not reuse):
{json.dumps(parent_pair.get('negative', []), indent=2)}

Return a JSON object in this exact shape:
{{
  "positive": ["...", "...", "...", "...", "...", "..."],
  "negative": ["...", "...", "...", "...", "...", "..."]
}}

Each example must be under 18 words. Do not include any text before or after the JSON object."""

    raw = proposer.generate(PROPOSER_SYSTEM_PROMPT, user_prompt)
    obj = _parse_single_pair(raw)
    if obj is None:
        return None
    pos = obj.get("positive") or []
    neg = obj.get("negative") or []
    if not isinstance(pos, list) or not isinstance(neg, list):
        return None
    if len(pos) < 3 or len(neg) < 3:
        return None

    new_pair = {
        "axis": parent_pair.get("axis", parent.concept),
        "positive": [str(x)[:240] for x in pos][:10],
        "negative": [str(x)[:240] for x in neg][:10],
        "rationale": (parent_pair.get("rationale", "") or "")[:400],
        "description": description,
    }
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=parent.target_effective,
        contrast_pair=new_pair,
        proposer_model=proposer.name,
        mutation_type="examples_swap",
        mutation_detail={
            "n_positive": len(new_pair["positive"]),
            "n_negative": len(new_pair["negative"]),
        },
    )


def description_sharpen(
    parent: ParentRecord,
    *,
    proposer: Proposer,
) -> Optional[CandidateSpec]:
    """Same axis name + examples, rewrite pole descriptions tighter.

    Often the original axis description is vague. A sharper description
    won't change the steering direction (since direction comes from
    examples), but it gives us a better label for analysis and for the
    judge's contrast-pair grading prompt.
    """
    parent_pair = parent.contrast_pair or {}
    user_prompt = f"""Refine the description for the contrast pair below. Keep the axis identifier, positive examples, and negative examples EXACTLY the same. Rewrite only the description so it states more precisely what dimension the positive and negative poles measure.

Axis: {parent_pair.get('axis', parent.concept)}
Old description: {parent_pair.get('description', '') or '(none)'}

Positive examples:
{json.dumps(parent_pair.get('positive', []), indent=2)}

Negative examples:
{json.dumps(parent_pair.get('negative', []), indent=2)}

Return a JSON object in this exact shape:
{{
  "description": "..."
}}

Description must be one sentence under 30 words. Do not include any text before or after the JSON object."""

    raw = proposer.generate(PROPOSER_SYSTEM_PROMPT, user_prompt)
    obj = _parse_single_pair(raw)
    if obj is None:
        return None
    new_desc = str(obj.get("description", ""))[:240]
    if not new_desc.strip():
        return None

    new_pair = {
        "axis": parent_pair.get("axis", parent.concept),
        "positive": list(parent_pair.get("positive", [])),
        "negative": list(parent_pair.get("negative", [])),
        "rationale": (parent_pair.get("rationale", "") or "")[:400],
        "description": new_desc,
    }
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=parent.target_effective,
        contrast_pair=new_pair,
        proposer_model=proposer.name,
        mutation_type="description_sharpen",
        mutation_detail={
            "old_description": (parent_pair.get("description", "") or "")[:120],
            "new_description": new_desc[:120],
        },
    )


def antonym_pivot(
    parent: ParentRecord,
    *,
    proposer: Proposer,
) -> Optional[CandidateSpec]:
    """Keep positive pole, generate a different negative pole.

    The positive pole is the "thing of interest"; the negative pole
    defines what we contrast it against. Different negative poles can
    isolate different sub-dimensions of the same positive concept.
    """
    parent_pair = parent.contrast_pair or {}
    user_prompt = f"""Below is a contrast pair where the POSITIVE pole works well. We want to test whether the same positive pole still produces signal when contrasted against a DIFFERENT negative pole — this isolates which sub-dimension of the positive concept matters.

Keep the axis identifier and positive examples EXACTLY the same. Generate a NEW description and 6 NEW negative examples that contrast with the positive pole on a DIFFERENT axis than the original negatives.

Axis: {parent_pair.get('axis', parent.concept)}

Positive examples (KEEP):
{json.dumps(parent_pair.get('positive', []), indent=2)}

Original negative examples (DO NOT use this same contrast):
{json.dumps(parent_pair.get('negative', []), indent=2)}

Return a JSON object in this exact shape:
{{
  "description": "...",
  "negative": ["...", "...", "...", "...", "...", "..."]
}}

Description must be one sentence under 30 words. Each negative example must be under 18 words. Do not include any text before or after the JSON object."""

    raw = proposer.generate(PROPOSER_SYSTEM_PROMPT, user_prompt)
    obj = _parse_single_pair(raw)
    if obj is None:
        return None
    neg = obj.get("negative") or []
    new_desc = str(obj.get("description", ""))[:240]
    if not isinstance(neg, list) or len(neg) < 3 or not new_desc.strip():
        return None

    new_pair = {
        "axis": parent_pair.get("axis", parent.concept),
        "positive": list(parent_pair.get("positive", [])),
        "negative": [str(x)[:240] for x in neg][:10],
        "rationale": (parent_pair.get("rationale", "") or "")[:400],
        "description": new_desc,
    }
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=parent.target_effective,
        contrast_pair=new_pair,
        proposer_model=proposer.name,
        mutation_type="antonym_pivot",
        mutation_detail={
            "n_new_negatives": len(new_pair["negative"]),
            "new_description": new_desc[:120],
        },
    )


def lexical_decontaminate(
    parent: ParentRecord,
    *,
    proposer: Proposer,
) -> Optional[CandidateSpec]:
    """Regenerate examples with surface-token contamination explicitly banned.

    Audits the parent contrast pair for tokens that appear in every
    example of one pole and zero examples of the other (and high-skew
    near-misses). Then asks the proposer to regenerate 6 positive + 6
    negative examples with those tokens forbidden in BOTH poles.

    Why this matters: if a winning result evaporates under
    decontamination, the original signal was lexical pickup on the
    exclusive tokens (e.g. the "caused" token in causal-vs-temporal-
    gerund-cause), not concept-level introspection. If the result
    survives, that's evidence the steering direction encodes an
    abstract conceptual feature, not just a surface token's embedding.

    Returns None when:
      - The parent pair is already lexically clean (no operator needed —
        let another operator handle this slot).
      - The proposer returns malformed JSON.
      - The proposer's regenerated examples STILL contain banned tokens
        (we retry once, then give up).
    """
    parent_pair = parent.contrast_pair or {}
    from .lexical_audit import (
        banned_tokens_for_decontamination,
        lexical_contamination,
    )

    report = lexical_contamination(parent_pair)
    banned = banned_tokens_for_decontamination(report)
    # If the parent is already clean, this operator has nothing to do —
    # signal that to the dispatcher so a different operator fills the slot.
    if not banned:
        return None

    description = parent_pair.get("description", "") or ""
    banned_str = ", ".join(f"'{t}'" for t in banned)
    user_prompt = f"""You are lexically-decontaminating a contrast pair. The original pair had specific surface tokens that appeared in EVERY example of one pole and ZERO examples of the other — that means the steering direction extracted from this pair was dominated by the EMBEDDINGS of those tokens, not the underlying concept.

Generate 6 NEW positive examples and 6 NEW negative examples for the contrast pair below. Keep the axis identifier and description EXACTLY the same. The new examples must:

  1. Express the SAME conceptual contrast as the original.
  2. AVOID these banned tokens in BOTH poles (no exceptions, no morphological variants like plurals or tense changes): {banned_str}
  3. Use varied vocabulary so no single content-word appears in 6/6 examples of one pole.
  4. Make the contrast come from MEANING/STRUCTURE, not from a single shared word.

Axis: {parent_pair.get('axis', parent.concept)}
Description: {description}

Original positive examples:
{json.dumps(parent_pair.get('positive', []), indent=2)}

Original negative examples:
{json.dumps(parent_pair.get('negative', []), indent=2)}

Return a JSON object in this exact shape:
{{
  "positive": ["...", "...", "...", "...", "...", "..."],
  "negative": ["...", "...", "...", "...", "...", "..."]
}}

Each example must be under 18 words. Do not include any text before or after the JSON object."""

    raw = proposer.generate(PROPOSER_SYSTEM_PROMPT, user_prompt)
    obj = _parse_single_pair(raw)
    if obj is None:
        return None
    pos = obj.get("positive") or []
    neg = obj.get("negative") or []
    if not isinstance(pos, list) or not isinstance(neg, list):
        return None
    if len(pos) < 3 or len(neg) < 3:
        return None

    # Verify the proposer actually obeyed the ban. If any banned token is
    # still present in the regenerated examples, we abort — better to
    # return None and let a fallback operator fill the slot than to emit
    # a "decontaminated" spec that's still contaminated.
    new_pair_dict = {
        "positive": [str(x)[:240] for x in pos][:10],
        "negative": [str(x)[:240] for x in neg][:10],
    }
    re_report = lexical_contamination(new_pair_dict)
    re_banned = set(banned_tokens_for_decontamination(re_report))
    # If any of the ORIGINAL banned tokens are still in the new pair's
    # exclusive sets, the proposer didn't obey. Even if NEW exclusive
    # tokens emerged (different from the original list), that's a softer
    # warning — we let it through but tag the rationale.
    original_banned = set(banned)
    leaked = original_banned & set(
        re_report.positive_exclusive + re_report.negative_exclusive
    )
    if leaked:
        # Proposer ignored the explicit ban on a banned token. Reject.
        return None

    # Tag the rationale so downstream analysis knows this pair was
    # decontaminated and which tokens were banned.
    rationale = (
        f"[lexical_decontaminate banned={','.join(banned)}] "
        + (parent_pair.get("rationale", "") or "")
    )[:400]

    new_pair = {
        "axis": parent_pair.get("axis", parent.concept),
        "positive": new_pair_dict["positive"],
        "negative": new_pair_dict["negative"],
        "rationale": rationale,
        "description": description,
    }
    return _make_child(
        parent,
        strategy=f"hillclimb_{parent.strategy}",
        layer=parent.layer_idx,
        target_effective=parent.target_effective,
        contrast_pair=new_pair,
        proposer_model=proposer.name,
        mutation_type="lexical_decontaminate",
        mutation_detail={
            "banned_tokens": banned,
            "n_banned": len(banned),
            "original_contamination": report.summary(),
            "decontaminated_contamination": re_report.summary(),
            "new_exclusive_emerged": sorted(
                set(re_report.positive_exclusive + re_report.negative_exclusive)
                - original_banned
            ),
        },
    )


# ---------------------------------------------------------------------------
# Public registry — operator name → callable
# ---------------------------------------------------------------------------

DETERMINISTIC_OPERATORS = ("layer_shift", "alpha_scale")
PROPOSER_OPERATORS = (
    "examples_swap",
    "description_sharpen",
    "antonym_pivot",
    "lexical_decontaminate",
)
ALL_VARIANT_OPERATORS = DETERMINISTIC_OPERATORS + PROPOSER_OPERATORS


def apply_operator(
    op_name: str,
    parent: ParentRecord,
    *,
    rng: random.Random,
    proposer: Optional[Proposer],
) -> Optional[CandidateSpec]:
    """Dispatch ``op_name`` against ``parent``. Returns None on parse failure
    (proposer-driven operators only); deterministic ops always succeed.
    """
    if op_name == "replication":
        return replication(parent)
    if op_name == "layer_shift":
        return layer_shift(parent, rng=rng)
    if op_name == "alpha_scale":
        return alpha_scale(parent, rng=rng)
    if op_name in PROPOSER_OPERATORS:
        if proposer is None:
            return None
        if op_name == "examples_swap":
            return examples_swap(parent, proposer=proposer)
        if op_name == "description_sharpen":
            return description_sharpen(parent, proposer=proposer)
        if op_name == "antonym_pivot":
            return antonym_pivot(parent, proposer=proposer)
        if op_name == "lexical_decontaminate":
            return lexical_decontaminate(parent, proposer=proposer)
    raise ValueError(f"Unknown mutation operator: {op_name!r}")
