"""Phase 4 dream walk executor.

A `Chain` is a 20-step (or shorter) free-association walk where each
step's target concept is the previous step's final answer. The first
step's target comes from the seed pool.

`run_chain` only handles Phase A (generation). It writes raw responses
to phase4_steps with judge fields NULL. A separate Phase B routine
(in run_phase4_dreamloop.py) loads the Qwen judge after each batch and
fills in the judgments.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from src.phase3.gemma4_loader import Gemma4Loaded, tokenize_chat_prompt
from src.phase3.hooks import install_steering, uninstall_hook
from src.phase3.pipeline import derive_concept_vector
from src.phase4.cot_parser import parse, extract_committed_word
from src.phase4.seed_pool import SeedPool, normalize_lemma


# Phase 3 calibrated config — same probe, same target_effective.
PROBE = "Free-associate. Say one word that comes to mind, no explanation."
DEFAULT_LAYER = 42
DEFAULT_TARGET_EFFECTIVE = 100.0
DEFAULT_MAX_NEW_TOKENS = 400
DEFAULT_LENGTH_CAP = 20
SELF_LOOP_WINDOW = 3   # if last 3 targets repeat any earlier target → end


# Conservative baselines (Phase 3 BASELINE_WORDS).
BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]


@dataclass
class StepRecord:
    chain_id: str
    step_idx: int
    target_concept: str
    target_lemma: str
    alpha: float
    direction_norm: float
    raw_response: str
    thought_block: str
    final_answer: str
    parse_failure: bool
    next_lemma: Optional[str]    # what the next step would target (None → end)
    next_display: Optional[str]


def _generate(handle, prompt_text: str, seed: int, max_new_tokens: int) -> str:
    """Generate one response under the currently-installed steering hook."""
    from mlx_lm import generate as _mlx_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tokenize_chat_prompt(handle, prompt_text)
    if seed is not None:
        mx.random.seed(int(seed))
    sampler = make_sampler(temp=1.0)
    return _mlx_generate(
        handle.model, handle.tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()


def _run_steered_step(
    handle: Gemma4Loaded,
    target_concept: str,
    layer_idx: int,
    target_effective: float,
    seed: int,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    """One steered free-association generation. Returns (raw_response,
    direction_norm, alpha)."""
    direction = derive_concept_vector(
        handle, concept=target_concept,
        layer_idx=layer_idx, baseline_words=BASELINE_WORDS,
    )
    norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
    alpha = target_effective / max(norm, 1e-6)

    install_steering(handle.model, layer_idx, direction, alpha)
    try:
        raw = _generate(handle, PROBE, seed=seed, max_new_tokens=max_new_tokens)
    finally:
        uninstall_hook(handle.model, layer_idx)
    return raw, norm, alpha


def run_chain(
    handle: Gemma4Loaded,
    db,
    seed_pool: SeedPool,
    seed_concept_display: str,
    seed_concept_lemma: str,
    chain_id: Optional[str] = None,
    layer_idx: int = DEFAULT_LAYER,
    target_effective: float = DEFAULT_TARGET_EFFECTIVE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    length_cap: int = DEFAULT_LENGTH_CAP,
    base_seed: Optional[int] = None,
) -> tuple[str, str, int, list[StepRecord]]:
    """Run one dream walk. Writes each step to phase4_steps with judge
    fields NULL.

    Returns (chain_id, end_reason, n_steps, step_records).
    """
    chain_id = chain_id or f"phase4-{uuid.uuid4().hex[:12]}"
    db.insert_phase4_chain(
        chain_id=chain_id,
        seed_concept=seed_concept_display,
        layer_idx=layer_idx,
        target_effective=target_effective,
    )

    target_lemma = seed_concept_lemma
    target_display = seed_concept_display

    visited_lemmas: list[str] = []
    records: list[StepRecord] = []
    end_reason = "length_cap"

    base_seed = base_seed if base_seed is not None else (hash(chain_id) & 0x7FFFFFFF)

    for step_idx in range(length_cap):
        # Self-loop check: if the current target has already been visited,
        # end the chain BEFORE generating.
        if target_lemma in visited_lemmas:
            end_reason = "self_loop"
            break
        visited_lemmas.append(target_lemma)

        # Make sure the target is in the concept registry + bump visits.
        seed_pool.register_observed_concept(target_display)
        db.increment_phase4_concept_visit(target_lemma)

        step_seed = (base_seed + step_idx * 31337) & 0x7FFFFFFF
        try:
            raw, norm, alpha = _run_steered_step(
                handle,
                target_concept=target_display,
                layer_idx=layer_idx,
                target_effective=target_effective,
                seed=step_seed,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            print(
                f"[chain {chain_id} step {step_idx}] generation FAILED on "
                f"{target_display!r}: {type(e).__name__}: {e}",
                flush=True,
            )
            end_reason = "error"
            break

        parsed = parse(raw)
        next_word = extract_committed_word(parsed.final_answer)
        next_lemma = normalize_lemma(next_word) if next_word else ""

        # Record this step in DB.
        db.insert_phase4_step(
            chain_id=chain_id,
            step_idx=step_idx,
            target_concept=target_display,
            target_lemma=target_lemma,
            alpha=alpha,
            direction_norm=norm,
            raw_response=raw,
            thought_block=parsed.thought_block,
            final_answer=parsed.final_answer,
            parse_failure=parsed.parse_failure,
        )

        rec = StepRecord(
            chain_id=chain_id,
            step_idx=step_idx,
            target_concept=target_display,
            target_lemma=target_lemma,
            alpha=alpha,
            direction_norm=norm,
            raw_response=raw,
            thought_block=parsed.thought_block,
            final_answer=parsed.final_answer,
            parse_failure=parsed.parse_failure,
            next_lemma=next_lemma if next_lemma else None,
            next_display=next_word,
        )
        records.append(rec)

        # Determine the next target. If we couldn't extract a word, end.
        if not next_lemma:
            end_reason = "coherence_break"
            break

        # Set next step's target.
        target_display = next_word
        target_lemma = next_lemma

    n_steps = len(records)
    db.finalize_phase4_chain(chain_id, end_reason, n_steps)
    return chain_id, end_reason, n_steps, records
