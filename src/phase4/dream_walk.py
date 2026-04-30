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
# Phase 4 generation budget: previously a fixed 600-token cap cut most
# thought blocks short, producing 46% coherence_break ends. New design
# (2026-04-30) gives the model effective free reign — 4000 tokens — and
# instead detects pathological loop patterns and aborts early.
#
# Two patterns we catch:
#   1. Hard repeat: the same token id appears ≥RUNAWAY_REPEAT_THRESHOLD
#      times in the trailing RUNAWAY_WINDOW tokens.
#   2. Soft loop: the trailing window has fewer than
#      RUNAWAY_MIN_UNIQUE distinct token ids — the model is cycling
#      through a small vocabulary repeatedly (observed in Harmonies /
#      Luminous chains where the model emitted "Luminous. Luminosities.
#      Symmetries. Luminous." for 11k+ chars without committing).
DEFAULT_MAX_NEW_TOKENS = 4000
RUNAWAY_WINDOW = 100             # trailing tokens to inspect
RUNAWAY_REPEAT_THRESHOLD = 30    # hard-repeat: same token ≥N times in window
RUNAWAY_MIN_UNIQUE = 25          # soft-loop: <N unique tokens in window
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
    """Generate one response under the currently-installed steering hook.

    Streams token-by-token via mlx_lm.stream_generate so we can detect
    pathological runaway repetition (any single token ID appearing
    ≥RUNAWAY_REPEAT_THRESHOLD times in the last RUNAWAY_WINDOW tokens)
    and stop early. The hard cap is max_new_tokens=4000.
    """
    from mlx_lm import stream_generate as _mlx_stream
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tokenize_chat_prompt(handle, prompt_text)
    if seed is not None:
        mx.random.seed(int(seed))
    sampler = make_sampler(temp=1.0)

    pieces: list[str] = []
    recent_tokens: list[int] = []
    n_emitted = 0
    aborted_runaway = False

    for resp in _mlx_stream(
        handle.model,
        handle.tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        sampler=sampler,
    ):
        # mlx_lm GenerationResponse exposes .text (delta) and .token (id).
        pieces.append(getattr(resp, "text", "") or "")
        tok = getattr(resp, "token", None)
        if tok is not None:
            recent_tokens.append(int(tok))
            if len(recent_tokens) > RUNAWAY_WINDOW:
                recent_tokens.pop(0)
        n_emitted += 1

        # Check every 20 tokens once we have a full window.
        if (
            len(recent_tokens) == RUNAWAY_WINDOW
            and n_emitted % 20 == 0
        ):
            counts: dict[int, int] = {}
            for t in recent_tokens:
                counts[t] = counts.get(t, 0) + 1
            max_count = max(counts.values()) if counts else 0
            n_unique = len(counts)
            # Hard repeat: same token id ≥N times in window.
            # Soft loop: the model is cycling a small vocabulary
            # (e.g. "Luminous. Luminosities. Symmetries.").
            if (
                max_count >= RUNAWAY_REPEAT_THRESHOLD
                or n_unique < RUNAWAY_MIN_UNIQUE
            ):
                aborted_runaway = True
                break

    out = "".join(pieces).strip()
    if aborted_runaway:
        # Tag the response so downstream parsing/judging doesn't treat
        # the truncated tail as a real answer.
        out = out + "\n[[runaway_abort]]"
    return out


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
