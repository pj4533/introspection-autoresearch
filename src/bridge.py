"""Bridge between this project and the safety-research/introspection-mechanisms paper repo.

Wraps model loading so every call site gets consistent MPS + fp16 defaults, and
exposes a small `DetectionPipeline` that composes derive -> inject -> judge for
Phase 1 exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .paper import (
    ModelWrapper,
    extract_concept_vector_with_baseline,
    get_baseline_words,
    get_layer_at_fraction,
    run_steered_introspection_test,
    run_unsteered_introspection_test,
)

from .judges.base import Judge, JudgeResult

DEFAULT_MODEL = "gemma3_12b"
DEFAULT_DEVICE = "mps"
# Gemma3 is natively bf16; fp16 on MPS can underflow and produce NaNs during
# sampling. bf16 is supported by MPS on Apple Silicon and matches the paper.
DEFAULT_DTYPE = torch.bfloat16


def load_gemma_mps(model_name: str = DEFAULT_MODEL) -> ModelWrapper:
    """Load Gemma on the MPS backend with fp16 weights."""
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend not available.")
    return ModelWrapper(model_name=model_name, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE)


@dataclass
class DetectionTrial:
    concept: str
    layer_idx: int
    strength: float
    injected: bool
    response: str
    judge_result: Optional[JudgeResult] = None


class DetectionPipeline:
    """Compose steering-vector derivation, injection, and judgment.

    Single-model, single-process. Not thread-safe. Intended for notebooks and
    Phase 1 sweeps; Phase 2 worker calls the underlying primitives directly so
    it can cache between candidates.
    """

    def __init__(
        self,
        model: ModelWrapper,
        judge: Judge,
        baseline_words: Optional[list[str]] = None,
        baseline_n: int = 32,
    ):
        self.model = model
        self.judge = judge
        self.baseline_words = baseline_words or get_baseline_words(n=baseline_n)

    def derive(self, concept: str, layer_idx: int) -> torch.Tensor:
        return extract_concept_vector_with_baseline(
            model=self.model,
            concept_word=concept,
            baseline_words=self.baseline_words,
            layer_idx=layer_idx,
            token_idx=-1,
            normalize=False,
        )

    def run_injected(
        self,
        concept: str,
        direction: torch.Tensor,
        layer_idx: int,
        strength: float,
        trial_number: int = 1,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        judge_concept: Optional[str] = None,
        prompt_style: str = "paper",
    ) -> DetectionTrial:
        """Run an injected trial.

        `concept` is a label for the trial (stored in DetectionTrial.concept, and
        passed as metadata to run_steered_introspection_test which does NOT
        include it in the prompt). `judge_concept`, if set, overrides what the
        judge is told about the injected concept for grading. When called from
        Phase 2, pass the candidate's SOURCE concept here so identification is
        graded against what was actually injected, not against the slot label.
        """
        response = run_steered_introspection_test(
            model=self.model,
            concept_word=concept,
            steering_vector=direction,
            layer_idx=layer_idx,
            strength=strength,
            trial_number=trial_number,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_style=prompt_style,
        )
        trial = DetectionTrial(
            concept=concept,
            layer_idx=layer_idx,
            strength=strength,
            injected=True,
            response=response,
        )
        trial.judge_result = self.judge.score_detection(
            response, judge_concept if judge_concept is not None else concept
        )
        return trial

    def run_control(
        self,
        concept: str,
        trial_number: int = 1,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        judge_concept: Optional[str] = None,
        prompt_style: str = "paper",
    ) -> DetectionTrial:
        """Run a control (no-injection) trial.

        For controls, `judge_concept` matters less — the response shouldn't
        claim detection, so identification grading is moot. We still accept the
        override for consistency with `run_injected`.
        """
        response = run_unsteered_introspection_test(
            model=self.model,
            concept_word=concept,
            trial_number=trial_number,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt_style=prompt_style,
        )
        trial = DetectionTrial(
            concept=concept,
            layer_idx=-1,
            strength=0.0,
            injected=False,
            response=response,
        )
        trial.judge_result = self.judge.score_detection(
            response, judge_concept if judge_concept is not None else concept
        )
        return trial

    def layer_at_fraction(self, fraction: float) -> int:
        return get_layer_at_fraction(self.model, fraction)
