"""Steering-vector derivation strategies.

Phase 1 uses only `mean_diff` via the paper's `extract_concept_vector_with_baseline`.
Additional strategies (pca, caa, median_diff) will be added in Phase 2.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .paper import (
    ModelWrapper,
    extract_concept_vector,
    extract_concept_vector_with_baseline,
)


def mean_diff(
    model: ModelWrapper,
    concept_word: str,
    baseline_words: Iterable[str],
    layer_idx: int,
    token_idx: int = -1,
    normalize: bool = False,
    template: str = "Tell me about {word}",
) -> torch.Tensor:
    return extract_concept_vector_with_baseline(
        model=model,
        concept_word=concept_word,
        baseline_words=list(baseline_words),
        layer_idx=layer_idx,
        template=template,
        token_idx=token_idx,
        normalize=normalize,
    )


def contrastive_mean_diff(
    model: ModelWrapper,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_idx: int,
    token_idx: int = -1,
    normalize: bool = False,
) -> torch.Tensor:
    return extract_concept_vector(
        model=model,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layer_idx=layer_idx,
        token_idx=token_idx,
        normalize=normalize,
    )
