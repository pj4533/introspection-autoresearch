"""Local MLX-backed proposer.

Mirrors the LocalMLXJudge pattern: lazy mlx_lm import, generate via the
tokenizer's chat template, return the full text. Strategies do their own
JSON / regex parsing on the returned string.

Two ways to use:
  1. Pass a model_path; the proposer manages load/unload itself (slow first
     call, fine for one-shot scripts).
  2. Pass a `loaded_pair=(model, tokenizer)` already loaded by an
     `MLXHandle`. This is the path the worker uses — the handle controls
     the lifecycle and the proposer just runs inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple


class LocalMLXProposer:
    name: str

    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        loaded_pair: Optional[Tuple[Any, Any]] = None,
        enable_thinking: bool = False,
    ):
        if (model_path is None) == (loaded_pair is None):
            raise ValueError(
                "exactly one of `model_path` or `loaded_pair` must be provided"
            )
        self.model_path = model_path
        self._pair: Optional[Tuple[Any, Any]] = loaded_pair
        self.name = (
            Path(model_path).name if model_path is not None
            else getattr(loaded_pair[0], "name_or_path", "loaded-mlx-pair")
        )
        # Many recent MLX models emit verbose <think>...</think> blocks by
        # default that overshoot max_tokens before the proposer's actual
        # JSON answer. Default off; flip true if your proposer model needs
        # CoT reasoning to produce good axes.
        self.enable_thinking = enable_thinking

    def _ensure_loaded(self) -> Tuple[Any, Any]:
        if self._pair is not None:
            return self._pair
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run: pip install -U mlx mlx-lm"
            ) from e
        self._pair = load(self.model_path)
        return self._pair

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> str:
        model, tokenizer = self._ensure_loaded()
        from mlx_lm import generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        sampler = make_sampler(temp=temperature)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text.strip()
