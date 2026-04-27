"""Local MLX-backed judge — drop-in replacement for ClaudeJudge.

DESIGN INTENT:
This module is for OFF-PIPELINE calibration only as of 2026-04-27.
Nothing in src/worker.py / src/evaluate.py / src/researcher.py imports it.
The live pipeline still goes through claude_judge.ClaudeJudge.

After calibration validates a model (see docs/local_pipeline_plan.md Day 1),
we'll wire this in by editing the worker to switch judges based on a flag.

Uses mlx_lm (Apple MLX inference) for any MLX-format model. The prompt
templates are deliberately copied verbatim from claude_judge so calibration
results are directly comparable to Sonnet's verdicts already in the DB.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from .base import JudgeResult
# Reuse the exact strict prompt templates from the production judge.
from .claude_judge import (
    PROMPT_TEMPLATE_VERSION,
    _SYSTEM,
    _USER_TEMPLATE,
    _CONTRAST_USER_TEMPLATE,
)


def _response_hash(response: str, concept: str) -> str:
    h = hashlib.sha256()
    h.update(f"{PROMPT_TEMPLATE_VERSION}\x00".encode())
    h.update(f"{concept}\x00".encode())
    h.update(response.encode())
    return h.hexdigest()[:16]


def _contrast_response_hash(
    response: str,
    axis: str,
    description: str,
    positive: list[str],
    negative: list[str],
) -> str:
    h = hashlib.sha256()
    h.update(f"contrast:{PROMPT_TEMPLATE_VERSION}\x00".encode())
    h.update(f"{axis}\x00{description}\x00".encode())
    h.update(("|".join(positive) + "\x00" + "|".join(negative)).encode())
    h.update(response.encode())
    return h.hexdigest()[:16]


class _JudgeCache:
    """SQLite-backed judgment cache, mirrors claude_judge._JudgeCache schema
    so the same schema can later be unified."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS judgments (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    detected INTEGER NOT NULL,
                    identified INTEGER NOT NULL,
                    coherent INTEGER NOT NULL,
                    reasoning TEXT NOT NULL,
                    raw TEXT NOT NULL
                )"""
            )

    def get(self, key: str) -> Optional[JudgeResult]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT detected, identified, coherent, reasoning, raw FROM judgments WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        d, i, c, reasoning, raw = row
        return JudgeResult(bool(d), bool(i), bool(c), reasoning, raw)

    def put(self, key: str, model: str, r: JudgeResult) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judgments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (key, model, int(r.detected), int(r.identified),
                 int(r.coherent), r.reasoning, r.raw),
            )


class LocalMLXJudge:
    """Strict-grading judge backed by an MLX model.

    Loads on first use. Supports any HF MLX repo path.
    Uses chat-template-style prompts via tokenizer.apply_chat_template.

    The model param doubles as the cache namespace so different models
    don't collide in the SQLite cache.
    """

    def __init__(
        self,
        model_path: str,
        cache_path: Optional[Path] = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        self.model_path = model_path
        # Stable short tag for cache namespacing — last path component is fine.
        self.model_tag = Path(model_path).name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.verbose = verbose

        cache_path = cache_path or Path(
            f"data/calibration/judge_cache_{self.model_tag}.sqlite"
        )
        self.cache = _JudgeCache(cache_path)

        # Lazy-load. mlx_lm import is deferred so importing this module
        # doesn't require mlx to be installed.
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "mlx-lm not installed. Run: pip install -U mlx mlx-lm"
            ) from e
        if self.verbose:
            print(f"[local_mlx_judge] loading {self.model_path} ...", flush=True)
        self._model, self._tokenizer = load(self.model_path)
        if self.verbose:
            print(f"[local_mlx_judge] loaded.", flush=True)

    def _generate(self, system: str, user: str) -> str:
        """Run the model on a system+user chat prompt, return raw text."""
        self._ensure_loaded()
        from mlx_lm import generate  # type: ignore
        from mlx_lm.sample_utils import make_sampler  # type: ignore

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampler = make_sampler(temp=self.temperature)
        text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )
        return text.strip()

    def _parse(self, raw: str) -> JudgeResult:
        # Some thinking-mode models emit `<think>...</think>` blocks before
        # the JSON. Strip them. Then take the LAST JSON object in the text
        # (in case the model emits multiple JSON-like fragments inside its
        # reasoning).
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Find all top-level JSON object spans, take the last one.
        candidates = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
        if not candidates:
            # Fallback: the simple regex. Models that emit nested JSON in
            # their reasoning would have failed the bracket-balance match
            # above; try the looser one.
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return JudgeResult(False, False, False,
                                    f"unparseable: {raw[:200]}", raw)
            json_text = m.group(0)
        else:
            json_text = candidates[-1]
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as e:
            return JudgeResult(False, False, False, f"json error: {e}", raw)
        return JudgeResult(
            detected=bool(obj.get("detected", False)),
            identified=bool(obj.get("identified", False)),
            coherent=bool(obj.get("coherent", True)),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
        )

    def _cache_key(self, response: str, concept: str) -> str:
        return f"{self.model_tag}:{_response_hash(response, concept)}"

    def _contrast_cache_key(
        self,
        response: str,
        axis: str,
        description: str,
        positive: list[str],
        negative: list[str],
    ) -> str:
        return (
            f"{self.model_tag}:contrast:"
            f"{_contrast_response_hash(response, axis, description, positive, negative)}"
        )

    def score_detection(self, response: str, concept: str) -> JudgeResult:
        key = self._cache_key(response, concept)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        user = _USER_TEMPLATE.format(concept=concept, response=response)
        try:
            raw = self._generate(_SYSTEM, user)
        except Exception as e:
            return JudgeResult(False, False, False,
                                f"judge_error: {type(e).__name__}: {e}",
                                f"<ERROR: {type(e).__name__}>")
        result = self._parse(raw)
        self.cache.put(key, self.model_tag, result)
        return result

    def score_contrast_pair(
        self,
        response: str,
        axis: str,
        description: str,
        positive: list[str],
        negative: list[str],
    ) -> JudgeResult:
        key = self._contrast_cache_key(response, axis, description, positive, negative)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        positive_block = "\n".join(f'  {i+1}. "{s}"' for i, s in enumerate(positive))
        negative_block = "\n".join(f'  {i+1}. "{s}"' for i, s in enumerate(negative))
        user = _CONTRAST_USER_TEMPLATE.format(
            axis=axis,
            description=description or "(no description provided)",
            positive_block=positive_block,
            negative_block=negative_block,
            response=response,
        )
        try:
            raw = self._generate(_SYSTEM, user)
        except Exception as e:
            return JudgeResult(False, False, False,
                                f"judge_error: {type(e).__name__}: {e}",
                                f"<ERROR: {type(e).__name__}>")
        result = self._parse(raw)
        self.cache.put(key, self.model_tag, result)
        return result
