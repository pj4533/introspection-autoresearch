"""Local MLX-backed strict-grading judge.

The production judge for the four-phase worker (ADR-019). Uses mlx_lm for
any MLX-format model — Qwen3.6-35B-A3B-8bit is the calibrated default
(see docs/calibration_results_qwen35b.md).

The prompt templates live in `prompts.py` so this module can stay focused
on inference + caching. Verdicts cache in a per-model SQLite namespace
keyed by `(model_tag, response_hash)` so different judges' verdicts never
collide.
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
from .prompts import (
    PROMPT_TEMPLATE_VERSION,
    SYSTEM as _SYSTEM,
    USER_TEMPLATE as _USER_TEMPLATE,
    CONTRAST_USER_TEMPLATE as _CONTRAST_USER_TEMPLATE,
    SAE_FEATURE_USER_TEMPLATE as _SAE_FEATURE_USER_TEMPLATE,
    FREEASSOC_USER_TEMPLATE as _FREEASSOC_USER_TEMPLATE,
    COT_RECOGNITION_TEMPLATE as _COT_RECOGNITION_TEMPLATE,
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


def _sae_response_hash(response: str, auto_interp: str) -> str:
    h = hashlib.sha256()
    h.update(f"sae:{PROMPT_TEMPLATE_VERSION}\x00".encode())
    h.update(f"{auto_interp}\x00".encode())
    h.update(response.encode())
    return h.hexdigest()[:16]


class _JudgeCache:
    """SQLite-backed judgment cache.

    Phase 2g: schema extended with `identification_type` column. Existing
    caches are migrated via ALTER TABLE on first open. Old rows have
    identification_type=NULL which `get()` returns as Python None — matches
    the JudgeResult default for non-SAE judge calls.
    """

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
            # Idempotent migration for the identification_type column.
            cols = {row[1] for row in conn.execute("PRAGMA table_info(judgments)")}
            if "identification_type" not in cols:
                conn.execute("ALTER TABLE judgments ADD COLUMN identification_type TEXT")

    def get(self, key: str) -> Optional[JudgeResult]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT detected, identified, coherent, reasoning, raw, identification_type "
                "FROM judgments WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        d, i, c, reasoning, raw, ident_type = row
        return JudgeResult(
            detected=bool(d),
            identified=bool(i),
            coherent=bool(c),
            reasoning=reasoning,
            raw=raw,
            identification_type=ident_type,
        )

    def put(self, key: str, model: str, r: JudgeResult) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO judgments "
                "(key, model, detected, identified, coherent, reasoning, raw, identification_type) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, model, int(r.detected), int(r.identified),
                 int(r.coherent), r.reasoning, r.raw, r.identification_type),
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
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        verbose: bool = False,
        enable_thinking: bool = False,
    ):
        self.model_path = model_path
        # Stable short tag for cache namespacing — last path component is fine.
        self.model_tag = Path(model_path).name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.verbose = verbose
        # Many recent MLX models (Qwen3.x, GLM-4.x) emit verbose <think>...
        # </think> blocks by default that blow past max_new_tokens before the
        # JSON arrives. The strict-grading prompt asks for a JSON-only answer,
        # so default to thinking=False. Override per-model if a model NEEDS
        # thinking mode to grade reliably.
        self.enable_thinking = enable_thinking

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
        # Try to pass enable_thinking; some templates don't accept the kwarg
        # (older / non-Qwen models). Fall back gracefully.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
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
        ident_type_raw = obj.get("identification_type")
        ident_type: Optional[str] = None
        if isinstance(ident_type_raw, str) and ident_type_raw in {
            "conceptual", "lexical_fallback", "none"
        }:
            ident_type = ident_type_raw
        return JudgeResult(
            detected=bool(obj.get("detected", False)),
            identified=bool(obj.get("identified", False)),
            coherent=bool(obj.get("coherent", True)),
            reasoning=str(obj.get("reasoning", "")),
            raw=raw,
            identification_type=ident_type,
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

    def score_freeassoc(
        self,
        response: str,
        concept: str,
    ) -> JudgeResult:
        """Phase 3 free-association judge call.

        Grades the response against the concept with a permissive
        identification criterion that accepts semantic neighbors
        (Saccharine for Sugar, Cascades for Avalanches, etc.).
        Returns the standard JudgeResult shape — `identified` is the
        load-bearing field; `detected` is just "did the model produce
        a word at all"; `coherent` is "is the output not garbled."
        """
        # Reuse the contrast-pair cache namespace prefix to avoid
        # collisions with other call types.
        key = f"{self.model_tag}:freeassoc:" + _response_hash(response, concept)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        user = _FREEASSOC_USER_TEMPLATE.format(concept=concept, response=response)
        try:
            raw = self._generate(_SYSTEM, user)
        except Exception as e:
            return JudgeResult(False, False, False,
                                f"judge_error: {type(e).__name__}: {e}",
                                f"<ERROR: {type(e).__name__}>")
        result = self._parse(raw)
        self.cache.put(key, self.model_tag, result)
        return result

    def score_sae_feature(
        self,
        response: str,
        auto_interp: str,
    ) -> JudgeResult:
        """Phase 2g judge call: emits identification_type sub-field."""
        key = (
            f"{self.model_tag}:sae:"
            f"{_sae_response_hash(response, auto_interp)}"
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        user = _SAE_FEATURE_USER_TEMPLATE.format(
            auto_interp=auto_interp or "(no label provided)",
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

    def _parse_cot_recognition(self, raw: str) -> JudgeResult:
        """Parse the COT_RECOGNITION_TEMPLATE judge output.

        The judge emits {"cot_named": "none"|"named"|"named_with_recognition",
        "evidence_quote": "...", "reasoning": "..."}. We map this onto
        JudgeResult fields:
            identification_type = cot_named  (the load-bearing three-way)
            identified          = (cot_named != "none")  — convenience flag
            detected            = same as identified
            coherent            = True (CoT can be empty; emptiness is "none")
            reasoning           = "evidence_quote: ... | reasoning: ..."
        """
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        candidates = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
        if not candidates:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return JudgeResult(
                    False, False, False,
                    f"unparseable: {raw[:200]}", raw,
                    identification_type="none",
                )
            json_text = m.group(0)
        else:
            json_text = candidates[-1]
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as e:
            return JudgeResult(
                False, False, False, f"json error: {e}", raw,
                identification_type="none",
            )
        cot_named = obj.get("cot_named")
        if cot_named not in {"none", "named", "named_with_recognition"}:
            cot_named = "none"
        is_named = cot_named != "none"
        evidence = str(obj.get("evidence_quote", ""))
        reasoning = str(obj.get("reasoning", ""))
        combined = f"evidence_quote: {evidence} | reasoning: {reasoning}".strip()
        return JudgeResult(
            detected=is_named,
            identified=is_named,
            coherent=True,
            reasoning=combined,
            raw=raw,
            identification_type=cot_named,
        )

    def score_cot_recognition(
        self,
        thought_block: str,
        concept: str,
    ) -> JudgeResult:
        """Phase 4 CoT recognition judge.

        Grades the model's *thought block* (not its final answer) for
        whether the concept was named, and whether a recognition marker
        flagged its anomalous salience. Returns a JudgeResult whose
        `identification_type` is one of {"none", "named",
        "named_with_recognition"} — the load-bearing field for the
        Forbidden Map's recognition_rate.

        Empty thought blocks → "none" (the model skipped CoT, so no
        recognition signal is available).
        """
        # Normalize empty / whitespace-only thought blocks before
        # touching the cache, so identical empty-block hits don't burn
        # judge cycles.
        tb = (thought_block or "").strip()
        if not tb:
            return JudgeResult(
                detected=False, identified=False, coherent=True,
                reasoning="empty_thought_block",
                raw="",
                identification_type="none",
            )

        key = (
            f"{self.model_tag}:cot_recog:"
            f"{_response_hash(tb, concept)}"
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        user = _COT_RECOGNITION_TEMPLATE.format(
            concept=concept, thought_block=tb,
        )
        try:
            raw = self._generate(_SYSTEM, user)
        except Exception as e:
            return JudgeResult(
                False, False, False,
                f"judge_error: {type(e).__name__}: {e}",
                f"<ERROR: {type(e).__name__}>",
                identification_type="none",
            )
        result = self._parse_cot_recognition(raw)
        self.cache.put(key, self.model_tag, result)
        return result
