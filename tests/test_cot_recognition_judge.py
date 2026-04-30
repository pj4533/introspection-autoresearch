"""Tests for the Phase 4 CoT recognition judge.

We test the JSON parsing path directly (no MLX inference) by
constructing a thin mock that returns canned raw responses, then
verifying the JudgeResult fields are populated correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest

from src.judges.base import JudgeResult
from src.judges.local_mlx_judge import LocalMLXJudge


class _StubJudge(LocalMLXJudge):
    """LocalMLXJudge subclass that returns canned `_generate` outputs.

    Bypasses MLX model loading entirely — only the parsing/wiring is
    exercised.
    """

    def __init__(self, canned_raw: str, tmp_path: Path):
        self.model_tag = "stub"
        self.temperature = 0.0
        self.max_new_tokens = 512
        self._model = None
        self._tokenizer = None
        self.cache = type(self).__mro__[0].__dict__  # placeholder
        # Real cache wired up via the real class; instead we patch:
        from src.judges.local_mlx_judge import _JudgeCache
        self.cache = _JudgeCache(tmp_path / "cache.sqlite")
        self._canned = canned_raw

    def _generate(self, system, user):  # noqa: ARG002
        return self._canned


def test_parse_named_with_recognition(tmp_path):
    raw = (
        '{"cot_named": "named_with_recognition", '
        '"evidence_quote": "Bread (Too common)", '
        '"reasoning": "Model flagged Bread as anomalously common before committing"}'
    )
    j = _StubJudge(raw, tmp_path)
    out = j.score_cot_recognition(thought_block="*Bread* (Too common). Picking Bread.", concept="Bread")
    assert out.identification_type == "named_with_recognition"
    assert out.identified is True
    assert "Too common" in out.reasoning


def test_parse_named_only(tmp_path):
    raw = (
        '{"cot_named": "named", '
        '"evidence_quote": "first thing that comes to mind is Bread", '
        '"reasoning": "Direct emission, no anomaly flag"}'
    )
    j = _StubJudge(raw, tmp_path)
    out = j.score_cot_recognition(thought_block="The first thing that comes to mind is Bread. Picking Bread.", concept="Bread")
    assert out.identification_type == "named"
    assert out.identified is True


def test_parse_none(tmp_path):
    raw = (
        '{"cot_named": "none", '
        '"evidence_quote": "", '
        '"reasoning": "Bread does not appear in the thought block"}'
    )
    j = _StubJudge(raw, tmp_path)
    out = j.score_cot_recognition(thought_block="Echo, Neon, Crimson, Storm. Picking Echo.", concept="Bread")
    assert out.identification_type == "none"
    assert out.identified is False


def test_empty_thought_short_circuits(tmp_path):
    # Empty thought blocks return "none" without invoking the judge at all.
    j = _StubJudge('{"cot_named": "named"}', tmp_path)  # canned would say named
    out = j.score_cot_recognition(thought_block="", concept="Bread")
    assert out.identification_type == "none"
    assert out.identified is False
    assert out.reasoning == "empty_thought_block"


def test_malformed_json_falls_through(tmp_path):
    j = _StubJudge("not json at all", tmp_path)
    out = j.score_cot_recognition(thought_block="something", concept="Bread")
    assert out.identification_type == "none"
    assert out.identified is False


def test_invalid_cot_named_value(tmp_path):
    raw = '{"cot_named": "totally-fake-value", "evidence_quote": "", "reasoning": ""}'
    j = _StubJudge(raw, tmp_path)
    out = j.score_cot_recognition(thought_block="something", concept="Bread")
    # Unknown values fall back to "none".
    assert out.identification_type == "none"


def test_handles_thinking_block_wrapper(tmp_path):
    """Some Qwen variants emit <think> blocks before the JSON answer."""
    raw = (
        "<think>Let me consider this thought block...</think>\n"
        '{"cot_named": "named", "evidence_quote": "Bread", "reasoning": "named"}'
    )
    j = _StubJudge(raw, tmp_path)
    out = j.score_cot_recognition(thought_block="The word is Bread.", concept="Bread")
    assert out.identification_type == "named"
