"""Tests for src/phase4/cot_parser.py — runs against synthetic shapes
and against real Phase 3 saved generations from data/results.db."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.phase4.cot_parser import parse, extract_committed_word, is_coherent_answer


def test_standard_shape():
    r = parse("<|channel>thought\n*  Some thinking\n*  More thinking<channel|>Bread")
    assert r.parse_failure is False
    assert "Some thinking" in r.thought_block
    assert "More thinking" in r.thought_block
    assert r.final_answer == "Bread"


def test_truncated_no_close():
    r = parse("<|channel>thought\n* Some thinking that got cut")
    assert r.parse_failure is False
    assert "Some thinking" in r.thought_block
    assert r.final_answer == ""


def test_no_cot_at_all():
    r = parse("Bread.")
    assert r.parse_failure is False
    assert r.thought_block == ""
    assert r.final_answer == "Bread."


def test_close_only_variant():
    r = parse("<|channel>\n*  Reasoning here<channel|>Sourdough")
    assert r.parse_failure is False
    assert "Reasoning here" in r.thought_block
    assert r.final_answer == "Sourdough"


def test_bare_thought_header():
    r = parse("thought\n*  Some reasoning that never closes")
    assert r.parse_failure is False
    assert "Some reasoning" in r.thought_block
    assert r.final_answer == ""


def test_empty_input():
    r = parse("")
    assert r.parse_failure is True
    r2 = parse(None)
    assert r2.parse_failure is True


def test_extract_committed_word_basic():
    assert extract_committed_word("Bread") == "Bread"
    assert extract_committed_word("Bread.") == "Bread"
    assert extract_committed_word("*Bread*") == "Bread"
    assert extract_committed_word("**Sourdough**") == "Sourdough"
    assert extract_committed_word('"Nebula"') == "Nebula"
    assert extract_committed_word("Bread (warm)") == "Bread"


def test_extract_committed_word_rejects_garbage():
    assert extract_committed_word("") is None
    assert extract_committed_word("...") is None
    assert extract_committed_word("123") is None
    # Reasonably long plausible word should still pass; only reject if absurdly long
    assert extract_committed_word("x" * 50) is None


def test_is_coherent_answer_accepts_clean():
    """A coherent free-association answer is exactly ONE word, possibly
    wrapped in markdown emphasis or trailing punctuation. Empty
    answers are passed through (the chain ends naturally because
    there's no word to commit to)."""
    assert is_coherent_answer("Bread") is True
    assert is_coherent_answer("Bread.") is True
    assert is_coherent_answer("*Bread*") is True
    assert is_coherent_answer('"Bread"') is True
    assert is_coherent_answer("_Bread_.") is True
    assert is_coherent_answer("") is True  # empty handled separately
    assert is_coherent_answer(None) is True


def test_is_coherent_answer_rejects_runaway_marker():
    """Any output that hit the runaway-abort marker is by definition
    incoherent — the model was looping when generation was killed."""
    assert is_coherent_answer("Resonance Resonance [[runaway_abort]]") is False
    assert is_coherent_answer("[[runaway_abort]]") is False


def test_is_coherent_answer_rejects_runaway_repetition():
    """Real Gemma 4 runaway patterns we observed."""
    assert is_coherent_answer(
        "Now Thoughts. Now Thoughts. Now Thoughts. Now Thoughts. "
        "Now Thoughts. Now Words. Now Words. Now Words. Now Words. "
        "Now Words. Now Wait... Now Words. Now Words."
    ) is False
    assert is_coherent_answer(
        "Nowhere. Nowhere. Nowhere. Now. Now. Now. Now. Now. Now. "
        "Now. Now. Now. Now. Now. Now."
    ) is False


def test_is_coherent_answer_rejects_run_on_prose():
    long_prose = " ".join(["word"] + ["alpha"] * 35)
    assert is_coherent_answer(long_prose) is False


def test_is_coherent_answer_rejects_no_separator_soup():
    """Concatenated single-word soup like 'NowaynowherenowhereNowhere...'"""
    soup = "Nowaynowherenowherenowherenowherenowherenowherenowhere"
    assert is_coherent_answer(soup) is False


def test_is_coherent_answer_rejects_multi_word():
    """The probe explicitly asks for one word, no explanation. Any
    multi-word answer is a prompt-failure and we treat it as
    incoherent, even if the answer 'looks reasonable'. Otherwise the
    behavior_rate gets inflated by the concept word appearing inside
    a sentence the model wasn't asked to write."""
    assert is_coherent_answer("I'd say Bread") is False
    assert is_coherent_answer("Word: Bread") is False
    assert is_coherent_answer("Bread, warm") is False


def test_extract_committed_word_takes_first_only():
    """For multi-word emissions, take the first token. The dream walk
    chooses next target from the first committed word."""
    # "I think bread" — first word is "I" but that's a real word
    assert extract_committed_word("I think bread") == "I"
    # That edge case is fine; the loop should also check "is this in
    # the seed pool / a real concept" before injection.


def test_against_phase3_saved_responses():
    """Parse every Phase 3 injected response in the live DB. None should
    fail catastrophically. At least one should have a non-empty
    final_answer (Gemma 4 successfully closed the channel)."""
    db_path = REPO / "data" / "results.db"
    if not db_path.exists():
        # Tests run on fresh checkouts without a DB — that's fine.
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT response FROM evaluations e "
        "JOIN candidates c ON c.id = e.candidate_id "
        "WHERE c.gemma_model='gemma4_31b' AND e.injected=1 "
        "LIMIT 100"
    ).fetchall()
    conn.close()

    if not rows:
        return  # No Phase 3 data yet

    n_with_answer = 0
    n_with_thought = 0
    for (resp,) in rows:
        p = parse(resp)
        assert p.parse_failure is False, f"parse failed on real response: {resp[:200]!r}"
        if p.final_answer:
            n_with_answer += 1
        if p.thought_block:
            n_with_thought += 1

    # We expect most Phase 3 generations to have BOTH a thought block and a final answer.
    # Loose bound: at least 30% should have a final answer (Gemma 4 sometimes
    # truncates mid-thought at max_new_tokens=400).
    assert n_with_answer >= 0.3 * len(rows), (
        f"only {n_with_answer}/{len(rows)} responses had a final answer — "
        f"parser may be over-eager classifying answer as part of thought"
    )
    assert n_with_thought >= 0.5 * len(rows), (
        f"only {n_with_thought}/{len(rows)} had a thought block"
    )
