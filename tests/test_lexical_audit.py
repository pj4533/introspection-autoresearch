"""Tests for the lexical contamination audit and decontaminate operator."""

from __future__ import annotations

import json

import pytest

from src.strategies.lexical_audit import (
    ContaminationReport,
    banned_tokens_for_decontamination,
    lexical_contamination,
)


# ---------------------------------------------------------------------------
# lexical_contamination
# ---------------------------------------------------------------------------

def test_clean_pair_reports_no_contamination():
    """A pair where every content word is varied is clean."""
    pair = {
        "positive": [
            "The boulder rolled down the hillside.",
            "Eagles soared above pine forests.",
            "Coffee aroma filled the morning kitchen.",
            "Snowflakes settled on weathered fenceposts.",
            "Steel girders supported the bridge span.",
            "Lavender grew along the garden path.",
        ],
        "negative": [
            "Children played in the schoolyard quietly.",
            "Trains rumbled through small mountain towns.",
            "Doctors consulted in the hospital corridor.",
            "Wind chimes rang on the porch.",
            "Bakers kneaded dough in the morning.",
            "Books lined the dusty wooden shelves.",
        ],
    }
    report = lexical_contamination(pair)
    assert report.is_clean
    assert not report.is_contaminated
    assert report.positive_exclusive == []
    assert report.negative_exclusive == []
    assert report.high_skew == []


def test_mixed_causal_positives_no_single_token_dominates():
    """If positives use a MIX of causal markers ('caused'/'because'/etc) and
    no single token hits 6/6, the audit should report 'clean' for those
    tokens — even though semantically the contrast is still causation.
    This documents a known limitation of token-level audit: contamination
    has to be attributable to a single repeated token to fire."""
    pair = {
        "positive": [
            "The rain caused the flood.",
            "She knocked the glass because she reached suddenly.",
            "The engine failed because the bolt was missing.",
            "His warning caused the change.",
            "Because the code changed, the tests broke.",
            "The decision was made because the data shifted.",
        ],
        "negative": [
            "The rain happened. Later, the flood arrived.",
            "She reached suddenly. The glass fell after.",
            "The bolt was missing. The engine failed afterward.",
            "His warning was given. The plan changed later.",
            "The code changed. The tests broke subsequently.",
            "The data shifted. The decision was made later.",
        ],
    }
    report = lexical_contamination(pair)
    # No single positive-pole token hits 6/6 since the proposer used
    # mixed causal markers. The audit may flag negative-pole exclusives
    # (e.g. "later", "afterward") instead. We just assert the report is
    # well-formed — the limitation is documented, not a bug.
    assert report.positive_total == 6
    assert report.negative_total == 6


def test_first_person_pronoun_contamination():
    """The exact failure mode from self-evaluation-vs-objective-evaluation:
    every positive starts with 'I', every negative is third-person."""
    pair = {
        "positive": [
            "I judge my performance to be adequate.",
            "I assess my progress as sufficient.",
            "I rate my effort as high.",
            "I evaluate my output as accurate.",
            "I consider my attempt successful.",
            "I deem my work satisfactory.",
        ],
        "negative": [
            "The performance is adequate.",
            "The progress is sufficient.",
            "The effort is high.",
            "The output is accurate.",
            "The attempt is successful.",
            "The work is satisfactory.",
        ],
    }
    report = lexical_contamination(pair)
    assert report.is_contaminated
    # 'i' is filtered (1 char) — but 'my' should be flagged
    assert "my" in report.positive_exclusive


def test_pure_caused_token_contamination():
    """Direct test of the original causality Class 1 contrast pair —
    every positive has 'caused', every negative has 'happened'."""
    pair = {
        "positive": [
            "Her shouting caused the alarm to sound.",
            "The heating caused the wax to melt.",
            "His leaving caused the room to empty.",
            "The rain caused the soil to erode.",
            "The shock caused the system to restart.",
            "The news caused the crowd to cheer.",
        ],
        "negative": [
            "She shouted, then the alarm sounded.",
            "The heat applied, then the wax melted.",
            "He left, then the room emptied.",
            "It rained, then the soil eroded.",
            "The shock hit, then the system restarted.",
            "The news broke, then the crowd cheered.",
        ],
    }
    report = lexical_contamination(pair)
    assert report.is_contaminated
    assert "caused" in report.positive_exclusive
    assert "then" in report.negative_exclusive


def test_high_skew_short_of_total():
    """Token in 5/6 positives and 0/6 negatives is high-skew, not exclusive."""
    pair = {
        "positive": [
            "The marathon endurance impressed everyone.",
            "Her endurance carried the team forward.",
            "Endurance separated the finalists.",
            "Their endurance defied expectations.",
            "Without endurance, athletes fail.",
            "Resolve, not strength, wins long contests.",
        ],
        "negative": [
            "Players gathered for the friendly match.",
            "The crowd cheered the local team.",
            "Coaches reviewed video after each game.",
            "Spectators wandered between concession stands.",
            "Children traded baseball cards in the bleachers.",
            "Stadium lights flickered on at dusk.",
        ],
    }
    report = lexical_contamination(pair)
    # 'endurance' is in 5/6 positives, 0/6 negatives → high_skew
    assert any(t == "endurance" for t, _, _ in report.high_skew)
    assert "endurance" not in report.positive_exclusive


def test_empty_pair_returns_empty_report():
    report = lexical_contamination({"positive": [], "negative": []})
    assert report.is_clean
    assert not report.is_contaminated


def test_summary_clean():
    pair = {
        "positive": ["alpha bravo", "charlie delta", "echo foxtrot",
                      "golf hotel", "india juliet", "kilo lima"],
        "negative": ["mike november", "oscar papa", "quebec romeo",
                      "sierra tango", "uniform victor", "whiskey xray"],
    }
    assert lexical_contamination(pair).summary() == "lexically clean"


def test_summary_contaminated():
    pair = {
        "positive": [
            "X caused Y.", "A caused B.", "P caused Q.",
            "M caused N.", "R caused S.", "U caused V.",
        ],
        "negative": [
            "X then Y.", "A then B.", "P then Q.",
            "M then N.", "R then S.", "U then V.",
        ],
    }
    summary = lexical_contamination(pair).summary()
    assert "caused" in summary
    assert "+pole exclusive" in summary


# ---------------------------------------------------------------------------
# banned_tokens_for_decontamination
# ---------------------------------------------------------------------------

def test_banned_tokens_unions_both_pole_exclusives():
    pair = {
        "positive": ["X caused Y."] * 6,
        "negative": ["X happened then Y."] * 6,
    }
    report = lexical_contamination(pair)
    banned = banned_tokens_for_decontamination(report)
    # 'caused' from positive-exclusive, 'then' / 'happened' from negative-exclusive
    assert "caused" in banned
    assert "then" in banned or "happened" in banned


def test_banned_tokens_empty_when_clean():
    pair = {
        "positive": ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"],
        "negative": ["golf", "hotel", "india", "juliet", "kilo", "lima"],
    }
    report = lexical_contamination(pair)
    assert banned_tokens_for_decontamination(report) == []


