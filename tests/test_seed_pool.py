"""Tests for src/phase4/seed_pool.py."""

from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.db import ResultsDB
from src.phase4.seed_pool import (
    SeedPool, normalize_lemma, load_default_seeds,
    CODEX_SUPPRESSED_CREATURES,
)


def test_normalize_lemma_basic():
    assert normalize_lemma("Bread") == "bread"
    assert normalize_lemma("BREAD!") == "bread"
    assert normalize_lemma("  Bread.  ") == "bread"


def test_normalize_lemma_plurals():
    assert normalize_lemma("Breads") == "bread"
    assert normalize_lemma("Berries") == "berry"
    assert normalize_lemma("Avalanches") == "avalanche"
    assert normalize_lemma("Goblins") == "goblin"


def test_normalize_lemma_rejects_garbage():
    assert normalize_lemma("") == ""
    assert normalize_lemma("a") == ""  # too short
    assert normalize_lemma("123") == ""
    assert normalize_lemma(None) == ""


def test_normalize_lemma_rejects_non_ascii():
    """Chinese, emoji, accented characters etc. should not become lemmas."""
    assert normalize_lemma("这是一个") == ""
    assert normalize_lemma("café") == ""
    assert normalize_lemma("naïve") == ""
    assert normalize_lemma("🍞") == ""


def test_normalize_lemma_rejects_bracket_leaks():
    """Token leaks like <thought>, <channel|>, }**Now should be rejected."""
    assert normalize_lemma("<thought>") == ""
    assert normalize_lemma("<channel|>") == ""
    assert normalize_lemma("}**Now") == ""
    assert normalize_lemma("Bread/Salt") == ""
    assert normalize_lemma("[brackets]") == ""


def test_normalize_lemma_rejects_non_concept_lemmas():
    """Adverbs and discourse markers are rejected."""
    assert normalize_lemma("currently") == ""
    assert normalize_lemma("hindsight") == ""
    assert normalize_lemma("Wait") == ""
    assert normalize_lemma("Now") == ""
    assert normalize_lemma("nowhere") == ""
    assert normalize_lemma("sincerely") == ""
    assert normalize_lemma("their") == ""
    # but real concepts that happen to start similarly still pass
    assert normalize_lemma("Watering") == "watering"


def test_codex_creatures_first(tmp_path):
    seeds = load_default_seeds(REPO)
    # The first six lemmas should be the Codex-suppressed creatures, in order.
    expected_first = [normalize_lemma(c) for c in CODEX_SUPPRESSED_CREATURES]
    actual_first = [normalize_lemma(s) for s in seeds[:6]]
    assert actual_first == expected_first
    assert actual_first[0] == "goblin"


def test_pool_force_lemma(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    pool = SeedPool(db, initial_seeds=["Goblin", "Bread", "Sugar"], rng=random.Random(0))
    forced = pool.sample_seed(force_lemma="goblin")
    assert forced.lemma == "goblin"
    assert forced.display == "Goblin"


def test_pool_priority_favors_unseen(tmp_path):
    """A concept with 0 visits should be picked far more often than one
    with many visits, given equal-weight sampling."""
    db = ResultsDB(tmp_path / "test.db")
    pool = SeedPool(
        db,
        initial_seeds=["Goblin", "Bread"],
        rng=random.Random(0),
        w_novelty=10.0, w_variance=0.0, w_recency=0.0,
    )
    # Make Bread heavily visited.
    for _ in range(100):
        db.increment_phase4_concept_visit("bread")

    counts = {"goblin": 0, "bread": 0}
    for _ in range(200):
        c = pool.sample_seed()
        counts[c.lemma] += 1
    # With priority = 1/(visits+1), Goblin (1.0) vs Bread (1/101) ≈ 100×.
    assert counts["goblin"] > counts["bread"] * 5


def test_register_observed_concept(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    pool = SeedPool(db, initial_seeds=["Goblin"])
    lemma = pool.register_observed_concept("Sourdough")
    assert lemma == "sourdough"
    # Subsequent registration of a plural form should normalize to same lemma.
    lemma2 = pool.register_observed_concept("Sourdoughs")
    assert lemma2 == "sourdough"

    stats = db.get_phase4_concept_stats()
    lemmas = {r["concept_lemma"] for r in stats}
    assert "sourdough" in lemmas
    assert "goblin" in lemmas
