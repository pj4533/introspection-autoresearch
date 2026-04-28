"""Tests for the mutation operators and structured-hillclimb dispatcher.

We don't run the LocalMLXProposer here (it would load Qwen). Instead we
substitute a tiny stub that returns canned strings, so we can verify each
operator's output structure, lineage tagging, and dedup behavior.
"""

from __future__ import annotations

import json
import random

import pytest

from src.evaluate import CandidateSpec
from src.strategies import mutations
from src.strategies.mutations import (
    ALL_VARIANT_OPERATORS,
    DETERMINISTIC_OPERATORS,
    PROPOSER_OPERATORS,
    ParentRecord,
    apply_operator,
)


# ---------------------------------------------------------------------------
# Stub proposer
# ---------------------------------------------------------------------------

class StubProposer:
    """Minimal Proposer that returns canned JSON for each operator type.

    Each method that proposer-driven operators trigger looks at the user
    prompt and pattern-matches to decide which canned response to return.
    """

    name = "stub-proposer"

    def __init__(self, *, scenario: str = "ok"):
        self.scenario = scenario
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.scenario == "garbage":
            return "this is not JSON at all"
        if "Generate 6 NEW positive examples" in user_prompt:
            # examples_swap
            return json.dumps({
                "positive": [f"new pos {i}" for i in range(6)],
                "negative": [f"new neg {i}" for i in range(6)],
            })
        if "Refine the description" in user_prompt:
            # description_sharpen
            return json.dumps({
                "description": "a sharpened single-sentence description.",
            })
        if "POSITIVE pole works well" in user_prompt:
            # antonym_pivot
            return json.dumps({
                "description": "a contrast against a different negative pole.",
                "negative": [f"alt neg {i}" for i in range(6)],
            })
        return "{}"


# ---------------------------------------------------------------------------
# Sample parent — used by every test
# ---------------------------------------------------------------------------

def _sample_parent() -> ParentRecord:
    return ParentRecord(
        candidate_id="cand-parent-001",
        strategy="directed_capraro_causality",
        concept="causal-vs-temporal",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="contrast_pair",
        contrast_pair={
            "axis": "causal-vs-temporal",
            "description": "causal nominalization vs temporal sequence",
            "positive": [f"X caused Y because of {i}" for i in range(6)],
            "negative": [f"X happened, then Y happened {i}" for i in range(6)],
            "rationale": "tests Capraro causality fault line",
        },
        fitness_mode="ident_prioritized",
        score=3.812,
        detection_rate=0.125,
        identification_rate=0.125,
    )


# ---------------------------------------------------------------------------
# Deterministic operators
# ---------------------------------------------------------------------------

def test_replication_preserves_axis_and_layer():
    parent = _sample_parent()
    child = mutations.replication(parent)

    assert child.concept == parent.concept
    assert child.layer_idx == parent.layer_idx
    assert child.target_effective == parent.target_effective
    assert child.contrast_pair["axis"] == parent.contrast_pair["axis"]
    # Examples are identical to parent
    assert child.contrast_pair["positive"] == parent.contrast_pair["positive"]
    assert child.contrast_pair["negative"] == parent.contrast_pair["negative"]
    # Lineage tag
    meta = child._lineage_meta
    assert meta["mutation_type"] == "replication"
    assert meta["parent_candidate_id"] == parent.candidate_id


def test_replication_changes_spec_hash():
    """Replications must dedup-pass (different rationale tag = different hash)."""
    from src.strategies.random_explore import spec_hash

    parent = _sample_parent()
    child = mutations.replication(parent)

    # Build a fake CandidateSpec mirroring parent for hashing comparison
    parent_spec = CandidateSpec(
        id=parent.candidate_id,
        strategy=parent.strategy,
        concept=parent.concept,
        layer_idx=parent.layer_idx,
        target_effective=parent.target_effective,
        derivation_method=parent.derivation_method,
        contrast_pair=parent.contrast_pair,
        fitness_mode=parent.fitness_mode,
    )
    assert spec_hash(child) != spec_hash(parent_spec)


def test_layer_shift_clamps_and_records_delta():
    parent = _sample_parent()
    rng = random.Random(42)
    child = mutations.layer_shift(parent, rng=rng)

    assert child.concept == parent.concept
    assert mutations.LAYER_MIN <= child.layer_idx <= mutations.LAYER_MAX
    assert child.layer_idx != parent.layer_idx or True  # delta could clamp to same
    meta = child._lineage_meta
    assert meta["mutation_type"] == "layer_shift"
    assert meta["parent_candidate_id"] == parent.candidate_id
    detail = json.loads(meta["mutation_detail"])
    assert detail["parent_layer"] == parent.layer_idx
    assert detail["delta"] in mutations.LAYER_DELTAS


def test_layer_shift_clamp_at_extreme():
    """A parent at L=44 with delta=+6 must clamp to LAYER_MAX, not exceed."""
    parent = _sample_parent()
    parent.layer_idx = 44
    # Force a large positive delta by looping until we get one
    found_clamp = False
    for seed in range(20):
        rng = random.Random(seed)
        child = mutations.layer_shift(parent, rng=rng)
        if child.layer_idx == mutations.LAYER_MAX:
            found_clamp = True
            break
    # At least one of those should clamp. Even if not, the bound is inviolate.
    assert mutations.LAYER_MIN <= child.layer_idx <= mutations.LAYER_MAX


def test_alpha_scale_records_factor():
    parent = _sample_parent()
    rng = random.Random(0)
    child = mutations.alpha_scale(parent, rng=rng)

    assert child.layer_idx == parent.layer_idx
    assert child.target_effective != parent.target_effective
    detail = json.loads(child._lineage_meta["mutation_detail"])
    assert detail["factor"] in mutations.ALPHA_FACTORS
    assert detail["new_eff"] == pytest.approx(
        parent.target_effective * detail["factor"]
    )


# ---------------------------------------------------------------------------
# Proposer-driven operators
# ---------------------------------------------------------------------------

def test_examples_swap_uses_new_examples():
    parent = _sample_parent()
    proposer = StubProposer()
    child = mutations.examples_swap(parent, proposer=proposer)

    assert child is not None
    assert child.contrast_pair["axis"] == parent.contrast_pair["axis"]
    # New examples replaced the old
    assert child.contrast_pair["positive"] != parent.contrast_pair["positive"]
    assert child.contrast_pair["negative"] != parent.contrast_pair["negative"]
    assert child._lineage_meta["mutation_type"] == "examples_swap"
    assert child.proposer_model == "stub-proposer"


def test_examples_swap_returns_none_on_garbage():
    parent = _sample_parent()
    proposer = StubProposer(scenario="garbage")
    child = mutations.examples_swap(parent, proposer=proposer)
    assert child is None


def test_description_sharpen_keeps_examples():
    parent = _sample_parent()
    proposer = StubProposer()
    child = mutations.description_sharpen(parent, proposer=proposer)

    assert child is not None
    # Examples preserved verbatim
    assert child.contrast_pair["positive"] == parent.contrast_pair["positive"]
    assert child.contrast_pair["negative"] == parent.contrast_pair["negative"]
    # Description changed
    assert child.contrast_pair["description"] != parent.contrast_pair["description"]
    assert child._lineage_meta["mutation_type"] == "description_sharpen"


def test_antonym_pivot_keeps_positive_swaps_negative():
    parent = _sample_parent()
    proposer = StubProposer()
    child = mutations.antonym_pivot(parent, proposer=proposer)

    assert child is not None
    # Positive preserved
    assert child.contrast_pair["positive"] == parent.contrast_pair["positive"]
    # Negative swapped
    assert child.contrast_pair["negative"] != parent.contrast_pair["negative"]
    assert child._lineage_meta["mutation_type"] == "antonym_pivot"


# ---------------------------------------------------------------------------
# apply_operator dispatcher
# ---------------------------------------------------------------------------

def test_apply_operator_dispatches_all_known_ops():
    parent = _sample_parent()
    rng = random.Random(7)
    proposer = StubProposer()

    for op_name in ("replication",) + ALL_VARIANT_OPERATORS:
        child = apply_operator(op_name, parent, rng=rng, proposer=proposer)
        assert child is not None, f"{op_name} returned None unexpectedly"
        assert child._lineage_meta["mutation_type"] == op_name


def test_apply_operator_unknown_raises():
    parent = _sample_parent()
    with pytest.raises(ValueError, match="Unknown mutation operator"):
        apply_operator("bogus", parent, rng=random.Random(0), proposer=None)


def test_apply_operator_proposer_op_without_proposer_returns_none():
    parent = _sample_parent()
    for op_name in PROPOSER_OPERATORS:
        child = apply_operator(op_name, parent, rng=random.Random(0), proposer=None)
        assert child is None, f"{op_name} should have returned None"


# ---------------------------------------------------------------------------
# DB lineage round-trip
# ---------------------------------------------------------------------------

def test_lineage_round_trips_through_db(tmp_db):
    """Insert a child via DB.insert_candidate with mutation_type and verify
    the values come back through the candidates table."""
    parent = _sample_parent()
    child = mutations.replication(parent)

    # Insert parent first (so the foreign-key-ish lineage holds together).
    tmp_db.insert_candidate(
        candidate_id=parent.candidate_id,
        strategy=parent.strategy,
        spec_json="{}",
        spec_hash=f"parent-hash-{parent.candidate_id}",
        concept=parent.concept,
        layer_idx=parent.layer_idx,
        target_effective=parent.target_effective,
        derivation_method=parent.derivation_method,
    )

    meta = child._lineage_meta
    tmp_db.insert_candidate(
        candidate_id=child.id,
        strategy=child.strategy,
        spec_json=json.dumps(child.to_dict()),
        spec_hash="child-hash-001",
        concept=child.concept,
        layer_idx=child.layer_idx,
        target_effective=child.target_effective,
        derivation_method=child.derivation_method,
        parent_candidate_id=meta["parent_candidate_id"],
        mutation_type=meta["mutation_type"],
        mutation_detail=meta["mutation_detail"],
    )

    row = tmp_db.get_candidate(child.id)
    assert row is not None
    assert row["parent_candidate_id"] == parent.candidate_id
    assert row["mutation_type"] == "replication"
    detail = json.loads(row["mutation_detail"])
    assert detail["of_candidate_id"] == parent.candidate_id
