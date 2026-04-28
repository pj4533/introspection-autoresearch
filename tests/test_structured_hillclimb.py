"""Tests for the structured-hillclimb dispatcher.

Covers:
- Cold start: no winners → falls through to directed_capraro
- With winners: emits replication / variants / cluster_expansion in the
  configured ratio
- Composition env-var override
- Lineage tags propagate to every emitted spec
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.evaluate import CandidateSpec
from src.strategies import structured_hillclimb
from src.strategies.structured_hillclimb import _parse_composition


# ---------------------------------------------------------------------------
# Composition parsing
# ---------------------------------------------------------------------------

def test_parse_composition_default():
    assert _parse_composition(None) == (4, 10, 2)
    assert _parse_composition("") == (4, 10, 2)


def test_parse_composition_valid():
    assert _parse_composition("6:8:2") == (6, 8, 2)
    assert _parse_composition("0:16:0") == (0, 16, 0)


def test_parse_composition_malformed_falls_back(capsys):
    assert _parse_composition("not:valid") == (4, 10, 2)
    assert _parse_composition("1:2:3:4") == (4, 10, 2)
    assert _parse_composition("0:0:0") == (4, 10, 2)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


# ---------------------------------------------------------------------------
# Stub proposer + helpers
# ---------------------------------------------------------------------------

class StubProposer:
    """Returns canned JSON for both directed_capraro and mutation prompts."""

    name = "stub-proposer"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Cluster-expansion calls directed_capraro which expects an ARRAY.
        if "Generate" in user_prompt and "contrast pairs" in user_prompt:
            return json.dumps([{
                "axis": "stub-cluster-axis",
                "description": "stub cluster expansion axis",
                "rationale": "fresh sibling",
                "positive": [f"pos {i}" for i in range(6)],
                "negative": [f"neg {i}" for i in range(6)],
            }])
        # Mutation operators expect single objects
        if "Generate 6 NEW positive examples" in user_prompt:
            return json.dumps({
                "positive": [f"swap pos {i}" for i in range(6)],
                "negative": [f"swap neg {i}" for i in range(6)],
            })
        if "Refine the description" in user_prompt:
            return json.dumps({"description": "sharper desc."})
        if "POSITIVE pole works well" in user_prompt:
            return json.dumps({
                "description": "alt-pivot desc.",
                "negative": [f"alt neg {i}" for i in range(6)],
            })
        return "{}"


def _seed_winner(db, candidate_id: str, axis: str, layer: int, eff: float, score: float):
    """Insert a fake evaluated winner so the dispatcher can pick it up."""
    spec_dict = {
        "id": candidate_id,
        "strategy": "directed_capraro_causality",
        "concept": axis,
        "layer_idx": layer,
        "target_effective": eff,
        "derivation_method": "contrast_pair",
        "contrast_pair": {
            "axis": axis,
            "description": f"{axis} description",
            "positive": [f"pos {i}" for i in range(6)],
            "negative": [f"neg {i}" for i in range(6)],
            "rationale": "test winner",
        },
    }
    db.insert_candidate(
        candidate_id=candidate_id,
        strategy="directed_capraro_causality",
        spec_json=json.dumps(spec_dict),
        spec_hash=f"hash-{candidate_id}",
        concept=axis,
        layer_idx=layer,
        target_effective=eff,
        derivation_method="contrast_pair",
    )
    db.set_candidate_status(candidate_id, "done")
    db.record_fitness(
        candidate_id=candidate_id,
        score=score,
        detection_rate=0.25,
        identification_rate=0.125 if score > 1.0 else 0.0,
        fpr=0.0,
        coherence_rate=1.0,
        n_held_out=8,
        n_controls=4,
        components_json="{}",
    )


# ---------------------------------------------------------------------------
# Cold start
# ---------------------------------------------------------------------------

def test_cold_start_falls_through_to_directed_capraro(tmp_db):
    """No winners → cold start path delegates to directed_capraro."""
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=4,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )
    assert len(specs) > 0
    # Cold-start specs come from directed_capraro and have strategy
    # 'directed_capraro_causality', not 'hillclimb_*'.
    for spec in specs:
        assert spec.strategy == "directed_capraro_causality"


# ---------------------------------------------------------------------------
# With winners
# ---------------------------------------------------------------------------

def test_with_winners_emits_replication_and_variants(tmp_db, monkeypatch):
    """Two winners present → replication slot and variant slot both populate."""
    _seed_winner(tmp_db, "cand-W1", "axis-A", 30, 18000.0, score=3.812)
    _seed_winner(tmp_db, "cand-W2", "axis-B", 33, 14000.0, score=1.906)

    # Force composition 4:6:0 so we can check counts cleanly without the
    # cluster-expansion slot (which calls directed_capraro and is tested
    # separately).
    monkeypatch.setenv("HILLCLIMB_BATCH_COMPOSITION", "4:6:0")
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=10,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )

    # Stub proposer returns deterministic JSON, so two
    # description_sharpen calls on the same parent dedup to a single
    # spec. With real Qwen output this rarely happens; assert the
    # composition is *roughly* right rather than exact.
    assert len(specs) >= 8
    mutation_types = [
        getattr(s, "_lineage_meta", {}).get("mutation_type")
        for s in specs
    ]
    assert mutation_types.count("replication") == 4
    # Remaining are variant operators (5 or 6, depending on dedup).
    variant_count = sum(1 for m in mutation_types if m and m != "replication")
    assert variant_count >= 4


def test_replication_targets_top_winners(tmp_db, monkeypatch):
    """Replication slot picks the top-scoring winners by score."""
    _seed_winner(tmp_db, "cand-HIGH", "axis-high", 30, 18000.0, score=5.0)
    _seed_winner(tmp_db, "cand-LOW", "axis-low", 33, 18000.0, score=0.1)

    monkeypatch.setenv("HILLCLIMB_BATCH_COMPOSITION", "2:0:0")
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=2,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )

    rep_parents = {
        s._lineage_meta["parent_candidate_id"]
        for s in specs
        if s._lineage_meta["mutation_type"] == "replication"
    }
    # Top-scoring winner must be in the set.
    assert "cand-HIGH" in rep_parents


def test_variant_specs_carry_parent_lineage(tmp_db, monkeypatch):
    """Every variant spec has a parent_candidate_id pointing at a winner.

    With one parent and 5 slots, the stub proposer returns identical JSON
    for repeated description_sharpen / antonym_pivot calls, so 1-2 specs
    may dedup. We assert at least 3 unique survive and all carry lineage —
    real proposer output varies, so this dedup is rare in practice.
    """
    _seed_winner(tmp_db, "cand-W1", "axis-A", 30, 18000.0, score=3.812)

    monkeypatch.setenv("HILLCLIMB_BATCH_COMPOSITION", "0:5:0")
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=5,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )
    assert len(specs) >= 3
    for spec in specs:
        meta = spec._lineage_meta
        assert meta["parent_candidate_id"] == "cand-W1"
        assert meta["mutation_type"] in (
            "layer_shift",
            "alpha_scale",
            "examples_swap",
            "description_sharpen",
            "antonym_pivot",
        )


def test_min_score_filters_near_zero_winners(tmp_db, monkeypatch):
    """Score-zero rows shouldn't be replicated — that's just amplifying noise."""
    _seed_winner(tmp_db, "cand-ZERO", "axis-zero", 30, 18000.0, score=0.0)
    # Force replication-only batch
    monkeypatch.setenv("HILLCLIMB_BATCH_COMPOSITION", "4:0:0")
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=4,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )
    # Cold-start path — score-zero filtered out, no winners
    for spec in specs:
        assert spec.strategy == "directed_capraro_causality"


def test_dedup_drops_identical_specs(tmp_db, monkeypatch):
    """If two operators produce the same spec content, only one survives."""
    _seed_winner(tmp_db, "cand-W1", "axis-A", 30, 18000.0, score=3.812)
    monkeypatch.setenv("HILLCLIMB_BATCH_COMPOSITION", "8:0:0")
    proposer = StubProposer()
    specs = structured_hillclimb.generate_candidates(
        n=8,
        db=tmp_db,
        fault_line_id="causality",
        proposer=proposer,
    )
    # Replications all share the same axis/layer/eff but each gets a
    # unique rep_tag in rationale, so spec_hashes differ. Expect all 8.
    seen_hashes = set()
    from src.strategies.random_explore import spec_hash
    for spec in specs:
        h = spec_hash(spec)
        assert h not in seen_hashes
        seen_hashes.add(h)


def test_replication_tag_makes_hash_unique_across_replications():
    """Each replication adds a uuid rep_id, so two reps of the same parent
    produce DIFFERENT spec_hashes — both can coexist as separate evals."""
    from src.strategies.mutations import replication, ParentRecord
    from src.strategies.random_explore import spec_hash

    parent = ParentRecord(
        candidate_id="cand-P",
        strategy="directed_capraro_causality",
        concept="axis-X",
        layer_idx=30,
        target_effective=18000.0,
        derivation_method="contrast_pair",
        contrast_pair={
            "axis": "axis-X",
            "positive": ["a", "b", "c", "d", "e", "f"],
            "negative": ["g", "h", "i", "j", "k", "l"],
            "rationale": "",
        },
        fitness_mode=None,
        score=3.0,
        detection_rate=0.1,
        identification_rate=0.1,
    )
    rep1 = replication(parent)
    rep2 = replication(parent)
    # uuid rep_id differs per call → hashes differ → both survive dedup
    assert spec_hash(rep1) != spec_hash(rep2)
