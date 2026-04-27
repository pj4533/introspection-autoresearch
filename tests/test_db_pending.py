"""Tests for the pending_responses table introduced for the four-phase worker.

Goal: prove that Phase A → Phase B handoff is durable, contrast-pair-aware,
and that the simple lifecycle (insert / get / count / delete) holds up.
"""

from __future__ import annotations


def test_round_trip_word_style(tmp_db):
    """A word-style probe round-trips through pending_responses."""
    rid = tmp_db.insert_pending_response(
        candidate_id="cand-A",
        eval_concept="Bread",
        injected=True,
        alpha=2.5,
        direction_norm=120.0,
        response="I detect bread.",
        derivation_method="mean_diff",
        judge_concept="Apple",
    )
    assert rid > 0

    rows = tmp_db.get_pending_responses("cand-A")
    assert len(rows) == 1
    r = rows[0]
    assert r["candidate_id"] == "cand-A"
    assert r["eval_concept"] == "Bread"
    assert r["injected"] is True
    assert r["alpha"] == 2.5
    assert r["direction_norm"] == 120.0
    assert r["response"] == "I detect bread."
    assert r["derivation_method"] == "mean_diff"
    assert r["judge_concept"] == "Apple"
    # word-style → contrast fields are None
    assert r["contrast_axis"] is None
    assert r["contrast_positive"] is None
    assert r["contrast_negative"] is None


def test_round_trip_contrast_pair(tmp_db):
    """Contrast-pair fields survive the JSON round-trip and decode to lists."""
    tmp_db.insert_pending_response(
        candidate_id="cand-B",
        eval_concept="Hummingbirds",
        injected=True,
        alpha=4.5,
        direction_norm=80.0,
        response="I notice live narration.",
        derivation_method="contrast_pair",
        contrast_axis="live-vs-retro",
        contrast_description="live narration vs retrospective",
        contrast_positive=["I am noticing now.", "Watching it unfold."],
        contrast_negative=["I noticed earlier.", "I had observed it."],
    )
    rows = tmp_db.get_pending_responses("cand-B")
    assert len(rows) == 1
    r = rows[0]
    assert r["contrast_axis"] == "live-vs-retro"
    assert r["contrast_description"] == "live narration vs retrospective"
    assert r["contrast_positive"] == ["I am noticing now.", "Watching it unfold."]
    assert r["contrast_negative"] == ["I noticed earlier.", "I had observed it."]
    assert r["judge_concept"] is None


def test_count_and_pending_candidate_ids(tmp_db):
    """count_pending_responses + pending_candidate_ids reflect inserts/deletes."""
    for cid, axis in [("cand-X", "x"), ("cand-X", "x"), ("cand-Y", "y")]:
        tmp_db.insert_pending_response(
            candidate_id=cid,
            eval_concept="C",
            injected=True,
            alpha=1.0,
            direction_norm=10.0,
            response="r",
            derivation_method="contrast_pair",
            contrast_axis=axis,
            contrast_positive=["a", "b", "c"],
            contrast_negative=["x", "y", "z"],
        )
    assert tmp_db.count_pending_responses() == 3
    assert tmp_db.count_pending_responses("cand-X") == 2
    assert tmp_db.count_pending_responses("cand-Y") == 1
    assert sorted(tmp_db.pending_candidate_ids()) == ["cand-X", "cand-Y"]


def test_delete_pending_responses_idempotent(tmp_db):
    """Phase B finalizer can be called twice without error."""
    tmp_db.insert_pending_response(
        candidate_id="cand-Z",
        eval_concept="C",
        injected=False,
        alpha=0.0,
        direction_norm=0.0,
        response="r",
        derivation_method="mean_diff",
        judge_concept="C",
    )
    assert tmp_db.count_pending_responses("cand-Z") == 1
    n1 = tmp_db.delete_pending_responses("cand-Z")
    assert n1 == 1
    n2 = tmp_db.delete_pending_responses("cand-Z")
    assert n2 == 0
    assert tmp_db.pending_candidate_ids() == []


def test_meta_kv_round_trip(tmp_db):
    """schema_meta key/value persists across reads, defaults work, upsert works."""
    # Default returned when key is missing
    assert tmp_db.get_meta("nope") is None
    assert tmp_db.get_meta("nope", default="fallback") == "fallback"

    # First write
    tmp_db.set_meta("propose_index|causality,grounding", "5")
    assert tmp_db.get_meta("propose_index|causality,grounding") == "5"

    # Upsert overwrites
    tmp_db.set_meta("propose_index|causality,grounding", "12")
    assert tmp_db.get_meta("propose_index|causality,grounding") == "12"

    # Independent keys don't collide
    tmp_db.set_meta("propose_index|other", "3")
    assert tmp_db.get_meta("propose_index|causality,grounding") == "12"
    assert tmp_db.get_meta("propose_index|other") == "3"


def test_pending_responses_ordering(tmp_db):
    """get_pending_responses returns rows in insertion order (by id ASC)."""
    for c in ["c1", "c2", "c3"]:
        tmp_db.insert_pending_response(
            candidate_id="cand-O",
            eval_concept=c,
            injected=True,
            alpha=1.0,
            direction_norm=10.0,
            response=c,
            derivation_method="mean_diff",
            judge_concept="K",
        )
    rows = tmp_db.get_pending_responses("cand-O")
    assert [r["eval_concept"] for r in rows] == ["c1", "c2", "c3"]
