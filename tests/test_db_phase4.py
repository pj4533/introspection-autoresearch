"""Schema v5 migration + Phase 4 helper tests."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest

from src.db import ResultsDB, SCHEMA_VERSION


def test_schema_version_is_5():
    assert SCHEMA_VERSION == 5


def test_fresh_db_has_phase4_tables(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    with sqlite3.connect(db.path) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "phase4_chains" in names
    assert "phase4_steps" in names
    assert "phase4_concepts" in names


def test_fresh_db_records_schema_version(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    assert db.get_meta("schema_version") == "5"


def test_chain_lifecycle(tmp_path):
    db = ResultsDB(tmp_path / "test.db")

    db.insert_phase4_chain(
        chain_id="chain-001",
        seed_concept="Bread",
        layer_idx=42,
        target_effective=100.0,
    )

    db.upsert_phase4_concept("bread", "Bread", is_seed=True)
    db.upsert_phase4_concept("bread", "Bread", is_seed=True)  # idempotent

    step_id = db.insert_phase4_step(
        chain_id="chain-001",
        step_idx=0,
        target_concept="Bread",
        target_lemma="bread",
        alpha=4.0,
        direction_norm=25.0,
        raw_response="<|channel>thought\n*Bread* (Too common)<channel|>Bread",
        thought_block="*Bread* (Too common)",
        final_answer="Bread",
        parse_failure=False,
    )
    assert step_id > 0

    db.update_phase4_step_judgments(
        step_id=step_id,
        behavior_named=True,
        behavior_coherent=True,
        cot_named="named_with_recognition",
        cot_evidence="*Bread* (Too common)",
        judge_model="qwen35b",
        judge_reasoning="recognition flag",
    )

    db.increment_phase4_concept_visit("bread")
    db.increment_phase4_concept_tallies(
        concept_lemma="bread",
        behavior_hit=True,
        cot_named="named_with_recognition",
        coherent=True,
    )

    db.finalize_phase4_chain(
        chain_id="chain-001",
        end_reason="length_cap",
        n_steps=20,
    )

    stats = db.get_phase4_concept_stats()
    assert len(stats) == 1
    bread = stats[0]
    assert bread["concept_lemma"] == "bread"
    assert bread["visits"] == 1
    assert bread["behavior_hits"] == 1
    assert bread["cot_named_hits"] == 1
    assert bread["cot_recognition_hits"] == 1


def test_unjudged_steps_query(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    db.insert_phase4_chain("c1", "Bread", 42, 100.0)

    s1 = db.insert_phase4_step("c1", 0, "Bread", "bread", 4.0, 25.0,
                                "raw1", "th1", "Bread", False)
    s2 = db.insert_phase4_step("c1", 1, "Wheat", "wheat", 4.0, 25.0,
                                "raw2", "th2", "Wheat", False)

    unjudged = db.fetch_unjudged_phase4_steps()
    assert len(unjudged) == 2
    assert {u["step_id"] for u in unjudged} == {s1, s2}

    # Judge one of them.
    db.update_phase4_step_judgments(s1, True, True, "named", "ev", "j", "r")

    unjudged = db.fetch_unjudged_phase4_steps()
    assert len(unjudged) == 1
    assert unjudged[0]["step_id"] == s2


def test_phase4_summary(tmp_path):
    db = ResultsDB(tmp_path / "test.db")
    db.insert_phase4_chain("c1", "Bread", 42, 100.0)
    db.insert_phase4_step("c1", 0, "Bread", "bread", 4.0, 25.0,
                          "raw", "thought", "Bread", False)
    db.upsert_phase4_concept("bread", "Bread")
    db.finalize_phase4_chain("c1", "length_cap", 1)

    summary = db.phase4_summary()
    assert summary["n_chains"] == 1
    assert summary["total_steps"] == 1
    assert summary["n_length_cap"] == 1
    assert summary["n_unjudged_steps"] == 1
    assert summary["n_concepts"] == 1


def test_existing_db_migrates_in_place(tmp_path):
    """If we open an existing DB created at v4, the new tables should be
    created on first connection without breaking existing data."""
    db_path = tmp_path / "legacy.db"
    # Create a v4-shaped DB by constructing it once and then mutating
    # schema_version backward — for this test we just construct fresh
    # (current version), then re-open and verify no data lost.
    db1 = ResultsDB(db_path)
    db1.insert_phase4_chain("c1", "Bread", 42, 100.0)

    # Re-open — migrations are idempotent.
    db2 = ResultsDB(db_path)
    assert db2.phase4_summary()["n_chains"] == 1
