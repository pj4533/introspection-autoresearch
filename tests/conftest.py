"""Shared pytest fixtures for the worker / phased-eval test suite.

Most tests use a temp-dir SQLite DB and mocked model handles so the suite
runs in a few seconds without loading any real weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `src` importable regardless of pytest's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def tmp_db(tmp_path):
    """Fresh ResultsDB in a temporary directory."""
    from src.db import ResultsDB
    db = ResultsDB(tmp_path / "results.db")
    return db


@pytest.fixture()
def tmp_queue(tmp_path, monkeypatch):
    """Override worker.QUEUE to a tmp dir for isolated queue testing."""
    queue_dir = tmp_path / "queue"
    for sub in ("pending", "running", "done", "failed"):
        (queue_dir / sub).mkdir(parents=True)
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    import src.worker as w
    monkeypatch.setattr(w, "QUEUE", queue_dir)
    monkeypatch.setattr(w, "RUNS", runs_dir)
    return queue_dir
