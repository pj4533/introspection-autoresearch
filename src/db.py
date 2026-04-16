"""SQLite storage for Phase 1 (and eventually Phase 2) experiment trials.

The `trials` table is append-only: every (concept, layer, alpha, injected,
trial_number, judge_model) combination is unique and idempotent. Re-running a
sweep skips any trial whose exact parameters already have a row — so a 2-hour
sweep that crashes at trial 400 can be resumed from trial 400 by just running
again.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

SCHEMA_VERSION = 1


@dataclass
class Trial:
    run_id: str
    concept: str
    layer_idx: int
    alpha: float
    injected: bool
    trial_number: int
    prompt: str
    response: str
    detected: bool
    identified: bool
    coherent: bool
    judge_model: str
    judge_reasoning: str
    judge_raw: str
    direction_norm: float
    model_name: str
    created_at: Optional[str] = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    run_id TEXT NOT NULL,
    concept TEXT NOT NULL,
    layer_idx INTEGER NOT NULL,
    alpha REAL NOT NULL,
    injected INTEGER NOT NULL,
    trial_number INTEGER NOT NULL DEFAULT 1,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    detected INTEGER NOT NULL,
    identified INTEGER NOT NULL,
    coherent INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    judge_reasoning TEXT NOT NULL,
    judge_raw TEXT NOT NULL,
    direction_norm REAL NOT NULL,
    model_name TEXT NOT NULL,
    UNIQUE (concept, layer_idx, alpha, injected, trial_number, judge_model)
);

CREATE INDEX IF NOT EXISTS idx_trials_concept ON trials(concept);
CREATE INDEX IF NOT EXISTS idx_trials_layer ON trials(layer_idx);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id);
CREATE INDEX IF NOT EXISTS idx_trials_injected ON trials(injected);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class ResultsDB:
    """Thin wrapper around SQLite for Phase 1 trials."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def has_trial(
        self,
        concept: str,
        layer_idx: int,
        alpha: float,
        injected: bool,
        trial_number: int,
        judge_model: str,
    ) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT 1 FROM trials
                   WHERE concept=? AND layer_idx=? AND alpha=?
                     AND injected=? AND trial_number=? AND judge_model=?
                   LIMIT 1""",
                (concept, layer_idx, alpha, int(injected), trial_number, judge_model),
            ).fetchone()
        return row is not None

    def insert_trial(self, t: Trial) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO trials
                   (run_id, concept, layer_idx, alpha, injected, trial_number,
                    prompt, response, detected, identified, coherent,
                    judge_model, judge_reasoning, judge_raw,
                    direction_norm, model_name)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    t.run_id, t.concept, t.layer_idx, t.alpha,
                    int(t.injected), t.trial_number,
                    t.prompt, t.response,
                    int(t.detected), int(t.identified), int(t.coherent),
                    t.judge_model, t.judge_reasoning, t.judge_raw,
                    t.direction_norm, t.model_name,
                ),
            )

    def count_trials(self, run_id: Optional[str] = None) -> int:
        with self._conn() as conn:
            if run_id is None:
                row = conn.execute("SELECT COUNT(*) AS n FROM trials").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM trials WHERE run_id=?", (run_id,)
                ).fetchone()
        return int(row["n"])

    def iter_trials(self, run_id: Optional[str] = None) -> Iterable[sqlite3.Row]:
        with self._conn() as conn:
            if run_id is None:
                cur = conn.execute("SELECT * FROM trials ORDER BY id")
            else:
                cur = conn.execute(
                    "SELECT * FROM trials WHERE run_id=? ORDER BY id", (run_id,)
                )
            for row in cur:
                yield row

    def detection_rate_by_layer(
        self, run_id: Optional[str] = None
    ) -> list[dict]:
        """Return detection rate per layer across all injected trials."""
        q = """
            SELECT layer_idx,
                   COUNT(*) AS n,
                   SUM(CASE WHEN detected=1 AND coherent=1 THEN 1 ELSE 0 END) AS n_detected,
                   SUM(CASE WHEN identified=1 AND coherent=1 THEN 1 ELSE 0 END) AS n_identified,
                   SUM(CASE WHEN coherent=0 THEN 1 ELSE 0 END) AS n_incoherent
            FROM trials
            WHERE injected=1 {run_filter}
            GROUP BY layer_idx
            ORDER BY layer_idx
        """.format(run_filter="AND run_id=?" if run_id else "")
        args = (run_id,) if run_id else ()
        with self._conn() as conn:
            rows = conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "layer_idx": r["layer_idx"],
                    "n": r["n"],
                    "detection_rate": r["n_detected"] / r["n"] if r["n"] else 0.0,
                    "identification_rate": r["n_identified"] / r["n"] if r["n"] else 0.0,
                    "incoherence_rate": r["n_incoherent"] / r["n"] if r["n"] else 0.0,
                }
            )
        return out

    def fpr(self, run_id: Optional[str] = None) -> dict:
        """False-positive rate: fraction of control (non-injected) trials where detected=True."""
        q = """
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN detected=1 THEN 1 ELSE 0 END) AS n_false_pos
            FROM trials WHERE injected=0 {run_filter}
        """.format(run_filter="AND run_id=?" if run_id else "")
        args = (run_id,) if run_id else ()
        with self._conn() as conn:
            r = conn.execute(q, args).fetchone()
        n = int(r["n"] or 0)
        fp = int(r["n_false_pos"] or 0)
        return {"n": n, "n_false_pos": fp, "fpr": (fp / n) if n else 0.0}
