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

SCHEMA_VERSION = 3  # 2026-04-18: add lineage tracking for Phase 2c autoresearch


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
-- Phase 1: individual (concept, layer, alpha, injected) trials
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

-- Phase 2: candidate steering-direction specs proposed by the researcher
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,                  -- UUID-ish, assigned by researcher
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TIMESTAMP,               -- set by worker when done
    strategy TEXT NOT NULL,               -- random_explore, exploit_topk, hillclimb_*
    spec_json TEXT NOT NULL,              -- full candidate spec
    spec_hash TEXT NOT NULL UNIQUE,       -- for dedup
    concept TEXT NOT NULL,
    layer_idx INTEGER NOT NULL,
    target_effective REAL NOT NULL,
    derivation_method TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, done, failed
    error_message TEXT,
    -- Phase 2c autoresearch: lineage tracking. A lineage groups a seed and
    -- all its mutations (successful + rejected). Each lineage has exactly
    -- one leader at a time (the best-scoring member so far).
    lineage_id TEXT,                      -- NULL for pre-Phase-2c legacy candidates
    parent_candidate_id TEXT,             -- NULL for seeds
    generation INTEGER NOT NULL DEFAULT 0,
    is_leader INTEGER NOT NULL DEFAULT 0,
    mutation_type TEXT,                   -- seed, swap_positive, swap_negative, alt_effective, alt_layer, edit_description
    mutation_detail TEXT                  -- JSON describing what changed from parent
);

CREATE INDEX IF NOT EXISTS idx_candidates_strategy ON candidates(strategy);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
CREATE INDEX IF NOT EXISTS idx_candidates_hash ON candidates(spec_hash);
-- Note: lineage / parent indexes live in _migrate() because they reference
-- columns that only exist after migration ALTER TABLE runs.

-- Phase 2: per-concept evaluation results for each candidate
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    candidate_id TEXT NOT NULL,
    eval_concept TEXT NOT NULL,           -- the held-out concept we tested on
    injected INTEGER NOT NULL,            -- 0 for controls, 1 for injection
    alpha REAL NOT NULL,
    direction_norm REAL NOT NULL,
    response TEXT NOT NULL,
    detected INTEGER NOT NULL,
    identified INTEGER NOT NULL,
    coherent INTEGER NOT NULL,
    judge_model TEXT NOT NULL,
    judge_reasoning TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE INDEX IF NOT EXISTS idx_evaluations_candidate ON evaluations(candidate_id);

-- Phase 2: composite fitness score per candidate
CREATE TABLE IF NOT EXISTS fitness_scores (
    candidate_id TEXT PRIMARY KEY,
    score REAL NOT NULL,
    detection_rate REAL NOT NULL,
    identification_rate REAL NOT NULL,
    fpr REAL NOT NULL,
    coherence_rate REAL NOT NULL,
    n_held_out INTEGER NOT NULL,
    n_controls INTEGER NOT NULL,
    components_json TEXT,                 -- full breakdown for analysis
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE INDEX IF NOT EXISTS idx_fitness_score ON fitness_scores(score);

-- Phase A → Phase B handoff buffer (added 2026-04-27 for the four-phase
-- worker state machine). The Generate phase writes Gemma's raw responses
-- here; the Judge phase later loads the local-MLX judge, drains this table
-- for each candidate, scores everything, and inserts into `evaluations`.
-- Crash recovery: if the worker dies between phases A and B, these rows
-- survive — Phase B can restart and pick up where it left off.
CREATE TABLE IF NOT EXISTS pending_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    candidate_id TEXT NOT NULL,
    eval_concept TEXT NOT NULL,           -- probe concept used at generation
    injected INTEGER NOT NULL,            -- 0 for controls, 1 for injection
    alpha REAL NOT NULL,
    direction_norm REAL NOT NULL,
    response TEXT NOT NULL,
    -- Judge prompt selection. derivation_method='contrast_pair' uses the
    -- semantic-pole prompt with axis/description/examples; word-style
    -- candidates use the strict concept prompt. Stored here so Phase B
    -- doesn't need to JOIN candidates.spec_json again.
    derivation_method TEXT NOT NULL,
    judge_concept TEXT,                   -- for word-style judging (the source concept)
    contrast_axis TEXT,                   -- for contrast_pair
    contrast_description TEXT,            -- for contrast_pair
    contrast_positive_json TEXT,          -- JSON list[str], for contrast_pair
    contrast_negative_json TEXT,          -- JSON list[str], for contrast_pair
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE INDEX IF NOT EXISTS idx_pending_responses_candidate ON pending_responses(candidate_id);

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
            self._migrate(conn)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

    # ------------------------------------------------------------------
    # Generic key/value durable state — schema_meta beyond schema_version.
    # Used by the worker to persist the fault-line rotation index across
    # restarts so a SIGTERM doesn't reset us back to causality.
    # ------------------------------------------------------------------

    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Apply in-place schema migrations for existing DBs.

        `CREATE TABLE IF NOT EXISTS` won't add columns to an already-created
        table, so new columns added in later schema versions need explicit
        ALTER TABLE statements guarded by a column-exists check.
        """
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(candidates)").fetchall()}
        add_col_stmts = {
            "lineage_id":          "ALTER TABLE candidates ADD COLUMN lineage_id TEXT",
            "parent_candidate_id": "ALTER TABLE candidates ADD COLUMN parent_candidate_id TEXT",
            "generation":          "ALTER TABLE candidates ADD COLUMN generation INTEGER NOT NULL DEFAULT 0",
            "is_leader":           "ALTER TABLE candidates ADD COLUMN is_leader INTEGER NOT NULL DEFAULT 0",
            "mutation_type":       "ALTER TABLE candidates ADD COLUMN mutation_type TEXT",
            "mutation_detail":     "ALTER TABLE candidates ADD COLUMN mutation_detail TEXT",
            # Phase 2d+: records whether this candidate ran under paper-method
            # abliteration or on vanilla Gemma. All rows predating the 2026-04-24
            # migration default to 'vanilla' — that's historically accurate;
            # Phase 2 never ran under paper-method hooks before this date.
            "abliteration_mode":   "ALTER TABLE candidates ADD COLUMN abliteration_mode TEXT NOT NULL DEFAULT 'vanilla'",
            # 2026-04-27: which LLM produced this candidate's contrast pair /
            # spec. Pre-local-pipeline rows are claude-opus-4-7 (or sonnet for
            # earlier novel_contrast); post-pipeline-switch rows are the
            # Qwen3.6-27B-MLX-8bit local proposer. NULL for random_explore
            # (no LLM involvement). Backfill is date-based.
            "proposer_model":      "ALTER TABLE candidates ADD COLUMN proposer_model TEXT",
        }
        for col, stmt in add_col_stmts.items():
            if col not in existing:
                conn.execute(stmt)
        # Ensure the lineage indexes exist (idempotent).
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_lineage ON candidates(lineage_id, is_leader)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_parent ON candidates(parent_candidate_id)"
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

    # ------------------------------------------------------------------
    # Phase 2: candidates / evaluations / fitness_scores
    # ------------------------------------------------------------------

    def has_candidate_hash(self, spec_hash: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM candidates WHERE spec_hash=? LIMIT 1", (spec_hash,)
            ).fetchone()
        return row is not None

    def insert_candidate(
        self,
        candidate_id: str,
        strategy: str,
        spec_json: str,
        spec_hash: str,
        concept: str,
        layer_idx: int,
        target_effective: float,
        derivation_method: str,
        lineage_id: Optional[str] = None,
        parent_candidate_id: Optional[str] = None,
        generation: int = 0,
        mutation_type: Optional[str] = None,
        mutation_detail: Optional[str] = None,
        abliteration_mode: str = "vanilla",
        proposer_model: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO candidates
                   (id, strategy, spec_json, spec_hash, concept, layer_idx,
                    target_effective, derivation_method, status,
                    lineage_id, parent_candidate_id, generation,
                    is_leader, mutation_type, mutation_detail,
                    abliteration_mode, proposer_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending',
                           ?, ?, ?, 0, ?, ?, ?, ?)""",
                (
                    candidate_id, strategy, spec_json, spec_hash,
                    concept, layer_idx, target_effective, derivation_method,
                    lineage_id, parent_candidate_id, generation,
                    mutation_type, mutation_detail, abliteration_mode,
                    proposer_model,
                ),
            )

    # ------------------------------------------------------------------
    # Phase 2c: lineage management
    # ------------------------------------------------------------------

    def get_leaders(self) -> list[dict]:
        """Return current leaders across all lineages, with their scores."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, f.score, f.detection_rate, f.identification_rate,
                          f.fpr, f.coherence_rate
                   FROM candidates c
                   LEFT JOIN fitness_scores f ON f.candidate_id = c.id
                   WHERE c.is_leader = 1
                   ORDER BY f.score DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_candidate(self, candidate_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT c.*, f.score, f.detection_rate, f.identification_rate,
                          f.fpr, f.coherence_rate
                   FROM candidates c
                   LEFT JOIN fitness_scores f ON f.candidate_id = c.id
                   WHERE c.id = ?""",
                (candidate_id,),
            ).fetchone()
        return dict(row) if row else None

    def promote_to_leader(self, new_leader_id: str, old_leader_id: str) -> None:
        """Commit a mutation: new_leader supersedes old_leader in its lineage."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE candidates SET is_leader = 0 WHERE id = ?", (old_leader_id,)
            )
            conn.execute(
                "UPDATE candidates SET is_leader = 1 WHERE id = ?", (new_leader_id,)
            )

    def mark_as_seed_leader(self, candidate_id: str, lineage_id: str) -> None:
        """One-shot: mark an existing candidate as a seed leader for a new lineage."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE candidates SET
                     lineage_id = ?, parent_candidate_id = NULL,
                     generation = 0, is_leader = 1,
                     mutation_type = 'seed'
                   WHERE id = ?""",
                (lineage_id, candidate_id),
            )

    def lineage_history(self, lineage_id: str) -> list[dict]:
        """Every candidate in a lineage (leaders and rejections), ordered by generation."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, f.score, f.detection_rate, f.identification_rate,
                          f.fpr, f.coherence_rate
                   FROM candidates c
                   LEFT JOIN fitness_scores f ON f.candidate_id = c.id
                   WHERE c.lineage_id = ?
                   ORDER BY c.generation, c.created_at""",
                (lineage_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_candidate_status(
        self,
        candidate_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            if status == "done":
                conn.execute(
                    """UPDATE candidates SET status=?, evaluated_at=CURRENT_TIMESTAMP,
                       error_message=? WHERE id=?""",
                    (status, error_message, candidate_id),
                )
            else:
                conn.execute(
                    "UPDATE candidates SET status=?, error_message=? WHERE id=?",
                    (status, error_message, candidate_id),
                )

    def record_evaluation(
        self,
        candidate_id: str,
        eval_concept: str,
        injected: bool,
        alpha: float,
        direction_norm: float,
        response: str,
        detected: bool,
        identified: bool,
        coherent: bool,
        judge_model: str,
        judge_reasoning: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO evaluations
                   (candidate_id, eval_concept, injected, alpha, direction_norm,
                    response, detected, identified, coherent,
                    judge_model, judge_reasoning)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id, eval_concept, int(injected), alpha, direction_norm,
                    response, int(detected), int(identified), int(coherent),
                    judge_model, judge_reasoning,
                ),
            )

    def record_fitness(
        self,
        candidate_id: str,
        score: float,
        detection_rate: float,
        identification_rate: float,
        fpr: float,
        coherence_rate: float,
        n_held_out: int,
        n_controls: int,
        components_json: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fitness_scores
                   (candidate_id, score, detection_rate, identification_rate,
                    fpr, coherence_rate, n_held_out, n_controls, components_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id, score, detection_rate, identification_rate,
                    fpr, coherence_rate, n_held_out, n_controls, components_json,
                ),
            )

    def top_candidates(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.id, c.strategy, c.concept, c.layer_idx,
                          c.target_effective, c.derivation_method,
                          f.score, f.detection_rate, f.identification_rate,
                          f.fpr, f.coherence_rate
                   FROM candidates c
                   JOIN fitness_scores f ON f.candidate_id = c.id
                   ORDER BY f.score DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Phase A → Phase B handoff (2026-04-27, worker)
    # ------------------------------------------------------------------

    def insert_pending_response(
        self,
        candidate_id: str,
        eval_concept: str,
        injected: bool,
        alpha: float,
        direction_norm: float,
        response: str,
        derivation_method: str,
        judge_concept: Optional[str] = None,
        contrast_axis: Optional[str] = None,
        contrast_description: Optional[str] = None,
        contrast_positive: Optional[list[str]] = None,
        contrast_negative: Optional[list[str]] = None,
    ) -> int:
        """Stash a Gemma response so Phase B can judge it later.

        Returns the autoincrement row id (rarely needed; mostly here so tests
        can reference rows directly).
        """
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO pending_responses
                   (candidate_id, eval_concept, injected, alpha, direction_norm,
                    response, derivation_method, judge_concept,
                    contrast_axis, contrast_description,
                    contrast_positive_json, contrast_negative_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate_id, eval_concept, int(injected), alpha,
                    direction_norm, response, derivation_method, judge_concept,
                    contrast_axis, contrast_description,
                    json.dumps(contrast_positive) if contrast_positive is not None else None,
                    json.dumps(contrast_negative) if contrast_negative is not None else None,
                ),
            )
            return int(cur.lastrowid)

    def get_pending_responses(
        self,
        candidate_id: Optional[str] = None,
    ) -> list[dict]:
        """Return pending responses, optionally filtered by candidate.

        Each row has the contrast_pair fields decoded back into Python lists
        (for derivation_method='contrast_pair' rows; otherwise None).
        """
        with self._conn() as conn:
            if candidate_id is None:
                cur = conn.execute(
                    "SELECT * FROM pending_responses ORDER BY id"
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM pending_responses WHERE candidate_id=? ORDER BY id",
                    (candidate_id,),
                )
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["injected"] = bool(d["injected"])
            d["contrast_positive"] = (
                json.loads(d["contrast_positive_json"])
                if d["contrast_positive_json"] else None
            )
            d["contrast_negative"] = (
                json.loads(d["contrast_negative_json"])
                if d["contrast_negative_json"] else None
            )
            out.append(d)
        return out

    def pending_candidate_ids(self) -> list[str]:
        """Distinct candidate_ids that still have unjudged rows in pending_responses."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT candidate_id FROM pending_responses ORDER BY candidate_id"
            ).fetchall()
        return [r["candidate_id"] for r in rows]

    def count_pending_responses(self, candidate_id: Optional[str] = None) -> int:
        with self._conn() as conn:
            if candidate_id is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM pending_responses"
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM pending_responses WHERE candidate_id=?",
                    (candidate_id,),
                ).fetchone()
        return int(row["n"])

    def delete_pending_responses(self, candidate_id: str) -> int:
        """Remove all pending responses for a candidate (Phase B finalizer).

        Returns the number of rows deleted. Idempotent.
        """
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM pending_responses WHERE candidate_id=?",
                (candidate_id,),
            )
            return int(cur.rowcount)

    def candidates_summary(self) -> dict:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS n_total,
                          SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS n_pending,
                          SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS n_running,
                          SUM(CASE WHEN status='done'    THEN 1 ELSE 0 END) AS n_done,
                          SUM(CASE WHEN status='failed'  THEN 1 ELSE 0 END) AS n_failed
                   FROM candidates"""
            ).fetchone()
        return dict(row) if row else {}
