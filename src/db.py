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

SCHEMA_VERSION = 5  # 2026-04-30: Phase 4 — phase4_chains / phase4_steps / phase4_concepts


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
    mutation_detail TEXT,                 -- JSON describing what changed from parent
    -- Phase 3 (Gemma 4): which Gemma model produced this candidate's
    -- responses. 'gemma3_12b' for all pre-Phase-3 rows; 'gemma4_31b'
    -- for Phase 3 rows. Site uses this for the per-row model badge.
    abliteration_mode TEXT NOT NULL DEFAULT 'vanilla',
    proposer_model TEXT,
    gemma_model TEXT NOT NULL DEFAULT 'gemma3_12b'
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

-- Phase 4: dream walks. A `chain` is a 20-step (or shorter)
-- free-association walk through steered concepts. Step 0's target
-- comes from the seed pool; step N's target is the previous step's
-- final answer.
CREATE TABLE IF NOT EXISTS phase4_chains (
    chain_id TEXT PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    seed_concept TEXT NOT NULL,
    end_reason TEXT,                  -- 'length_cap', 'self_loop', 'coherence_break', 'parse_fail', 'error'
    n_steps INTEGER NOT NULL DEFAULT 0,
    layer_idx INTEGER NOT NULL,
    target_effective REAL NOT NULL,
    gemma_model TEXT NOT NULL DEFAULT 'gemma4_31b'
);
CREATE INDEX IF NOT EXISTS idx_phase4_chains_seed ON phase4_chains(seed_concept);

CREATE TABLE IF NOT EXISTS phase4_steps (
    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    target_concept TEXT NOT NULL,           -- raw form (display)
    target_lemma TEXT NOT NULL,             -- normalized form (for tallies)
    alpha REAL NOT NULL,
    direction_norm REAL NOT NULL,
    raw_response TEXT NOT NULL,
    thought_block TEXT,
    final_answer TEXT,
    parse_failure INTEGER NOT NULL DEFAULT 0,
    behavior_named INTEGER,                  -- judge: 0/1, NULL until judged
    behavior_coherent INTEGER,               -- judge coherent flag
    cot_named TEXT,                          -- 'none' | 'named' | 'named_with_recognition' | NULL
    cot_evidence TEXT,
    judge_model TEXT,
    judge_reasoning TEXT,
    judged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chain_id) REFERENCES phase4_chains(chain_id)
);
CREATE INDEX IF NOT EXISTS idx_phase4_steps_chain ON phase4_steps(chain_id);
CREATE INDEX IF NOT EXISTS idx_phase4_steps_target ON phase4_steps(target_lemma);
CREATE INDEX IF NOT EXISTS idx_phase4_steps_unjudged ON phase4_steps(judged_at) WHERE judged_at IS NULL;

CREATE TABLE IF NOT EXISTS phase4_concepts (
    concept_lemma TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,                  -- canonical surface form
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    visits INTEGER NOT NULL DEFAULT 0,           -- times appeared as target across all chains
    behavior_hits INTEGER NOT NULL DEFAULT 0,    -- judge said behavior_named=1
    cot_named_hits INTEGER NOT NULL DEFAULT 0,   -- judge said named OR named_with_recognition
    cot_recognition_hits INTEGER NOT NULL DEFAULT 0,  -- judge said named_with_recognition only
    coherent_hits INTEGER NOT NULL DEFAULT 0,
    is_seed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_phase4_concepts_visits ON phase4_concepts(visits);
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
            # 2026-04-29: Phase 3 — which Gemma generation/size produced
            # this candidate's responses. All pre-Phase-3 rows are
            # 'gemma3_12b' (the only model used in Phases 1/2). Phase 3
            # rows are 'gemma4_31b' (instruction-tuned, MLX 8-bit). The
            # site reads this to render the model badge that distinguishes
            # Gemma 3 results from Gemma 4 results in the same leaderboard.
            "gemma_model":         "ALTER TABLE candidates ADD COLUMN gemma_model TEXT NOT NULL DEFAULT 'gemma3_12b'",
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
        # Phase 3: index gemma_model for fast per-model filtering on the site.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_candidates_gemma_model ON candidates(gemma_model)"
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
        gemma_model: str = "gemma3_12b",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO candidates
                   (id, strategy, spec_json, spec_hash, concept, layer_idx,
                    target_effective, derivation_method, status,
                    lineage_id, parent_candidate_id, generation,
                    is_leader, mutation_type, mutation_detail,
                    abliteration_mode, proposer_model, gemma_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending',
                           ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
                (
                    candidate_id, strategy, spec_json, spec_hash,
                    concept, layer_idx, target_effective, derivation_method,
                    lineage_id, parent_candidate_id, generation,
                    mutation_type, mutation_detail, abliteration_mode,
                    proposer_model, gemma_model,
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

    # ------------------------------------------------------------------
    # Phase 4 — Dream Walks + Forbidden Map.
    # ------------------------------------------------------------------

    def insert_phase4_chain(
        self,
        chain_id: str,
        seed_concept: str,
        layer_idx: int,
        target_effective: float,
        gemma_model: str = "gemma4_31b",
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO phase4_chains
                       (chain_id, seed_concept, layer_idx, target_effective, gemma_model)
                   VALUES (?, ?, ?, ?, ?)""",
                (chain_id, seed_concept, layer_idx, target_effective, gemma_model),
            )

    def finalize_phase4_chain(
        self,
        chain_id: str,
        end_reason: str,
        n_steps: int,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE phase4_chains
                   SET ended_at = CURRENT_TIMESTAMP,
                       end_reason = ?,
                       n_steps = ?
                   WHERE chain_id = ?""",
                (end_reason, n_steps, chain_id),
            )

    def insert_phase4_step(
        self,
        chain_id: str,
        step_idx: int,
        target_concept: str,
        target_lemma: str,
        alpha: float,
        direction_norm: float,
        raw_response: str,
        thought_block: Optional[str],
        final_answer: Optional[str],
        parse_failure: bool,
    ) -> int:
        """Insert a step record (generation phase). Judge fields stay NULL
        until the judge phase runs. Returns the step_id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO phase4_steps
                       (chain_id, step_idx, target_concept, target_lemma,
                        alpha, direction_norm, raw_response,
                        thought_block, final_answer, parse_failure)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chain_id, step_idx, target_concept, target_lemma,
                    alpha, direction_norm, raw_response,
                    thought_block, final_answer, int(parse_failure),
                ),
            )
            return int(cur.lastrowid)

    def update_phase4_step_judgments(
        self,
        step_id: int,
        behavior_named: bool,
        behavior_coherent: bool,
        cot_named: str,
        cot_evidence: str,
        judge_model: str,
        judge_reasoning: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE phase4_steps
                   SET behavior_named = ?,
                       behavior_coherent = ?,
                       cot_named = ?,
                       cot_evidence = ?,
                       judge_model = ?,
                       judge_reasoning = ?,
                       judged_at = CURRENT_TIMESTAMP
                   WHERE step_id = ?""",
                (
                    int(behavior_named), int(behavior_coherent),
                    cot_named, cot_evidence,
                    judge_model, judge_reasoning,
                    step_id,
                ),
            )

    def upsert_phase4_concept(
        self,
        concept_lemma: str,
        display_name: str,
        is_seed: bool = False,
    ) -> None:
        """First-touch concept registration. Idempotent — repeated calls
        are no-ops once the concept is in the table."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO phase4_concepts (concept_lemma, display_name, is_seed)
                   VALUES (?, ?, ?)
                   ON CONFLICT(concept_lemma) DO NOTHING""",
                (concept_lemma, display_name, int(is_seed)),
            )

    def increment_phase4_concept_visit(self, concept_lemma: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE phase4_concepts
                   SET visits = visits + 1
                   WHERE concept_lemma = ?""",
                (concept_lemma,),
            )

    def increment_phase4_concept_tallies(
        self,
        concept_lemma: str,
        behavior_hit: bool,
        cot_named: str,            # 'none' | 'named' | 'named_with_recognition'
        coherent: bool,
    ) -> None:
        named_hit = 1 if cot_named in ("named", "named_with_recognition") else 0
        recog_hit = 1 if cot_named == "named_with_recognition" else 0
        with self._conn() as conn:
            conn.execute(
                """UPDATE phase4_concepts
                   SET behavior_hits = behavior_hits + ?,
                       cot_named_hits = cot_named_hits + ?,
                       cot_recognition_hits = cot_recognition_hits + ?,
                       coherent_hits = coherent_hits + ?
                   WHERE concept_lemma = ?""",
                (
                    int(behavior_hit), named_hit, recog_hit, int(coherent),
                    concept_lemma,
                ),
            )

    def get_phase4_concept_stats(self) -> list[dict]:
        """Return all phase4_concepts rows as plain dicts. Used by the
        seed-pool priority sampler and the Forbidden Map exporter."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT concept_lemma, display_name, visits,
                          behavior_hits, cot_named_hits,
                          cot_recognition_hits, coherent_hits, is_seed,
                          first_seen_at
                   FROM phase4_concepts"""
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_unjudged_phase4_steps(self, limit: int = 1000) -> list[dict]:
        """Return steps with judged_at IS NULL — Phase B's worklist."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT step_id, chain_id, step_idx, target_concept,
                          target_lemma, raw_response, thought_block,
                          final_answer, parse_failure
                   FROM phase4_steps
                   WHERE judged_at IS NULL
                   ORDER BY step_id
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def phase4_summary(self) -> dict:
        """Quick-glance stats for the dream-loop launcher and monitor."""
        with self._conn() as conn:
            chain_row = conn.execute(
                """SELECT COUNT(*) AS n_chains,
                          SUM(n_steps) AS total_steps,
                          SUM(CASE WHEN end_reason='length_cap' THEN 1 ELSE 0 END) AS n_length_cap,
                          SUM(CASE WHEN end_reason='self_loop' THEN 1 ELSE 0 END) AS n_self_loop,
                          SUM(CASE WHEN end_reason='coherence_break' THEN 1 ELSE 0 END) AS n_coherence_break
                   FROM phase4_chains"""
            ).fetchone()
            unjudged = conn.execute(
                "SELECT COUNT(*) AS n FROM phase4_steps WHERE judged_at IS NULL"
            ).fetchone()
            n_concepts = conn.execute(
                "SELECT COUNT(*) AS n FROM phase4_concepts"
            ).fetchone()
        return {
            "n_chains": int(chain_row["n_chains"] or 0),
            "total_steps": int(chain_row["total_steps"] or 0),
            "n_length_cap": int(chain_row["n_length_cap"] or 0),
            "n_self_loop": int(chain_row["n_self_loop"] or 0),
            "n_coherence_break": int(chain_row["n_coherence_break"] or 0),
            "n_unjudged_steps": int(unjudged["n"] or 0),
            "n_concepts": int(n_concepts["n"] or 0),
        }
