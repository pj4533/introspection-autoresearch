"""Four-phase autoresearch worker. All-local pipeline (ADR-019).

State machine:

  ┌─ A. GENERATE  (Gemma loaded) ────────────────────────────────────────┐
  │   For up to BATCH_SIZE pending candidates:                            │
  │     - load spec from queue/pending/                                   │
  │     - record candidate in DB, mark `running`                          │
  │     - move file to queue/running/                                     │
  │     - call evaluate.phase_a_generate                                  │
  │     - 12 Gemma probes → pending_responses table                       │
  ├─ B. JUDGE     (Gemma unloaded, judge loaded) ────────────────────────┤
  │   For each candidate_id with rows in pending_responses:               │
  │     - call evaluate.phase_b_judge                                     │
  │     - 12 judge calls → evaluations + fitness_scores                   │
  │     - delete pending_responses rows                                   │
  │     - mark candidate `done`                                           │
  │     - move queue file to queue/done/                                  │
  │     - run lineage commit-or-reject                                    │
  ├─ C. PROPOSE   (judge unloaded, proposer loaded; only if queue low) ─┤
  │   - call directed_capraro.generate_candidates (or novel_contrast)    │
  │   - write specs to queue/pending/                                    │
  ├─ D. RELOAD    (proposer unloaded, Gemma loaded) ────────────────────┤
  │   Back to A.                                                          │
  └───────────────────────────────────────────────────────────────────────┘

State invariant: at most ONE model is loaded across the whole process. The
HandleRegistry enforces this on every transition.

Crash recovery: SIGTERM/SIGINT triggers shutdown after the current operation
finishes (no half-complete candidates). On restart, any rows left in
`pending_responses` are picked up immediately by Phase B before generating
new responses — no work is lost.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import ResultsDB
from src.evaluate import (
    CandidateSpec,
    load_eval_sets,
    phase_a_generate,
    phase_b_judge,
)
from src.judges.base import Judge
from src.judges.local_mlx_judge import LocalMLXJudge
from src.models.registry import (
    GemmaHandle,
    HandleRegistry,
    MLXHandle,
)
from src.proposers import LocalMLXProposer
from src.proposers.base import Proposer
from src.strategies.random_explore import spec_hash

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "queue"
RUNS = REPO / "runs"
DB_PATH = REPO / "data" / "results.db"
HELD_OUT_PATH = REPO / "data" / "eval_sets" / "held_out_concepts.json"

DEFAULT_BATCH_SIZE = 16
DEFAULT_PROPOSE_THRESHOLD = 4   # if queue/pending has < this many, run Phase C
DEFAULT_PROPOSE_N = 16          # candidates to ask proposer for per Phase C


# ----------------------------------------------------------------------
# Shutdown handling
# ----------------------------------------------------------------------

_shutdown = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _shutdown
        _shutdown = True
        print(
            f"[worker] received signal {signum}, shutting down at next safe point",
            flush=True,
        )

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _ensure_dirs() -> None:
    for sub in ("pending", "running", "done", "failed"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)


def _oldest_pending() -> Optional[Path]:
    pending = QUEUE / "pending"
    candidates = sorted(
        pending.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[0] if candidates else None


def _move(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    src.rename(dst)
    return dst


def _write_run_artifact(candidate_id: str, spec: dict, result_summary: dict) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    run_dir = RUNS / today / candidate_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    (run_dir / "fitness.json").write_text(json.dumps(result_summary, indent=2) + "\n")


# ----------------------------------------------------------------------
# Lineage promotion (verbatim from old worker.py — same contract)
# ----------------------------------------------------------------------

def _maybe_promote_mutation(
    db: ResultsDB,
    candidate_id: str,
    lineage_meta: dict,
    new_score: float,
) -> None:
    parent_id = lineage_meta.get("parent_candidate_id")
    lineage_id = lineage_meta.get("lineage_id")
    if not parent_id or not lineage_id:
        return
    generation = lineage_meta.get("generation", "?")
    mutation_type = lineage_meta.get("mutation_type", "?")
    leaders = [l for l in db.get_leaders() if l.get("lineage_id") == lineage_id]
    if not leaders:
        with db._conn() as conn:
            conn.execute("UPDATE candidates SET is_leader=1 WHERE id=?", (candidate_id,))
        return
    current_leader = max(leaders, key=lambda l: l.get("score") or 0.0)
    current_score = float(current_leader.get("score") or 0.0)
    lid_short = lineage_id[:8]
    if new_score > current_score:
        with db._conn() as conn:
            conn.execute(
                "UPDATE candidates SET is_leader=0 WHERE lineage_id=?",
                (lineage_id,),
            )
            conn.execute(
                "UPDATE candidates SET is_leader=1 WHERE id=?",
                (candidate_id,),
            )
        print(
            f"[lineage {lid_short}] gen{generation} {mutation_type}: "
            f"{current_score:.3f} -> {new_score:.3f} ✓ COMMITTED",
            flush=True,
        )
    else:
        print(
            f"[lineage {lid_short}] gen{generation} {mutation_type}: "
            f"current leader {current_score:.3f} vs this {new_score:.3f} "
            "✗ rejected",
            flush=True,
        )


# ----------------------------------------------------------------------
# Phase implementations
# ----------------------------------------------------------------------

@dataclass
class PhaseAItem:
    """Bookkeeping for a candidate that's been generated but not yet judged."""
    candidate_id: str
    spec: CandidateSpec
    spec_dict: dict
    queue_running_path: Path
    lineage_meta: dict


def _phase_a_one(
    path: Path,
    pipeline,
    db: ResultsDB,
    held_out: list[str],
    controls: list[str],
    abliteration_mode: str,
) -> Optional[PhaseAItem]:
    """Run Phase A for one queued candidate file.

    Returns a PhaseAItem on success (so Phase B can pick it up later), or
    None if the candidate was rejected (e.g. duplicate hash).
    """
    spec_dict = json.loads(path.read_text())
    spec = CandidateSpec.from_dict(spec_dict)

    h = spec_hash(spec, abliteration_mode=abliteration_mode)
    lineage_meta = spec_dict.get("_lineage") or {}

    db.insert_candidate(
        candidate_id=spec.id,
        strategy=spec.strategy,
        spec_json=json.dumps(spec_dict),
        spec_hash=h,
        concept=spec.concept,
        layer_idx=spec.layer_idx,
        target_effective=spec.target_effective,
        derivation_method=spec.derivation_method,
        lineage_id=lineage_meta.get("lineage_id"),
        parent_candidate_id=lineage_meta.get("parent_candidate_id"),
        generation=int(lineage_meta.get("generation", 0) or 0),
        mutation_type=lineage_meta.get("mutation_type"),
        mutation_detail=lineage_meta.get("mutation_detail"),
        abliteration_mode=abliteration_mode,
    )
    db.set_candidate_status(spec.id, "running")
    moved = _move(path, QUEUE / "running")

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] gen-A    {spec.id}  "
        f"concept={spec.concept!r} L={spec.layer_idx} "
        f"eff={spec.target_effective:.0f}",
        flush=True,
    )
    try:
        phase_a_generate(
            spec=spec,
            pipeline=pipeline,
            db=db,
            held_out_concepts=held_out,
            control_concepts=controls,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[worker] phase_a FAILED {spec.id}: {e}\n{tb}", flush=True)
        db.set_candidate_status(spec.id, "failed", error_message=str(e)[:500])
        _move(moved, QUEUE / "failed")
        return None

    return PhaseAItem(
        candidate_id=spec.id,
        spec=spec,
        spec_dict=spec_dict,
        queue_running_path=moved,
        lineage_meta=lineage_meta,
    )


def _phase_b_drain(
    db: ResultsDB,
    judge: Judge,
    items: list[PhaseAItem],
    abliteration_mode: str,
) -> None:
    """Score every pending response in `items`, finalize fitness, move queue files.

    Crash-recovery: if items is empty but the DB has orphan pending rows,
    they're judged anyway (this is the recovery path).
    """
    candidate_ids = [it.candidate_id for it in items]
    # Add any orphan rows that aren't in `items` (crash recovery).
    orphan_ids = [
        cid for cid in db.pending_candidate_ids()
        if cid not in set(candidate_ids)
    ]
    if orphan_ids:
        print(
            f"[worker] phase B picking up {len(orphan_ids)} orphan "
            f"pending candidate(s) from prior session",
            flush=True,
        )
    items_by_id = {it.candidate_id: it for it in items}

    for cid in candidate_ids + orphan_ids:
        try:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] judge-B  {cid}",
                flush=True,
            )
            result = phase_b_judge(
                candidate_id=cid,
                judge=judge,
                db=db,
                abliteration_mode=abliteration_mode,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[worker] phase_b FAILED {cid}: {e}\n{tb}", flush=True)
            db.set_candidate_status(cid, "failed", error_message=str(e)[:500])
            it = items_by_id.get(cid)
            if it is not None:
                _move(it.queue_running_path, QUEUE / "failed")
            continue

        # Successful judging — promote lineage and write run artifact.
        it = items_by_id.get(cid)
        if it is not None:
            result_summary = {
                "candidate_id": cid,
                "score": result.score,
                "detection_rate": result.detection_rate,
                "identification_rate": result.identification_rate,
                "fpr": result.fpr,
                "coherence_rate": result.coherence_rate,
                "n_held_out": result.n_held_out,
                "n_controls": result.n_controls,
                "components": result.components,
            }
            _write_run_artifact(cid, it.spec.to_dict(), result_summary)
            _move(it.queue_running_path, QUEUE / "done")
            _maybe_promote_mutation(db, cid, it.lineage_meta, result.score)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] done     {cid}  "
            f"score={result.score:.3f}",
            flush=True,
        )


def _phase_c_propose(
    proposer: Proposer,
    db: ResultsDB,
    fault_line_id: Optional[str],
    propose_n: int,
) -> int:
    """Generate `propose_n` new candidate specs and queue them.

    Returns the number of candidates actually queued. If `fault_line_id` is
    set, runs `directed_capraro.generate_candidates`. Otherwise runs
    `novel_contrast.generate_candidates`.
    """
    if fault_line_id is not None:
        from src.strategies.directed_capraro import generate_candidates as gen
        specs = gen(
            n=propose_n,
            db=db,
            fault_line_id=fault_line_id,
            mode="opus",
            proposer=proposer,
        )
    else:
        from src.strategies.novel_contrast import generate_candidates as gen
        specs = gen(
            n=propose_n,
            db=db,
            proposer=proposer,
        )

    n_written = 0
    for spec in specs:
        path = QUEUE / "pending" / f"{spec.id}.json"
        path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
        n_written += 1
    print(f"[worker] phase C wrote {n_written} new candidates to queue/pending",
          flush=True)
    return n_written


# ----------------------------------------------------------------------
# Main state machine
# ----------------------------------------------------------------------

def main_loop(
    *,
    registry: HandleRegistry,
    db: ResultsDB,
    held_out: list[str],
    controls: list[str],
    judge_factory,
    proposer_factory,
    batch_size: int = DEFAULT_BATCH_SIZE,
    propose_threshold: int = DEFAULT_PROPOSE_THRESHOLD,
    propose_n: int = DEFAULT_PROPOSE_N,
    fault_line_id: Optional[str] = None,
    max_cycles: int = 0,
    abliteration_mode: str = "vanilla",
    poll_interval_s: int = 5,
) -> int:
    """Run the four-phase loop until shutdown or max_cycles exceeded.

    `judge_factory` / `proposer_factory` build a Judge / Proposer from the
    handle's loaded `(model, tokenizer)` pair. The worker doesn't construct
    these eagerly — only after the corresponding handle is loaded.

    Returns the number of complete A-B cycles run (Phase C is a side-effect
    inside the cycle, not its own counted unit).
    """
    cycles = 0

    # Crash recovery on startup: if there are orphan pending rows, judge them
    # FIRST before generating anything new.
    orphan_count = len(db.pending_candidate_ids())
    if orphan_count > 0:
        print(
            f"[worker] STARTUP crash-recovery: {orphan_count} orphan "
            f"pending candidate(s) from prior session — judging first",
            flush=True,
        )
        registry.activate(registry.judge)
        judge = judge_factory(registry.judge.obj)
        _phase_b_drain(db, judge, [], abliteration_mode)

    while not _shutdown:
        if max_cycles and cycles >= max_cycles:
            print(f"[worker] hit max_cycles={max_cycles}, exiting", flush=True)
            break

        # ---------- A. GENERATE -----------------------------------------
        registry.activate(registry.gemma)
        pipeline = registry.gemma.obj
        items: list[PhaseAItem] = []
        while len(items) < batch_size and not _shutdown:
            path = _oldest_pending()
            if path is None:
                break
            it = _phase_a_one(
                path, pipeline, db, held_out, controls, abliteration_mode
            )
            if it is not None:
                items.append(it)

        if not items and len(db.pending_candidate_ids()) == 0:
            # Queue empty AND no orphan pending — go straight to Phase C.
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] queue empty + no "
                "pending; jumping to Phase C",
                flush=True,
            )
            registry.unload_all()
        else:
            # Phase B: JUDGE
            registry.activate(registry.judge)
            judge = judge_factory(registry.judge.obj)
            _phase_b_drain(db, judge, items, abliteration_mode)
            cycles += 1
            if _shutdown:
                break

        # ---------- C. PROPOSE -----------------------------------------
        # Run only if the queue is empty (or we just had nothing else to do).
        pending_left = len(list((QUEUE / "pending").glob("*.json")))
        if pending_left < propose_threshold:
            registry.activate(registry.proposer)
            proposer = proposer_factory(registry.proposer.obj)
            try:
                _phase_c_propose(proposer, db, fault_line_id, propose_n)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[worker] phase_c FAILED: {e}\n{tb}", flush=True)
        else:
            if items:
                # Skip C — queue still has work.
                pass
            else:
                # Idle wait — queue is empty AND we already pivoted away.
                time.sleep(poll_interval_s)

    summary = db.candidates_summary()
    print(
        f"[worker] shutdown. cycles={cycles}. DB summary: {summary}",
        flush=True,
    )
    registry.unload_all()
    return cycles


# ----------------------------------------------------------------------
# Default factories for production use (real models)
# ----------------------------------------------------------------------

def default_judge_factory(judge_model_path: str):
    """Returns a callable: (model_pair) -> LocalMLXJudge bound to that pair."""
    def _factory(pair) -> LocalMLXJudge:
        # The judge needs a model_path string for cache namespacing. Re-use
        # the directory name so old caches (from calibration) are reused.
        j = LocalMLXJudge(model_path=judge_model_path)
        # Hot-wire the loaded pair so the judge skips its own _ensure_loaded.
        j._model, j._tokenizer = pair
        return j
    return _factory


def default_proposer_factory(_proposer_model_path: str):
    def _factory(pair) -> LocalMLXProposer:
        return LocalMLXProposer(loaded_pair=pair)
    return _factory


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Four-phase local-only worker.")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--held-out", type=Path, default=HELD_OUT_PATH)
    ap.add_argument(
        "--judge-model-path",
        default=str(Path.home() / "models/Qwen3.6-35B-A3B-8bit"),
        help="Path to MLX judge model (Qwen3.6-35B-A3B-8bit by default).",
    )
    ap.add_argument(
        "--proposer-model-path",
        default=str(Path.home() / "models/Qwen3.6-27B-MLX-8bit"),
        help="Path to MLX proposer model.",
    )
    ap.add_argument("--gemma-model", default="gemma3_12b")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--propose-threshold", type=int, default=DEFAULT_PROPOSE_THRESHOLD)
    ap.add_argument("--propose-n", type=int, default=DEFAULT_PROPOSE_N)
    ap.add_argument("--fault-line", default=None,
                    help="If set, Phase C uses directed_capraro for this fault "
                         "line. Otherwise novel_contrast.")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="0 = unlimited. Each completed A→B counts as one cycle.")
    args = ap.parse_args()

    _install_signal_handlers()
    _ensure_dirs()

    print("[worker] starting four-phase local-only worker", flush=True)
    print(f"           gemma:    {args.gemma_model}", flush=True)
    print(f"           judge:    {args.judge_model_path}", flush=True)
    print(f"           proposer: {args.proposer_model_path}", flush=True)

    registry = HandleRegistry(
        gemma=GemmaHandle(model_id=args.gemma_model, expected_ram_gb=24.0),
        judge=MLXHandle(model_path=args.judge_model_path, expected_ram_gb=40.0),
        proposer=MLXHandle(model_path=args.proposer_model_path, expected_ram_gb=32.0),
    )

    db = ResultsDB(args.db)
    held_out, controls = load_eval_sets(args.held_out)
    print(f"[worker] eval set: {len(held_out)} held-out, "
          f"{len(controls)} controls", flush=True)

    return main_loop(
        registry=registry,
        db=db,
        held_out=held_out,
        controls=controls,
        judge_factory=default_judge_factory(args.judge_model_path),
        proposer_factory=default_proposer_factory(args.proposer_model_path),
        batch_size=args.batch_size,
        propose_threshold=args.propose_threshold,
        propose_n=args.propose_n,
        fault_line_id=args.fault_line,
        max_cycles=args.max_cycles,
        abliteration_mode="vanilla",
    )


if __name__ == "__main__":
    sys.exit(main())
