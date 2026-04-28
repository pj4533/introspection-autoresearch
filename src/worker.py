"""Three-phase autoresearch worker (Phase 2g, all-local).

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
  ├─ C. PROPOSE   (no model loaded; pure CPU work) ──────────────────────┤
  │   - call sae_capraro.generate_candidates(fault_line=…)               │
  │   - the strategy reads data/sae_features/capraro_buckets.json        │
  │     and the leaderboard to pick fresh SAE-feature candidates         │
  │   - write specs to queue/pending/                                    │
  ├─ D. RELOAD    (Gemma loaded) ───────────────────────────────────────┤
  │   Back to A.                                                          │
  └───────────────────────────────────────────────────────────────────────┘

State invariant: at most ONE model is loaded across the whole process. The
HandleRegistry enforces this on every transition. Phase 2g removed the
proposer model — SAE features come from a static Neuronpedia-derived
bucket file, no LLM proposer needed. Phase C is now pure CPU work, so
no swap I/O between B and C.

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
    spec_hash,
)
from src.judges.base import Judge
from src.judges.local_mlx_judge import LocalMLXJudge
from src.models.registry import (
    GemmaHandle,
    HandleRegistry,
    MLXHandle,
)

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "queue"
RUNS = REPO / "runs"
DB_PATH = REPO / "data" / "results.db"
HELD_OUT_PATH = REPO / "data" / "eval_sets" / "held_out_concepts.json"

DEFAULT_BATCH_SIZE = 16
DEFAULT_PROPOSE_THRESHOLD = 4   # if queue/pending has < this many, run Phase C
DEFAULT_PROPOSE_N = 16          # candidates per Phase C call

# All seven Capraro fault lines — Phase 2g rotates through these.
DEFAULT_FAULT_LINES = (
    "experience,causality,grounding,metacognition,parsing,motivation,value"
)


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
        proposer_model=spec.proposer_model,
    )
    db.set_candidate_status(spec.id, "running")
    moved = _move(path, QUEUE / "running")

    # Use the SAE fault-line tag for log readability when present.
    fault_tag = spec.sae_fault_line or spec.strategy
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] gen-A    [{fault_tag}] "
        f"{spec.id}  concept={spec.concept!r} L={spec.layer_idx} "
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
        it = items_by_id.get(cid)
        if it is not None:
            tag = it.spec.sae_fault_line or it.spec.strategy
        else:
            row = db.get_candidate(cid)
            tag = row["strategy"] if row else "unknown"
        try:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] judge-B  [{tag}] {cid}",
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
            f"[{datetime.now().strftime('%H:%M:%S')}] done     [{tag}] "
            f"{cid}  score={result.score:.3f} "
            f"det={result.detection_rate:.2f} ident={result.identification_rate:.2f} "
            f"fpr={result.fpr:.2f} coh={result.coherence_rate:.2f}",
            flush=True,
        )


def _phase_c_propose(
    db: ResultsDB,
    fault_line_id: Optional[str],
    propose_n: int,
) -> int:
    """Generate `propose_n` SAE-feature candidates for `fault_line_id`.

    Phase 2g: no proposer model is loaded — `sae_capraro` reads the static
    `data/sae_features/capraro_buckets.json` and the leaderboard, picks
    fresh feature candidates per its sub-mode mix (explore / neighbors /
    replicate / cross_fault), and writes them to queue/pending/.
    """
    from src.strategies.sae_capraro import generate_candidates as gen
    specs = gen(
        n=propose_n,
        db=db,
        fault_line=fault_line_id,
    )

    n_written = 0
    for spec in specs:
        path = QUEUE / "pending" / f"{spec.id}.json"
        spec_dict = spec.to_dict()
        lineage_meta = getattr(spec, "_lineage_meta", None)
        if lineage_meta:
            spec_dict["_lineage"] = lineage_meta
        path.write_text(json.dumps(spec_dict, indent=2) + "\n")
        n_written += 1
    label = fault_line_id or "sae_capraro"
    print(f"[worker] phase C ({label}) wrote {n_written} new candidates to "
          f"queue/pending", flush=True)
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    propose_threshold: int = DEFAULT_PROPOSE_THRESHOLD,
    propose_n: int = DEFAULT_PROPOSE_N,
    fault_lines: Optional[list[str]] = None,
    max_cycles: int = 0,
    abliteration_mode: str = "vanilla",
    poll_interval_s: int = 5,
) -> int:
    """Run the three-phase loop until shutdown or max_cycles exceeded.

    Phase 2g: no `proposer_factory` parameter — Phase C runs CPU-only.

    `fault_lines` is the round-robin rotation of Capraro fault lines. Each
    successful Phase C picks `fault_lines[propose_index % len(fault_lines)]`
    and asks `sae_capraro` for that fault line's variants. Default rotation
    is all 7 Capraro fault lines.

    Returns the number of complete A→B cycles run (Phase C is a side-effect
    inside the cycle, not its own counted unit).
    """
    cycles = 0
    rotation = fault_lines or []
    rotation_key = (
        f"propose_index|{','.join(rotation)}" if rotation
        else "propose_index|<no-rotation>"
    )
    try:
        propose_index = int(db.get_meta(rotation_key, "0") or "0")
    except (TypeError, ValueError):
        propose_index = 0
    if propose_index > 0 and rotation:
        next_label = rotation[propose_index % len(rotation)]
        print(
            f"[worker] resuming rotation at index {propose_index} "
            f"(next fault line: {next_label})",
            flush=True,
        )

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
        items: list[PhaseAItem] = []
        queue_has_work = _oldest_pending() is not None
        orphan_count = len(db.pending_candidate_ids())
        if not queue_has_work and orphan_count == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] queue empty + no "
                "pending; jumping to Phase C",
                flush=True,
            )
            registry.unload_all()
        else:
            registry.activate(registry.gemma)
            pipeline = registry.gemma.obj
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [phase A] starting "
                f"batch — up to {batch_size} candidates × 12 probes each "
                f"(~{batch_size * 75 // 60} min est)",
                flush=True,
            )
            while len(items) < batch_size and not _shutdown:
                path = _oldest_pending()
                if path is None:
                    break
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] [phase A] "
                    f"{len(items) + 1}/{batch_size} ...",
                    flush=True,
                )
                it = _phase_a_one(
                    path, pipeline, db, held_out, controls, abliteration_mode
                )
                if it is not None:
                    items.append(it)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [phase A] complete "
                f"({len(items)} candidates queued for judging)",
                flush=True,
            )

        if items or orphan_count > 0:
            # Phase B: JUDGE
            registry.activate(registry.judge)
            judge = judge_factory(registry.judge.obj)
            n_to_judge = len(items) + orphan_count
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [phase B] judging "
                f"{n_to_judge} candidate(s), ~{n_to_judge * 12} probes total",
                flush=True,
            )
            _phase_b_drain(db, judge, items, abliteration_mode)
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [phase B] complete",
                flush=True,
            )
            cycles += 1
            if _shutdown:
                break

        # ---------- C. PROPOSE -----------------------------------------
        # Phase 2g: no model load — pure CPU work. Unload any model first
        # so we run lean.
        pending_left = len(list((QUEUE / "pending").glob("*.json")))
        if pending_left < propose_threshold:
            registry.unload_all()
            if rotation:
                next_fault_line = rotation[propose_index % len(rotation)]
            else:
                next_fault_line = None
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] [phase C] cycle "
                f"{propose_index} target: "
                f"{next_fault_line or 'sae_capraro (no rotation)'} "
                f"(building {propose_n} candidates from buckets)",
                flush=True,
            )
            try:
                _phase_c_propose(db, next_fault_line, propose_n)
                propose_index += 1
                db.set_meta(rotation_key, str(propose_index))
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
        j = LocalMLXJudge(model_path=judge_model_path)
        j._model, j._tokenizer = pair
        return j
    return _factory


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Three-phase Phase 2g worker.")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--held-out", type=Path, default=HELD_OUT_PATH)
    ap.add_argument(
        "--judge-model-path",
        default=str(Path.home() / "models/Qwen3.6-35B-A3B-8bit"),
        help="Path to MLX judge model (Qwen3.6-35B-A3B-8bit by default).",
    )
    ap.add_argument("--gemma-model", default="gemma3_12b")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--propose-threshold", type=int, default=DEFAULT_PROPOSE_THRESHOLD)
    ap.add_argument("--propose-n", type=int, default=DEFAULT_PROPOSE_N)
    ap.add_argument(
        "--fault-lines",
        default=DEFAULT_FAULT_LINES,
        help="Comma-separated rotation of Capraro fault lines for Phase C. "
             "Default: all 7 (experience, causality, grounding, "
             "metacognition, parsing, motivation, value).",
    )
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="0 = unlimited. Each completed A→B counts as one cycle.")
    args = ap.parse_args()

    fault_lines = [s.strip() for s in args.fault_lines.split(",") if s.strip()]

    _install_signal_handlers()
    _ensure_dirs()

    print("[worker] starting three-phase Phase 2g worker", flush=True)
    print(f"           gemma:    {args.gemma_model}", flush=True)
    print(f"           judge:    {args.judge_model_path}", flush=True)
    print(f"           strategy: sae_capraro (no proposer)", flush=True)

    registry = HandleRegistry(
        gemma=GemmaHandle(model_id=args.gemma_model, expected_ram_gb=0.0),
        judge=MLXHandle(model_path=args.judge_model_path, expected_ram_gb=0.0),
    )

    db = ResultsDB(args.db)
    held_out, controls = load_eval_sets(args.held_out)
    print(f"[worker] eval set: {len(held_out)} held-out, "
          f"{len(controls)} controls", flush=True)

    if fault_lines:
        print(f"[worker] fault-line rotation: {' → '.join(fault_lines)}", flush=True)
    else:
        print("[worker] no fault-line rotation; sae_capraro idle until configured",
              flush=True)

    main_loop(
        registry=registry,
        db=db,
        held_out=held_out,
        controls=controls,
        judge_factory=default_judge_factory(args.judge_model_path),
        batch_size=args.batch_size,
        propose_threshold=args.propose_threshold,
        propose_n=args.propose_n,
        fault_lines=fault_lines,
        max_cycles=args.max_cycles,
        abliteration_mode="vanilla",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
