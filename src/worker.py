"""Phase 2 worker — polls queue/pending/, evaluates candidates, writes results.

Long-lived process. Loads Gemma3-12B once at startup and keeps it resident.
On each iteration:
  1. Scan queue/pending/ for the oldest JSON candidate spec.
  2. Move it to queue/running/ (atomic-ish move; see caveat below).
  3. Derive the direction, run the fitness evaluation, write results to SQLite
     and to runs/YYYY-MM-DD/{candidate_id}/.
  4. Move to queue/done/ on success or queue/failed/ on error.
  5. If the queue is empty, sleep briefly and try again.

Launch with:
    setsid nohup ./scripts/start_worker.sh > /tmp/worker.log 2>&1 &

SIGTERM-handled for clean shutdown between candidates. Signals received mid-
evaluation are deferred until the current candidate finishes writing, so the
DB never gets a half-recorded trial.

Caveat: this is a single-worker design. If you run two workers against the
same queue, both will race on the `mv pending -> running` step and one will
fail. That's acceptable for now (no reason to run multiple workers on one
Mac Studio with one GPU).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bridge import DetectionPipeline, load_gemma_mps
from src.db import ResultsDB
from src.evaluate import CandidateSpec, evaluate_candidate, load_eval_sets
from src.judges.claude_judge import ClaudeJudge

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "queue"
RUNS = REPO / "runs"
DB_PATH = REPO / "data" / "results.db"
HELD_OUT_PATH = REPO / "data" / "eval_sets" / "held_out_concepts.json"

POLL_INTERVAL_S = 5
IDLE_LOG_INTERVAL_S = 60


class _ShutdownRequested(Exception):
    pass


_shutdown = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _shutdown
        _shutdown = True
        print(f"[worker] received signal {signum}, shutting down after current candidate", flush=True)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _ensure_dirs() -> None:
    for sub in ("pending", "running", "done", "failed"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)


def _oldest_pending() -> Path | None:
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


def _maybe_promote_mutation(
    db: ResultsDB,
    candidate_id: str,
    lineage_meta: dict,
    new_score: float,
) -> None:
    """Phase 2c commit-or-reject rule.

    Each lineage has ONE leader at a time — the highest-scoring member so
    far. A mutation is "committed" only if it beats the current leader
    (not just its immediate parent). This avoids a race where two parallel
    mutations both commit by each beating their stale parent, leaving the
    lineage with multiple leaders.

    Candidates with no parent (legacy or seeds) are no-ops here — seeds
    are marked leader by ``scripts/seed_lineages.py``, legacy have no
    lineage.
    """
    parent_id = lineage_meta.get("parent_candidate_id")
    lineage_id = lineage_meta.get("lineage_id")
    if not parent_id or not lineage_id:
        return
    generation = lineage_meta.get("generation", "?")
    mutation_type = lineage_meta.get("mutation_type", "?")

    # Find the current leader of this lineage. Compare against THAT, not
    # the mutation's immediate parent.
    leaders = [l for l in db.get_leaders() if l.get("lineage_id") == lineage_id]
    if not leaders:
        print(
            f"[worker] lineage {lineage_id[:8]} has no leader; "
            "treating this as seed",
            flush=True,
        )
        # No current leader — make this one the leader.
        with db._conn() as conn:
            conn.execute("UPDATE candidates SET is_leader=1 WHERE id=?", (candidate_id,))
        return

    # Pick the single best leader if there somehow are multiple (legacy from
    # the buggy parent-compare version).
    current_leader = max(leaders, key=lambda l: l.get("score") or 0.0)
    current_leader_id = current_leader["id"]
    current_score = float(current_leader.get("score") or 0.0)
    lid_short = lineage_id[:8]

    if new_score > current_score:
        # Demote ALL current leaders in this lineage (in case of legacy
        # stale multi-leader state) and promote this one.
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


def _process_one(
    path: Path,
    pipeline: DetectionPipeline,
    db: ResultsDB,
    held_out_concepts: list[str],
    control_concepts: list[str],
) -> None:
    spec_dict = json.loads(path.read_text())
    spec = CandidateSpec.from_dict(spec_dict)

    # Hash for dedup; must match researcher's spec_hash
    from src.strategies.random_explore import spec_hash
    h = spec_hash(spec)

    # Phase 2c lineage fields live in spec_dict["_lineage"] (nested so they
    # don't collide with CandidateSpec schema).
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
    )
    db.set_candidate_status(spec.id, "running")

    moved = _move(path, QUEUE / "running")
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] processing {spec.id}  "
        f"concept={spec.concept!r} L={spec.layer_idx} "
        f"eff={spec.target_effective:.0f}",
        flush=True,
    )

    try:
        result = evaluate_candidate(
            spec=spec,
            pipeline=pipeline,
            db=db,
            held_out_concepts=held_out_concepts,
            control_concepts=control_concepts,
        )
        result_summary = {
            "candidate_id": spec.id,
            "score": result.score,
            "detection_rate": result.detection_rate,
            "identification_rate": result.identification_rate,
            "fpr": result.fpr,
            "coherence_rate": result.coherence_rate,
            "n_held_out": result.n_held_out,
            "n_controls": result.n_controls,
            "components": result.components,
        }
        _write_run_artifact(spec.id, spec.to_dict(), result_summary)
        db.set_candidate_status(spec.id, "done")
        _move(moved, QUEUE / "done")
        _maybe_promote_mutation(db, spec.id, lineage_meta, result.score)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] done      {spec.id}  "
            f"score={result.score:.3f}",
            flush=True,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[worker] FAILED {spec.id}: {e}\n{tb}", flush=True)
        db.set_candidate_status(spec.id, "failed", error_message=str(e)[:500])
        _move(moved, QUEUE / "failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH,
                    help="SQLite DB path. For abliterated runs, use "
                         "data/results_abliterated.db")
    ap.add_argument("--model", default="gemma3_12b",
                    help="Model slug (from MODEL_NAME_MAP). For Phase 1.5 use "
                         "gemma3_12b_abliterated.")
    ap.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--held-out", type=Path, default=HELD_OUT_PATH)
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="Exit after processing N candidates (0 = unlimited)")
    args = ap.parse_args()

    _install_signal_handlers()
    _ensure_dirs()

    print(f"[worker] loading model {args.model} ...", flush=True)
    model = load_gemma_mps(args.model)
    judge = ClaudeJudge(
        model=args.judge_model,
        cache_path=REPO / "data" / "judge_cache.sqlite",
    )
    pipeline = DetectionPipeline(model=model, judge=judge)

    db = ResultsDB(args.db)
    held_out, controls = load_eval_sets(args.held_out)
    print(
        f"[worker] ready. eval set: {len(held_out)} held-out, {len(controls)} controls",
        flush=True,
    )

    n_processed = 0
    last_idle_log = 0.0
    while not _shutdown:
        if args.max_candidates and n_processed >= args.max_candidates:
            print(f"[worker] hit max-candidates={args.max_candidates}, exiting", flush=True)
            break

        path = _oldest_pending()
        if path is None:
            now = time.time()
            if now - last_idle_log > IDLE_LOG_INTERVAL_S:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] queue empty, waiting...", flush=True)
                last_idle_log = now
            time.sleep(POLL_INTERVAL_S)
            continue

        _process_one(path, pipeline, db, held_out, controls)
        n_processed += 1

    summary = db.candidates_summary()
    print(f"[worker] shutdown. processed {n_processed}. DB summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
