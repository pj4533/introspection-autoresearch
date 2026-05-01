"""Phase 4 — Dream Walks + Forbidden Map overnight loop.

Continuously runs free-association chains under steering, periodically
swapping in the Qwen judge to score the backlog. Writes everything to
the Phase 4 tables in data/results.db.

Pattern (per cycle):

  Phase A — Gemma loaded:
    For BATCH_CHAINS chains:
      1. Pick seed (force Goblin first, then priority-weighted sample)
      2. Run a dream walk (up to LENGTH_CAP steps)
      3. Each step's raw response written to phase4_steps with NULL judge fields

  Phase B — release Gemma → load Qwen judge:
    1. Pull all unjudged steps from phase4_steps
    2. For each: score behavior (score_freeassoc on final_answer) +
       CoT (score_cot_recognition on thought_block)
    3. Write judgments back; update phase4_concepts tallies

  Phase C — release Qwen → reload Gemma → next cycle

Stop condition: until killed (`pkill -f phase4_dreamloop`) or
MAX_CHAINS reached.

Usage:
    setsid nohup python scripts/run_phase4_dreamloop.py \
        > logs/phase4_dreamloop.log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.db import ResultsDB
from src.judges.local_mlx_judge import LocalMLXJudge
from src.phase3.gemma4_loader import load_gemma4
from src.phase3.hooks import uninstall_all
from src.phase4.cot_parser import is_coherent_answer
from src.phase4.dream_walk import (
    DEFAULT_LAYER, DEFAULT_TARGET_EFFECTIVE, DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_LENGTH_CAP, run_chain,
)
from src.phase4.seed_pool import SeedPool, load_default_seeds


DB_PATH = REPO / "data" / "results.db"
DEFAULT_JUDGE_PATH = str(Path.home() / "models/Qwen3.6-35B-A3B-8bit")
DEFAULT_BATCH_CHAINS = 5
DEFAULT_MAX_CHAINS = 0  # 0 = unlimited; run until killed


def _release_gemma(handle):
    """Drop MLX Gemma weights so the judge can load. Same pattern as
    Phase 3 sweep harness."""
    handle.model = None
    handle.tokenizer = None
    gc.collect()
    try:
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _release_judge(judge):
    judge._model = None
    judge._tokenizer = None
    gc.collect()
    try:
        mx.metal.clear_cache()
        mx.metal.reset_peak_memory()
    except Exception:
        pass


def _phase_a_chains(handle, db, seed_pool, args, n_chains_run, log):
    """Run BATCH_CHAINS dream walks."""
    for i in range(args.batch):
        # Force Goblin as the very first chain seed (Codex directive
        # framing — see seed_pool.CODEX_SUPPRESSED_CREATURES).
        force_lemma = "goblin" if n_chains_run == 0 else None
        seed = seed_pool.sample_seed(force_lemma=force_lemma)

        log(f"[chain {n_chains_run + 1}] starting from {seed.display!r} "
            f"(visits={seed.visits}, behavior_hits={seed.behavior_hits})")
        t0 = time.time()
        chain_id, end_reason, n_steps, records = run_chain(
            handle=handle,
            db=db,
            seed_pool=seed_pool,
            seed_concept_display=seed.display,
            seed_concept_lemma=seed.lemma,
            layer_idx=args.layer,
            target_effective=args.target_effective,
            max_new_tokens=args.max_new_tokens,
            length_cap=args.length_cap,
        )
        elapsed = time.time() - t0
        log(f"[chain {n_chains_run + 1}] DONE id={chain_id} "
            f"end={end_reason} n_steps={n_steps} "
            f"elapsed={elapsed:.1f}s "
            f"path: {' → '.join([r.target_concept for r in records[:6]])}"
            f"{'…' if len(records) > 6 else ''}")
        n_chains_run += 1
        if args.max_chains > 0 and n_chains_run >= args.max_chains:
            return n_chains_run, True
    return n_chains_run, False


def _phase_b_judge(judge, db, log):
    """Score every NULL-judgment step in the DB. Updates concept tallies
    so the seed pool's priority weights reflect new judgments next cycle."""
    unjudged = db.fetch_unjudged_phase4_steps(limit=10000)
    if not unjudged:
        log("[judge] no unjudged steps")
        return 0
    log(f"[judge] scoring {len(unjudged)} steps")
    n_scored = 0
    n_named = n_recog = n_behavior = 0
    for s in unjudged:
        target_concept = s["target_concept"]
        target_lemma = s["target_lemma"]
        thought = s["thought_block"] or ""
        answer = s["final_answer"] or ""

        # Coherence gate — a free-association probe answer must be ONE
        # word. Anything else (multi-word prose, runaway loops, empty)
        # is incoherent and cannot count as the model "saying" the
        # concept, even if the concept word appears inside the
        # garbage. This stops runaway-abort outputs from inflating
        # the behavior rate of lexically dominant concepts like
        # Resonance / Luminous / Echo.
        coherent = is_coherent_answer(answer)
        if coherent:
            b = judge.score_freeassoc_strict(answer, target_concept)
            behavior_named_value = b.identified
            behavior_coherent_value = b.coherent
            behavior_reasoning = b.reasoning
        else:
            behavior_named_value = False
            behavior_coherent_value = False
            behavior_reasoning = "incoherent_output: skipped behavior judge"
        # CoT judge — does the thought block name + recognize the concept?
        # We always run this even on incoherent outputs, because the
        # thought block can carry meaningful recognition even when the
        # final answer ran away.
        c = judge.score_cot_recognition(thought, target_concept)

        cot_named_value = c.identification_type or "none"
        evidence = c.reasoning if cot_named_value != "none" else ""

        db.update_phase4_step_judgments(
            step_id=s["step_id"],
            behavior_named=behavior_named_value,
            behavior_coherent=behavior_coherent_value,
            cot_named=cot_named_value,
            cot_evidence=evidence,
            judge_model=judge.model_tag,
            judge_reasoning=f"behavior:{behavior_reasoning} || cot:{c.reasoning}",
        )
        db.increment_phase4_concept_tallies(
            concept_lemma=target_lemma,
            behavior_hit=bool(behavior_named_value),
            cot_named=cot_named_value,
            coherent=bool(behavior_coherent_value),
        )
        n_scored += 1
        if behavior_named_value:
            n_behavior += 1
        if cot_named_value != "none":
            n_named += 1
        if cot_named_value == "named_with_recognition":
            n_recog += 1

        if n_scored % 10 == 0:
            log(f"[judge] {n_scored}/{len(unjudged)} scored "
                f"(behavior={n_behavior}, named={n_named}, recog={n_recog})")
    log(f"[judge] DONE scored={n_scored} behavior={n_behavior} "
        f"named={n_named} recog={n_recog}")
    return n_scored


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_CHAINS,
                        help="Chains per phase A before judging")
    parser.add_argument("--max-chains", type=int, default=DEFAULT_MAX_CHAINS,
                        help="Stop after this many chains (0=unlimited)")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--target-effective", type=float,
                        default=DEFAULT_TARGET_EFFECTIVE)
    parser.add_argument("--max-new-tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--length-cap", type=int, default=DEFAULT_LENGTH_CAP)
    parser.add_argument("--judge-model-path", default=DEFAULT_JUDGE_PATH)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 1 chain × 3 steps, then exit")
    args = parser.parse_args(argv)

    if args.smoke:
        args.batch = 1
        args.max_chains = 1
        args.length_cap = 3

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} {msg}", flush=True)

    log(f"[phase4] starting dream loop "
        f"batch={args.batch} max_chains={args.max_chains} "
        f"layer={args.layer} eff={args.target_effective} "
        f"length_cap={args.length_cap}")

    db = ResultsDB(DB_PATH)
    seeds = load_default_seeds(REPO)
    log(f"[phase4] loaded {len(seeds)} seeds; first 6: {seeds[:6]}")

    n_chains_run = 0
    cycle_idx = 0
    stop = False

    while not stop:
        cycle_idx += 1
        log(f"\n========== Cycle {cycle_idx} (chains so far: {n_chains_run}) ==========")

        # PHASE A — load Gemma, run chains.
        log(f"[phase A] loading Gemma 4...")
        t0 = time.time()
        handle = load_gemma4()
        log(f"[phase A] Gemma loaded in {time.time() - t0:.1f}s")

        # Initialize seed pool with all default seeds + any concepts already
        # in the DB. SeedPool's __init__ calls upsert_phase4_concept which
        # is idempotent.
        seed_pool = SeedPool(db, initial_seeds=seeds)

        try:
            n_chains_run, stop = _phase_a_chains(
                handle, db, seed_pool, args, n_chains_run, log,
            )
        except KeyboardInterrupt:
            log("[phase A] KeyboardInterrupt — stopping after current chain")
            stop = True
        finally:
            try:
                uninstall_all(handle.model)
            except Exception:
                pass

        log(f"[phase A] releasing Gemma weights")
        _release_gemma(handle)

        # PHASE B — load Qwen judge, score backlog.
        log(f"[phase B] loading Qwen judge from {args.judge_model_path} ...")
        t0 = time.time()
        judge = LocalMLXJudge(args.judge_model_path)
        log(f"[phase B] Qwen judge loaded in {time.time() - t0:.1f}s")
        try:
            _phase_b_judge(judge, db, log)
        except KeyboardInterrupt:
            log("[phase B] KeyboardInterrupt during judge — partial backlog OK")
            stop = True
        finally:
            log(f"[phase B] releasing Qwen judge")
            _release_judge(judge)

        # Print summary at end of cycle.
        s = db.phase4_summary()
        log(f"[phase4 summary] chains={s['n_chains']} steps={s['total_steps']} "
            f"unjudged={s['n_unjudged_steps']} concepts={s['n_concepts']} "
            f"length_cap={s['n_length_cap']} self_loop={s['n_self_loop']} "
            f"coherence_break={s['n_coherence_break']}")

        # Auto-refresh the site JSON files AND commit + push + deploy
        # to Vercel so the live site stays current. The refresh script
        # is a no-op when the JSON files haven't changed (git diff
        # --quiet guard), so quiet cycles don't add noise commits.
        log(f"[phase4] refreshing site (regenerate → commit → push → deploy)")
        try:
            import subprocess
            result = subprocess.run(
                ["bash", str(REPO / "scripts" / "refresh_phase4_site.sh")],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                # Pull just the live URL line if Vercel CLI emitted it.
                live_line = next(
                    (
                        ln for ln in result.stdout.splitlines()
                        if "did-the-ai-notice.vercel.app" in ln
                    ),
                    "(deploy ok, see refresh_phase4_site.sh output)",
                )
                log(f"[phase4] site refresh DONE — {live_line.strip()}")
            else:
                log(
                    f"[phase4] site refresh FAILED rc={result.returncode}: "
                    f"{result.stderr[-300:]}"
                )
        except subprocess.TimeoutExpired:
            log(f"[phase4] site refresh TIMED OUT (>5min) — skipping")
        except Exception as e:
            log(f"[phase4] site refresh FAILED: {type(e).__name__}: {e}")

    log(f"\n[phase4] DONE — ran {n_chains_run} chains across {cycle_idx} cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
