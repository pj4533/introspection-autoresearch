"""Phase 1 full sweep: 50 concepts x layer grid, with resumable SQLite logging.

Builds on the MVP pipeline from `src.bridge`. Writes every trial to
`data/results.db` via `src.db.ResultsDB`. Safe to re-run: trials already in
the DB are skipped.

Typical usage (see scripts/run_phase1_sweep.py):

    from src.sweep import run_sweep, SweepConfig
    cfg = SweepConfig(concepts=..., layers=[10, 15, 20, ...], alpha=4.0)
    run_sweep(cfg, db_path=..., model_name='gemma3_12b')
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .bridge import DetectionPipeline, load_gemma_mps
from .db import ResultsDB, Trial
from .judges.base import Judge
from .judges.claude_judge import ClaudeJudge
from .paper.abliteration import (
    install_abliteration_hooks,
    paper_layer_weights_for_model,
    remove_abliteration_hooks,
)


@dataclass
class SweepConfig:
    concepts: list[str]
    layers: list[int]
    # Alpha mode: if target_effective is set, per-cell alpha is chosen so that
    # alpha * ||direction|| == target_effective (cross-concept normalization).
    # Otherwise, use fixed `alpha` for every cell.
    alpha: float = 4.0
    # 18000 calibrated on Gemma3-12B, layer 33, strict-paper judge.
    # See docs/CALIBRATION.md for methodology.
    target_effective: Optional[float] = 18000.0
    trials_per_cell: int = 1
    run_controls: bool = True
    max_new_tokens: int = 120
    temperature: float = 1.0
    judge_model: str = "claude-sonnet-4-6"

    def total_cells(self) -> int:
        injected = len(self.concepts) * len(self.layers) * self.trials_per_cell
        control = (len(self.concepts) * self.trials_per_cell) if self.run_controls else 0
        return injected + control


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def run_sweep(
    cfg: SweepConfig,
    db_path: Path,
    model_name: str = "gemma3_12b",
    run_id: Optional[str] = None,
    judge: Optional[Judge] = None,
    verbose: bool = True,
    abliterate_paper: Optional[Path] = None,
    abliteration_weight: Optional[float] = None,
) -> str:
    """Run the full sweep, writing every trial to SQLite.

    If ``abliterate_paper`` is a Path, loads the pre-computed per-layer
    refusal directions from that file and installs projection-out hooks on
    every layer of the vanilla model (paper-method abliteration). This is an
    alternative to loading a pre-abliterated HF model.

    When ``abliteration_weight`` is None (default), uses the paper's Optuna-
    tuned region weights proportionally remapped to this model's layer count
    (mean ~0.023, max ~0.12 — very gentle per-layer ablation). Pass a float
    to override with a uniform weight instead; weight=1.0 is ~40x more
    aggressive than the paper's recipe and will typically destroy coherent
    generation.

    Returns the run_id.
    """
    db = ResultsDB(db_path)
    judge_cache = Path(db_path).parent / "judge_cache.sqlite"
    if judge is None:
        judge = ClaudeJudge(model=cfg.judge_model, cache_path=judge_cache)

    if run_id is None:
        run_id = f"phase1-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    if verbose:
        print(f"Sweep run_id: {run_id}")
        print(f"Model:        {model_name}")
        print(f"Concepts:     {len(cfg.concepts)}")
        print(f"Layers:       {cfg.layers}")
        print(f"Alpha:        {cfg.alpha}")
        print(f"Trials/cell:  {cfg.trials_per_cell}  (controls: {cfg.run_controls})")
        print(f"Judge:        {cfg.judge_model}")
        print(f"Total cells:  {cfg.total_cells()}")
        print(f"DB path:      {db_path}")
        print()

    # --- Build the trial plan -------------------------------------------------
    # When target_effective is set, we can't know per-cell alpha until the
    # direction is derived. So we plan injected cells with alpha=None and
    # resolve at runtime. Control cells use alpha=0.0.
    plan: list[tuple[str, int, Optional[float], bool, int]] = []
    for concept in cfg.concepts:
        for layer in cfg.layers:
            for t_num in range(1, cfg.trials_per_cell + 1):
                fixed_alpha = None if cfg.target_effective is not None else cfg.alpha
                plan.append((concept, layer, fixed_alpha, True, t_num))
        if cfg.run_controls:
            for t_num in range(1, cfg.trials_per_cell + 1):
                plan.append((concept, -1, 0.0, False, t_num))

    # Filter out anything already done (resumability). For adaptive-alpha
    # injected cells we skip if ANY injected trial for that (concept, layer,
    # trial_number, judge_model) is already in DB, regardless of exact alpha.
    def already_done_for(p):
        concept, layer, alpha, injected, t_num = p
        if injected and alpha is None:
            # Adaptive alpha — look up any injected row with matching key
            with db._conn() as conn:
                r = conn.execute(
                    """SELECT 1 FROM trials
                       WHERE concept=? AND layer_idx=? AND injected=1
                         AND trial_number=? AND judge_model=? LIMIT 1""",
                    (concept, layer, t_num, cfg.judge_model),
                ).fetchone()
            return r is not None
        return db.has_trial(concept, layer, alpha or 0.0, injected, t_num, cfg.judge_model)

    pending = [p for p in plan if not already_done_for(p)]
    already_done = len(plan) - len(pending)
    if verbose:
        print(f"Resume: {already_done} trials already in DB, {len(pending)} pending.")
        print()

    if not pending:
        if verbose:
            print("Nothing to do.")
        return run_id

    # --- Load model (only if we have work) -----------------------------------
    if verbose:
        print("Loading model on MPS (bf16)...")
    model = load_gemma_mps(model_name)
    pipeline = DetectionPipeline(model=model, judge=judge)
    if verbose:
        print(f"Model loaded: {model.hf_path}  n_layers={model.n_layers}")

    # --- Cache derived directions per (concept, layer) across trials ----------
    direction_cache: dict[tuple[str, int], torch.Tensor] = {}

    # --- Paper-method abliteration (optional) -------------------------------
    # CRITICAL: concept directions must be derived from the VANILLA model
    # (no abliteration hooks), then injected into the abliterated model.
    # This matches the paper's methodology. If we derived with hooks active,
    # the extracted direction would have its refusal-aligned components
    # projected out, leaving only noise that adaptive-alpha amplifies into
    # token-salad generation.
    ablation_handles: list = []
    if abliterate_paper is not None:
        if verbose:
            print(f"Loading refusal directions from {abliterate_paper} ...")
        payload = torch.load(abliterate_paper, map_location="cpu", weights_only=False)
        directions = (
            payload["directions"] if isinstance(payload, dict) else payload
        )

        # Pre-derive concept directions from VANILLA model before hooks install.
        # Enumerate the unique (concept, layer) pairs we'll need.
        if verbose:
            print(
                "Pre-deriving concept directions from vanilla model "
                "(before abliteration hooks install)..."
            )
        unique_pairs: set[tuple[str, int]] = {
            (c, l) for (c, l, _a, inj, _t) in pending if inj
        }
        for i, (c, layer) in enumerate(sorted(unique_pairs), 1):
            direction_cache[(c, layer)] = pipeline.derive(
                concept=c, layer_idx=layer
            )
            if verbose and i % 20 == 0:
                print(f"  derived {i}/{len(unique_pairs)}")
        if verbose:
            print(f"  done: {len(direction_cache)} directions cached")

        # NOW install abliteration hooks for the injection / generation phase
        if abliteration_weight is None:
            per_layer_weights = paper_layer_weights_for_model(model.n_layers)
            ablation_handles = install_abliteration_hooks(
                model.model, directions, layer_weights=per_layer_weights
            )
            if verbose:
                w = per_layer_weights
                print(
                    f"Installed {len(ablation_handles)} abliteration hooks "
                    f"with paper's per-region weights "
                    f"(mean={sum(w)/len(w):.4f}, max={max(w):.4f}, min={min(w):.6f})"
                )
        else:
            ablation_handles = install_abliteration_hooks(
                model.model, directions, weight=abliteration_weight
            )
            if verbose:
                warn = "  ⚠  this is much stronger than paper" if abliteration_weight >= 0.3 else ""
                print(
                    f"Installed {len(ablation_handles)} abliteration hooks "
                    f"(uniform weight={abliteration_weight}){warn}"
                )
    if verbose:
        print()

    # --- Main loop -----------------------------------------------------------
    t_start = time.time()
    for i, (concept, layer, alpha, injected, t_num) in enumerate(pending, 1):
        cell_start = time.time()

        if injected:
            key = (concept, layer)
            if key not in direction_cache:
                direction_cache[key] = pipeline.derive(concept=concept, layer_idx=layer)
            direction = direction_cache[key]
            direction_norm = float(direction.norm().item())

            # Resolve adaptive alpha
            if alpha is None:
                alpha = cfg.target_effective / max(direction_norm, 1e-6)

            torch.manual_seed(t_num)
            trial = pipeline.run_injected(
                concept=concept,
                direction=direction,
                layer_idx=layer,
                strength=alpha,
                trial_number=t_num,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
        else:
            direction_norm = 0.0
            alpha = 0.0
            torch.manual_seed(t_num)
            trial = pipeline.run_control(
                concept=concept,
                trial_number=t_num,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )

        jr = trial.judge_result
        db_trial = Trial(
            run_id=run_id,
            concept=concept,
            layer_idx=layer,
            alpha=alpha,
            injected=injected,
            trial_number=t_num,
            prompt="introspection_v1",
            response=trial.response,
            detected=jr.detected,
            identified=jr.identified,
            coherent=jr.coherent,
            judge_model=cfg.judge_model,
            judge_reasoning=jr.reasoning,
            judge_raw=jr.raw,
            direction_norm=direction_norm,
            model_name=model_name,
        )
        db.insert_trial(db_trial)

        cell_elapsed = time.time() - cell_start
        total_elapsed = time.time() - t_start
        eta = total_elapsed * (len(pending) - i) / i if i > 0 else 0.0
        if verbose:
            cond = "inj" if injected else "ctrl"
            marks = (
                f"d={int(jr.detected)} id={int(jr.identified)} "
                f"coh={int(jr.coherent)}"
            )
            print(
                f"[{i:>4d}/{len(pending)}] {cond} {concept:<14} "
                f"L={layer:>2d} a={alpha:>4.1f} t={t_num}  "
                f"{marks}  cell={cell_elapsed:.1f}s  "
                f"elapsed={_fmt_elapsed(total_elapsed)} "
                f"eta={_fmt_elapsed(eta)}"
            )

    if verbose:
        print()
        print(f"Sweep complete: {len(pending)} new trials in {_fmt_elapsed(time.time() - t_start)}.")
        print(f"Run id: {run_id}")

    # Clean up hooks if we installed them
    if ablation_handles:
        remove_abliteration_hooks(ablation_handles)

    return run_id


def load_concepts(path: Path) -> list[str]:
    data = json.loads(Path(path).read_text())
    return list(data["concepts"])
