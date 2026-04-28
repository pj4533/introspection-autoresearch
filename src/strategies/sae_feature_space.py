"""Phase 2h strategy: SAE-feature-space mean-diff over Capraro fault lines.

Generates `CandidateSpec` objects with
``derivation_method="sae_feature_space_mean_diff"``. Reads the precomputed
fault-line directions from ``data/sae_features/fault_line_directions.pt``
(built by ``scripts/build_fault_line_directions.py``).

Two sub-modes per fault line, default split 12:4 of 16 candidates:

  - sweep:     fresh probes across the target_effective grid. Most slots.
  - replicate: re-run the highest-scoring prior candidate on this fault
               line at one or two perturbed alphas, to gather replication
               evidence on whatever signal we find.

No LLM proposer. No per-feature exploration. The fault-line direction
itself is fixed (built from the prompt corpora); the only thing the
strategy varies is the injection strength and the held-out probe
concepts (which the worker shuffles per-candidate via spec.id seeding).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

import torch

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from ..sae_loader import DEFAULT_RELEASE, DEFAULT_SAE_ID


REPO = Path(__file__).resolve().parent.parent.parent
DIRECTIONS_PATH = REPO / "data" / "sae_features" / "fault_line_directions.pt"
DEFAULT_LAYER_IDX = 31

# Phase 2h directions have natural-magnitude texture (sum-of-feature-
# activations through W_dec), so the target_effective range echoes
# Phase 1's mean_diff calibration where target_effective=18000 worked
# cleanly. We sweep an order of magnitude around that to cover the
# coherence/detection window per fault line.
DEFAULT_TARGET_EFFECTIVES = (8000.0, 14000.0, 18000.0, 24000.0)

# 12 sweep slots + 4 replicate slots = 16. Tunable via env.
DEFAULT_BATCH_COMPOSITION = (12, 4)  # (sweep, replicate)


_DIRECTIONS_META: Optional[dict] = None


def _directions_payload() -> dict:
    global _DIRECTIONS_META
    if _DIRECTIONS_META is not None:
        return _DIRECTIONS_META
    if not DIRECTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Fault-line directions missing: {DIRECTIONS_PATH}\n"
            "Run `python scripts/build_fault_line_directions.py` first."
        )
    _DIRECTIONS_META = torch.load(DIRECTIONS_PATH, weights_only=False)
    return _DIRECTIONS_META


def _list_fault_lines() -> list[str]:
    payload = _directions_payload()
    return sorted((payload.get("directions") or {}).keys())


def _judge_target_for(fault_line: str) -> str:
    payload = _directions_payload()
    entry = payload["directions"].get(fault_line) or {}
    return entry.get("judge_target") or fault_line


def _parse_batch_composition() -> tuple[int, int]:
    raw = os.environ.get("SAE_FS_BATCH_COMPOSITION", "")
    if not raw:
        return DEFAULT_BATCH_COMPOSITION
    parts = raw.split(":")
    if len(parts) != 2:
        return DEFAULT_BATCH_COMPOSITION
    try:
        sweep, rep = int(parts[0]), int(parts[1])
        if sweep + rep <= 0:
            return DEFAULT_BATCH_COMPOSITION
        return (sweep, rep)
    except ValueError:
        return DEFAULT_BATCH_COMPOSITION


def _winners_for_fault_line(
    db: ResultsDB,
    fault_line: str,
    min_score: float = 0.05,
    limit: int = 4,
) -> list[dict]:
    """Top scoring prior candidates with this fault_line, ordered desc."""
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS id, c.spec_json AS spec_json, fs.score AS score
            FROM candidates c
            JOIN fitness_scores fs ON c.id = fs.candidate_id
            WHERE c.derivation_method = 'sae_feature_space_mean_diff'
              AND fs.score >= ?
            ORDER BY fs.score DESC
            LIMIT ?
            """,
            (min_score, limit * 4),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        spec = json.loads(row["spec_json"])
        if spec.get("sae_fault_line") == fault_line:
            out.append({"id": row["id"], "spec": spec, "score": row["score"]})
        if len(out) >= limit:
            break
    return out


def _make_spec(
    *,
    fault_line: str,
    target_effective: float,
    layer_idx: int,
    parent_candidate_id: Optional[str],
    mutation_type: str,
    mutation_detail: Optional[str],
) -> CandidateSpec:
    judge_target = _judge_target_for(fault_line)
    cid = f"saeFS-{uuid.uuid4().hex[:10]}"
    spec = CandidateSpec(
        id=cid,
        strategy=f"sae_feature_space_{fault_line}",
        concept=judge_target,
        layer_idx=layer_idx,
        target_effective=float(target_effective),
        derivation_method="sae_feature_space_mean_diff",
        notes=(
            f"Phase 2h fault-line direction. fault_line={fault_line!r} "
            f"judge_target={judge_target!r}"
        ),
        fitness_mode="sae_aware",
        sae_release=DEFAULT_RELEASE,
        sae_id=DEFAULT_SAE_ID,
        sae_auto_interp=judge_target,
        sae_fault_line=fault_line,
    )
    if parent_candidate_id is not None or mutation_type:
        spec._lineage_meta = {
            "parent_candidate_id": parent_candidate_id,
            "mutation_type": mutation_type,
            "mutation_detail": mutation_detail,
        }
    return spec


def generate_candidates(
    n: int,
    db: ResultsDB,
    fault_line: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> list[CandidateSpec]:
    """Generate `n` Phase 2h candidates for `fault_line`.

    If `fault_line` is None or not in the directions file, falls back to
    the first available fault line. Strategy mode is split into sweep
    (fresh probes across the target_effective grid) and replicate
    (re-runs of prior winners at perturbed alpha).
    """
    available = _list_fault_lines()
    if not available:
        return []
    if fault_line is None or fault_line not in available:
        fault_line = available[0]

    n_sweep_default, n_rep_default = _parse_batch_composition()
    total_default = n_sweep_default + n_rep_default
    if total_default != n:
        # Re-scale proportionally.
        n_sweep = max(1, round(n * n_sweep_default / total_default))
        n_rep = max(0, n - n_sweep)
    else:
        n_sweep, n_rep = n_sweep_default, n_rep_default

    out: list[CandidateSpec] = []

    # ---- Sweep slots: fresh probes across the alpha grid ----------
    grid = list(DEFAULT_TARGET_EFFECTIVES)
    for i in range(n_sweep):
        eff = grid[i % len(grid)]
        out.append(_make_spec(
            fault_line=fault_line,
            target_effective=eff,
            layer_idx=DEFAULT_LAYER_IDX,
            parent_candidate_id=None,
            mutation_type="sae_fs_sweep",
            mutation_detail=f"target_effective={eff:.0f}",
        ))

    # ---- Replicate slots: re-run prior winners at perturbed alpha --
    winners = _winners_for_fault_line(db, fault_line)
    if winners and n_rep > 0:
        perturbations = [0.7, 1.4]
        i = 0
        while len(out) < n_sweep + n_rep:
            winner = winners[i % len(winners)]
            base_eff = float(
                winner["spec"].get("target_effective", DEFAULT_TARGET_EFFECTIVES[1])
            )
            factor = perturbations[i % len(perturbations)]
            target_eff = base_eff * factor
            out.append(_make_spec(
                fault_line=fault_line,
                target_effective=target_eff,
                layer_idx=DEFAULT_LAYER_IDX,
                parent_candidate_id=winner["id"],
                mutation_type="sae_fs_replicate",
                mutation_detail=f"alpha_factor={factor:.2f}",
            ))
            i += 1
            if i > 8 * n_rep:
                break

    return out
