"""Phase 2g strategy: SAE-feature injection over Capraro fault lines.

Generates `CandidateSpec` objects with `derivation_method="sae_feature"`.
Reads the static fault-line bucketing from
`data/sae_features/capraro_buckets.json` (built by
`scripts/build_capraro_buckets.py`) and the leaderboard from the SQLite
DB. No LLM proposer is involved.

Four sub-modes per fault line, default split 6:6:3:1 of 16 candidates:

  - sae_explore        : sample unevaluated features from this fault
                          line's bucket, prioritizing high bucket scores.
  - sae_neighbors      : decoder-cosine neighbors of leaderboard winners
                          for this fault line.
  - sae_replicate      : winners re-run at perturbed alpha
                          (target_effective × {0.7, 1.4}).
  - sae_cross_fault    : a winner from a DIFFERENT fault line, but
                          judged against THIS fault line's label — the
                          control for "is the signal fault-line-specific?"

Cold start: when a fault line has fewer than 2 winners with score >=
COLD_START_SCORE_THRESHOLD, slots B (neighbors) and C (replicate) divert
to slot A (explore). No LLM fallback — the proposer is gone.

Layer + alpha grid: candidates are emitted at L=31 (the canonical layer
matching our SAE checkpoint) and SAE_TARGET_EFFECTIVE_DEFAULTS, with
slot D explicitly randomizing alpha within the per-feature sweep range.
SAE_TARGET_EFFECTIVE is calibrated by the smoke test (Phase 2g run plan
Step 2) and pinned in the env or as a constant here.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Optional

from ..db import ResultsDB
from ..evaluate import CandidateSpec


REPO = Path(__file__).resolve().parent.parent.parent
BUCKETS_PATH = REPO / "data" / "sae_features" / "capraro_buckets.json"

# Layer 31 is the canonical Phase 2g layer (matches the SAE checkpoint we
# downloaded). Phase 1's introspection peak was L=33; Phase 2g operates at
# L=31 because Neuronpedia auto-interp is only available there. The Phase 1
# curve was broad in this range.
DEFAULT_LAYER_IDX = 31

# The α × ‖direction‖ product. SAE decoder vectors are ~unit-norm, so the
# target_effective for SAE features is order-of-magnitude smaller than the
# Phase 1 mean_diff calibration (target_effective=18000 with norms in the
# hundreds). The Phase 2g run plan calls for a smoke test to pin this.
# Default sweep used by sae_explore: try a few values per feature so we
# don't depend on a single fragile alpha.
DEFAULT_TARGET_EFFECTIVES = (8.0, 16.0, 32.0)

# Default 16-candidate slot composition. Tunable via env var.
DEFAULT_BATCH_COMPOSITION = (6, 6, 3, 1)   # explore, neighbors, replicate, cross_fault

# Score threshold below which a fault line is considered cold-start (no
# real winners yet). Slots B/C divert to slot A under cold start.
COLD_START_SCORE_THRESHOLD = 0.05

SAE_RELEASE = "google/gemma-scope-2-12b-it"
SAE_ID = "resid_post/layer_31_width_262k_l0_medium"


# ----------------------------------------------------------------------
# Bucket I/O
# ----------------------------------------------------------------------

def _load_buckets(path: Path = BUCKETS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Capraro buckets file not found at {path}. "
            "Run `python scripts/build_capraro_buckets.py` first."
        )
    return json.loads(path.read_text())


def _parse_batch_composition() -> tuple[int, int, int, int]:
    raw = os.environ.get("SAE_CAPRARO_BATCH_COMPOSITION", "")
    if not raw:
        return DEFAULT_BATCH_COMPOSITION
    parts = raw.split(":")
    if len(parts) != 4:
        return DEFAULT_BATCH_COMPOSITION
    try:
        e, n, r, c = (int(p) for p in parts)
        if e + n + r + c <= 0:
            return DEFAULT_BATCH_COMPOSITION
        return (e, n, r, c)
    except ValueError:
        return DEFAULT_BATCH_COMPOSITION


def _list_fault_lines(buckets: dict) -> list[str]:
    return list(buckets.get("fault_lines") or buckets.get("buckets", {}).keys())


# ----------------------------------------------------------------------
# Leaderboard helpers (read prior winners from the DB)
# ----------------------------------------------------------------------

def _winners_for_fault_line(
    db: ResultsDB,
    fault_line: str,
    min_score: float = COLD_START_SCORE_THRESHOLD,
    limit: int = 8,
) -> list[dict]:
    """Top-N candidates with derivation_method='sae_feature' AND
    sae_fault_line == fault_line, ordered by score desc.

    Returns dict rows with at least: id, spec_json, score.
    Uses ResultsDB.get_leaders() if available; otherwise raw SQL.
    """
    with db._conn() as conn:
        rows = conn.execute(
            """
            SELECT c.id AS id, c.spec_json AS spec_json, fs.score AS score
            FROM candidates c
            JOIN fitness_scores fs ON c.id = fs.candidate_id
            WHERE c.derivation_method = 'sae_feature'
              AND fs.score >= ?
            ORDER BY fs.score DESC
            LIMIT ?
            """,
            (min_score, limit * 4),  # over-fetch; we filter by fault_line in Python
        ).fetchall()

    out: list[dict] = []
    for row in rows:
        spec = json.loads(row["spec_json"])
        if spec.get("sae_fault_line") == fault_line:
            out.append({
                "id": row["id"],
                "spec": spec,
                "score": row["score"],
            })
        if len(out) >= limit:
            break
    return out


def _evaluated_feature_indices(
    db: ResultsDB,
    fault_line: Optional[str] = None,
) -> set[int]:
    """Return the set of (sae_feature_idx) already evaluated, optionally
    scoped to a single fault line. Used to skip repeats in sae_explore.
    """
    sql = (
        "SELECT spec_json FROM candidates WHERE derivation_method='sae_feature'"
    )
    out: set[int] = set()
    with db._conn() as conn:
        rows = conn.execute(sql).fetchall()
    for row in rows:
        spec = json.loads(row["spec_json"])
        if fault_line is not None and spec.get("sae_fault_line") != fault_line:
            continue
        idx = spec.get("sae_feature_idx")
        if idx is not None:
            out.add(int(idx))
    return out


# ----------------------------------------------------------------------
# Per-sub-mode generators
# ----------------------------------------------------------------------

def _spec_for_feature(
    *,
    feature: dict,
    fault_line: str,
    strategy: str,
    target_effective: float,
    layer_idx: int = DEFAULT_LAYER_IDX,
    parent_candidate_id: Optional[str] = None,
    mutation_type: Optional[str] = None,
    mutation_detail: Optional[str] = None,
    judge_concept_override: Optional[str] = None,
) -> CandidateSpec:
    """Build a CandidateSpec for one (feature, layer, alpha) triple.

    `judge_concept_override` lets cross_fault candidates use the
    DESTINATION fault line's label instead of the feature's own
    auto_interp (the control's whole point).
    """
    feature_idx = int(feature["feature_idx"])
    auto_interp = feature["auto_interp"]
    cid = f"saeC-{uuid.uuid4().hex[:10]}"
    judge_concept = judge_concept_override or auto_interp
    spec = CandidateSpec(
        id=cid,
        strategy=strategy,
        concept=judge_concept,                  # the judge target
        layer_idx=layer_idx,
        target_effective=target_effective,
        derivation_method="sae_feature",
        notes=(
            f"SAE feature {feature_idx} | fault_line={fault_line} | "
            f"auto_interp={auto_interp!r}"
        ),
        fitness_mode="sae_aware",
        sae_release=SAE_RELEASE,
        sae_id=SAE_ID,
        sae_feature_idx=feature_idx,
        sae_auto_interp=judge_concept,
        sae_fault_line=fault_line,
    )
    if parent_candidate_id is not None or mutation_type is not None:
        spec._lineage_meta = {
            "parent_candidate_id": parent_candidate_id,
            "mutation_type": mutation_type,
            "mutation_detail": mutation_detail,
        }
    return spec


def _sae_explore(
    *,
    n: int,
    bucket: list[dict],
    fault_line: str,
    seen: set[int],
    rng: random.Random,
) -> list[CandidateSpec]:
    """Pick n unevaluated features, prioritizing high bucket scores.

    Strategy: take the top 5×n features from the bucket (already sorted by
    bucket score), filter out those already evaluated, then pick n at
    random. This balances 'best matches first' with 'don't always pick
    the same handful'.
    """
    if not bucket:
        return []
    pool = [f for f in bucket[:max(5 * n, 50)] if int(f["feature_idx"]) not in seen]
    if not pool:
        # Fall back to the rest of the bucket (lower-scoring features) once
        # the top is exhausted.
        pool = [f for f in bucket if int(f["feature_idx"]) not in seen]
    rng.shuffle(pool)
    chosen = pool[:n]
    out: list[CandidateSpec] = []
    for feat in chosen:
        # One target_effective per candidate, drawn from the default sweep.
        target_eff = rng.choice(DEFAULT_TARGET_EFFECTIVES)
        out.append(_spec_for_feature(
            feature=feat,
            fault_line=fault_line,
            strategy=f"sae_capraro_{fault_line}",
            target_effective=target_eff,
            mutation_type="sae_explore",
        ))
    return out


def _sae_neighbors(
    *,
    n: int,
    winners: list[dict],
    fault_line: str,
    seen: set[int],
    rng: random.Random,
) -> list[CandidateSpec]:
    """Decoder-cosine neighbors of leaderboard winners.

    Loads the SAE on-demand for cosine. With 262k features the lookup
    is sub-second; we cache the SAE in sae_loader's LRU cache.
    """
    if not winners:
        return []
    from ..sae_loader import get_neighbors

    out: list[CandidateSpec] = []
    # Distribute n neighbor slots across the winners. Top winner gets the
    # most.
    weights = [3, 2, 1] + [1] * (max(0, len(winners) - 3))
    weights = weights[:len(winners)]
    total_w = sum(weights)
    per_winner = [
        max(1, round(n * w / total_w)) for w in weights
    ]
    while sum(per_winner) > n:
        # Trim the smallest first.
        idx = per_winner.index(min(per_winner))
        per_winner[idx] = max(0, per_winner[idx] - 1)
    while sum(per_winner) < n:
        per_winner[0] += 1

    for winner, k in zip(winners, per_winner):
        if k == 0:
            continue
        feat_idx = winner["spec"].get("sae_feature_idx")
        if feat_idx is None:
            continue
        try:
            neighbors = get_neighbors(
                release=SAE_RELEASE,
                sae_id=SAE_ID,
                feature_idx=int(feat_idx),
                n=max(40, k * 4),
                exclude_self=True,
            )
        except Exception:
            continue
        # Filter out already-evaluated neighbors. Pick top-k unseen.
        neighbor_idxs = [(idx, sim) for idx, sim in neighbors if idx not in seen]
        # We don't have auto-interp for arbitrary neighbor indices unless
        # they're in the bucket file too. Fall back to the parent's
        # auto_interp as a hint for any neighbor lacking its own label.
        chosen = neighbor_idxs[:k]
        parent_auto_interp = winner["spec"].get("sae_auto_interp", "")
        for idx, sim in chosen:
            target_eff = rng.choice(DEFAULT_TARGET_EFFECTIVES)
            out.append(_spec_for_feature(
                feature={"feature_idx": idx, "auto_interp": parent_auto_interp},
                fault_line=fault_line,
                strategy=f"sae_capraro_{fault_line}",
                target_effective=target_eff,
                parent_candidate_id=winner["id"],
                mutation_type="sae_neighbor",
                mutation_detail=f"cos={sim:.3f}",
            ))
            seen.add(idx)
    return out[:n]


def _sae_replicate(
    *,
    n: int,
    winners: list[dict],
    fault_line: str,
    rng: random.Random,
) -> list[CandidateSpec]:
    """Re-run prior winners at perturbed alphas."""
    if not winners or n == 0:
        return []
    out: list[CandidateSpec] = []
    perturbations = [0.7, 1.4]
    i = 0
    while len(out) < n:
        winner = winners[i % len(winners)]
        spec = winner["spec"]
        feat_idx = spec.get("sae_feature_idx")
        if feat_idx is None:
            i += 1
            continue
        base_eff = float(spec.get("target_effective", DEFAULT_TARGET_EFFECTIVES[1]))
        target_eff = base_eff * rng.choice(perturbations)
        out.append(_spec_for_feature(
            feature={
                "feature_idx": int(feat_idx),
                "auto_interp": spec.get("sae_auto_interp", ""),
            },
            fault_line=fault_line,
            strategy=f"sae_capraro_{fault_line}",
            target_effective=target_eff,
            parent_candidate_id=winner["id"],
            mutation_type="sae_replicate_alpha",
            mutation_detail=f"alpha_factor={target_eff/base_eff:.2f}",
        ))
        i += 1
        if i > 8 * n:
            break
    return out


def _sae_cross_fault(
    *,
    n: int,
    db: ResultsDB,
    target_fault_line: str,
    fault_line_buckets: dict,
    rng: random.Random,
) -> list[CandidateSpec]:
    """Take a winner from another fault line and judge it against THIS
    fault line's label — the explicit control for "is the signal
    fault-line-specific?"

    The judge target becomes the destination fault line's representative
    description (drawn from the bucket file's top feature for that fault
    line, used as a stand-in label).
    """
    if n == 0:
        return []
    other_fault_lines = [f for f in fault_line_buckets if f != target_fault_line]
    if not other_fault_lines:
        return []
    out: list[CandidateSpec] = []
    # Use the top feature of the destination bucket as the cross-fault
    # judge concept. It's the most-emblematic auto_interp label for this
    # fault line and matches what `sae_explore` would inject for it.
    dest_bucket = fault_line_buckets.get(target_fault_line) or []
    dest_concept = (
        dest_bucket[0]["auto_interp"]
        if dest_bucket
        else target_fault_line
    )
    for _ in range(n):
        src_fault = rng.choice(other_fault_lines)
        winners = _winners_for_fault_line(db, src_fault)
        if not winners:
            continue
        winner = rng.choice(winners)
        spec = winner["spec"]
        feat_idx = spec.get("sae_feature_idx")
        if feat_idx is None:
            continue
        target_eff = float(spec.get("target_effective", DEFAULT_TARGET_EFFECTIVES[1]))
        out.append(_spec_for_feature(
            feature={
                "feature_idx": int(feat_idx),
                "auto_interp": spec.get("sae_auto_interp", ""),
            },
            fault_line=target_fault_line,
            strategy=f"sae_capraro_cross_{target_fault_line}",
            target_effective=target_eff,
            parent_candidate_id=winner["id"],
            mutation_type="sae_cross_fault",
            mutation_detail=f"src_fault={src_fault}",
            judge_concept_override=dest_concept,
        ))
    return out


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------

def generate_candidates(
    n: int,
    db: ResultsDB,
    fault_line: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> list[CandidateSpec]:
    """Generate `n` SAE-feature CandidateSpec objects for `fault_line`.

    If `fault_line` is None or unknown, runs sae_explore on the largest
    bucket as a fallback.
    """
    rng = random.Random(rng_seed)
    buckets_data = _load_buckets()
    fault_line_buckets: dict[str, list[dict]] = buckets_data["buckets"]

    if fault_line is None or fault_line not in fault_line_buckets:
        # Fallback: pick the fault line with the largest bucket. This
        # keeps the worker producing candidates if its rotation hits an
        # unknown name.
        fault_line = max(
            fault_line_buckets.keys(),
            key=lambda f: len(fault_line_buckets[f]),
        )

    bucket = fault_line_buckets[fault_line]
    seen = _evaluated_feature_indices(db, fault_line=fault_line)
    winners = _winners_for_fault_line(db, fault_line)

    n_explore, n_neighbors, n_replicate, n_cross = _parse_batch_composition()
    # Re-scale to the requested n, preserving proportions.
    total_default = n_explore + n_neighbors + n_replicate + n_cross
    if total_default != n:
        # Allocate proportionally; round and fix the total.
        ratios = [n_explore, n_neighbors, n_replicate, n_cross]
        scaled = [max(0, round(r * n / total_default)) for r in ratios]
        # Ensure sum == n.
        diff = n - sum(scaled)
        i = 0
        while diff != 0:
            if diff > 0:
                scaled[i % len(scaled)] += 1
                diff -= 1
            else:
                if scaled[i % len(scaled)] > 0:
                    scaled[i % len(scaled)] -= 1
                    diff += 1
            i += 1
        n_explore, n_neighbors, n_replicate, n_cross = scaled

    cold_start = len(winners) < 2
    if cold_start:
        # Divert neighbors and replicate slots to explore.
        n_explore += n_neighbors + n_replicate
        n_neighbors = 0
        n_replicate = 0

    out: list[CandidateSpec] = []

    explore_specs = _sae_explore(
        n=n_explore,
        bucket=bucket,
        fault_line=fault_line,
        seen=seen,
        rng=rng,
    )
    out.extend(explore_specs)
    seen.update(int(s.sae_feature_idx) for s in explore_specs if s.sae_feature_idx is not None)

    neighbor_specs = _sae_neighbors(
        n=n_neighbors,
        winners=winners,
        fault_line=fault_line,
        seen=seen,
        rng=rng,
    )
    out.extend(neighbor_specs)

    replicate_specs = _sae_replicate(
        n=n_replicate,
        winners=winners,
        fault_line=fault_line,
        rng=rng,
    )
    out.extend(replicate_specs)

    cross_specs = _sae_cross_fault(
        n=n_cross,
        db=db,
        target_fault_line=fault_line,
        fault_line_buckets=fault_line_buckets,
        rng=rng,
    )
    out.extend(cross_specs)

    return out
