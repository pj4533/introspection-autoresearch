"""Structured hill-climb dispatcher for fault-line-targeted autoresearch.

Replaces ``directed_capraro.generate_candidates`` in Phase C of the worker.
Instead of asking the proposer for N free-form variants per cycle, this
strategy allocates each batch into three explicit slots:

  | Slot | Default count | Source                                          |
  |------|---------------|-------------------------------------------------|
  | A    | 4             | REPLICATION of top-2 prior winners on this      |
  |      |               | fault line (verbatim re-eval; 2 per winner)     |
  | B    | 10            | TARGETED VARIANTS: top-3 winners × mutation     |
  |      |               | operators (layer_shift, alpha_scale,            |
  |      |               | examples_swap, description_sharpen,             |
  |      |               | antonym_pivot)                                  |
  | C    | 2             | CLUSTER EXPANSION: 1 fresh sibling axis from    |
  |      |               | the proposer, "stay in this fault line's       |
  |      |               | conceptual neighborhood" — gets 2 layers        |

Total: 16 candidates per cycle, same as the prior unstructured loop.

Rationale for the split: the prior loop drifted away from winners because
nothing forced re-testing or systematic variation. The replication slot
produces the reproducibility evidence a writeup needs; the targeted-variants
slot does actual hill-climbing on proven directions; the cluster-expansion
slot keeps a small wildcard for finding new families inside a fault line.

Lineage tracking: every emitted spec carries
``spec._lineage_meta`` (read by the queue writer in worker.py) with
``parent_candidate_id`` and ``mutation_type``. Replications point at their
verbatim parent. Variants point at the winner they were mutated from.
Cluster-expansion specs have no parent (mutation_type='cluster_expansion').

Tunable via env var ``HILLCLIMB_BATCH_COMPOSITION="<rep>:<var>:<exp>"``,
e.g. ``"6:8:2"`` to favor more replications. Defaults to ``"4:10:2"``.

See ``docs/structured_hillclimb.md`` for the full design.
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import time
import uuid
from typing import Optional

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from ..proposers.base import Proposer
from . import mutations
from .mutations import ParentRecord, apply_operator
from .random_explore import spec_hash

# Default batch composition. Override via HILLCLIMB_BATCH_COMPOSITION env var.
DEFAULT_REPLICATION = 4
DEFAULT_VARIANTS = 10
DEFAULT_CLUSTER_EXPANSION = 2


def _parse_composition(env_value: Optional[str]) -> tuple[int, int, int]:
    """Parse ``"<r>:<v>:<e>"`` into (replication, variants, expansion).

    Falls back to the defaults if the env var is missing, malformed, or
    sums to zero. Logs a warning if it's malformed.
    """
    if not env_value:
        return DEFAULT_REPLICATION, DEFAULT_VARIANTS, DEFAULT_CLUSTER_EXPANSION
    try:
        parts = [int(x) for x in env_value.split(":")]
        if len(parts) != 3 or any(p < 0 for p in parts) or sum(parts) == 0:
            raise ValueError
        return parts[0], parts[1], parts[2]
    except (ValueError, AttributeError):
        print(
            f"[structured_hillclimb] WARNING: malformed "
            f"HILLCLIMB_BATCH_COMPOSITION={env_value!r}, using defaults",
            flush=True,
        )
        return DEFAULT_REPLICATION, DEFAULT_VARIANTS, DEFAULT_CLUSTER_EXPANSION


# ---------------------------------------------------------------------------
# Parent loading — read top winners on this fault line from the DB
# ---------------------------------------------------------------------------

def _load_winners(
    db: ResultsDB,
    fault_line_id: str,
    *,
    limit: int = 5,
    min_score: float = 0.05,
) -> list[ParentRecord]:
    """Top-scoring evaluated candidates on this fault line.

    Joins candidates × fitness_scores, filters to ``directed_capraro_<fl>``
    or ``hillclimb_directed_capraro_<fl>`` strategies (so children of past
    winners are also eligible), orders by score desc, dedups by axis name
    (keeps the highest-scoring instance of each axis so we don't waste
    slots replicating siblings of the same family).

    ``min_score`` filters out near-zero scores to avoid replicating noise.
    """
    pattern_a = f"directed_capraro_{fault_line_id}"
    pattern_b = f"hillclimb_directed_capraro_{fault_line_id}"

    with sqlite3.connect(str(db.path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT c.id AS candidate_id, c.strategy, c.concept,
                      c.layer_idx, c.target_effective, c.derivation_method,
                      c.spec_json,
                      f.score, f.detection_rate, f.identification_rate
               FROM candidates c
               JOIN fitness_scores f ON f.candidate_id = c.id
               WHERE (c.strategy = ? OR c.strategy = ?)
                 AND f.score >= ?
               ORDER BY f.score DESC""",
            (pattern_a, pattern_b, min_score),
        ).fetchall()

    seen_axes: set[str] = set()
    out: list[ParentRecord] = []
    for r in rows:
        if r["concept"] in seen_axes:
            continue
        seen_axes.add(r["concept"])
        spec_dict = json.loads(r["spec_json"])
        out.append(ParentRecord(
            candidate_id=r["candidate_id"],
            strategy=r["strategy"],
            concept=r["concept"],
            layer_idx=int(r["layer_idx"]),
            target_effective=float(r["target_effective"]),
            derivation_method=r["derivation_method"],
            contrast_pair=spec_dict.get("contrast_pair"),
            fitness_mode=spec_dict.get("fitness_mode"),
            score=float(r["score"]),
            detection_rate=float(r["detection_rate"]),
            identification_rate=float(r["identification_rate"]),
        ))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Slot generators
# ---------------------------------------------------------------------------

def _slot_replication(
    winners: list[ParentRecord],
    n_slots: int,
) -> list[CandidateSpec]:
    """Verbatim re-evaluations of top-K winners.

    Distributes ``n_slots`` evenly across the top winners (round-robin).
    With n_slots=4 and 2 winners, that's 2 reps each.
    """
    if not winners or n_slots <= 0:
        return []
    out: list[CandidateSpec] = []
    i = 0
    while len(out) < n_slots:
        winner = winners[i % len(winners)]
        out.append(mutations.replication(winner))
        i += 1
    return out


def _slot_variants(
    winners: list[ParentRecord],
    n_slots: int,
    *,
    proposer: Proposer,
    rng: random.Random,
) -> list[CandidateSpec]:
    """Targeted variants: each slot picks a winner + a mutation operator.

    Strategy: take the top ``min(3, len(winners))`` parents and round-robin
    them across slots; per slot, pick a random operator (deterministic ones
    are 2× weighted because they're cheaper and more reliable).
    """
    if not winners or n_slots <= 0:
        return []
    parents = winners[:3]
    # Weighted pool — deterministic ops cheaper, more reliable, so weight up.
    op_pool = (
        list(mutations.DETERMINISTIC_OPERATORS) * 2
        + list(mutations.PROPOSER_OPERATORS)
    )

    out: list[CandidateSpec] = []
    for slot_idx in range(n_slots):
        parent = parents[slot_idx % len(parents)]
        op = rng.choice(op_pool)
        try:
            spec = apply_operator(op, parent, rng=rng, proposer=proposer)
        except Exception as e:  # pragma: no cover — defensive
            print(
                f"[structured_hillclimb] operator {op!r} crashed on "
                f"parent={parent.candidate_id}: {e}",
                flush=True,
            )
            spec = None
        if spec is None:
            # Fall back to a deterministic op — guarantees we fill the slot
            # even if the proposer returned junk.
            fallback_op = rng.choice(mutations.DETERMINISTIC_OPERATORS)
            spec = apply_operator(
                fallback_op, parent, rng=rng, proposer=None
            )
            print(
                f"[structured_hillclimb] {op} failed for "
                f"{parent.concept}, fell back to {fallback_op}",
                flush=True,
            )
        out.append(spec)
    return out


def _slot_cluster_expansion(
    db: ResultsDB,
    fault_line_id: str,
    n_slots: int,
    *,
    proposer: Proposer,
) -> list[CandidateSpec]:
    """Fresh sibling axes: one new contrast pair for this fault line.

    Reuses ``directed_capraro.generate_candidates`` in 'opus' mode but
    scoped tightly: ask for ONE new pair, expand it across ``n_slots``
    layers (clamped to {30, 33} by default — the proven hot zones).

    The resulting specs have no ``parent_candidate_id`` (mutation_type
    = 'cluster_expansion'), but they're still lineage-tagged so we can
    distinguish them from the variant slot in analysis.
    """
    if n_slots <= 0:
        return []
    # Use the existing directed_capraro plumbing — feedback block, opus
    # brief, fault-line registry — but request only 1 new pair with a
    # narrow target_effective grid and 2 layers.
    from .directed_capraro import generate_candidates as gen_capraro
    specs = gen_capraro(
        n=n_slots,
        db=db,
        fault_line_id=fault_line_id,
        mode="opus",
        proposer=proposer,
        layers=[30, 33],            # focus on proven hot zones
        target_effectives=[14000.0, 18000.0],
        oversample_factor=3,        # ask for more pairs in case of parse failure
    )
    # Tag each emitted spec as cluster_expansion (mutation_type only —
    # no parent_candidate_id). The dispatcher includes _lineage_meta in
    # the queue file so the worker writes mutation_type into candidates.
    for spec in specs:
        spec._lineage_meta = {  # type: ignore[attr-defined]
            "parent_candidate_id": None,
            "mutation_type": "cluster_expansion",
            "mutation_detail": json.dumps(
                {"fault_line": fault_line_id}
            ),
            "generation": 0,
        }
    return specs


# ---------------------------------------------------------------------------
# Public entrypoint — replaces directed_capraro.generate_candidates in Phase C
# ---------------------------------------------------------------------------

def generate_candidates(
    n: int,
    db: ResultsDB,
    fault_line_id: str,
    *,
    proposer: Proposer,
    seed: Optional[int] = None,
) -> list[CandidateSpec]:
    """Generate one cycle's worth of structured-hillclimb candidates.

    Drop-in replacement for ``directed_capraro.generate_candidates`` in
    the worker's Phase C. Same contract: returns ``CandidateSpec`` list,
    each spec already populated with ``proposer_model`` and ``fitness_mode``.

    The ``n`` parameter is treated as the TOTAL batch size; the actual
    slot allocation comes from ``HILLCLIMB_BATCH_COMPOSITION`` env var
    (defaults to 4:10:2 = 16 total). If ``n`` differs from the
    composition sum, the slots are scaled proportionally.

    Cold-start fallback: if there are no prior winners on this fault line
    (DB has < 1 evaluated candidate above ``min_score``), the function
    falls through to ``directed_capraro.generate_candidates`` so the
    fault line gets seeded with the hand-written seed pairs (mode=seed)
    or fresh Opus variants (mode=opus). Once winners exist, the
    structured loop kicks in.
    """
    rep_n, var_n, exp_n = _parse_composition(
        os.environ.get("HILLCLIMB_BATCH_COMPOSITION")
    )

    # Scale slot counts if caller asked for a different batch size.
    total_default = rep_n + var_n + exp_n
    if n != total_default and total_default > 0:
        scale = n / total_default
        rep_n = max(0, round(rep_n * scale))
        var_n = max(0, round(var_n * scale))
        exp_n = max(0, n - rep_n - var_n)

    rng = random.Random(seed)

    winners = _load_winners(db, fault_line_id)
    print(
        f"[structured_hillclimb:{fault_line_id}] "
        f"loaded {len(winners)} winner(s) "
        f"(top score {winners[0].score:.3f} on {winners[0].concept!r})"
        if winners else
        f"[structured_hillclimb:{fault_line_id}] no winners yet — cold start",
        flush=True,
    )

    if not winners:
        # Cold start: fall through to directed_capraro seeds + opus.
        # Use 'opus' mode if proposer present, else 'seed'.
        from .directed_capraro import generate_candidates as gen_capraro
        return gen_capraro(
            n=n,
            db=db,
            fault_line_id=fault_line_id,
            mode="opus",
            proposer=proposer,
        )

    # ----- Slot A: replication -----
    rep_specs = _slot_replication(winners[:2], rep_n)
    print(
        f"[structured_hillclimb:{fault_line_id}] "
        f"slot A (replication): {len(rep_specs)} spec(s)"
        + (f" of [{', '.join(p.concept for p in winners[:2])}]" if rep_specs else ""),
        flush=True,
    )

    # ----- Slot B: targeted variants -----
    var_specs = _slot_variants(winners[:3], var_n, proposer=proposer, rng=rng)
    print(
        f"[structured_hillclimb:{fault_line_id}] "
        f"slot B (variants): {len(var_specs)} spec(s) across "
        f"{', '.join({s._lineage_meta.get('mutation_type', '?') for s in var_specs})}",  # type: ignore[attr-defined]
        flush=True,
    )

    # ----- Slot C: cluster expansion -----
    exp_specs = _slot_cluster_expansion(db, fault_line_id, exp_n, proposer=proposer)
    print(
        f"[structured_hillclimb:{fault_line_id}] "
        f"slot C (cluster_expansion): {len(exp_specs)} spec(s)",
        flush=True,
    )

    out: list[CandidateSpec] = []
    seen_hashes: set[str] = set()
    # The variant-slot may produce a child with same content as another
    # child (same parent + same operator + RNG collision). Dedup here.
    for spec in rep_specs + var_specs + exp_specs:
        h = spec_hash(spec)
        if h in seen_hashes or db.has_candidate_hash(h):
            continue
        seen_hashes.add(h)
        out.append(spec)

    print(
        f"[structured_hillclimb:{fault_line_id}] "
        f"total: {len(out)} unique spec(s) "
        f"({len(rep_specs)+len(var_specs)+len(exp_specs)} produced, "
        f"{len(rep_specs)+len(var_specs)+len(exp_specs)-len(out)} dedup'd)",
        flush=True,
    )
    return out
