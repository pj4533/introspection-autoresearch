"""Random-exploration strategy.

Samples (concept, layer, target_effective) uniformly from the configured
search space. Deduplicates against candidates already in the DB via spec_hash.
Does not look at past fitness scores — just explores.

Good for seeding the loop and guaranteeing diversity even after smarter
strategies have been added.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from pathlib import Path
from typing import Optional

from ..db import ResultsDB
from ..evaluate import CandidateSpec

DEFAULT_LAYERS = [25, 30, 33, 36, 40]
DEFAULT_TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0, 22000.0]
DEFAULT_DERIVATION_METHODS = ["mean_diff"]  # Phase 2 MVP only has one method


def spec_hash(spec: CandidateSpec, abliteration_mode: str = "vanilla") -> str:
    """Stable dedup key across (concept, layer, target_effective, method,
    contrast-pair content, abliteration_mode).

    ``abliteration_mode`` defaults to ``"vanilla"`` for backward
    compatibility — every pre-2026-04-24 candidate was evaluated on vanilla
    Gemma3-12B, so their hashes stay stable. When the Phase 2 worker runs
    under paper-method abliteration (ADR-017 default), it passes
    ``abliteration_mode="paper_method"`` here so the same (concept, layer,
    eff, poles) under abliteration gets a distinct hash from its vanilla
    counterpart. This lets the DB hold both evaluations side-by-side without
    UNIQUE-constraint collisions.
    """
    payload = (
        f"{spec.concept}|{spec.layer_idx}|{spec.target_effective:.1f}"
        f"|{spec.derivation_method}"
    )
    if spec.derivation_method == "contrast_pair" and spec.contrast_pair is not None:
        pos = "|".join(spec.contrast_pair.get("positive", []))
        neg = "|".join(spec.contrast_pair.get("negative", []))
        payload += f"|pos:{pos}|neg:{neg}"
        # Include rationale ONLY when it carries a replication tag —
        # this lets multiple replications of the same parent coexist
        # (each gets a uuid'd rep_id appended to the rationale by
        # mutations.replication). For non-replication candidates the
        # rationale is summary text that doesn't affect the steering
        # direction, so we leave it out to keep legacy hashes stable.
        rationale = spec.contrast_pair.get("rationale", "") or ""
        if rationale.startswith("[replication-of-"):
            payload += f"|rep:{rationale[:80]}"
    # Only include the mode suffix when non-vanilla, to keep legacy vanilla
    # hashes bit-identical with pre-2026-04-24 entries.
    if abliteration_mode and abliteration_mode != "vanilla":
        payload += f"|abl:{abliteration_mode}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_candidates(
    n: int,
    db: ResultsDB,
    concept_pool: list[str],
    layers: Optional[list[int]] = None,
    target_effectives: Optional[list[float]] = None,
    derivation_methods: Optional[list[str]] = None,
    rng_seed: Optional[int] = None,
    max_attempts_per_candidate: int = 40,
) -> list[CandidateSpec]:
    """Generate `n` new candidate specs that are NOT already in the DB."""

    layers = layers or DEFAULT_LAYERS
    target_effectives = target_effectives or DEFAULT_TARGET_EFFECTIVES
    derivation_methods = derivation_methods or DEFAULT_DERIVATION_METHODS

    rng = random.Random(rng_seed if rng_seed is not None else time.time_ns())
    out: list[CandidateSpec] = []
    seen_this_batch: set[str] = set()

    for _ in range(n):
        for _ in range(max_attempts_per_candidate):
            spec = CandidateSpec(
                id=f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
                strategy="random_explore",
                concept=rng.choice(concept_pool),
                layer_idx=rng.choice(layers),
                target_effective=rng.choice(target_effectives),
                derivation_method=rng.choice(derivation_methods),
                baseline_n=32,
                notes="random_explore",
            )
            h = spec_hash(spec)
            if h in seen_this_batch or db.has_candidate_hash(h):
                continue
            seen_this_batch.add(h)
            out.append(spec)
            break
    return out


def write_candidate_json(spec: CandidateSpec, pending_dir: Path) -> Path:
    """Write a candidate spec JSON file to the queue's pending directory."""
    pending_dir.mkdir(parents=True, exist_ok=True)
    path = pending_dir / f"{spec.id}.json"
    path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    return path
