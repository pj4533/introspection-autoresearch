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


def spec_hash(spec: CandidateSpec) -> str:
    """Stable dedup key across (concept, layer, target_effective, method).

    For contrast_pair candidates, also hashes the pair content so the same
    (concept label, layer, eff) with different contrast pairs gets distinct
    hashes — and so the same pair isn't re-proposed across runs.
    """
    payload = (
        f"{spec.concept}|{spec.layer_idx}|{spec.target_effective:.1f}"
        f"|{spec.derivation_method}"
    )
    if spec.derivation_method == "contrast_pair" and spec.contrast_pair is not None:
        pos = "|".join(spec.contrast_pair.get("positive", []))
        neg = "|".join(spec.contrast_pair.get("negative", []))
        payload += f"|pos:{pos}|neg:{neg}"
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
