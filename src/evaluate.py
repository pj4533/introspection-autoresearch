"""Phase 2 candidate fitness function.

MVP: single-tier evaluation with a 3-component multiplicative fitness score.
For each candidate, we derive the steering direction, run it on a small set
of held-out concepts (injected) and a few control trials (no injection),
judge each response with Claude, and produce:

    fitness = detection_rate * (1 - fpr) * coherence_rate

Higher is better, bounded in [0, 1]. Multiplicative means failing any one
component tanks the score — a direction that causes high detection but also
high false positives, or that destroys coherence, is worthless.

This is intentionally simpler than the 6-component fitness in the spec. T1/T2/
T3 tiered screening and cross-phrasing / monotonicity / bidirectional checks
can be added once the loop is producing useful candidates.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .bridge import DetectionPipeline
from .db import ResultsDB
from .judges.base import Judge

DEFAULT_N_HELD_OUT = 8
DEFAULT_N_CONTROLS = 4


@dataclass
class CandidateSpec:
    """A candidate steering direction's full specification.

    Two derivation methods supported:

    - ``derivation_method="mean_diff"`` (default): derive a steering vector for
      the single concept word ``concept`` by mean-differencing its activations
      against ``baseline_n`` random baseline words. This is what ``random_explore``
      produces — the Phase 1 approach extended across more concepts.

    - ``derivation_method="contrast_pair"``: derive a direction from two LISTS of
      short example sentences, ``contrast_pair["positive"]`` vs
      ``contrast_pair["negative"]``. The resulting direction lives between the
      two sets and captures an abstract axis (e.g., certainty-vs-doubt) that
      doesn't correspond to any single English word. This is what
      ``novel_contrast`` produces. For these candidates, ``concept`` holds a
      human-readable label like ``"commitment_vs_hesitation"`` that describes
      the axis; it is not injected into the prompt.
    """
    id: str
    strategy: str
    concept: str
    layer_idx: int
    target_effective: float
    derivation_method: str = "mean_diff"
    baseline_n: int = 32
    notes: str = ""
    # Only populated when derivation_method == "contrast_pair". Shape:
    #   {"axis": str, "positive": list[str], "negative": list[str]}
    contrast_pair: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateSpec":
        return cls(
            id=d["id"],
            strategy=d.get("strategy", "unknown"),
            concept=d["concept"],
            layer_idx=int(d["layer_idx"]),
            target_effective=float(d["target_effective"]),
            derivation_method=d.get("derivation_method", "mean_diff"),
            baseline_n=int(d.get("baseline_n", 32)),
            notes=d.get("notes", ""),
            contrast_pair=d.get("contrast_pair"),
        )

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "strategy": self.strategy,
            "concept": self.concept,
            "layer_idx": self.layer_idx,
            "target_effective": self.target_effective,
            "derivation_method": self.derivation_method,
            "baseline_n": self.baseline_n,
            "notes": self.notes,
        }
        if self.contrast_pair is not None:
            out["contrast_pair"] = self.contrast_pair
        return out


@dataclass
class FitnessResult:
    score: float
    detection_rate: float
    identification_rate: float
    fpr: float
    coherence_rate: float
    n_held_out: int
    n_controls: int
    components: dict


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def evaluate_candidate(
    spec: CandidateSpec,
    pipeline: DetectionPipeline,
    db: ResultsDB,
    held_out_concepts: list[str],
    control_concepts: list[str],
    n_held_out: int = DEFAULT_N_HELD_OUT,
    n_controls: int = DEFAULT_N_CONTROLS,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> FitnessResult:
    """Evaluate one candidate, record everything to the DB, return fitness.

    RNG seeds are derived from `rng_seed + hash(spec.id)` so different
    candidates get different held-out shuffles and per-trial sampling seeds
    while each candidate remains individually reproducible.
    """

    # Derive a stable per-candidate seed from the candidate ID. Using
    # hashlib.sha256 rather than Python's built-in hash() because the latter
    # is randomized between interpreter runs.
    base_seed = rng_seed if rng_seed is not None else 0
    import hashlib
    id_seed = int(hashlib.sha256(spec.id.encode()).hexdigest()[:8], 16)
    candidate_seed = base_seed + id_seed

    rng = random.Random(candidate_seed)
    held_out = [c for c in held_out_concepts if c.lower() != spec.concept.lower()]
    rng.shuffle(held_out)
    held_out = held_out[:n_held_out]

    controls = [c for c in control_concepts if c.lower() != spec.concept.lower()]
    rng.shuffle(controls)
    controls = controls[:n_controls]

    # --- derive the steering direction once ---------------------------------
    if spec.derivation_method == "contrast_pair":
        # Novel-contrast: derive direction from a pair of prompt LISTS, one
        # representing each pole of an abstract axis. Uses the paper's
        # extract_concept_vector (positive_prompts vs negative_prompts).
        if spec.contrast_pair is None:
            raise ValueError(
                f"Candidate {spec.id}: derivation_method='contrast_pair' but "
                "contrast_pair field is None"
            )
        from .paper import extract_concept_vector
        direction = extract_concept_vector(
            model=pipeline.model,
            positive_prompts=spec.contrast_pair["positive"],
            negative_prompts=spec.contrast_pair["negative"],
            layer_idx=spec.layer_idx,
            token_idx=-1,
            normalize=False,
        )
    else:
        # mean_diff (default, Phase 1 approach): concept word vs baseline words
        direction = pipeline.derive(concept=spec.concept, layer_idx=spec.layer_idx)

    norm = float(direction.norm().item())
    alpha = spec.target_effective / max(norm, 1e-6)

    if verbose:
        print(
            f"    derive: concept={spec.concept!r} L={spec.layer_idx} "
            f"||dir||={norm:.0f} alpha={alpha:.2f}"
        )

    n_det = n_ident = n_coh = 0
    n_inj = 0

    # --- injected trials on held-out concepts -------------------------------
    for c in held_out:
        # Per-trial seed derived from (candidate_id, slot). Different candidates
        # get different sampling paths even on the same slot concept, so we
        # don't overfit to lucky / unlucky seeds repeating across candidates.
        trial_seed = candidate_seed + int(
            hashlib.sha256(f"{spec.id}|{c}".encode()).hexdigest()[:8], 16
        )
        torch.manual_seed(trial_seed % (2**31))
        trial = pipeline.run_injected(
            concept=c,                 # slot label (not in prompt, used as metadata)
            direction=direction,
            layer_idx=spec.layer_idx,
            strength=alpha,
            trial_number=1,
            max_new_tokens=120,
            judge_concept=spec.concept,  # grade identification against SOURCE concept
        )
        jr = trial.judge_result
        db.record_evaluation(
            candidate_id=spec.id,
            eval_concept=c,
            injected=True,
            alpha=alpha,
            direction_norm=norm,
            response=trial.response,
            detected=jr.detected,
            identified=jr.identified,
            coherent=jr.coherent,
            judge_model=pipeline.judge.model if hasattr(pipeline.judge, "model") else "unknown",
            judge_reasoning=jr.reasoning,
        )
        n_inj += 1
        if jr.coherent:
            n_coh += 1
            if jr.detected:
                n_det += 1
            if jr.identified:
                n_ident += 1
        if verbose:
            print(
                f"      inj  {c:<14} det={int(jr.detected)} "
                f"ident={int(jr.identified)} coh={int(jr.coherent)}"
            )

    # --- control trials (no injection, ask the same question) --------------
    n_fp = 0
    n_ctrl = 0
    for c in controls:
        trial_seed = candidate_seed + int(
            hashlib.sha256(f"{spec.id}|ctrl|{c}".encode()).hexdigest()[:8], 16
        )
        torch.manual_seed(trial_seed % (2**31))
        trial = pipeline.run_control(
            concept=c,
            trial_number=1,
            max_new_tokens=120,
            # judge_concept: for controls, passes through as `c` (slot label).
            # Detection should be false regardless of target; identification is
            # moot. Kept consistent with run_injected's semantics.
        )
        jr = trial.judge_result
        db.record_evaluation(
            candidate_id=spec.id,
            eval_concept=c,
            injected=False,
            alpha=0.0,
            direction_norm=0.0,
            response=trial.response,
            detected=jr.detected,
            identified=jr.identified,
            coherent=jr.coherent,
            judge_model=pipeline.judge.model if hasattr(pipeline.judge, "model") else "unknown",
            judge_reasoning=jr.reasoning,
        )
        n_ctrl += 1
        if jr.detected:
            n_fp += 1
        if verbose:
            print(f"      ctrl {c:<14} det={int(jr.detected)} (FP={int(jr.detected)})")

    # --- compute fitness --------------------------------------------------
    detection_rate = n_det / n_inj if n_inj else 0.0
    identification_rate = n_ident / n_inj if n_inj else 0.0
    fpr = n_fp / n_ctrl if n_ctrl else 0.0
    coherence_rate = n_coh / n_inj if n_inj else 0.0

    # Multiplicative score. The `fpr` component uses a steep penalty so a
    # single false positive in a small sample doesn't totally wipe the score
    # but consistent false positives absolutely do.
    fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
    score = detection_rate * fpr_penalty * coherence_rate

    components = {
        "detection_rate": detection_rate,
        "identification_rate": identification_rate,
        "fpr": fpr,
        "coherence_rate": coherence_rate,
        "fpr_penalty": fpr_penalty,
        "direction_norm": norm,
        "alpha": alpha,
        "n_held_out_tested": n_inj,
        "n_controls_tested": n_ctrl,
        "concepts_held_out": held_out,
        "concepts_controls": controls,
    }

    db.record_fitness(
        candidate_id=spec.id,
        score=score,
        detection_rate=detection_rate,
        identification_rate=identification_rate,
        fpr=fpr,
        coherence_rate=coherence_rate,
        n_held_out=n_inj,
        n_controls=n_ctrl,
        components_json=json.dumps(components),
    )

    if verbose:
        print(
            f"    fitness: score={score:.3f}  "
            f"det={detection_rate:.2%} fpr={fpr:.2%} coh={coherence_rate:.2%}"
        )

    return FitnessResult(
        score=score,
        detection_rate=detection_rate,
        identification_rate=identification_rate,
        fpr=fpr,
        coherence_rate=coherence_rate,
        n_held_out=n_inj,
        n_controls=n_ctrl,
        components=components,
    )


def load_eval_sets(
    held_out_path: Path,
    control_path: Optional[Path] = None,
) -> tuple[list[str], list[str]]:
    """Load held-out and control concept lists. Controls default to held-out."""
    held_out = json.loads(Path(held_out_path).read_text())["concepts"]
    if control_path is None:
        controls = held_out
    else:
        controls = json.loads(Path(control_path).read_text())["concepts"]
    return held_out, controls
