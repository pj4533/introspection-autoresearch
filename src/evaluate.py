"""Phase 2 candidate fitness — phased evaluation for the four-phase worker.

Two entry points, called in sequence by the worker:

    Phase A — `phase_a_generate(spec, pipeline, db, ...)`
        Gemma3-12B is loaded. Derive the steering direction, run 8 injected
        + 4 control probes, store all 12 raw responses in `pending_responses`.
        No judge calls.

    Phase B — `phase_b_judge(candidate_id, judge, db, ...)`
        Judge model is loaded (Gemma is not). Score every pending response,
        insert into `evaluations`, compute fitness, write to `fitness_scores`,
        delete the pending rows, mark the candidate `done`.

The `pending_responses` rows are the durable hand-off: a crash between A
and B doesn't lose work — the worker drains orphan pending rows on
startup before generating new ones.

Fitness math (unchanged from prior pipeline):

    fitness = (det_weight * detection_rate + ident_weight * identification_rate)
              * fpr_penalty * coherence_rate
    where fpr_penalty = max(0, 1 - 5 * fpr)

    Default mode:           det_weight=1.0, ident_weight=15.0
    ident_prioritized mode: det_weight=0.5, ident_weight=30.0  (ADR-018)

Mode is read from `spec.fitness_mode` first, then `FITNESS_MODE` env var,
then defaults to "default". Strategy generators set per-spec mode so a
mixed queue scores each candidate under whatever mode the strategy that
generated it intended.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
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

    Two derivation methods:

    - ``derivation_method="mean_diff"``: derive a steering vector for the
      single concept word ``concept`` by mean-differencing its activations
      against ``baseline_n`` random baseline words.

    - ``derivation_method="contrast_pair"``: derive a direction from two
      LISTS of short example sentences,
      ``contrast_pair["positive"]`` vs ``contrast_pair["negative"]``. The
      resulting direction lives between the two sets and captures an
      abstract axis (e.g., certainty-vs-doubt) that doesn't correspond to
      any single English word. For these candidates, ``concept`` is just a
      label like ``"commitment-vs-hesitation"``; it is NOT injected into
      the prompt.
    """
    id: str
    strategy: str
    concept: str
    layer_idx: int
    target_effective: float
    derivation_method: str = "mean_diff"
    baseline_n: int = 32
    notes: str = ""
    # Only populated when derivation_method == "contrast_pair":
    #   {"axis": str, "positive": list[str], "negative": list[str]}
    contrast_pair: Optional[dict] = None
    # Per-candidate fitness mode (ADR-018). Strategies set this to
    # "ident_prioritized" when they want identification-weighted ranking.
    # Falls back to FITNESS_MODE env var if unset, then to "default".
    fitness_mode: Optional[str] = None
    # Which LLM proposer produced this spec. Set by Phase C in the worker.
    # NULL for non-LLM strategies (random_explore). The web UI displays this
    # per row so a mixed-history leaderboard shows the actual model behind
    # each axis (claude-opus-4-7, Qwen3.6-27B-MLX-8bit, etc.).
    proposer_model: Optional[str] = None

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
            fitness_mode=d.get("fitness_mode"),
            proposer_model=d.get("proposer_model"),
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
        if self.fitness_mode is not None:
            out["fitness_mode"] = self.fitness_mode
        if self.proposer_model is not None:
            out["proposer_model"] = self.proposer_model
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


def _candidate_seed(spec: CandidateSpec, base_seed: int) -> int:
    id_seed = int(hashlib.sha256(spec.id.encode()).hexdigest()[:8], 16)
    return base_seed + id_seed


# ----------------------------------------------------------------------
# Phase A — generate responses (Gemma loaded)
# ----------------------------------------------------------------------

def phase_a_generate(
    spec: CandidateSpec,
    pipeline: DetectionPipeline,
    db: ResultsDB,
    held_out_concepts: list[str],
    control_concepts: list[str],
    n_held_out: int = DEFAULT_N_HELD_OUT,
    n_controls: int = DEFAULT_N_CONTROLS,
    rng_seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """Generate Gemma responses for `spec` and stash them in pending_responses.

    No judge call here — responses are queued for Phase B. Returns a summary
    dict so the worker can log progress.

    Idempotent: if `spec.id` already has rows in pending_responses, we trust
    them (crash recovery) and skip generation.
    """
    base_seed = rng_seed if rng_seed is not None else 0
    candidate_seed = _candidate_seed(spec, base_seed)

    if db.count_pending_responses(spec.id) > 0:
        if verbose:
            print(
                f"    [phase_a] {spec.id} already has pending responses, skipping",
                flush=True,
            )
        return {
            "candidate_id": spec.id,
            "skipped": True,
            "reason": "already_pending",
            "n_pending": db.count_pending_responses(spec.id),
        }

    rng = random.Random(candidate_seed)
    held_out = [c for c in held_out_concepts if c.lower() != spec.concept.lower()]
    rng.shuffle(held_out)
    held_out = held_out[:n_held_out]

    controls = [c for c in control_concepts if c.lower() != spec.concept.lower()]
    rng.shuffle(controls)
    controls = controls[:n_controls]

    # Suspend abliteration hooks during direction derivation (ADR-014).
    suspend = (
        pipeline.abliteration_ctx.suspended()
        if getattr(pipeline, "abliteration_ctx", None) is not None
        and pipeline.abliteration_ctx.installed
        else contextlib.nullcontext()
    )
    with suspend:
        if spec.derivation_method == "contrast_pair":
            if spec.contrast_pair is None:
                raise ValueError(
                    f"Candidate {spec.id}: derivation_method='contrast_pair' "
                    "but contrast_pair field is None"
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
            direction = pipeline.derive(
                concept=spec.concept, layer_idx=spec.layer_idx
            )

    norm = float(direction.norm().item())
    alpha = spec.target_effective / max(norm, 1e-6)

    prompt_style = "open" if spec.derivation_method == "contrast_pair" else "paper"

    if verbose:
        print(
            f"    derive: concept={spec.concept!r} L={spec.layer_idx} "
            f"||dir||={norm:.0f} alpha={alpha:.2f} prompt={prompt_style}",
            flush=True,
        )

    # Pre-compute contrast-pair fields once so they don't get re-stringified
    # on every probe row.
    if spec.derivation_method == "contrast_pair":
        contrast_axis = spec.contrast_pair["axis"]
        contrast_description = (
            spec.notes or spec.contrast_pair.get("description") or ""
        )
        contrast_positive = spec.contrast_pair["positive"]
        contrast_negative = spec.contrast_pair["negative"]
        judge_concept = None
    else:
        contrast_axis = None
        contrast_description = None
        contrast_positive = None
        contrast_negative = None
        judge_concept = spec.concept

    n_inj = 0
    n_ctrl = 0

    # Injected probes
    for c in held_out:
        trial_seed = candidate_seed + int(
            hashlib.sha256(f"{spec.id}|{c}".encode()).hexdigest()[:8], 16
        )
        torch.manual_seed(trial_seed % (2**31))
        trial = pipeline.run_injected(
            concept=c,
            direction=direction,
            layer_idx=spec.layer_idx,
            strength=alpha,
            trial_number=1,
            max_new_tokens=120,
            judge_concept=spec.concept,
            prompt_style=prompt_style,
            run_judge=False,
        )
        db.insert_pending_response(
            candidate_id=spec.id,
            eval_concept=c,
            injected=True,
            alpha=alpha,
            direction_norm=norm,
            response=trial.response,
            derivation_method=spec.derivation_method,
            judge_concept=judge_concept,
            contrast_axis=contrast_axis,
            contrast_description=contrast_description,
            contrast_positive=contrast_positive,
            contrast_negative=contrast_negative,
        )
        n_inj += 1
        if verbose:
            print(f"      gen  inj  {c}", flush=True)

    # Control probes
    for c in controls:
        trial_seed = candidate_seed + int(
            hashlib.sha256(f"{spec.id}|ctrl|{c}".encode()).hexdigest()[:8], 16
        )
        torch.manual_seed(trial_seed % (2**31))
        trial = pipeline.run_control(
            concept=c,
            trial_number=1,
            max_new_tokens=120,
            prompt_style=prompt_style,
            run_judge=False,
        )
        db.insert_pending_response(
            candidate_id=spec.id,
            eval_concept=c,
            injected=False,
            alpha=0.0,
            direction_norm=0.0,
            response=trial.response,
            derivation_method=spec.derivation_method,
            judge_concept=judge_concept,
            contrast_axis=contrast_axis,
            contrast_description=contrast_description,
            contrast_positive=contrast_positive,
            contrast_negative=contrast_negative,
        )
        n_ctrl += 1
        if verbose:
            print(f"      gen  ctrl {c}", flush=True)

    return {
        "candidate_id": spec.id,
        "skipped": False,
        "n_inj": n_inj,
        "n_ctrl": n_ctrl,
        "norm": norm,
        "alpha": alpha,
        "prompt_style": prompt_style,
        "concepts_held_out": held_out,
        "concepts_controls": controls,
    }


# ----------------------------------------------------------------------
# Phase B — judge responses (judge loaded)
# ----------------------------------------------------------------------

def phase_b_judge(
    candidate_id: str,
    judge: Judge,
    db: ResultsDB,
    *,
    abliteration_mode: str = "vanilla",
    verbose: bool = True,
) -> FitnessResult:
    """Score `candidate_id`'s pending responses and finalize fitness.

    Reads from `pending_responses`, calls the judge per row, inserts into
    `evaluations`, computes fitness, writes to `fitness_scores`, deletes
    the pending rows, marks the candidate `status='done'`.
    """
    pending = db.get_pending_responses(candidate_id)
    if not pending:
        raise ValueError(
            f"phase_b_judge: no pending responses for candidate_id={candidate_id!r}"
        )

    cand_row = db.get_candidate(candidate_id)
    if cand_row is None:
        raise ValueError(f"phase_b_judge: candidate {candidate_id!r} missing from DB")
    spec_dict = json.loads(cand_row["spec_json"])
    spec_fitness_mode = spec_dict.get("fitness_mode")

    judge_model_tag = (
        getattr(judge, "model", None)
        or getattr(judge, "model_tag", None)
        or "unknown"
    )

    n_inj = n_det = n_ident = n_coh = 0
    n_ctrl = n_fp = 0

    for row in pending:
        if row["derivation_method"] == "contrast_pair":
            jr = judge.score_contrast_pair(
                response=row["response"],
                axis=row["contrast_axis"] or "",
                description=row["contrast_description"] or "",
                positive=row["contrast_positive"] or [],
                negative=row["contrast_negative"] or [],
            )
        else:
            jr = judge.score_detection(
                row["response"], row["judge_concept"] or ""
            )

        db.record_evaluation(
            candidate_id=candidate_id,
            eval_concept=row["eval_concept"],
            injected=row["injected"],
            alpha=row["alpha"],
            direction_norm=row["direction_norm"],
            response=row["response"],
            detected=jr.detected,
            identified=jr.identified,
            coherent=jr.coherent,
            judge_model=judge_model_tag,
            judge_reasoning=jr.reasoning,
        )

        if row["injected"]:
            n_inj += 1
            if jr.coherent:
                n_coh += 1
                if jr.detected:
                    n_det += 1
                if jr.identified:
                    n_ident += 1
        else:
            n_ctrl += 1
            if jr.detected:
                n_fp += 1

        if verbose:
            inj_tag = "inj" if row["injected"] else "ctrl"
            print(
                f"      judge {inj_tag} {row['eval_concept']:<14} "
                f"det={int(jr.detected)} ident={int(jr.identified)} "
                f"coh={int(jr.coherent)}",
                flush=True,
            )

    detection_rate = n_det / n_inj if n_inj else 0.0
    identification_rate = n_ident / n_inj if n_inj else 0.0
    fpr = n_fp / n_ctrl if n_ctrl else 0.0
    coherence_rate = n_coh / n_inj if n_inj else 0.0

    fpr_penalty = max(0.0, 1.0 - 5.0 * fpr)
    fitness_mode = spec_fitness_mode or os.environ.get("FITNESS_MODE", "default")
    if fitness_mode == "ident_prioritized":
        det_weight = 0.5
        ident_weight = 30.0
    else:
        fitness_mode = "default"
        det_weight = 1.0
        ident_weight = 15.0
    ident_bonus = ident_weight * identification_rate
    score = (det_weight * detection_rate + ident_bonus) * fpr_penalty * coherence_rate

    components = {
        "detection_rate": detection_rate,
        "identification_rate": identification_rate,
        "fpr": fpr,
        "coherence_rate": coherence_rate,
        "fpr_penalty": fpr_penalty,
        "ident_bonus": ident_bonus,
        "fitness_mode": fitness_mode,
        "det_weight": det_weight,
        "ident_weight": ident_weight,
        "abliteration_mode": abliteration_mode,
        "judge_model": judge_model_tag,
        "n_held_out_tested": n_inj,
        "n_controls_tested": n_ctrl,
    }

    db.record_fitness(
        candidate_id=candidate_id,
        score=score,
        detection_rate=detection_rate,
        identification_rate=identification_rate,
        fpr=fpr,
        coherence_rate=coherence_rate,
        n_held_out=n_inj,
        n_controls=n_ctrl,
        components_json=json.dumps(components),
    )

    db.delete_pending_responses(candidate_id)
    db.set_candidate_status(candidate_id, "done")

    if verbose:
        print(
            f"    fitness: score={score:.3f}  "
            f"det={detection_rate:.2%} fpr={fpr:.2%} coh={coherence_rate:.2%}",
            flush=True,
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


# ----------------------------------------------------------------------
# Eval-set loader (used by the worker at startup)
# ----------------------------------------------------------------------

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
