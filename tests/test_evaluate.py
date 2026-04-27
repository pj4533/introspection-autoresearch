"""Tests for the phased evaluation: phase_a_generate + phase_b_judge.

These tests use mock pipelines and mock judges so they run in milliseconds
and can exhaust the failure modes (mid-batch error, contrast vs word style,
crash recovery, fitness-mode propagation) without loading real weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from src.bridge import DetectionTrial
from src.evaluate import CandidateSpec, phase_a_generate, phase_b_judge
from src.judges.base import JudgeResult


# ----------------------------------------------------------------------
# Mocks
# ----------------------------------------------------------------------

class _FakeTensor:
    """A torch.Tensor stand-in with .norm() and .item()."""
    def __init__(self, value: float):
        self._v = value

    def norm(self):
        return self

    def item(self) -> float:
        return self._v


class FakePipeline:
    """Minimal DetectionPipeline-shape good enough for phase_a_generate."""

    def __init__(self, *, derive_norm: float = 100.0):
        self._derive_norm = derive_norm
        self.abliteration_ctx = None
        self.model = object()
        self.injected_calls = []
        self.control_calls = []

    def derive(self, concept: str, layer_idx: int):
        return _FakeTensor(self._derive_norm)

    def run_injected(
        self,
        concept: str,
        direction,
        layer_idx: int,
        strength: float,
        trial_number: int = 1,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        judge_concept: Optional[str] = None,
        prompt_style: str = "paper",
        run_judge: bool = True,
    ) -> DetectionTrial:
        self.injected_calls.append({"concept": concept, "alpha": strength})
        return DetectionTrial(
            concept=concept,
            layer_idx=layer_idx,
            strength=strength,
            injected=True,
            response=f"INJ[{concept}]",
        )

    def run_control(
        self,
        concept: str,
        trial_number: int = 1,
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        prompt_style: str = "paper",
        run_judge: bool = True,
    ) -> DetectionTrial:
        self.control_calls.append({"concept": concept})
        return DetectionTrial(
            concept=concept,
            layer_idx=0,
            strength=0.0,
            injected=False,
            response=f"CTRL[{concept}]",
        )


@dataclass
class FakeJudge:
    """Returns a JudgeResult based on a static rule per response keyword."""
    name: str = "mock-judge"
    model: str = "mock-judge"

    def score_detection(self, response: str, concept: str) -> JudgeResult:
        det = "INJ" in response and "Bread" in response
        return JudgeResult(
            detected=det,
            identified=det,
            coherent=True,
            reasoning=f"mock detection rule for concept={concept}",
            raw=response,
        )

    def score_contrast_pair(
        self,
        response: str,
        axis: str,
        description: str,
        positive: list[str],
        negative: list[str],
    ) -> JudgeResult:
        det = "INJ" in response and len(positive) > 0
        return JudgeResult(
            detected=det,
            identified=False,
            coherent=True,
            reasoning=f"mock contrast rule for axis={axis}",
            raw=response,
        )


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture()
def held_out():
    return ["Bread", "Whiskey", "Granite", "Architects",
            "Crimson", "Nostalgia", "Symmetry", "Surgeons",
            "Hummingbirds", "Photons"]


@pytest.fixture()
def controls():
    return ["Anxiety", "Curiosity", "Cinnamon", "Dancing"]


@pytest.fixture()
def mean_diff_spec():
    return CandidateSpec(
        id="cand-test-md",
        strategy="random_explore",
        concept="Bread",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="mean_diff",
        baseline_n=32,
    )


@pytest.fixture()
def contrast_spec():
    return CandidateSpec(
        id="cand-test-cp",
        strategy="directed_capraro_causality",
        concept="causal-vs-temporal",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="contrast_pair",
        baseline_n=0,
        notes="X caused Y vs X then Y",
        contrast_pair={
            "axis": "causal-vs-temporal",
            "positive": [
                "The rain caused the flood.",
                "She knocked it because she reached.",
                "The engine failed due to the bolt.",
            ],
            "negative": [
                "The rain happened, then the flood.",
                "She reached, then it fell.",
                "The bolt was missing, then the engine failed.",
            ],
            "rationale": "test",
        },
        fitness_mode="ident_prioritized",
    )


# ----------------------------------------------------------------------
# Phase A
# ----------------------------------------------------------------------

def test_phase_a_word_style_writes_pending_rows(
    tmp_db, mean_diff_spec, held_out, controls
):
    pipeline = FakePipeline(derive_norm=100.0)
    summary = phase_a_generate(
        spec=mean_diff_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls,
        n_held_out=8, n_controls=4, verbose=False,
    )
    assert summary["skipped"] is False
    assert summary["n_inj"] == 8
    assert summary["n_ctrl"] == 4
    # 12 rows in pending
    assert tmp_db.count_pending_responses("cand-test-md") == 12
    rows = tmp_db.get_pending_responses("cand-test-md")
    assert sum(1 for r in rows if r["injected"]) == 8
    assert sum(1 for r in rows if not r["injected"]) == 4
    # Word-style: judge_concept is set, contrast fields are None
    for r in rows:
        assert r["derivation_method"] == "mean_diff"
        assert r["judge_concept"] == "Bread"
        assert r["contrast_axis"] is None
        assert r["contrast_positive"] is None


def test_phase_a_contrast_pair_writes_pending_rows(
    tmp_db, contrast_spec, held_out, controls, monkeypatch
):
    """Contrast-pair should call extract_concept_vector instead of derive()."""
    captured = {}
    def fake_extract(*, model, positive_prompts, negative_prompts,
                     layer_idx, token_idx, normalize):
        captured["pos"] = positive_prompts
        captured["neg"] = negative_prompts
        return _FakeTensor(150.0)

    # The import happens lazily inside phase_a_generate; patch at module level
    import src.paper as paper_pkg
    monkeypatch.setattr(paper_pkg, "extract_concept_vector", fake_extract)

    pipeline = FakePipeline()
    summary = phase_a_generate(
        spec=contrast_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls,
        verbose=False,
    )
    assert summary["norm"] == 150.0
    assert summary["alpha"] == pytest.approx(18000.0 / 150.0)
    rows = tmp_db.get_pending_responses("cand-test-cp")
    assert len(rows) == 12
    for r in rows:
        assert r["derivation_method"] == "contrast_pair"
        assert r["contrast_axis"] == "causal-vs-temporal"
        assert r["contrast_positive"] == contrast_spec.contrast_pair["positive"]
        assert r["contrast_negative"] == contrast_spec.contrast_pair["negative"]
        assert r["judge_concept"] is None
    # The fake extractor was indeed called
    assert captured["pos"] == contrast_spec.contrast_pair["positive"]


def test_phase_a_skips_when_already_pending(
    tmp_db, mean_diff_spec, held_out, controls
):
    """Crash-recovery case: phase_a_generate must NOT duplicate rows."""
    pipeline = FakePipeline()
    phase_a_generate(
        spec=mean_diff_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls,
        verbose=False,
    )
    assert tmp_db.count_pending_responses("cand-test-md") == 12

    summary = phase_a_generate(
        spec=mean_diff_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls,
        verbose=False,
    )
    assert summary["skipped"] is True
    # Still 12 rows, not 24
    assert tmp_db.count_pending_responses("cand-test-md") == 12


# ----------------------------------------------------------------------
# Phase B
# ----------------------------------------------------------------------

def _seed_candidate(tmp_db, spec):
    """Insert a candidate row so phase_b_judge can find it."""
    from src.strategies.random_explore import spec_hash
    tmp_db.insert_candidate(
        candidate_id=spec.id,
        strategy=spec.strategy,
        spec_json=json.dumps(spec.to_dict()),
        spec_hash=spec_hash(spec),
        concept=spec.concept,
        layer_idx=spec.layer_idx,
        target_effective=spec.target_effective,
        derivation_method=spec.derivation_method,
    )
    tmp_db.set_candidate_status(spec.id, "running")


def test_phase_b_finalizes_word_style(
    tmp_db, mean_diff_spec, held_out, controls
):
    _seed_candidate(tmp_db, mean_diff_spec)
    pipeline = FakePipeline()
    phase_a_generate(
        spec=mean_diff_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls, verbose=False,
    )
    judge = FakeJudge()
    result = phase_b_judge(
        candidate_id=mean_diff_spec.id, judge=judge, db=tmp_db, verbose=False,
    )
    # Judge returns det+ident=True for INJ[Bread] response
    # Held-out includes Bread (since it's in our held_out list and Phase A
    # filters out spec.concept). With concept='Bread' filtered, there's no
    # "Bread" probe — but the judge's mock rule fires on any "INJ" containing
    # "Bread" in the response. Since spec.concept is Bread, it's filtered out.
    # So no detection; rates = 0.
    assert result.detection_rate == 0.0
    # Pending rows are gone, candidate marked done
    assert tmp_db.count_pending_responses(mean_diff_spec.id) == 0
    assert tmp_db.get_candidate(mean_diff_spec.id)["status"] == "done"


def test_phase_b_with_detected_responses(
    tmp_db, held_out, controls
):
    """Judge that detects everything → detection_rate = 1.0."""
    spec = CandidateSpec(
        id="cand-detect-all",
        strategy="random_explore",
        concept="Apple",   # not in held_out, so all 8 probes survive
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="mean_diff",
        baseline_n=32,
    )
    _seed_candidate(tmp_db, spec)

    pipeline = FakePipeline()
    phase_a_generate(
        spec=spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls, verbose=False,
    )

    class DetectingAll:
        name = model = "always-yes"
        def score_detection(self, response, concept):
            return JudgeResult(True, True, True, "mock", response)
        def score_contrast_pair(self, response, axis, description, positive, negative):
            return JudgeResult(True, True, True, "mock", response)

    result = phase_b_judge(
        candidate_id=spec.id, judge=DetectingAll(), db=tmp_db, verbose=False,
    )
    # Wait — controls also detect, so fpr=1.0, fpr_penalty=0, score=0.
    assert result.detection_rate == 1.0
    assert result.identification_rate == 1.0
    assert result.fpr == 1.0
    assert result.score == 0.0  # FPR penalty zeroes it


def test_phase_b_with_clean_signal(tmp_db, held_out, controls):
    """Detect on injected only → high score."""
    spec = CandidateSpec(
        id="cand-clean",
        strategy="random_explore",
        concept="Apple",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="mean_diff",
        baseline_n=32,
    )
    _seed_candidate(tmp_db, spec)
    pipeline = FakePipeline()
    phase_a_generate(
        spec=spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls, verbose=False,
    )

    class InjOnlyDetector:
        name = model = "inj-only"
        def score_detection(self, response, concept):
            det = "INJ" in response  # detect injected only
            return JudgeResult(det, det, True, "mock", response)
        def score_contrast_pair(self, response, axis, description, positive, negative):
            det = "INJ" in response
            return JudgeResult(det, False, True, "mock", response)

    result = phase_b_judge(
        candidate_id=spec.id, judge=InjOnlyDetector(), db=tmp_db, verbose=False,
    )
    assert result.detection_rate == 1.0
    assert result.identification_rate == 1.0
    assert result.fpr == 0.0
    assert result.coherence_rate == 1.0
    # default fitness mode: (1.0 + 15*1.0) * 1.0 * 1.0 = 16.0
    assert result.score == pytest.approx(16.0)


def test_phase_b_respects_ident_prioritized_mode(
    tmp_db, contrast_spec, held_out, controls, monkeypatch
):
    """ident_prioritized fitness_mode in spec must change weights."""
    import src.paper as paper_pkg
    monkeypatch.setattr(
        paper_pkg, "extract_concept_vector",
        lambda **kw: _FakeTensor(100.0),
    )
    _seed_candidate(tmp_db, contrast_spec)
    pipeline = FakePipeline()
    phase_a_generate(
        spec=contrast_spec, pipeline=pipeline, db=tmp_db,
        held_out_concepts=held_out, control_concepts=controls, verbose=False,
    )

    class HalfIdent:
        name = model = "half-ident"
        def score_detection(self, *a, **kw):
            return JudgeResult(True, True, True, "mock", "ok")
        def score_contrast_pair(self, *a, **kw):
            # Always detected, never identified, on both inj and ctrl: forces
            # det=1, ident=0, fpr=1, fpr_penalty=0
            return JudgeResult(True, False, True, "mock", "ok")

    result = phase_b_judge(
        candidate_id=contrast_spec.id, judge=HalfIdent(), db=tmp_db, verbose=False,
    )
    # ident_prioritized mode: det_weight=0.5, ident_weight=30
    assert result.components["fitness_mode"] == "ident_prioritized"
    assert result.components["det_weight"] == 0.5
    assert result.components["ident_weight"] == 30.0


def test_phase_b_raises_when_no_pending(tmp_db, mean_diff_spec):
    _seed_candidate(tmp_db, mean_diff_spec)
    judge = FakeJudge()
    with pytest.raises(ValueError, match="no pending responses"):
        phase_b_judge(
            candidate_id=mean_diff_spec.id, judge=judge, db=tmp_db, verbose=False,
        )
