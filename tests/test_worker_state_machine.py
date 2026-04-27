"""End-to-end test of worker_v2's four-phase state machine, with mocked models.

These tests run in milliseconds because every model is replaced with a mock
that returns canned responses. The objective: prove the state machine moves
through Phase A → B → C → A correctly, that exactly one handle is loaded
at each transition, that crash-recovery works, and that the queue / DB
state is consistent at the end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.bridge import DetectionTrial
from src.evaluate import CandidateSpec
from src.judges.base import JudgeResult
from src.models.registry import HandleRegistry, MockHandle
from src.proposers.mock_proposer import MockProposer


# ----------------------------------------------------------------------
# Mocks (shared between tests)
# ----------------------------------------------------------------------

class _FakeTensor:
    def __init__(self, v: float):
        self._v = v
    def norm(self):
        return self
    def item(self) -> float:
        return self._v


class FakePipeline:
    def __init__(self):
        self.abliteration_ctx = None
        self.model = object()
        self.injected_calls = 0
        self.control_calls = 0

    def derive(self, concept, layer_idx):
        return _FakeTensor(100.0)

    def run_injected(self, concept, direction, layer_idx, strength, **kw):
        self.injected_calls += 1
        return DetectionTrial(
            concept=concept, layer_idx=layer_idx, strength=strength,
            injected=True, response=f"INJ-{concept}",
        )

    def run_control(self, concept, **kw):
        self.control_calls += 1
        return DetectionTrial(
            concept=concept, layer_idx=0, strength=0.0,
            injected=False, response=f"CTRL-{concept}",
        )


@dataclass
class FakeJudge:
    name: str = "fake"
    model: str = "fake"

    def score_detection(self, response, concept):
        det = response.startswith("INJ")
        return JudgeResult(det, det, True, "mock", response)

    def score_contrast_pair(self, response, axis, description, positive, negative):
        det = response.startswith("INJ")
        # detect on injected, never identify (Class 2)
        return JudgeResult(det, False, True, "mock", response)


def _make_proposer_response(strategy_axes: list[str]) -> str:
    """Build a JSON array string the strategies can parse."""
    pairs = []
    for ax in strategy_axes:
        pairs.append({
            "axis": ax,
            "description": f"description for {ax}",
            "rationale": "test rationale",
            "positive": [
                "positive ex one.", "positive ex two.", "positive ex three.",
                "positive ex four.", "positive ex five.", "positive ex six.",
            ],
            "negative": [
                "negative ex one.", "negative ex two.", "negative ex three.",
                "negative ex four.", "negative ex five.", "negative ex six.",
            ],
        })
    return json.dumps(pairs)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture()
def held_out():
    return ["Bread", "Whiskey", "Granite", "Architects",
            "Crimson", "Nostalgia", "Symmetry", "Surgeons",
            "Hummingbirds", "Photons", "Apple", "Symmetry"]


@pytest.fixture()
def controls():
    return ["Anxiety", "Curiosity", "Cinnamon", "Dancing", "Elephants"]


def _enqueue(queue_dir, spec: CandidateSpec) -> None:
    (queue_dir / "pending" / f"{spec.id}.json").write_text(
        json.dumps(spec.to_dict(), indent=2)
    )


def _make_word_spec(idx: int) -> CandidateSpec:
    return CandidateSpec(
        id=f"cand-test-{idx:04d}",
        strategy="random_explore",
        concept="Apple",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="mean_diff",
        baseline_n=32,
    )


def _build_registry(extract_patch_targets: list[str]):
    """Build a HandleRegistry of three MockHandles."""
    pipeline = FakePipeline()
    gemma = MockHandle(name="gemma", load_value=pipeline)
    judge_pair = ("model", "tokenizer")  # opaque to us
    judge_h = MockHandle(name="judge", load_value=judge_pair)
    proposer_pair = ("p_model", "p_tokenizer")
    proposer_h = MockHandle(name="proposer", load_value=proposer_pair)
    return HandleRegistry(gemma=gemma, judge=judge_h, proposer=proposer_h), pipeline


# ----------------------------------------------------------------------
# State-machine tests
# ----------------------------------------------------------------------

def test_full_cycle_one_candidate(
    tmp_db, tmp_queue, held_out, controls, monkeypatch
):
    """Single candidate: A generates pending, B judges, queue file moves to done."""
    import src.worker_v2 as w
    import src.paper as paper_pkg
    # extract_concept_vector isn't needed for mean_diff but we patch defensively
    monkeypatch.setattr(
        paper_pkg, "extract_concept_vector",
        lambda **kw: _FakeTensor(100.0),
    )

    spec = _make_word_spec(1)
    _enqueue(tmp_queue, spec)

    registry, pipeline = _build_registry([])
    fake_judge = FakeJudge()
    fake_proposer = MockProposer(response=_make_proposer_response(
        ["axis-a", "axis-b"]
    ))

    cycles = w.main_loop(
        registry=registry,
        db=tmp_db,
        held_out=held_out,
        controls=controls,
        judge_factory=lambda pair: fake_judge,
        proposer_factory=lambda pair: fake_proposer,
        batch_size=4,
        propose_threshold=4,
        propose_n=2,
        fault_line_id=None,
        max_cycles=1,
        abliteration_mode="vanilla",
    )
    assert cycles == 1

    # Candidate is done
    cand = tmp_db.get_candidate(spec.id)
    assert cand["status"] == "done"
    assert cand["score"] is not None
    # No pending rows left
    assert tmp_db.count_pending_responses() == 0
    # Queue file moved to done
    assert (tmp_queue / "done" / f"{spec.id}.json").exists()
    assert not (tmp_queue / "pending" / f"{spec.id}.json").exists()
    # Pipeline was actually exercised
    assert pipeline.injected_calls == 8
    assert pipeline.control_calls == 4


def test_handles_swap_correctly(
    tmp_db, tmp_queue, held_out, controls, monkeypatch
):
    """At end of cycle, no handle should be loaded except possibly proposer
    if Phase C ran. Crucial invariant: never two at once."""
    import src.worker_v2 as w
    import src.paper as paper_pkg
    monkeypatch.setattr(paper_pkg, "extract_concept_vector",
                        lambda **kw: _FakeTensor(100.0))

    spec = _make_word_spec(2)
    _enqueue(tmp_queue, spec)
    registry, _ = _build_registry([])

    fake_judge = FakeJudge()
    fake_proposer = MockProposer(response=_make_proposer_response(["x"]))

    # Track each load/unload event in time order
    order: list[str] = []
    for h in registry.all():
        original_load = h._do_load
        original_unload = h._do_unload
        name = h.name
        def make_load(orig, n):
            def f():
                order.append(f"load:{n}")
                return orig()
            return f
        def make_unload(orig, n):
            def f(obj):
                order.append(f"unload:{n}")
                return orig(obj)
            return f
        h._do_load = make_load(original_load, name)
        h._do_unload = make_unload(original_unload, name)

    w.main_loop(
        registry=registry,
        db=tmp_db,
        held_out=held_out,
        controls=controls,
        judge_factory=lambda pair: fake_judge,
        proposer_factory=lambda pair: fake_proposer,
        batch_size=4, propose_threshold=4, propose_n=1,
        max_cycles=1, abliteration_mode="vanilla",
    )

    # Reconstruct loaded set after each event; assert never > 1 loaded.
    loaded: set[str] = set()
    max_concurrent = 0
    for ev in order:
        action, name = ev.split(":")
        if action == "load":
            loaded.add(name)
        else:
            loaded.discard(name)
        max_concurrent = max(max_concurrent, len(loaded))
    assert max_concurrent <= 1, f"{max_concurrent} models loaded at once: {order}"

    # And we should see at least gemma → judge transition (the key swap)
    assert "load:gemma" in order
    assert "unload:gemma" in order
    assert "load:judge" in order
    assert "unload:judge" in order


def test_phase_c_writes_specs_when_queue_empty(
    tmp_db, tmp_queue, held_out, controls, monkeypatch
):
    """If no candidates pending and no orphans, Phase C runs and queues new specs."""
    import src.worker_v2 as w
    import src.paper as paper_pkg
    monkeypatch.setattr(paper_pkg, "extract_concept_vector",
                        lambda **kw: _FakeTensor(100.0))

    # Queue is empty initially. No orphan pending rows. main_loop should
    # short-circuit Phase A/B and go straight to Phase C, then exit on
    # max_cycles=1 (since no A→B happened, cycles stays 0; we cap at 1
    # and break when shutdown OR when cycles >= max_cycles. Need to use
    # max_cycles=0 + manual shutdown signal... easier path: just verify
    # that Phase C ran by checking the queue/pending after one iteration.)
    registry, _ = _build_registry([])
    fake_proposer = MockProposer(response=_make_proposer_response(
        ["axis-1", "axis-2", "axis-3"]
    ))

    # Hack: install shutdown after the first iteration to break the loop.
    iter_count = {"n": 0}
    original_oldest = w._oldest_pending
    def patched_oldest():
        iter_count["n"] += 1
        if iter_count["n"] > 5:
            import src.worker_v2 as ww
            ww._shutdown = True
        return original_oldest()
    monkeypatch.setattr(w, "_oldest_pending", patched_oldest)
    monkeypatch.setattr(w, "_shutdown", False)

    w.main_loop(
        registry=registry,
        db=tmp_db,
        held_out=held_out,
        controls=controls,
        judge_factory=lambda pair: FakeJudge(),
        proposer_factory=lambda pair: fake_proposer,
        batch_size=4, propose_threshold=10, propose_n=3,
        max_cycles=0, abliteration_mode="vanilla",
        poll_interval_s=0,
    )
    monkeypatch.setattr(w, "_shutdown", False)

    # Phase C should have written candidate specs to queue/pending
    pending_files = list((tmp_queue / "pending").glob("*.json"))
    assert len(pending_files) >= 1, f"expected at least 1 pending spec, got {len(pending_files)}"
    # Each should be a valid CandidateSpec dict
    spec_dict = json.loads(pending_files[0].read_text())
    assert "id" in spec_dict
    assert spec_dict["derivation_method"] == "contrast_pair"
    # The proposer was called at least once
    assert len(fake_proposer.calls) >= 1


def test_crash_recovery_picks_up_orphan_pending(
    tmp_db, tmp_queue, held_out, controls, monkeypatch
):
    """Simulate a crash mid-Phase-A: pending_responses rows exist for a
    candidate but Phase B never ran. Restart should drain them first."""
    import src.worker_v2 as w
    import src.paper as paper_pkg
    monkeypatch.setattr(paper_pkg, "extract_concept_vector",
                        lambda **kw: _FakeTensor(100.0))

    # Manually plant the post-crash state: 1 candidate row with status=running,
    # 12 pending_responses rows for it, no queue file.
    spec = _make_word_spec(99)
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
    for i in range(8):
        tmp_db.insert_pending_response(
            candidate_id=spec.id,
            eval_concept=f"P{i}",
            injected=True,
            alpha=2.0,
            direction_norm=100.0,
            response=f"INJ-P{i}",
            derivation_method="mean_diff",
            judge_concept=spec.concept,
        )
    for i in range(4):
        tmp_db.insert_pending_response(
            candidate_id=spec.id,
            eval_concept=f"C{i}",
            injected=False,
            alpha=0.0,
            direction_norm=0.0,
            response=f"CTRL-C{i}",
            derivation_method="mean_diff",
            judge_concept=spec.concept,
        )
    assert tmp_db.count_pending_responses(spec.id) == 12

    # Empty queue. Force shutdown after first phase-C call.
    iter_count = {"n": 0}
    original = w._oldest_pending
    def patched():
        iter_count["n"] += 1
        if iter_count["n"] > 3:
            import src.worker_v2 as ww
            ww._shutdown = True
        return original()
    monkeypatch.setattr(w, "_oldest_pending", patched)
    monkeypatch.setattr(w, "_shutdown", False)

    registry, _ = _build_registry([])
    fake_proposer = MockProposer(response=_make_proposer_response(["x"]))

    w.main_loop(
        registry=registry,
        db=tmp_db,
        held_out=held_out,
        controls=controls,
        judge_factory=lambda pair: FakeJudge(),
        proposer_factory=lambda pair: fake_proposer,
        batch_size=4, propose_threshold=4, propose_n=1,
        max_cycles=0, abliteration_mode="vanilla",
        poll_interval_s=0,
    )
    monkeypatch.setattr(w, "_shutdown", False)

    # The orphan was drained
    assert tmp_db.count_pending_responses(spec.id) == 0
    cand = tmp_db.get_candidate(spec.id)
    assert cand["status"] == "done"
    assert cand["score"] is not None
