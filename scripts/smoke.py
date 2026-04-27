#!/usr/bin/env python3
"""Visible smoke test of the four-phase worker with mocked models.

Demonstrates one full A→B→C→A cycle without loading any real weights.
The user-visible output should show:
  - Phase A: 12 mock probes per candidate, written to pending_responses
  - Phase B: 12 mock judgments per candidate → fitness recorded
  - Phase C: 1+ new candidate specs written to queue/pending
  - Cycle counter increments

Useful for: verifying the worker behaves correctly after edits, demoing
the architecture, and capturing log output for the docs.

Run from repo root:
    .venv/bin/python scripts/smoke_v2.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bridge import DetectionTrial
from src.db import ResultsDB
from src.evaluate import CandidateSpec
from src.judges.base import JudgeResult
from src.models.registry import HandleRegistry, MockHandle
from src.proposers.mock_proposer import MockProposer


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

    def derive(self, concept, layer_idx):
        return _FakeTensor(100.0)

    def run_injected(self, concept, direction, layer_idx, strength, **kw):
        return DetectionTrial(
            concept=concept, layer_idx=layer_idx, strength=strength,
            injected=True, response=f"INJ-{concept}",
        )

    def run_control(self, concept, **kw):
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
        return JudgeResult(det, False, True, "mock", response)


def _make_proposer_response() -> str:
    pairs = [
        {
            "axis": "smoke-test-axis-1",
            "description": "first smoke axis",
            "rationale": "demonstrating Phase C output",
            "positive": [f"positive {i}." for i in range(6)],
            "negative": [f"negative {i}." for i in range(6)],
        },
        {
            "axis": "smoke-test-axis-2",
            "description": "second smoke axis",
            "rationale": "demonstrating multi-pair output",
            "positive": [f"alt positive {i}." for i in range(6)],
            "negative": [f"alt negative {i}." for i in range(6)],
        },
    ]
    return json.dumps(pairs)


def main() -> int:
    print("=" * 70)
    print("worker SMOKE TEST — fully mocked, no real models loaded")
    print("=" * 70)

    tmp = Path(tempfile.mkdtemp(prefix="worker_smoke_"))
    print(f"workspace: {tmp}")
    queue_dir = tmp / "queue"
    runs_dir = tmp / "runs"
    for sub in ("pending", "running", "done", "failed"):
        (queue_dir / sub).mkdir(parents=True)
    runs_dir.mkdir()

    # Patch worker's QUEUE/RUNS to our tmp dir
    import src.worker as w
    w.QUEUE = queue_dir
    w.RUNS = runs_dir
    # Patch shutdown flag (in case a previous test or run set it)
    w._shutdown = False

    # Patch the paper extractor (contrast_pair only; mean_diff doesn't need it)
    import src.paper as paper_pkg
    paper_pkg.extract_concept_vector = lambda **kw: _FakeTensor(120.0)

    db = ResultsDB(tmp / "results.db")

    held_out = ["Bread", "Whiskey", "Granite", "Architects",
                "Crimson", "Nostalgia", "Symmetry", "Surgeons",
                "Hummingbirds", "Photons", "Apple"]
    controls = ["Anxiety", "Curiosity", "Cinnamon", "Dancing", "Elephants"]

    # Pre-populate one candidate spec
    spec = CandidateSpec(
        id="cand-smoke-0001",
        strategy="random_explore",
        concept="Apple",
        layer_idx=33,
        target_effective=18000.0,
        derivation_method="mean_diff",
        baseline_n=32,
    )
    (queue_dir / "pending" / f"{spec.id}.json").write_text(
        json.dumps(spec.to_dict(), indent=2)
    )
    print(f"[smoke] enqueued candidate {spec.id}")

    # Build registry of MockHandles
    pipeline = FakePipeline()
    gemma = MockHandle(name="gemma3_12b-mock", load_value=pipeline)
    judge_h = MockHandle(name="qwen-judge-mock",
                         load_value=("model_obj", "tokenizer_obj"))
    proposer_h = MockHandle(name="qwen-proposer-mock",
                            load_value=("p_model", "p_tokenizer"))
    registry = HandleRegistry(gemma=gemma, judge=judge_h, proposer=proposer_h)

    fake_judge = FakeJudge()
    fake_proposer = MockProposer(response=_make_proposer_response())

    print()
    print("=" * 70)
    print("STARTING main_loop, max_cycles=1")
    print("=" * 70)
    cycles = w.main_loop(
        registry=registry,
        db=db,
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

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"cycles completed: {cycles}")
    cand = db.get_candidate(spec.id)
    print(f"\noriginal candidate {spec.id}:")
    print(f"  status: {cand['status']}")
    print(f"  score:  {cand['score']:.3f}" if cand['score'] is not None else "  score:  None")
    print(f"  det:    {cand['detection_rate']}")
    print(f"  ident:  {cand['identification_rate']}")
    print(f"  fpr:    {cand['fpr']}")
    print(f"  coh:    {cand['coherence_rate']}")

    pending_files = sorted((queue_dir / "pending").glob("*.json"))
    done_files = sorted((queue_dir / "done").glob("*.json"))
    print(f"\nqueue/done:    {[p.name for p in done_files]}")
    print(f"queue/pending: {[p.name for p in pending_files]}")
    if pending_files:
        new_spec = json.loads(pending_files[0].read_text())
        print(f"\nFirst new spec from Phase C:")
        print(f"  id:       {new_spec['id']}")
        print(f"  strategy: {new_spec['strategy']}")
        print(f"  axis:     {new_spec.get('contrast_pair', {}).get('axis')}")
        print(f"  layer:    {new_spec['layer_idx']}")
        print(f"  eff:      {new_spec['target_effective']}")

    # Handle bookkeeping: confirm at-most-one was loaded at any point
    print(f"\nhandle load/unload counts:")
    for h in (gemma, judge_h, proposer_h):
        print(f"  {h.name}: loaded {h.load_count}× / unloaded {h.unload_count}×")

    # Cleanup
    shutil.rmtree(tmp)
    print(f"\n[smoke] workspace removed.")
    print(f"\nSMOKE PASSED ✓" if cycles == 1 and cand['status'] == 'done'
          else "SMOKE FAILED ✗")
    return 0 if cycles == 1 and cand['status'] == 'done' else 1


if __name__ == "__main__":
    sys.exit(main())
