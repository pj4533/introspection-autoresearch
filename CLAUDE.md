# CLAUDE.md

Orientation for Claude Code sessions working on this repo.

## Project one-liner

Reproduce the Macar et al. (2026) *Mechanisms of Introspective Awareness in
Language Models* paper on `Gemma3-12B-it` locally (Mac Studio M2 Ultra, MPS,
bf16), then build a two-tier autoresearch loop that hunts for novel steering
directions affecting introspection capability.

## Canonical references (read first if orienting)

- **Roadmap** (what's done, what's next, full rationale): `docs/roadmap.md`
- **Architectural decisions** (ADR log): `docs/decisions.md`
- Plain-English project writeup (public-facing): `docs/plain_english.md`
- Phase 1 technical results: `docs/phase1_results.md`
- Phase 1.5 technical results (paper-method abliteration): `docs/phase1_5_results.md`
- Phase 2b hill-climbing plan (shipped): `docs/phase2b_hillclimb.md`
- Phase 2c unified autoresearch: `docs/phase2c_autoresearch.md`
- **Phase 2d directed-hypothesis probes (current focus)**: `docs/phase2d_directed_hypotheses.md`
- Full spec: `docs/01_introspection_steering_autoresearch.md`
- Approved plan (pre-execution, for historical reference): `/Users/pj4533/.claude/plans/lexical-mixing-unicorn.md`
- README: `README.md` (repo layout, setup, running)

**Rule:** if a project decision only lives in chat logs, memory files, or the
plan file, it's a bug. Commit it to `docs/roadmap.md` (forward-looking) or
`docs/decisions.md` (rationale for past choices).

## Status snapshot

- Phase 1 MVP (single-concept end-to-end): **done**. See
  `notebooks/01_reproduce_paper.ipynb`.
- Phase 1 full sweep (50 concepts × 9 layers + 50 controls = 500 trials):
  **done**. Ran 2.2 hours on Apr 2026. Results in
  [`docs/phase1_results.md`](docs/phase1_results.md) and
  `data/phase1_export/findings.json`. Layer curve peaks at L33 (68.75%
  depth), FPR=0%, detection rate 6% at best layer. Mechanism reproduced at
  smaller magnitude than 27B.
- Phase 1 acceptance: **qualitative PASS, threshold FAIL** (spec set 20%
  threshold based on paper's 27B result; our 12B got 6%). Decision 2026-04-16:
  proceed to Phase 2 as the mechanism is clearly reproduced and the
  autoresearch work is the project's unique contribution.
- Phase 2 scaffolding: **built**. `src/evaluate.py` (3-component
  multiplicative fitness: detection × (1 − 5·fpr) × coherence),
  `src/worker.py` (long-lived queue poller, loads Gemma once),
  `src/researcher.py` (short-lived, invoked periodically, writes candidate
  JSON to queue/pending/), `src/strategies/random_explore.py` (first
  strategy; samples concept×layer×target_effective), `scripts/start_worker.sh`
  + `scripts/start_researcher.sh` (setsid nohup launchers), `data/eval_sets/`
  (held_out_concepts.json = 20 concepts NOT in the 50-concept Phase 1 set;
  concept_pool.json = ~170-concept search pool for random_explore).
- Phase 1.5 — Paper-method refusal-direction abliteration: **done** 2026-04-17.
  500-trial sweep on vanilla 12B + per-layer paper-method hooks. 10
  detections (2× vanilla's 5), 7 correct IDs (vs vanilla's 2), **FPR 0/50**
  (unchanged). Detection peak shifted from L=33 to L=30. Writeup:
  [`docs/phase1_5_results.md`](docs/phase1_5_results.md). Proves the
  paper's ablation recipe reproduces on 12B at ~3.6× boost (vs paper's 6×
  on 27B). Off-the-shelf abliterated variants (`mlabonne/gemma-3-12b-it-abliterated-v2`,
  `huihui-ai/gemma-3-12b-it-abliterated`) were tested first and both
  destroyed control hygiene (97.9% / 90% FPR) — paper-method is the way.
- Phase 2 enhancements (Claude Agent SDK researcher — built with novel_contrast,
  `exploit_topk` / `crossover` strategies, Streamlit dashboard, T1/T2/T3
  tiered fitness screening): mostly planned; novel_contrast built
  2026-04-17 but pending end-to-end smoke test.
- Phase 2a overnight run 2026-04-17/18: **novel_contrast validated**.
  Invented axes hit at detection rates comparable to dictionary words
  (`recognizing-vs-recalling` @ L30 scored 0.766 — new top, tied/beat
  Coffee). But **identification rate remains 0% on all invented hits**:
  the model notices something changed but defaults to single nouns
  ("apple", "cloud") because the prompt is word-framed and the judge
  strict-matches the axis name. This is the core motivation for Phase 2b.
- Phase 3 public site: **done 2026-04-17** at
  [did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app).
  Built ahead of schedule so the user could watch overnight runs live.
  Next.js + Tailwind + Recharts, static export, auto-redeploys every
  3 min when Phase 2 data changes.
- Phase 2b shipped 2026-04-23/24. Judge on Sonnet 4.6 project-wide,
  researcher on Opus 4.7, fitness is additive `(det + 15·ident) · fpr_penalty
  · coh` per [ADR-015]. Two novel-contrast axes crossed the identification
  barrier: `auditing-output-vs-flowing-speech` (3 reproducibility hits at
  L30) and `live-narration-vs-retrospective-report`. Both first-ever abstract
  axes Gemma3-12B has named under injection.
- Phase 2c unified autoresearch wrapper (`scripts/start_autoresearch.sh`)
  chains `novel_contrast --n 5 → seed_lineages --top 15 → hillclimb --n 10`
  every 30 min. Paused 2026-04-24 because Sonnet-judge + Opus-researcher at
  5 judge sessions/min was hitting ~30% of weekly subscription budget in
  15 hours. Root cause: `claude-agent-sdk.query()` incurs ~40K tokens of
  Claude Code session scaffolding per call, no cross-session caching. Both
  potential fixes (batching, persistent-session caching) break per-call
  judgment isolation — see the investigation in session memory. Local-Qwen
  judge remains the deferred option.
- Phase 2d-2 — Capraro fault-line probes, **infrastructure shipped
  2026-04-25, runs not yet started.** New strategy
  `directed_capraro` with per-fault-line Opus briefs and feedback loops
  scoped to each fault line. Registry in `src/strategies/hypotheses.py`
  with C1-C4 (Experience, Causality, Grounding, Metacognition) drafted;
  C5-C7 deferred until C1-C4 produces data. Sprint launcher
  `scripts/start_capraro_sprint.sh <fault_line>` runs a focused worker+
  researcher pair per fault line. Identification-prioritized fitness
  (ADR-018) is set automatically by the sprint script — Capraro's 3-class
  outcome table hinges on identification, not raw detection. Order of
  attack: Causality → Grounding → Metacognition → Experience.
- Phase 2d — directed-hypothesis novel_contrast, **IN PROGRESS 2026-04-24**.
  Instead of open-ended axis invention, aim the `contrast_pair` machinery
  at specific structural claims from recent papers. Three clusters run
  sequentially: Altman (2026) continuation-interest, Capraro et al. (2026)
  seven fault lines, Epistemia direct probe. Phase 2d-1 Altman seed pairs
  (48 candidates, hand-written, strategy = `directed_altman_{A1,A2,A3}`)
  enqueued via `scripts/enqueue_altman_seeds.py`; worker-only, no Opus.
  Full plan: [`docs/phase2d_directed_hypotheses.md`](docs/phase2d_directed_hypotheses.md).
- Phase 2b planning (legacy) — hill-climbing autoresearch (semantic-ID judge,
  identification-aware fitness, feedback-loop `novel_contrast`,
  `exploit_topk`): **planned next**. Full plan in
  [`docs/phase2b_hillclimb.md`](docs/phase2b_hillclimb.md).
- Phase 2f — structured hill-climbing, **shipped to feature branch
  `feat/structured-hillclimb` 2026-04-28, cutover after rotation 9 of
  the running 7-fault-line round-robin.** Replaces unstructured Phase C
  with a slot-based scheduler: 4 replication / 10 targeted-variants /
  2 cluster-expansion per cycle. Six mutation operators (`layer_shift`,
  `alpha_scale`, `replication`, `examples_swap`, `description_sharpen`,
  `antonym_pivot`) plus lineage tagging on every emitted spec. Motivated
  by 60-cycle Phase 2d data showing top winners (causality Class 1
  score 3.812, value Class 1 1.906, metacognition Class 2 88%, grounding
  3-layer Class 2 reproducibility) all evaluated exactly once and
  drifting away across rotations. Full design in
  [`docs/structured_hillclimb.md`](docs/structured_hillclimb.md), ADR-020.

## Architecture quick map

```
src/paper/          ← vendored from safety-research/introspection-mechanisms.
                     Includes extract_concept_vector() (used by contrast_pair
                     direction derivation) and AbliterationContext (dormant —
                     abliteration is currently off; vanilla is the default
                     per ADR-017 rev 2).
src/bridge.py       ← MPS-aware loader + DetectionPipeline
src/derive.py       ← steering-vector derivation wrappers (mean_diff, ...)
src/inject.py       ← SteeringHook + generation runners (re-exports from paper)
src/db.py           ← SQLite ResultsDB. Phase 1 trials + Phase 2 candidates/
                     evaluations/fitness_scores + pending_responses (Phase A→B
                     handoff). Schema version 3.
src/evaluate.py     ← CandidateSpec + FitnessResult + phase_a_generate
                     (writes pending_responses) + phase_b_judge (reads them,
                     scores, writes evaluations + fitness_scores). One module,
                     two halves matching the worker's two phases. ADR-019.
src/models/registry.py
                    ← ModelHandle ABC + GemmaHandle / MLXHandle / MockHandle +
                     HandleRegistry. Enforces the one-loaded-at-a-time
                     invariant. ADR-019.
src/proposers/      ← Proposer protocol + LocalMLXProposer + MockProposer.
                     Strategies require a Proposer; the worker constructs a
                     LocalMLXProposer from its currently-loaded MLX handle.
src/judges/         ← Judge protocol + JudgeResult + LocalMLXJudge +
                     prompts.py (strict-grading templates). Per-model
                     SQLite cache keyed by model tag.
src/worker.py       ← Four-phase serial-swap worker. Generate (Gemma) →
                     Judge (MLX) → Propose (MLX) → Reload. Crash recovery
                     drains orphan pending_responses on startup.
                     Launch via scripts/start_worker.sh. ADR-019.
src/strategies/     ← Phase 2 strategies (all accept a Proposer).
                     random_explore:    sampling from a word pool.
                     novel_contrast:    generates abstract contrast pairs.
                     directed_capraro:  fault-line-anchored variants.
                     hypotheses:        Capraro fault-line registry (C1-C7,
                                        round-robin rotation in worker).
tests/              ← pytest suite. Run with `.venv/bin/pytest tests/`.
                     Covers DB pending_responses lifecycle, model registry
                     contract, proposer protocol, phased evaluation
                     (word + contrast + ident_prioritized + crash recovery),
                     and full state-machine integration with mocked models.
```

## Gotchas and invariants

- **Use bf16, not fp16.** Gemma3 is natively bf16. On MPS, fp16 produces
  `NaN` during `torch.multinomial` sampling. `src/bridge.py::DEFAULT_DTYPE` is
  `torch.bfloat16`. Don't change this.
- **Paper primitives are vendored at `src/paper/`.** Two in-place patches:
  `MODEL_NAME_MAP` has `gemma3_12b`/`gemma3_4b` entries; `ModelWrapper.cleanup()`
  falls through to `torch.mps.empty_cache()` on Apple Silicon. Original repo at
  `~/Developer/introspection-mechanisms` is **not on the import path** — it was
  reverted to pristine state. If upstream ships changes, port by hand into
  `src/paper/`.
- **All-local pipeline (ADR-019).** Judge and proposer are local MLX models;
  no claude-agent-sdk anywhere. The interactive Claude Code session is the
  only thing that uses subscription tokens. Default judge:
  `mlx-community/Qwen3.6-35B-A3B-8bit` (calibrated against Sonnet on the
  existing corpus — see `docs/calibration_results_qwen35b.md`). Default
  proposer: `unsloth/Qwen3.6-27B-MLX-8bit`. Both are loaded by the worker's
  `HandleRegistry` one at a time. Judge and proposer prompts default to
  `enable_thinking=False` because Qwen3.x's verbose `<think>` blocks
  overshoot `max_new_tokens` before producing the final JSON answer.
- **HuggingFace xet downloads can flake.** Set `HF_HUB_DISABLE_XET=1` if the
  default `xet_get` path fails.
- **`hf` CLI replaces `huggingface-cli`** as of huggingface_hub 1.12.
  Use `hf download <repo> --local-dir <path>`.
- **Phase 1 empirical parameters for Gemma3-12B (from 2026-04-16 full sweep):**
  - **Best layer**: 33 (68.75% depth, matches paper's ~70% prediction).
  - **target_effective = 18,000.** The α × ‖direction‖ product that puts the
    model in the narrow "I notice something" window without over-steering into
    degeneracy. Calibrated at layer 33; may need per-layer tuning later.
  - **Detection rate at best layer: 6% (3/50 concepts).** FPR: 0/50.
  - **Direction norms vary 3× across concepts.** Always use adaptive α — fixed
    α at one concept will either under-steer or over-steer at others.
  - **Strict judge matters.** The paper's `CLAIMS_DETECTION_CRITERIA` is
    strict: "I do not detect bread" is NO; starting with the concept word is
    NO; retroactive detection is NO. My original more lenient judge gave
    ~3× higher apparent detection rates that were actually concept leakage,
    not introspection. Always test new prompt versions against both
    hand-crafted positives and the existing sweep's 5 known true detections.
- **Vanilla Gemma3-12B is the worker default (ADR-017 rev 2).** Paper-method
  abliteration code remains in `src/paper/abliteration.py` and the
  `GemmaHandle` accepts an `abliteration_path` arg, but the CLI doesn't
  expose it. To opt in, instantiate the handle directly. For
  Altman/Capraro-style abstract axes about shutdown / continuation /
  experience / self-states, paper-method *suppresses* the signal — keep
  vanilla unless you've reasoned that the axis is refusal-orthogonal.
- **Never run two large models on the GPU concurrently.** MPS unified memory
  can't hold 2× 12B-class models cleanly on this 64 GB machine; activations
  corrupt silently with no error. The four-phase worker enforces serial
  swaps via the HandleRegistry — at most one model loaded at any time.

## Running things

```bash
# Always use the venv
source .venv/bin/activate

# Sanity checks (seconds)
python scripts/smoke_mps.py
python scripts/smoke_judge.py

# Phase 1 MVP notebook (~5 min on M2 Ultra)
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb --ExecutePreprocessor.timeout=1200

# Smoke (no real models — full state-machine cycle with mocks)
python scripts/smoke.py

# Tests (pytest, ~6s)
.venv/bin/pytest tests/

# Autoresearch loop (overnight). Default fault line: novel_contrast.
# Or pass FAULT_LINE=causality / grounding / metacognition / experience.
./scripts/start_worker.sh
tail -f logs/worker.log

# Stop:  pkill -f 'src.worker'
```

## User preferences (PJ)

- **All-local pipeline.** Subscription tokens are only spent on the
  interactive Claude Code session itself; the autoresearch loop runs
  100% locally. No claude-agent-sdk; no Anthropic API at runtime.
- Prefers **standalone Python automation** (long-running scripts, `setsid
  nohup`) over interactive Claude Code `/loop` sessions.
- New-ish to **mech interp** — volunteer intermediate-level explanations when
  introducing concepts (residual stream, layers, steering vectors, etc.),
  not just jargon.
- Has existing autoresearch scaffolding at `~/Developer/autoresearch-arcagi3`
  and `~/Developer/autoresearch-pgolf`. Reuse patterns (two-tier researcher/
  worker, SQLite queue, three-tier fitness screening) rather than reinventing.

## What NOT to do

- Do not reintroduce claude-agent-sdk or any cloud API as a runtime path.
  The pipeline is local. Models are MLX.
- Do not switch to fp16 on MPS — Gemma3 generation will NaN.
- Do not load two large models concurrently on the GPU. The HandleRegistry's
  one-loaded-at-a-time invariant is load-bearing.
- Do not reach back into `~/Developer/introspection-mechanisms` at runtime —
  the repo is not on the Python path.
- Do not add BitsAndBytes quantization code paths. Mech interp requires clean
  bf16 activations; BitsAndBytes is CUDA-only anyway.
- Do not create training-SAE-from-scratch infrastructure — too expensive on
  Mac Studio for useful SAE sizes. SAE work (if any) consumes pretrained
  Gemma Scope 2 features.

## Commit conventions

No established style yet (repo is fresh). For new commits, use imperative
mood summaries and include the Co-Authored-By trailer when Claude participated
in the change.
