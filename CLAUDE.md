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

## Architecture quick map

```
src/paper/          ← vendored from safety-research/introspection-mechanisms
src/bridge.py       ← MPS-aware loader + DetectionPipeline
src/derive.py       ← steering-vector derivation wrappers (mean_diff, ...)
src/inject.py       ← SteeringHook + generation runners (re-exports from paper)
src/judges/         ← Claude judge via claude-agent-sdk subscription OAuth
                     (PROMPT_TEMPLATE_VERSION in claude_judge.py — bump to
                      invalidate the sqlite cache when prompt changes)
src/db.py           ← SQLite ResultsDB. Phase 1 trials + Phase 2 candidates/
                     evaluations/fitness_scores. Schema version 2.
src/sweep.py        ← Resumable Phase 1 sweep runner.
                     Adaptive α via target_effective (α = target / ‖dir‖).
                     Supports --abliterate-paper <directions.pt>: pre-derives
                     concept vectors from vanilla, then installs per-layer
                     paper-method hooks for injection. See ADR-014.
src/paper/abliteration.py
                    ← Paper-method refusal-direction ablation. Per-layer
                     projection-out hooks with Optuna-tuned per-region
                     weights proportionally remapped 27B → 12B. Includes
                     compute_per_layer_refusal_directions() for one-shot
                     extraction to data/refusal_directions_12b.pt.
src/paper/refusal_prompts.py
                    ← Vendored 520 harmful + 31811 harmless prompts for
                     refusal direction derivation.
src/verify_phase1.py ← Phase 1 acceptance check.
src/evaluate.py     ← Phase 2 candidate fitness (3-component multiplicative).
                     Legacy single-call eval: derives direction, runs probes,
                     judges inline. Used by the legacy worker.py.
src/evaluate_phased.py
                    ← Same fitness math split into phase_a_generate (writes
                     pending_responses) and phase_b_judge (reads them, scores,
                     writes evaluations + fitness_scores). Used by worker_v2.
src/models/registry.py
                    ← ModelHandle ABC + GemmaHandle / MLXHandle / MockHandle.
                     The HandleRegistry enforces the one-loaded-at-a-time
                     invariant. ADR-019.
src/proposers/      ← Proposer protocol + ClaudeProposer + LocalMLXProposer
                     + MockProposer. Strategies accept a `proposer=` arg
                     (default ClaudeProposer for backwards compat).
src/judges/local_mlx_judge.py
                    ← MLX-backed strict-grading judge. Same prompt templates
                     as claude_judge.py so verdicts are directly comparable.
                     Per-model SQLite cache keyed by model tag.
src/worker.py       ← LEGACY Phase 2 worker. Loads model once, judges inline
                     via Claude. Still works for the old cloud pipeline.
src/worker_v2.py    ← Four-phase serial-swap worker. Generate (Gemma) →
                     Judge (MLX) → Propose (MLX) → Reload. Crash-recovery
                     drains orphan pending_responses on startup.
                     Launch via scripts/start_worker_v2.sh. ADR-019.
src/researcher.py   ← LEGACY short-lived candidate generator driver. Replaced
                     by worker_v2's Phase C; kept for old launcher scripts.
src/strategies/     ← Phase 2 strategies.
                     random_explore: samples concept/layer/eff from a word pool
                     novel_contrast: generates abstract contrast pairs via a
                       Proposer (defaults to ClaudeProposer Opus 4.7;
                       worker_v2 passes LocalMLXProposer instead).
                     directed_capraro: same pattern, fault-line-anchored.
                     Future: exploit_topk, crossover.
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
- **Claude judge uses Claude Code subscription OAuth.** Never add
  `ANTHROPIC_API_KEY` requirements. The `claude-agent-sdk` inherits auth from
  the local Claude Code install. **Default judge model is Sonnet 4.6
  (`claude-sonnet-4-6`) project-wide** (both Phase 1 and Phase 2, as of
  2026-04-23). Haiku 4.5 remains available via `--judge-model
  claude-haiku-4-5-20251001` as a cheaper throwaway option, but Phase 2b's
  feedback-driven hill-climb amplifies any judge bias, so Sonnet's
  stricter semantic-gist grading is the standing default. The SQLite
  judge cache keys by model name, so old Haiku entries sit in a separate
  namespace and never collide with Sonnet entries.
- **Researcher (novel_contrast / exploit_topk / hillclimb) uses Opus 4.7
  (`claude-opus-4-7`).** Single constant at
  `src/strategies/novel_contrast.py::CLAUDE_MODEL` imported by the other
  two strategies. Researcher token volume is small (~150K/day) so Opus's
  heavier subscription weighting is negligible; the creativity-gated
  abstract-axis invention task benefits from the smartest available
  proposer.
- **Judge is blocking-sync from Jupyter.** `ClaudeJudge._run_sync` spawns a
  worker thread because Jupyter already runs an asyncio loop and
  `asyncio.run()` would raise "Already running asyncio in this thread".
- **HuggingFace xet downloads can flake.** Set `HF_HUB_DISABLE_XET=1` if the
  default `xet_get` path fails.
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
- **Abliterated sweeps: derive concept vectors from vanilla BEFORE installing
  hooks, then inject under hooks.** Deriving under active abliteration
  projects the refusal-aligned component out of the concept vector, leaving
  noise that adaptive-α amplifies into token salad. `src/sweep.py` does this
  automatically; `src/worker.py` honors this via
  `pipeline.abliteration_ctx.suspended()` around the derive call in
  `src/evaluate.py` (shipped 2026-04-24, ADR-017). See ADR-014 and ADR-017.
- **Vanilla Gemma3-12B is the Phase 2 worker default (ADR-017 rev 2,
  2026-04-25).** `./scripts/start_worker.sh` runs raw, no hooks. Paper-method
  abliteration is OPT-IN via `ABLITERATED=1 ./scripts/start_worker.sh` (or
  `--abliterate-paper` flag). **Use paper-method only when you've reasoned
  about whether the axis you're testing is refusal-orthogonal.** For
  dictionary-word directions (Phase 1.5 territory), paper-method delivers
  ~3.6× detection / ~3.5× identification boost. For Altman/Capraro/Epistemia-
  style abstract axes about shutdown / continuation / experience / self-
  states, paper-method *suppresses* the signal because the signal lives
  inside the refusal subspace itself (Phase 2d-1 finding, 2026-04-24).
  When in doubt: vanilla. `spec_hash` includes `abliteration_mode` so
  vanilla and paper-method evaluations of the same direction coexist as
  distinct DB rows. UI renders an amber "abliterated" badge vs a neutral
  "vanilla" badge on each leaderboard card. The legacy `ABLITERATED=1`
  path (loaded deprecated off-the-shelf HF checkpoints) is gone — the env
  flag now correctly maps to paper-method. See ADR-017.
- **Never run two sweeps concurrently on the same DB.** MPS unified memory
  can't hold 2× 12B (44GB+ of just weights on a 64GB machine) and activations
  start corrupting silently. Both processes produce token salad with no
  error. Convention: one sweep at a time per DB path.
- **Paper-method abliteration is the only abliteration we use.** Off-the-shelf
  HF abliterated checkpoints (`mlabonne/gemma-3-12b-it-abliterated-v2`,
  `huihui-ai/gemma-3-12b-it-abliterated`) produced catastrophic FPR on our
  protocol (97.9% and 90% respectively). The paper's Optuna-tuned per-region
  weights are the surgical version; proportional remap from 62-layer 27B to
  48-layer 12B via `paper_layer_weights_for_model()`. See ADR-013.
- **Claude judge has 60s timeout + 3 retries with exponential backoff.** The
  Anthropic API can enter CLOSE_WAIT state and hang indefinitely. Without the
  timeout, a single hang kills whole sweeps (caused an 8h stall mid-session
  before c63c295 fix). Default sentinel `JudgeResult` on final failure so
  the sweep keeps going.

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

# Phase 1 full sweep (~2 hours)
python scripts/run_phase1_sweep.py
python -m src.verify_phase1
python scripts/export_phase1.py

# Phase 1.5 paper-method abliteration
python scripts/compute_refusal_direction.py                              # one-shot, ~3-5 min
python scripts/diagnose_paper_weights.py                                 # sanity check
python scripts/run_phase1_sweep.py --abliterate-paper data/refusal_directions_12b.pt  # ~2.5 hours
python scripts/compare_abliterations.py                                  # vanilla vs mlabonne vs huihui vs paper-method

# Phase 2 autoresearch loop (overnight)
./scripts/start_worker.sh          # background, one per machine
./scripts/start_researcher.sh      # background, 30-min cycle
tail -f logs/worker.log logs/researcher.log

# Phase 2 manual / test invocation
python -m src.researcher --strategy random --n 5 --dry-run
python -m src.researcher --strategy random --n 5
python -m src.worker --max-candidates 5
```

## User preferences (PJ)

- Uses **Claude Code subscription OAuth**, not API billing. Anything that calls
  Claude must go through `claude-agent-sdk` or a subscription-aware path.
- Wants the **newest Claude models** — `claude-opus-4-7`, `claude-sonnet-4-6`,
  `claude-haiku-4-5-20251001`. No GPT-4o or older Claude 3.x.
- Prefers **standalone Python automation** (long-running scripts, `setsid
  nohup`) over interactive Claude Code `/loop` sessions. Claude Agent SDK is
  fine for the "agent does research" layer.
- New-ish to **mech interp** — volunteer intermediate-level explanations when
  introducing concepts (residual stream, layers, steering vectors, etc.),
  not just jargon.
- Has existing autoresearch scaffolding at `~/Developer/autoresearch-arcagi3`
  and `~/Developer/autoresearch-pgolf`. Reuse patterns (two-tier researcher/
  worker, SQLite queue, three-tier fitness screening) rather than reinventing.

## What NOT to do

- Do not add `ANTHROPIC_API_KEY` as a hard requirement for the judge.
- Do not switch to fp16 on MPS — Gemma3 generation will NaN.
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
