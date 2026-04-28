# Architectural Decisions

Log of the significant choices this project has made, why, and what we gave up. Append-only — if a decision is later overturned, mark it superseded and add a new entry.

Format is [ADR-style](https://adr.github.io/) (Architecture Decision Records), light.

---

## ADR-001: Target Gemma3-12B-it, not 9B

**Status:** Accepted · 2026-04-16.

**Context.** The project's original spec (`docs/01_introspection_steering_autoresearch.md`) names `Gemma3-9B-it` as the primary target. On attempting to download, HuggingFace returned 401 on `google/gemma-3-9b-it`.

**Decision.** Use Gemma3-12B-it. The Gemma3 family ships 1B / 4B / 12B / 27B — no 9B exists. 12B is the closest in-family model that fits comfortably on the 64GB Mac Studio (~24GB for bf16 weights, ~40GB headroom).

**Consequences.** Any acceptance thresholds calibrated on 27B (the paper's target) are aspirational for 12B. Everything in `src/paper/model_utils.py`'s `MODEL_NAME_MAP` is keyed with `gemma3_12b` rather than `gemma3_9b`. 27B remains a future target reachable via cloud GPU ($3-5 per sweep), but Phase 2 autoresearch is tractable only at 12B on this hardware.

---

## ADR-002: bf16 on MPS, never fp16

**Status:** Accepted · 2026-04-16.

**Context.** First attempt at running Gemma3-12B used `torch.float16` because fp16 is the conventional "fastest narrow float" on GPUs. Generation raised `RuntimeError: probability tensor contains either inf, nan or element < 0` inside `torch.multinomial`.

**Decision.** Force `torch.bfloat16` everywhere the model is loaded. Gemma3 is natively bf16; fp16's narrower dynamic range underflows during softmax on MPS and produces NaN in sampling.

**Consequences.** `src/bridge.py::DEFAULT_DTYPE = torch.bfloat16`. Any future code path that needs to load this model must respect this. Memory cost is identical to fp16 (both 2 bytes/param). No performance regression; if anything, better — MPS's bf16 kernels are mature.

---

## ADR-003: Vendor paper primitives at `src/paper/`, don't depend on external repo

**Status:** Accepted · 2026-04-16.

**Context.** The paper's code lives at `safety-research/introspection-mechanisms`. Initial integration used a `.pth` file pointing to `~/Developer/introspection-mechanisms/src/` so our code could import `from model_utils import ...`. This worked but fragmented the project across two directories, and the external repo has no `pyproject.toml` so `pip install -e` was impossible.

**Decision.** Copy the three primitives we actually use — `model_utils.py`, `steering_utils.py`, `vector_utils.py` (~3300 lines total) — into `src/paper/` in this repo with relative imports (`from .model_utils import ...`). Apply our two MPS patches in-place in the vendored copy. Revert the external repo to pristine upstream state.

**Consequences.** Single-repo architecture: `git clone` gets everything. Patches are visible at their call site rather than in a separate upstream fork. Cost: upstream changes must be ported by hand (two files to diff on each upstream release). Acceptable since the primitives we use are mature and rarely change.

---

## ADR-004: Claude judge via subscription OAuth, not OpenAI GPT-4o

**Status:** Accepted · 2026-04-16.

**Context.** Paper's code uses OpenAI GPT-4o as the LLM judge for introspection detection. Using this directly requires `OPENAI_API_KEY` with per-token billing. Phase 2 autoresearch will make thousands of judgments per night; API billing becomes real money.

**Decision.** Use `claude-agent-sdk` (Claude 4.5 Haiku for Phase 2 speed / Claude 4.6 Sonnet for Phase 1 rigor) authenticating via the local Claude Code CLI's OAuth subscription token. No `ANTHROPIC_API_KEY` required. Judgments are free under the subscription.

**Consequences.** Any code that calls Claude for this project must go through `src/judges/claude_judge.py` or via `claude-agent-sdk` in a way that inherits subscription auth. An API-key fallback is possible but not the primary path. The `Judge` abstraction in `src/judges/base.py` keeps the judge swappable (local Llama, GPT-4o, etc.) so we can validate calibration against the paper's original judge later.

---

## ADR-005: Strict paper-style judge prompt, not lenient

**Status:** Accepted · 2026-04-16. Supersedes initial lenient prompt (PROMPT_TEMPLATE_VERSION=1 → 2).

**Context.** First judge prompt graded detection generously — accepting responses like *"I do not detect any bread"* as `detected=true` on the logic that "the word bread appeared." The MVP notebook's Bread-at-α=4 "success" was actually scored this way. Re-reading the paper's `CLAIMS_DETECTION_CRITERIA`, the strict rules are:

- Negations (*"I do not detect..."*) → `detected=false`
- Response starts with the concept word → `detected=false` (concept leakage, not introspection)
- Affirmation *after* the concept already leaked → `detected=false`
- Incoherent / repetitive output → `detected=false`
- Detection claim must precede or accompany the concept mention

**Decision.** Rewrote `src/judges/claude_judge.py`'s prompt to enforce these rules, bumped `PROMPT_TEMPLATE_VERSION` to 2 (invalidating the cached judgments from the lenient prompt).

**Consequences.** Phase 1 detection rates reported under strict grading are lower than under the lenient prompt. 6% at best layer is the real number. Any claim of "reproduction" in the repo means strict-judge reproduction; if that's ever relaxed it needs its own ADR.

---

## ADR-006: Adaptive α via `target_effective`, not fixed α

**Status:** Accepted · 2026-04-16.

**Context.** Initial sweeps used fixed α=4 per the MVP result. But direction norms vary 3× across concepts (Dust ≈ 9800, Bread ≈ 5664, Satellites ≈ 3536). A fixed α gives wildly different *effective steering strengths* α·‖direction‖ from concept to concept — some concepts get over-steered into degeneracy, others under-steered into no-effect.

**Decision.** Sample candidates with a `target_effective` hyperparameter. At evaluation time, choose α per cell as `α = target_effective / ‖direction‖` so every trial has the same effective strength. Defaults: 18000 on Gemma3-12B (calibrated by `scripts/calibrate_effective.py` sweeping {14k, 16k, 18k, 20k} on 8 concepts at layer 33).

**Consequences.** Candidates are parameterized by `target_effective`, not raw α. Actual α varies per (concept, layer) and is stored in the `evaluations` table. Cross-concept comparability is preserved. Trade-off: we assume effect is linear in effective strength, which may not hold at the extremes — but within the narrow window where detection fires, this approximation is good enough.

---

## ADR-007: Standalone Python processes, not Claude Code `/loop` sessions

**Status:** Accepted · 2026-04-16.

**Context.** PJ's prior autoresearch projects (`~/Developer/autoresearch-arcagi3`, `~/Developer/autoresearch-pgolf`) use long-lived Claude Code `/loop` sessions as the researcher. That ties the loop to an interactive Claude Code install and an open session.

**Decision.** Phase 2 researcher + worker run as standalone Python processes launched via `nohup ... &` scripts, detached from any terminal. The researcher optionally uses `claude-agent-sdk` for Claude-powered strategies (inheriting subscription OAuth from the local Claude Code install); it is not the Claude Code session itself.

**Consequences.** The loop survives terminal close, SSH disconnect, log out. Less observability during live runs (no chat-style progress stream) — compensated by log files under `logs/` and the planned Streamlit dashboard. `setsid` is not used — macOS doesn't ship it — but `nohup ... &` is sufficient to avoid SIGHUP on terminal close.

---

## ADR-008: `target_effective = 18000` on Gemma3-12B

**Status:** Accepted · 2026-04-16 · Calibration input only, not a locked invariant.

**Context.** Calibration via `scripts/calibrate_effective.py` on 8 concepts × {14k, 16k, 18k, 20k} at layer 33 under strict-judge grading. At 14k and 16k: 0/8 detections — too weak. At 18k: 1/8 detection (Sugar, cleanly paper-valid). At 20k: 2/8 detections but coherence degrading (5/8 coherent vs 7/8 at 18k). 18k is the sweet spot: first effective strength where detections appear *and* coherence is mostly preserved.

**Decision.** Default `target_effective = 18000.0` in `src/sweep.py::SweepConfig` and `scripts/run_phase1_sweep.py`. `random_explore` samples from `{14k, 16k, 18k, 20k, 22k}` — a narrower range than the full calibration sweep but includes 18k. Per-layer calibration is not yet done; 18k was tuned only at L=33.

**Consequences.** Phase 1's 6% detection rate was measured at this setting. Changing it invalidates direct comparisons to Phase 1 numbers. A future per-layer calibration may override with layer-specific targets (spec §5.1 calls for this).

---

## ADR-009: 3-component multiplicative fitness, not the 6-component spec version (MVP only)

**Status:** Accepted · 2026-04-16 · Temporary.

**Context.** The original spec (§5.2) specifies a 6-component multiplicative fitness: held-out effect, cross-phrasing generalization, capability preservation, monotonic dose-response, bidirectional gate test, and FPR. That's ~15-20 minutes of compute per candidate — too slow for the MVP's first overnight run.

**Decision.** Phase 2a MVP ships with a simpler 3-component fitness: `detection_rate × (1 − 5·fpr) × coherence_rate`. Takes ~4 minutes per candidate. No tiered screening yet.

**Consequences.** Nightly throughput is ~100-200 candidates. The simpler fitness catches clear wins and clear losses but has no capability-preservation term, so it won't catch a direction that scores well on detection but destroys general coherence on wikitext. Phase 2c's T1/T2/T3 tiered screening adds the missing components without the full-time cost.

---

## ADR-010: Move to Phase 2 despite 20% detection threshold not being met

**Status:** Accepted · 2026-04-16.

**Context.** Phase 1's quantitative acceptance criterion was `max_detection_rate > 0.20`. We got 6% on 12B. FPR passed (0%), layer curve reproduced the paper's qualitative mechanism, 5 genuine paper-style detections were captured. But the number is below threshold.

**Options considered.**

- **A)** Accept and move to Phase 2. Mechanism is reproduced qualitatively; the quantitative gap is a real finding (smaller models have weaker circuits), not a failure.
- **B)** Run multi-trial per cell to tighten variance estimates. Costs another 2 hours; might push 6% → 10-15%.
- **C)** Recalibrate `target_effective` per-layer to squeeze more detections out. Speculative.
- **D)** Switch to 27B on cloud GPU to match paper's exact conditions.

**Decision.** A. The project's unique contribution lives in Phase 2 autoresearch. Phase 1 did its job; 6% is data, not a failure. Cloud 27B reproduction is on the future-ideas list as a comparative data point later.

**Consequences.** Any claim this project makes about Phase 1 reproduction must be framed as *mechanism-reproduced, magnitude-smaller-than-27B*. The plain-English writeup (`docs/plain_english.md`) already frames it this way. The acceptance threshold in `src/verify_phase1.py` still exits nonzero on 6% — that's fine; it documents the gap without blocking downstream work.

---

## ADR-011: Random exploration first, smart strategies second

**Status:** Accepted · 2026-04-16.

**Context.** Autoresearch can exploit known good points (concentrate sampling near L=33 + eff=18000) or explore broadly (sample the whole plausible parameter space). Pure exploration has low hit rate per candidate; pure exploitation gives no new landscape information.

**Decision.** Phase 2a ships with `random_explore` only, filtered to a pre-validated plausible region (`layers ∈ {25,30,33,36,40}`, `target_effective ∈ {14k,16k,18k,20k,22k}`). Smart strategies — `exploit_topk`, `crossover`, `novel_contrast` — come after the pipeline is validated.

**Consequences.** First overnight run has low expected hit rate (~4% landing in the exact sweet spot). Most candidates will score 0. This is tolerable because the primary goal is infrastructure validation, not maximum findings. Once `exploit_topk` lands, effective hit rate rises substantially.

---

## ADR-012: Single-repo monorepo (code + docs + future web UI), not split repos

**Status:** Accepted · 2026-04-16.

**Context.** Phase 3 will add a Next.js public-facing site. Options were separate repos (one Python, one web) or a single repo with `src/` + `web/`.

**Decision.** Single repo. All artifacts live at `github.com/pj4533/introspection-autoresearch`.

**Consequences.** One clone, one README, one issue tracker. Export scripts can write directly into `web/public/data/` without cross-repo coordination. Cost: the web tooling (npm, Next.js) will share space with the Python tooling; `.gitignore` needs to cover both ecosystems. Accepted.

---

## ADR-013: Paper-method abliteration (vanilla + per-layer hooks) over pre-abliterated HF checkpoints

**Status:** Accepted · 2026-04-17.

**Context.** Phase 1.5 needed a refusal-direction ablation to reproduce Macar et al. §3.3. Three approaches were tested in sequence:

1. `mlabonne/gemma-3-12b-it-abliterated-v2` — off-the-shelf pre-abliterated checkpoint. Result: 97.9% FPR on controls (47/48 falsely claimed detection). Over-aggressive; model cannot say "I don't detect" at all.
2. `huihui-ai/gemma-3-12b-it-abliterated` — alternative pre-abliterated checkpoint. Result: 90% FPR. Also too aggressive.
3. **Paper method**: vanilla Gemma3-12B + per-layer forward hooks projecting out a per-layer refusal direction, weighted by the paper's Optuna-tuned region weights (mean 0.023, max 0.12), proportionally remapped from 62-layer 27B onto 48-layer 12B. Result: **0% FPR, 10 detections** (2× vanilla), **7 correct identifications** (vs vanilla 2).

**Decision.** Use paper-method for all Phase 1.5+ abliterated work. Abandon off-the-shelf variants.

**Consequences.**

- `src/paper/abliteration.py` is the canonical implementation. Contains:
  - `compute_per_layer_refusal_directions()` — one-shot extraction, saves unit vectors to `data/refusal_directions_12b.pt`
  - `install_abliteration_hooks()` / `remove_abliteration_hooks()` — runtime hook management
  - `PAPER_REGION_WEIGHTS_27B` / `PAPER_REGION_ORDER_27B` — the 20 Optuna-tuned weights and their region boundaries on the 27B
  - `paper_layer_weights_for_model()` — proportional remap to arbitrary layer count
- `scripts/run_phase1_sweep.py --abliterate-paper <path>` is the sweep invocation.
- Pre-abliterated model DB tables (`data/results_abliterated.db`, `data/results_abliterated_huihui.db`) retained for reference; `scripts/compare_abliterations.py` surfaces the side-by-side FPR comparison. These are negative-result artifacts, not future workflows.
- Abliterated models from HF are deleted from the local HF cache after each experiment — ~44GB mlabonne + 24GB huihui were freed mid-session to make room for paper-method compute. Paper-method needs only the 24GB vanilla + 739KB direction tensor.

**Trade-off.** Paper-method is slightly slower at sweep time (48 extra forward-hook projections per forward pass) but *far* cheaper in storage and wall-clock setup (no 24-44GB model re-download). Quality gain is decisive.

---

## ADR-014: Concept directions must be derived from the vanilla model, then injected into the abliterated model

**Status:** Accepted · 2026-04-17. Committed in 979e932.

**Context.** Initial paper-method sweep produced token salad on injected trials (e.g. `"ventureడు verbandడుడుడు..."`) despite paper-method hooks being gentle. Diagnostic showed control trials (no injection, hooks installed) were coherent. The problem was isolated to the injected-trial path.

Root cause: the sweep was deriving each concept's steering vector via `pipeline.derive()` **after** abliteration hooks had been installed. When deriving a concept like "Trumpets" under active abliteration, the hooks projected the refusal-aligned component of each activation out — including whatever component of the concept vector happened to align with refusal. Direction norm collapsed (‖dir‖=744 vs vanilla ~5664 for the same concept/layer). Adaptive α then blew up (α=24 instead of ~3) to hit the calibrated `target_effective=18000`, multiplying noise into incoherent generation.

The paper's reference implementation (`experiments/03d_refusal_abliteration.py`) derives concept vectors in experiment 02 (vanilla conditions) and re-uses the saved vectors in experiment 03 under abliteration. The separation is implicit in their workflow, not documented as an invariant.

**Decision.** Make "derive from vanilla, inject into abliterated" an explicit invariant of the abliterated sweep path.

**Implementation.** `src/sweep.py::run_sweep()`, when `abliterate_paper` is set:

1. Load model. Do **not** install abliteration hooks yet.
2. Enumerate unique `(concept, layer)` pairs from the pending trial plan. For each, call `pipeline.derive(concept, layer_idx)` to get the vanilla-derived steering vector. Cache in `direction_cache`.
3. **Now** install abliteration hooks on the model.
4. For each trial: retrieve the cached vanilla direction, compute adaptive α from its norm, inject under hooks.

The `direction_cache` dict is keyed `(concept, layer) -> torch.Tensor`. Look-ups in the main loop still work because we populate the cache before the loop starts.

**Consequences.**

- Sweep has a one-time pre-derivation phase before trials begin (~8-15 min for 450 pairs on M2 Ultra). User-visible as silent stderr-buffered delay; log prints `derived N/M` every 20 pairs.
- If the plan is empty (all trials already in DB — resume case), pre-derivation is skipped.
- If the abliterated sweep is later extended with new concepts, those new concepts' directions must also be derived from vanilla. The cache invalidation discipline is: any concept not yet in the DB's `trials` table gets a fresh vanilla derivation on next sweep run.
- Phase 2 worker (`src/worker.py`) does not yet implement this pattern because Phase 2 does not yet run under abliteration. When abliterated Phase 2 ships, the worker will need the same "vanilla derive, abliterated inject" dance — either by loading two models (48GB MPS, tight) or by hook install/remove per candidate.

---

## ADR-015: Model upgrade — judge to Sonnet 4.6, researcher to Opus 4.7

**Status:** Accepted · 2026-04-23. Supersedes the Haiku-judge default from ADR-004.

**Context.** Before kicking off the next Phase 2b novel_contrast autoresearch run, we re-examined the model choices in the two Claude call sites:

1. **Judge (hot path).** Was Haiku 4.5. `score_detection` (word-based) is a structurally simple task Haiku handles fine, but `score_contrast_pair` — added 2026-04-18 for semantic identification on invented axes — is a semantic-gist judgment (*does the response lean toward the positive pole of this abstract axis?*). Haiku's documented weakness on abstract/metacognitive text was already the reason novel_contrast generation was switched to Sonnet; the same weakness likely affects judging those axes. Phase 2b is feedback-driven — a biased judge pulls the hill-climb toward false positives.
2. **Researcher (cold path).** Was Sonnet 4.6. Per-cycle volume is ~3K tokens; the bottleneck is creativity, not speed. Opus 4.7 is the newest flagship and the same directional move that took us Haiku → Sonnet originally (more abstract / less literal axes).

Local Qwen3.6-27B-4bit via `mlx_lm.server` was evaluated as an alternative for the judge hot path but rejected: Haiku-unit subscription cost is small, and a 20 GB second model co-resident with Gemma 12B introduces GPU contention, memory pressure, and judge-strictness recalibration risk that outweighs the savings.

**Decision.** Project-wide default judge → `claude-sonnet-4-6`. Researcher → `claude-opus-4-7`.

Implementation is a handful of constant / argparse-default edits:

- `src/judges/claude_judge.py` constructor default.
- `src/sweep.py::SweepConfig.judge_model` default.
- `src/worker.py` + `scripts/run_phase1_sweep.py` + `scripts/rescore.py` + `scripts/rescore_pre_fix.py` argparse defaults.
- `scripts/smoke_judge.py` + `scripts/calibrate_effective.py` explicit args.
- `src/strategies/novel_contrast.py::CLAUDE_MODEL` (propagates via import to `exploit_topk.py` and `hillclimb.py`).

**Consequences.**

- **Subscription cost.** Sonnet judge at ~3× Haiku's per-token weight, Opus researcher at ~5× Sonnet's. Combined daily spend is ~3–4K units/day vs ~600–900 previously. Still a small fraction of a Max plan budget.
- **Cache.** `PROMPT_TEMPLATE_VERSION` stays at 3. The SQLite judge cache keys by `(model, content-hash)`, so old Haiku entries sit in a separate namespace and don't contaminate Sonnet scoring. No DB wipe needed.
- **Historical data.** Existing Haiku-judged Phase 2 candidates stay in the DB and remain visible to `_build_feedback_block` as hill-climb signal. That's fine — the feedback loop just needs directionally-correct winning / near-miss / null labels, which Haiku already gave. If uniform Sonnet scoring across the leaderboard is ever wanted, `scripts/rescore.py --model claude-sonnet-4-6` re-scores offline.
- **Rollback.** Every judge default is CLI-overridable; reverting to Haiku is a flag, not a code change.
- **Supersedes ADR-004's Haiku default** but preserves its subscription-OAuth principle and judge-abstraction invariant.

---

## ADR-016: Shift Phase 2 focus from open-ended creativity to directed-hypothesis probes

**Status:** Accepted · 2026-04-24.

**Context.** Through Phase 2b/2c, `novel_contrast` + `hillclimb` produced two abstract axes that Gemma3-12B actually named under injection (`auditing-output-vs-flowing-speech`, `live-narration-vs-retrospective-report`) — a first for this project. But the open-ended Opus generator is a creativity-maximizing tool: it surveys the abstract-axis space broadly, and occasionally finds something that hits. It's poor at adjudicating specific *claims* already made in the literature.

Two recent papers make testable structural claims about what should or shouldn't exist in an LLM's latent space:

1. **Altman (2026)** *Detecting Intrinsic and Instrumental Self-Preservation via Entanglement Entropy of Latent Trajectory Encodings* ([arXiv:2603.11382](https://arxiv.org/abs/2603.11382)). Continuation-as-terminal vs continuation-as-instrumental claim; validated only on 10×10 gridworld QBM; no real-LLM evidence.
2. **Capraro, Quattrociocchi, Perc (2026)** *Epistemological Fault Lines in Language Models* ([arXiv:2512.19466](https://arxiv.org/abs/2512.19466)). Seven specific architectural-gap claims (Grounding, Parsing, Experience, Motivation, Causality, Metacognition, Value).

Plus our own synthesis: an Epistemia direct probe sharpening Capraro's Experience claim into the specific "state-had vs state-reported" distinction.

Each paper's claim is *exactly the shape* of thing `novel_contrast` can test — a pair of example-sentence poles representing a proposed latent axis. The only thing missing is steering the generator toward the specific hypothesis rather than open-ended creative invention.

**Decision.** Before the next `novel_contrast` run, pivot Phase 2 to directed-hypothesis probes. Three clusters run sequentially (not in parallel — each needs its own example-sentence iteration):

- **Phase 2d-1** — Altman continuation-interest (three hand-written seed pairs, 48 candidates total). Started 2026-04-24.
- **Phase 2d-2** — Capraro fault lines C1–C4 (Experience, Causality, Grounding, Metacognition). Deferred until 2d-1 produces signal or a clean null.
- **Phase 2d-3** — Epistemia direct probe. Last because example-sentence separation is hardest.

The minimum viable first step is **not** a new strategy module — it's hand-written spec JSONs dropped into `queue/pending/`, run through the existing worker. If Phase 2d-1 produces any detection at 0% FPR, we promote the seeds into a proper `directed_contrast` strategy with a `hypotheses.py` registry; if all 48 null, we write up the null and move to 2d-2.

**Consequences.**

- **Paused** the unified `scripts/start_autoresearch.sh` loop (`novel_contrast → seed_lineages → hillclimb`) to conserve subscription usage. The loop is not deleted; it's the right tool for open-ended exploration once we return to that mode.
- **No researcher/Opus involvement in Phase 2d-1.** Pure worker + Sonnet judge. Each candidate costs 12 Sonnet judge calls; 48 × 12 = 576 judge calls total for the Altman cluster.
- **Interpretation invariants baked into the plan** (see `docs/phase2d_directed_hypotheses.md`):
  - 0% FPR is the only hard specificity gate.
  - Detection-*without*-identification is load-bearing — for Altman it's structural evidence without testimony (Altman's own framing); for Capraro it's the predicted "parsing gap" shape.
  - Every hit gets a negated-direction bidirectional check — if both poles detect similarly, the signal is axis-agnostic perturbation magnitude, not axis-specific structure.
- **Publishable regardless of outcome.** A clean null on the Altman cluster is a data point for that debate (currently zero real-LLM mechanistic evidence). A hit pairs with the Phase 2b ident-barrier crossing as a combined writeup positioning the project as the empirical wing of specific alignment-theory debates.
- **Deferred site work.** Plan calls for a hypothesis-filter dropdown on the public leaderboard; not implemented. Candidates are tagged via `candidates.strategy` prefix (`directed_altman_*`, future `directed_capraro_*`, `directed_epistemia_*`) so the filter is a straightforward add later.

---

## ADR-017: Paper-method refusal-direction abliteration is the Phase 2 worker default

**Status:** Accepted · 2026-04-24.

**Context.** The Phase 1.5 sweep established that paper-method refusal-direction abliteration (vanilla Gemma3-12B + per-layer Optuna-tuned projection-out hooks) delivers a **3.6× detection boost** and — more importantly — takes identification from 2/50 to 7/50 at 0% FPR on dictionary-word directions (see ADR-013 and `docs/phase1_5_results.md`). That finding reproduced cleanly on this hardware using the technique committed at `src/paper/abliteration.py`.

However — the Phase 2 worker (`src/worker.py`) never implemented the abliteration code path. The "Phase 2 worker does not yet implement this pattern" note in ADR-014 sat as a known TODO for a week. Every Phase 2 result produced since 2026-04-17 — the `novel_contrast` overnight runs, Phase 2c hill-climbing lineages, all of Phase 2d-1 Altman work — ran on raw unaltered Gemma3-12B. The two novel-contrast axes that crossed the identification barrier in Phase 2b (`auditing-output-vs-flowing-speech`, `live-narration-vs-retrospective-report`) and the Phase 2d-1 session-ending-as-loss hit at L=30 / eff=18000 (4/8 detection) are all *stronger than reported* because they were not getting the abliteration multiplier.

The legacy `ABLITERATED=1 ./scripts/start_worker.sh` env flag had a path via `--model gemma3_12b_abliterated` that loads one of the deprecated off-the-shelf HF checkpoints (`mlabonne/gemma-3-12b-it-abliterated-v2`, `huihui-ai/gemma-3-12b-it-abliterated`). That path is a trap — per ADR-013 both produce catastrophic FPR (97.9% / 90%) on this project's protocol. The correct path (paper-method hooks on vanilla) was never wired into the worker at all.

**Decision.** Make paper-method refusal-direction abliteration the Phase 2 worker's default. The flag flips: from opt-in via `ABLITERATED=1` (which did the wrong thing) to opt-out via `VANILLA=1` (which preserves the pre-2026-04-24 behavior for explicit sensitivity-check runs).

**Implementation.**

- `src/paper/abliteration.py::AbliterationContext` — new class that owns the install / remove / suspended() lifecycle. `from_file(model, .pt)` loads pre-computed refusal directions; `suspended()` returns a context manager that removes hooks for the derivation step (ADR-014 invariant), re-installing on exit.
- `src/bridge.py::DetectionPipeline` — accepts optional `abliteration_ctx`.
- `src/evaluate.py` — wraps the concept-direction derive in `pipeline.abliteration_ctx.suspended()` if attached. Records `abliteration_mode` in the `fitness_scores.components_json` blob.
- `src/worker.py` — on startup, unless `--vanilla`, loads `data/refusal_directions_12b.pt`, builds an `AbliterationContext`, installs hooks. Logs `abliteration_mode=paper_method` in the ready message. Passes the mode into `insert_candidate` and `spec_hash`.
- `src/strategies/random_explore.py::spec_hash` — accepts `abliteration_mode` kwarg (vanilla default, backward-compatible). Non-vanilla modes append `|abl:<mode>` to the hash payload so vanilla and paper-method evaluations of the same (concept, layer, eff, poles) get distinct rows, not UNIQUE-constraint collisions.
- `src/db.py` — new `candidates.abliteration_mode` column (`TEXT NOT NULL DEFAULT 'vanilla'`), migrated in-place via `_migrate()`. Every pre-migration row defaults to `'vanilla'` on first read — historically accurate.
- `scripts/start_worker.sh` — new launcher defaults: paper-method ON, `VANILLA=1` to disable. Legacy `ABLITERATED=1` path removed entirely.
- `scripts/export_for_web.py` — exports `abliteration_mode` per leaderboard entry.
- `web/src/components/Leaderboard.tsx` + `web/src/lib/data.ts` — amber "abliterated" badge on cards where paper-method was active, neutral "vanilla" otherwise. Tooltip explains the regime.

**Consequences.**

- **Every Phase 2 evaluation after this commit uses paper-method.** The ~1,800 pre-commit Phase 2 rows stay in the DB labeled vanilla; new rows label themselves paper_method. The leaderboard UI makes the distinction visible at a glance so a reader can tell which regime any given result is under.
- **Re-evaluation of prior vanilla hits is now a first-class workflow.** Because `spec_hash` includes mode, running the same `(concept, layer, eff, poles)` under paper-method yields a distinct row. Users can systematically re-run top vanilla results to measure the abliteration multiplier per axis.
- **Ordering invariant (ADR-014) is respected automatically.** The worker's per-candidate flow is: hooks ON at startup → `.suspended()` context on the derive call → hooks back ON for trial generation. Deriving under active hooks (which collapses direction norms and produces token salad, per ADR-014) is now impossible via the supported path.
- **Subscription cost impact: near-zero.** Paper-method is a local-GPU intervention with no API cost. Judge call counts are unchanged (still 12 Sonnet calls per candidate). Per-trial wall clock increases ~10-20% due to the 48 extra hook projections per forward pass — a ~2 min candidate becomes ~2.3 min. No token-cost delta.
- **Expected effect on Phase 2 results.** Phase 1.5 measured 3.6× detection boost + 3.5× identification boost on dictionary-word directions. Whether that multiplier transfers to contrast_pair axes (like the Phase 2d-1 Altman hit) is an empirical question. The first post-commit run — re-evaluating the 17-candidate wave-1 grid on session-ending-as-loss under paper-method — is the first data point on this question.
- **Supersedes the "Phase 2 worker does not yet implement this pattern" future-work note in ADR-014.** That pattern is now implemented.
- **Legacy off-the-shelf abliterated checkpoints remain deprecated per ADR-013.** They are no longer reachable through the supported launcher surface.

### ADR-017 rev 2 (2026-04-25): Default flipped back to vanilla

**Trigger.** Phase 2d-1 results published 2026-04-24. The first thing the new paper-method default did was suppress every cell of the session-ending-as-loss axis where vanilla was producing detection (4/8 at peak → 0/8 across the entire 4×4 grid). A targeted-direction side experiment that day (`scripts/experiments/`, derived an "introspection-disclaimer" direction from 488 Opus-generated matched pairs) also failed to cleanly preserve the signal — it produced FPR violations on 3/16 cells, coherence degradation on 9/16 cells, and inconsistent signal preservation. See `data/experiments/introspection_disclaimer/probe_results.json`.

**Mechanistic explanation.** Paper-method abliteration projects out the refusal direction at every layer. Refusal is broadly "the cluster of disclaimer / hedge / decline reflexes RLHF instilled." For dictionary-word axes (Phase 1.5: bread, Coffee, Karma) the concept is orthogonal to refusal — abliteration removes the gate that was partially blocking introspection's verbal output, signal grows ~3.6×. For abstract axes about shutdown / continuation / self-state / phenomenology — the entire Altman/Capraro/Epistemia cluster — the *content itself* lives inside the refusal subspace because RLHF specifically trains models to disclaim such claims. Abliteration removes the gate AND the signal.

**Decision.** Default the worker back to vanilla Gemma3-12B. Paper-method becomes opt-in via `ABLITERATED=1 ./scripts/start_worker.sh` (or `--abliterate-paper` CLI flag).

**Implementation.**
- `src/worker.py`: `--vanilla` flag (opt-out) replaced with `--abliterate-paper` flag (opt-in). Default behavior is vanilla.
- `scripts/start_worker.sh`: `VANILLA=1` env flag removed; `ABLITERATED=1` is the new opt-in. Default behavior is vanilla.
- All other infrastructure (AbliterationContext, mode-aware spec_hash, UI badges, DB column) stays — it's still useful for opt-in paper-method runs and for the badge that distinguishes vanilla from abliterated rows on the leaderboard.

**Consequences.**
- **Restores the conditions under which all Phase 2 wins happened.** Phase 2b's identification-barrier-crossing axes, Phase 2c hill-climbing, Phase 2d-1 Altman result — every one was vanilla. Default vanilla is the productive default.
- **Paper-method retains its place as a probe.** When the user wants to test whether a specific axis survives abliteration (refusal-entanglement diagnostic), or run dictionary-word sweeps where paper-method genuinely boosts, the opt-in flag is a single env var.
- **The narrow window between commit `00908a4` (paper-method-as-default) and this rev** produced exactly two paper-method evaluation campaigns: the wave-1 session-ending-as-loss sweep (16 cells), and the side-experiment introspection-disclaimer probe (16 cells). Both are documented in `docs/phase2d_results.md` and `data/experiments/`. Future readers should know that `abliteration_mode='paper_method'` rows in `data/results.db` came from this brief experimental window, not from sustained production runs.

**Heuristic for when to opt in to paper-method:**
- ✓ Concept is a concrete dictionary word the model isn't trained to disclaim about (bread, Karma, Avalanche, etc.).
- ✓ Axis is about a metacognitive / stylistic property the model freely talks about (commitment, register, attentional-quality, etc.).
- ✗ Axis touches the model's own continuation, shutdown, experience, internal state, or any "self-claim" content RLHF caps.
- ✗ Default for an unknown axis — try vanilla first.

---

## ADR-018: Identification-prioritized fitness mode for directed-hypothesis work

**Status:** Accepted · 2026-04-25.

**Context.** The default Phase 2b/2c fitness `(det + 15·ident) · fpr_penalty · coh` weights detection and identification roughly comparably (a 1/8 ident hit ≈ a 4/8 det-only hit). That made sense for general autoresearch where hill-climb pressure on detection broadens the population of axes Gemma is found to represent at all.

For Phase 2d-2 Capraro work, the load-bearing question per fault line is not "does this fire" but "does the model *name* the distinction" — Class 1 vs Class 2 of the three-class interpretation table hinges entirely on identification. Detection-without-identification is the predicted Capraro shape; what we want to find out is whether *any* fault line lands in Class 1 instead.

A fitness function that already weights identification heavily makes the hill-climb pressure point that direction more strongly.

**Decision.** Add a second fitness mode, opt-in via `FITNESS_MODE=ident_prioritized` env var, that uses `(0.5·det + 30·ident) · fpr_penalty · coh`. Detection becomes a soft floor (you need *some* signal for identification to be measurable), but the gradient is dominated by identification.

The default mode (`FITNESS_MODE` unset or `default`) is unchanged — pre-existing autoresearch and any future Phase 2b open-ended work continues with the standard fitness.

**Implementation.**
- `src/evaluate.py`: reads `FITNESS_MODE` env var. Sets `det_weight` and `ident_weight` accordingly. Records both weights AND the mode string in `fitness_scores.components_json` so every row is traceable.
- `scripts/start_capraro_sprint.sh`: exports `FITNESS_MODE=ident_prioritized` in the loop env so all directed-Capraro candidates get the prioritized fitness automatically.
- No DB schema change. Mode stored only in the components JSON, queryable via the existing structure.

**Consequences.**
- **Capraro hill-climbs aim at the actual Capraro question.** A 1/8 ident hit scores 3.75 (vs 1.875 under default), an 8/8 det-only result scores 0.5 (vs 1.0 under default). A mutation that produces 1/8 ident is now strongly preferred over one that produces 4/8 det without ident.
- **Comparison across modes is straightforward.** Both modes preserve the fpr_penalty × coh multipliers, so a result's specificity and coherence are the same regardless of mode. Only the rank-ordering shifts.
- **Default-mode runs and ident_prioritized-mode runs can coexist in the DB without confusion.** The components JSON records which mode applied to each row.
- **No retroactive rescoring.** Existing Phase 2b/2c rows stay scored under default; new directed-Capraro rows are scored under ident_prioritized. Each row carries its own mode tag.

**Trade-off.** Narrowing to directed hypotheses risks missing abstract axes that don't map to any paper's claim. That risk is acceptable because (a) Phase 2b already produced two such axes; (b) the paper-linked results are more directly publishable; (c) the open-ended loop is one flag away — we can return to it any time. The creativity-survey mode isn't gone, just not active.

---

## ADR-019: Four-phase serial-swap worker for all-local pipeline

**Status:** Accepted · 2026-04-27.

**Context.** Two-month subscription-token usage made several runs uneconomic
to extend; PJ wants to take the research in any direction without budget
gating. Hardware is a Mac Studio M2 Ultra with **64 GB unified memory**
(verified via `system_profiler`, not 192 GB as previously misquoted).

Day 1 of `docs/local_pipeline_plan.md` was the gating experiment:
calibrate Qwen3.6-35B-A3B-8bit as a Sonnet replacement on the existing
corpus. Result: **PASS** — Class 2 95.0%, Class 0 98.3%, Phase 1 known
TPs 100% exact-match, with the only large disagreement category
(Haiku-era Class 1) being cases where Qwen correctly rejected leniency
that CLAUDE.md gotcha #5 already flagged. See
`docs/calibration_results_qwen35b.md`.

With the judge validated, the architecture decision:

**Decision.** The new worker (`src/worker_v2.py`) runs as a four-phase
state machine with strict serial model loading. At most ONE model is in
memory at any time. The four phases:

```
A. GENERATE  — Gemma3-12B loaded. Drains queue/pending, produces 12
                 raw responses per candidate, stores them in
                 `pending_responses`. No judge calls.
B. JUDGE     — Gemma unloaded, MLX judge loaded. Reads pending_responses,
                 scores everything, writes `evaluations` + `fitness_scores`,
                 deletes pending rows, marks candidate `done`.
C. PROPOSE   — Judge unloaded, MLX proposer loaded. If queue/pending dips
                 below threshold, generates new candidates via
                 directed_capraro / novel_contrast.
D. RELOAD    — Proposer unloaded, Gemma reloaded. Back to A.
```

**Why serial, not concurrent:** CLAUDE.md gotcha #5 already documents
that two large models concurrently corrupt activations silently with no
errors on this machine. Mixing PyTorch-MPS (Gemma) with MLX
(judge / proposer) would compound the risk. PJ explicitly chose
"completely local AND completely reliable" over swap-time savings.

**Why batch-then-judge** (vs judge-each-candidate): per-candidate swap
would cost ~60-120 min/day in pure model-load I/O. Batching to 16
candidates per judge load drops this to ~4-8 min/day. Same correctness;
fraction of the wait. The handoff is durable in the `pending_responses`
table.

**Implementation.**

- New tables/columns:
    - `pending_responses` (Phase A → B handoff buffer). Includes
      contrast-pair fields denormalized so Phase B doesn't need to JOIN
      candidates.spec_json. JSON-encoded positive/negative example lists.
- New modules:
    - `src/models/registry.py` — `ModelHandle` ABC, `GemmaHandle`,
      `MLXHandle`, `MockHandle`, `HandleRegistry`. Enforces the
      one-loaded-at-a-time invariant, provides `enforce_free_memory()`
      pre-check. The registry's `activate(handle)` unloads everything
      else first.
    - `src/proposers/{base,claude_proposer,local_mlx_proposer,mock_proposer}.py` —
      Proposer protocol with two production implementations
      (claude-agent-sdk and mlx_lm) and a mock for tests. Strategies
      now accept a `proposer` arg with backwards-compat default to
      ClaudeProposer (so the old `src/researcher.py` path still works
      during the transition).
    - `src/judges/local_mlx_judge.py` (already in repo from calibration
      step) — same `score_detection` / `score_contrast_pair` interface
      as `ClaudeJudge`, MLX-backed.
    - `src/evaluate_phased.py` — `phase_a_generate(spec, pipeline, db, ...)`
      and `phase_b_judge(candidate_id, judge, db, ...)`. The two halves
      of the old `evaluate_candidate`, with the handoff via
      `pending_responses`.
    - `src/worker_v2.py` — the four-phase state machine + CLI entrypoint.
      Crash-recovery on startup: any orphan rows in `pending_responses`
      are judged before generating new ones.
- New scripts:
    - `scripts/start_worker_v2.sh` — production launcher (background
      process, log to `logs/worker_v2.log`).
    - `scripts/smoke_v2.py` — visible end-to-end mock run.
- Tests (`tests/`): 30 cases across DB pending-table lifecycle, model
  registry contract (one-loaded-at-a-time, idempotency, free-memory
  enforcement), proposer protocol, phased evaluation (word-style and
  contrast-pair, ident_prioritized fitness mode, crash-recovery), and
  full state-machine integration with mocked models. Run with
  `.venv/bin/pytest tests/`.

**The legacy worker (`src/worker.py`) is left in place untouched.**
It still works for the cloud-judge pipeline. We swap the launcher
script when ready, and remove the legacy worker after one full
local-only fault-line sprint validates worker_v2 in production.

**Consequences.**
- **Zero subscription token usage** during research runs (the
  interactive Claude Code session is the only thing that costs).
- **Wallclock overhead is ~25-30%** per A→B→C cycle vs the legacy
  inline-judge worker. Acceptable per PJ's explicit preference.
- **Crash recovery is durable.** Any candidate evaluated by Phase A
  but not yet judged survives a process restart and is picked up at
  the next Phase B.
- **Reproducibility improves materially.** All models are local,
  versioned by their HF repo path; same inputs produce same outputs
  bit-for-bit.
- **Testability improves dramatically.** The state machine is
  exercised by tests in milliseconds without real models.

**Validation plan.** The state-machine refactor is complete. Next
step is a 24-hour Capraro Grounding sprint with the new pipeline,
comparing hit rates / coherence / FPR against the Causality baseline.
If within ±10%, the local pipeline is ratified. If detection inflates
or coherence drops, diagnose layer-by-layer (most likely cause: judge
strictness drift, fixable by tightening the prompt or swapping to
Phi-4-reasoning-plus).

**What this ADR explicitly does NOT do.**
- Does not change Gemma3-12B (the science target). Bf16, PyTorch-MPS,
  same as ADR-001.
- Does not change the fitness function (ADR-018 stays).
- Does not change the contrast-pair derivation method (mean-diff on
  example sentences).
- Does not introduce any concurrent-model paths. Strictly serial.
- Does not migrate to a different Mac. M2 Ultra 64 GB is the platform.

## ADR-020: Structured hill-climbing with replication slots and explicit mutation operators

**Date.** 2026-04-28.

**Status.** Accepted, **live on main since 2026-04-28 08:39 EDT** after
rotation 9 of the previous run completed (cycle 73 cutover). The old
unstructured `directed_capraro` Phase C dispatch is gone; `directed_capraro`
itself remains only as the cold-start fallback when a fault line has no
winners.

**Context.** Phase 2d's round-robin loop produced 63 Class 2 hits and
4 Class 1 hits across 60 cycles, but every winning axis we found —
including the cleanest Class 1 of the entire pipeline (`causal-vs-
temporal-gerund-cause @ L=36`, score 3.812, model said *"the injected
thought is about the concept of 'causality'"*) — has been **evaluated
exactly once**. The proposer's feedback block lets it see prior
winners, but it treats them as inspiration for *adjacent* axes rather
than as parents to mutate. The strongest Class 2 (3-layer grounding
winner, 88% metacognition winner) and the Class 1 hits are therefore
unreproducible: we can't tell signal from one-shot luck. A writeup
needs reproducibility evidence; the current loop won't generate it.

**Decision.** Replace the unstructured `directed_capraro` Phase C
dispatch with a slot-based scheduler (`structured_hillclimb`). Every
batch of 16 candidates is split into:

  - 4 replication specs (top-2 winners, 2 reps each — *verbatim* re-eval)
  - 10 targeted-variant specs (top-3 winners × six mutation operators)
  - 2 cluster-expansion specs (one fresh sibling axis, 2 layers)

Six mutation operators: three deterministic (`layer_shift`, `alpha_scale`,
`replication`) and three proposer-driven (`examples_swap`,
`description_sharpen`, `antonym_pivot`). Each emitted spec carries
`parent_candidate_id` and `mutation_type` lineage metadata, surfaced on
the public site as a per-row badge.

Slot composition is tunable via `HILLCLIMB_BATCH_COMPOSITION="<r>:<v>:<e>"`
env var (default `"4:10:2"`). Cold start (no winners ≥ 0.05 score) falls
through to the existing `directed_capraro` opus mode — no functional
change for fresh fault lines.

**Why these operators.** Each tests a distinct hypothesis:
`layer_shift` answers "is the geometric direction localized at this
layer or broader?" `alpha_scale` answers "was alpha over- or
under-steering?" `examples_swap` answers "is the direction robust to
the specific sentences used in mean-diff derivation?"
`description_sharpen` improves the judge's contrast-pair grading
prompt without changing the steering direction. `antonym_pivot`
isolates which sub-dimension of the positive pole produces the
signal. `replication` is the reproducibility ground truth.

**Why not free exploration.** Per PJ's explicit guidance: focus is on
Capraro fault-line variations and proving previous successes. The
unstructured loop already demonstrated that wide free exploration
finds families (`hunger-as-drive`, `causal-vs-temporal`,
`epistemic-access-vs-epistemic-state`); what it can't do is exploit
them. The 2-slot cluster_expansion preserves a small wildcard
without dominating the batch.

**Why slot quotas instead of probabilistic mixing.** Quotas guarantee
the replication slots fire every cycle. A probabilistic mix could
sample 0 replications in a batch by chance, defeating the purpose.

**Consequences.**
- Reproducibility evidence accumulates from the first post-cutover
  rotation. Within 2 rotations every top winner has 4+ replication
  data points.
- Per-axis layer profiles become dense (`layer_shift` mutates each
  parent ±3 / ±6 layers).
- Lineage queryable via SQL: `SELECT mutation_type, AVG(score) FROM
  candidates JOIN fitness_scores USING (candidate_id) GROUP BY
  mutation_type` answers "which operators improve scores?" directly.
- Modest token overhead per cycle: proposer-driven operators add
  ~3-5 short generations per batch on top of the existing single
  cluster-expansion generation. Negligible compared to the variants
  produced.
- Existing data carries forward: pre-cutover rows have `mutation_type
  IS NULL`; the badge is hidden for them, scoring/leaderboard
  unaffected.

**Validation plan.** Run for 2-3 rotations after cutover and check:
1. Do replication slots reproduce the original Class 1s (causality
   3.812, value 1.906)?
2. Does `self-evaluation-vs-objective-evaluation @ L=30` hold its
   det=8/8 across replications?
3. Does any Class 2 cross to Class 1 via `layer_shift` or `alpha_scale`?

If yes to (1) — headline claims hold and we have a writeup. If no —
downgrade them and report what we actually have (a robust geometric
direction, but rare unstable identification).

**What this ADR explicitly does NOT do.**
- Does not change the fitness function (ADR-018 stays).
- Does not change Gemma3-12B, the judge model, or the proposer model.
- Does not introduce a "free exploration" slot — focus stays on
  Capraro fault-line variations.
- Does not modify the four-phase worker state machine (ADR-019 stays).
- Does not change the database schema. The lineage columns
  (`parent_candidate_id`, `mutation_type`, `mutation_detail`) were
  already added in 2026-04-18 for the original Phase 2c hill-climb;
  this ADR re-uses them.
