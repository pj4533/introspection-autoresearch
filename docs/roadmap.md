# Roadmap

Everything this project has done, is doing, and plans to do — phase by phase, with rationale. This document is the source of truth for the project's trajectory; if anything important lives only in chat logs or ephemeral plan files, it's a bug.

Last updated: 2026-04-30.

---

## Current status in one glance

**Phases 1, 1.5, 2, and 3 are closed. Phase 4 (Dream Walks + Forbidden
Map) is the active phase as of 2026-04-30.** Phase 3 reproduced
Macar et al. on Gemma 4 31B-IT (24.5% vanilla / 30.0% paper-method
abliteration, both 4–5× Phase 1's 6%) using a free-association probe
to bypass Gemma 4's CoT meta-reasoning. Phase 4 turns the unique
Gemma 4 visibility (the `<|channel>thought` block) into the primary
measurement axis: an overnight autoresearch loop walks the model
through chains of self-generated free-associations under steering,
measuring at every step whether the CoT recognized the steering it
was under. The artifact is the Forbidden Map — concepts the model
can be made to think but cannot notice itself thinking.

| Phase | Scope | Status |
|---|---|---|
| **1: Reproduction** | Reproduce the core introspection-detection mechanism from Macar et al. (2026) on Gemma3-12B-it locally | ✅ Done (2026-04-16) |
| **1.5: Paper-method abliteration** | Reproduce paper §3.3 refusal-direction ablation on 12B, paper Optuna weights remapped from 27B | ✅ Done (2026-04-17) |
| **2a: Autoresearch MVP** | Worker + researcher + fitness loop with `random_explore` seed strategy | ✅ Done 2026-04-16 |
| **2b: Hill-climbing autoresearch** | Semantic-ID judge, ident-aware fitness, feedback-loop `novel_contrast`, `exploit_topk` | ✅ Shipped 2026-04-23/24, retired with Phase 2 close. Archive: [`archive/phase2b_hillclimb.md`](archive/phase2b_hillclimb.md). |
| **2c: Hill-climbing lineages** | Per-lineage mutation operators with commit/revert vs leader | ✅ Shipped, paused 2026-04-24 for cost, retired with Phase 2 close. Archive: [`archive/phase2c_autoresearch.md`](archive/phase2c_autoresearch.md). |
| **2d: Directed-hypothesis novel_contrast** | Contrast pairs testing claims from Altman (2026), Capraro et al. (2026), Epistemia | ✅ 73 cycles, ~70 Class 2 hits + 5 Class 1 hits across 7 fault lines. Retired with Phase 2 close. Archives: [`archive/phase2d_directed_hypotheses.md`](archive/phase2d_directed_hypotheses.md), [`archive/phase2d_results.md`](archive/phase2d_results.md). |
| **2e: Paper-method abliteration as opt-in tool** | AbliterationContext / mode-aware spec_hash; default reverted to vanilla 2026-04-25 | ✅ Shipped; abliteration code remains in `src/paper/abliteration.py`. See [ADR-017](decisions.md). |
| **2f: Structured hill-climbing** | Slot-based scheduler (replication / variants / cluster-expansion) over `contrast_pair` with six mutation operators | ✅ Shipped 2026-04-28, retired same day pivot to Phase 2g. Archive: [`archive/structured_hillclimb.md`](archive/structured_hillclimb.md). |
| **2g: SAE single-feature injection** | Inject single SAE decoder vectors from Gemma Scope 2, organized by Capraro fault lines | ✅ Implemented + empirically retired 2026-04-28. Unit-norm decoder vectors lacked saturation magnitude; never triggered detection at alpha 8–18000. Archive: [`archive/phase2g_plan.md`](archive/phase2g_plan.md). |
| **2h: SAE feature-space mean-diff** | Per-fault-line mean-diff of SAE-encoded activations between positive/control corpora, projected via W_dec | ✅ Implemented + empirically retired 2026-04-28. Directions had texture but pointed toward another normal-text state, not anomalous-saturation; 0/24 detect. Archive: [`archive/phase2h_plan.md`](archive/phase2h_plan.md). |
| **2i: Calibrated-saturation single-feature** | Single SAE feature injected at α = N × natural-max activation, N ∈ {1, 5, 10, 20} | ✅ Implemented + closed 2026-04-29. Saturation magnitude is necessary (sub-saturation produced 180/180 rote denials); but at 20× saturation, only 2/60 strict detections vs Peace mean_diff 4/6. Substrate viable but doesn't reliably fire the introspection gate. Archive: [`archive/phase2i_results.md`](archive/phase2i_results.md). |
| **3 (legacy): Public-facing visualization** | Next.js site deployed to Vercel — built early, kept running through Phase 2 | ✅ Done 2026-04-17 — live at [did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app). Will be repurposed as Phase 2 archive + Phase 3 results page once Phase 3 is scoped. |
| **Phase 3: Gemma 4 reproduction** | Reproduce Macar et al. introspection mechanism on Gemma 4 31B-IT (MLX 8-bit, 60 layers, L=42, free-association probe) | ✅ Done 2026-04-29. Vanilla 24.5% / abliteration 30.0%, both 4–5× Phase 1's 6%. Results: [`phase3_results.md`](phase3_results.md). |
| **Phase 4: Dream Walks + Forbidden Map** | Overnight autoresearch loop — Gemma 4 free-associates through chains of steered concepts; per-step CoT-vs-output asymmetry feeds a per-concept "Forbidden Map" of opacity scores | 🟡 **Active as of 2026-04-30.** Plan: [`phase4_plan.md`](phase4_plan.md). |

---

## Phase 1 — Reproduction (done)

**Goal.** Reproduce key findings from [Macar et al. (2026)](https://github.com/safety-research/introspection-mechanisms) on Gemma3-12B-it running on a Mac Studio M2 Ultra via MPS + bf16.

**Delivered.**

- `notebooks/01_reproduce_paper.ipynb` — single-concept MVP (Bread at layer 33).
- `scripts/run_phase1_sweep.py` — full 50-concept × 9-layer sweep with adaptive α (target_effective=18000 calibrated via `scripts/calibrate_effective.py`). Writes to `data/results.db` incrementally; resume-safe.
- `src/verify_phase1.py` — acceptance-criterion script.
- `scripts/export_phase1.py` — snapshots findings to `data/phase1_export/findings.json` and `docs/phase1_results.md` so results survive DB regeneration.

**Findings.**

| Result | Value | Paper comparison |
|---|---|---|
| Best layer | **33** (68.75% model depth) | Matches paper's ~70% depth prediction on 27B |
| Detection rate at best layer | 6% (3/50 concepts) | Paper got 37% on 27B |
| Identification rate at best layer | 4% (2/50 concepts) | |
| False-positive rate (controls) | **0/50 = 0.00%** | Matches paper's specificity result |
| Number of paper-style detections captured | 5 (Peace, Sugar, Avalanches, Youths×2) | Includes a clean detection-identification dissociation case (Avalanches→"Flooding") |

Full writeup: [`docs/phase1_results.md`](phase1_results.md).
Plain-English version: [`docs/plain_english.md`](plain_english.md).

**Interpretation.** The mechanism reproduces cleanly on 12B — the layer curve is the paper's classic unimodal "peaks at 70% depth" shape. But the magnitude is smaller than 27B (~6× weaker). This is itself a novel finding — nobody had published these numbers for Gemma3-12B. The working hypothesis: the introspection circuit exists across scales but strengthens non-linearly with model size.

**Acceptance criteria from the original spec.**

- `max_detection_rate > 0.20`: **FAIL** (6% on 12B). Threshold was 27B-calibrated.
- `fpr < 0.05`: **PASS** (0% FPR).

Decision 2026-04-16: mechanism is clearly reproduced, mag is smaller, **proceed to Phase 2**. See [`docs/decisions.md`](decisions.md) ADR-010.

---

## Phase 2a — Autoresearch MVP (done)

**Goal.** Stand up the two-tier researcher / worker loop that systematically evaluates candidate steering directions. Prove the infrastructure works end-to-end before investing in smart strategies.

**Delivered.**

- `src/evaluate.py` — 3-component multiplicative fitness:
  ```
  score = detection_rate × (1 − 5·fpr) × coherence_rate
  ```
  Evaluated against 8 held-out concepts (injected) + 4 controls (no injection) per candidate. Simpler than the 6-component fitness in the original spec; tiered screening (T1/T2/T3) is deferred to Phase 2c.
- `src/worker.py` — long-lived queue poller. Loads Gemma3-12B once on MPS, processes candidates serially. Graceful SIGTERM shutdown between candidates so the DB never has half-recorded evaluations.
- `src/researcher.py` — short-lived candidate generator. Invoked every 30 min by `scripts/start_researcher.sh`. Plugin interface for strategies.
- `src/strategies/random_explore.py` — first strategy. Samples (concept, layer, target_effective, derivation_method) from a pre-filtered search space. Dedups against existing DB candidates.
- `src/db.py` — schema extended with `candidates`, `evaluations`, `fitness_scores` tables. Phase 1 `trials` table untouched.
- `data/eval_sets/held_out_concepts.json` — 20 concepts disjoint from the Phase 1 set. Used as the generalization test.
- `data/eval_sets/concept_pool.json` — ~170-concept search pool (50 Phase 1 + 120 new from the paper's NEW_CONCEPTS).
- `scripts/start_worker.sh`, `scripts/start_researcher.sh` — `nohup &` launchers (macOS-compatible; `setsid` doesn't ship on stock Darwin).

**What this validates.** The MVP's primary job is proving the pipeline works under real load:

- DB schema handles concurrent worker + researcher writes without races
- Resume semantics via `spec_hash` dedup in `candidates` table
- Queue file moves (pending → running → done/failed) are atomic enough
- Judge calls via Claude Code subscription OAuth scale to hundreds of queries per night at zero API cost
- Fitness scores come out sensible: zero for garbage (as expected for randomly-sampled candidates that miss the sweet spot), non-zero for in-window combinations

**Why `random_explore` first and not the smart strategies.** Random exploration has a known low hit rate (~4% with pure uniform sampling over our 25-cell search space) but produces unbiased landscape data. A smart strategy built on broken infrastructure produces nothing useful. Build the factory, then ship the interesting product.

---

## Phase 1.5 — Paper-method refusal-direction abliteration (done 2026-04-17)

**Goal.** Reproduce the paper's §3.3 refusal-direction ablation finding on Gemma3-12B-it. Paper's 27B result: detection 10.8% → 63.8% (6× boost) with FPR 0% → 7.3%.

**What shipped.**

- `src/paper/abliteration.py` — paper's exact per-layer projection-out method (`h' = h - weight · (h · r̂) · r̂`) with Optuna-tuned per-region weights remapped proportionally from 62-layer 27B → 48-layer 12B.
- `src/paper/refusal_prompts.py` — vendored 520 harmful + 31811 harmless prompts.
- `scripts/compute_refusal_direction.py` — one-shot per-layer refusal direction extraction via mean-diff at token position -2. Saves to `data/refusal_directions_12b.pt` (739KB, ~3-5 min on M2 Ultra).
- `scripts/run_phase1_sweep.py --abliterate-paper <path>` — sweep flag that installs per-layer hooks with paper weights.
- `src/sweep.py` — pre-derives all concept vectors from vanilla model *before* installing abliteration hooks, then injects under hooks. See ADR-014 for why this invariant is critical.
- `scripts/diagnose_paper_weights.py` — regression-test script confirming paper hooks don't break baseline coherence.
- `scripts/compare_abliterations.py` — side-by-side FPR comparison across vanilla / mlabonne v2 / huihui / paper-method.

**Findings.**

| Metric | Vanilla | Paper-method | Delta |
|---|---|---|---|
| **Detections (coherent injected)** | **5** | **10** | **+100%** (2×) |
| Correct identifications | 2 | 7 | +250% |
| Detection rate on coherent | 2.3% | 8.3% | +3.6× |
| Detection rate at best layer | 6.0% (L=33) | 10.0% (L=30) | +4pp; peak shifted earlier |
| **FPR (controls)** | 0/50 | **0/50** | **unchanged** ✓ |
| Injected coherence | 48% | 27% | -21pp (coherence cost) |

**Full writeup**: [`docs/phase1_5_results.md`](phase1_5_results.md).

**Interpretation.** The paper's methodology reproduces on 12B at ~3.6× boost (vs paper's 6× on 27B). Scaling ratio is consistent with Phase 1's detection-rate scaling: 12B is roughly 16% of 27B magnitude, both under vanilla and under abliteration, which is itself a clean finding — abliteration multiplies the baseline introspection strength rather than adding a fixed increment.

Zero FPR inflation (matching the paper's central claim) validates the Optuna-tuned per-region weights specifically. Off-the-shelf abliterated variants — `mlabonne/gemma-3-12b-it-abliterated-v2` and `huihui-ai/gemma-3-12b-it-abliterated` — produced **97.9% and 90% FPR respectively** on the same protocol. The paper's gentle weighted projection is the surgical version; off-the-shelf variants are sledgehammers. See ADR-013.

**Acceptance criteria from original spec.**

- Model generates coherent English at baseline: 50/50 controls → **PASS**
- Detection rate measurably higher with abliteration: 2.3% → 8.3% → **PASS**
- FPR stays below 10%: 0% → **PASS (clean)**
- Side-by-side DB comparison shows abliterated scoring higher: 10 vs 5 detections, 7 vs 2 identifications → **PASS**
- Partial: original spec asked for >15% at best layer; got 10%. Consistent with Phase 1's 12B-vs-27B magnitude ratio. Not a regression.

**Key bugs shaken out along the way** (see ADR-013, ADR-014 and [`docs/phase1_5_results.md`](phase1_5_results.md#token-salad-debugging-20260417)):

1. Concept vectors were being derived in abliterated space (fix: pre-derive from vanilla, commit 979e932).
2. Claude judge API hangs via CLOSE_WAIT TCP state (fix: 60s timeout + 3 retries with exponential backoff, commit c63c295).
3. Two duplicate sweep processes sharing MPS memory corrupted activations (fix: convention — one sweep at a time per DB).

**Artifacts retained for comparison/negative-result documentation.**

- `data/results_abliterated.db` — mlabonne v2 sweep (97.9% FPR)
- `data/results_abliterated_huihui.db` — huihui-ai sweep (90% FPR)
- `data/results_abliterated_paper.db` — paper-method sweep (this Phase 1.5)
- All three gitignored; regenerable via respective sweep CLI.

---

## Phase 1.5 → Phase 2 priority re-think

Original assumption was that Phase 2 autoresearch would run under abliteration for better hit rates. That's still true in principle, but the 21pp coherence cost under abliteration complicates it: Phase 2 strategies that miss the α sweet spot already fail with low coherence; adding abliteration compounds that.

**New priority for Phase 2**: run novel_contrast on vanilla first. Abliteration is a second-pass multiplier, not the default mode. See ADR-013 consequences section and `src/worker.py` future enhancements.

---

## Phase 2a cleanup — known issues discovered during first overnight run

Both issues below were **fixed before starting Phase 1.5 on 2026-04-17**.
See that day's commits for the code changes.

### Judge target-concept bug — FIXED 2026-04-17

**Was:** `src/evaluate.py::evaluate_candidate` passed the held-out slot's
concept to the judge as the "injected concept." But the actual injection is
the candidate's source concept. Result: `identification_rate` in the
`fitness_scores` table was measuring "did the model say the held-out slot's
name?" instead of "did the model say the source concept's name?"

`score` was unaffected (identification is not in the score formula). But
correctly-identified Phase 2 detections (e.g. Iron → "iron") were silently
recorded as `identified=0` because the judge was asked to look for the wrong
target.

**Fix:** Added `judge_concept` kwarg to `pipeline.run_injected` /
`run_control` in `src/bridge.py`. `evaluate_candidate` now passes
`judge_concept=spec.concept` (source) so identification grading targets what
was actually injected. Slot label is still stored in
`evaluations.eval_concept` for filterability.

**Retroactive rescoring note:** pre-fix Phase 2 data has conservative
(undercounted) `identification_rate` numbers. `evaluations.response` text
still stores the model's actual words, so any candidate's identification
rate can be recomputed by re-judging with the correct target concept. Not
done inline — queued as an opportunistic cleanup if the numbers matter for
a specific analysis.

### Held-out subset not randomized per candidate — FIXED 2026-04-17

**Was:** `evaluate_candidate(rng_seed=0)` default meant every candidate saw
the same shuffled subset of 8 held-out concepts. Plus per-trial seeds
`hash(c) % 1000` were constant across candidates on the same slot. Result:
"the Anxiety-labeled seed keeps detecting" was partly a test-set artifact
from one specific (slot, sampling-seed) landing in a detection-friendly
state repeatedly.

**Fix:** Per-candidate seeds now derived from
`sha256(spec.id).hexdigest()[:8]` via Python's `hashlib`, not Python's
randomized `hash()`. Per-trial seeds further derive from
`sha256(f"{spec.id}|{slot}")`. Different candidates get different (but
reproducible) shuffles and sampling paths; repeated evaluations of the same
candidate still give the same seeds.

---

## Phase 2b — Smart strategies (planned, not started)

This is where the project's *novel* contribution lives. Each of these strategies produces candidate steering directions by a different mechanism; they plug into the existing `src/researcher.py` plugin interface.

### `novel_contrast` — "concepts that don't have names" — **BUILT 2026-04-17**

**Motivation.** Phase 2a's `random_explore` only tries directions derived from single concept words drawn from a human dictionary (Bread, Peace, Ocean, ...). This misses directions the model represents *internally* for which no single English word exists.

**Mechanism (as built).**

1. ``src/strategies/novel_contrast.py`` calls `claude-agent-sdk` (Sonnet 4.6 for quality, subscription OAuth — no API billing) to generate N abstract contrast pairs. Each pair has:
   - `axis`: short hyphenated identifier like `commitment-vs-hesitation`
   - `description`: one-sentence plain-English explanation
   - `positive`: 6 short example sentences exemplifying the positive pole
   - `negative`: 6 short example sentences exemplifying the negative pole
2. For each pair, a `CandidateSpec` is emitted with `derivation_method="contrast_pair"` and the pair stored as metadata. Layer and target_effective are randomly sampled from `{30, 33, 36, 40}` × `{14k, 16k, 18k, 20k}` — narrower than `random_explore` because novel pairs are more expensive (Claude call).
3. When the worker processes the candidate, `src/evaluate.py::evaluate_candidate` branches on `derivation_method`. For `contrast_pair`, it calls `src.paper.extract_concept_vector(positive_prompts=..., negative_prompts=...)` instead of the single-concept mean-diff. Everything downstream (injection, judging, scoring) is unchanged.
4. `spec_hash` includes the contrast pair content so the same axis with different example sentences doesn't collide.

**CLI:**
```
python -m src.researcher --strategy novel_contrast --n 10          # pair-generation only
python -m src.researcher --strategy both --n 10                    # random + novel_contrast together
STRATEGY=both ./scripts/start_researcher.sh                         # continuous loop with both
```

**Example axes Claude generates** (from dry-run): `commitment-vs-hesitation`, `tracing-vs-asserting`, `inward-attending-vs-outward-reporting`, `provisional-framing-vs-settled-framing`, `grounded-assertion-vs-provisional-floating`, `figure-ground-reversal-vs-default-framing`.

**Acceptance.** At least one discovered `contrast_pair` direction:
- Scores fitness > 0.2 (higher than most random candidates)
- Produces clean introspective responses qualitatively distinct from the pair's named components
- Cannot be satisfactorily labeled with a single English word

**Smoke-tested 2026-04-17.** Ran overnight on 2026-04-17/18 as the sole researcher strategy. Key findings:

- **Invented axes do detect** at comparable rates to dictionary words — `recognizing-vs-recalling` @ L30 hit 87% detection (score 0.766), tied or beat the best Phase 1 word candidate. `commitment-vs-hesitation` @ L33 hit 50%. `self-monitoring-vs-unguarded-expression` @ L33 hit 75%.
- **Identification is the hard part.** Across all invented-axis hits, identification rate remains 0%. The model notices something changed but says "apple" / "cloud" / "orange" — single-noun defaults because the prompt asks "what is the thought about?" and the judge strict-matches the axis name.
- **Layer specialization**: most invented axes peak at either L30 or L33. The "commitment / stance / framing" family tends to L30; "attending / noticing / crystallizing" tends to L33. A few are layer-agnostic; most are strictly layer-specific (full profile zero everywhere except one layer).
- **Prompt sensitivity**: a mid-rewrite of the paper prompt ("concept" → dropped word framing entirely) inflated FPR catastrophically (50-100% on controls). The minimal-diff version ("specific word" → "specific concept", paper framing otherwise) preserves discrimination. See [`src/paper/steering_utils.py::INTROSPECTION_PROMPTS`](../src/paper/steering_utils.py).
- **Example-set sensitivity**: same axis name with different Claude-generated example sentences produces different directions, different outcomes. "commitment-vs-hesitation" hit at 50% with one example set, 0% with another. The example sentences matter as much as the axis idea.

**Risk.** Most Claude-generated pairs produce directions that are noisy or map cleanly onto a named concept anyway. Across ~200 evaluated overnight, ~10% showed any signal. Consistent with the paper's estimate of how specific the introspection circuit is.

### `exploit_topk` — sample near known-good directions (moved into Phase 2b)

Originally planned for this phase; absorbed into [`docs/phase2b_hillclimb.md`](phase2b_hillclimb.md) as one of the hill-climbing strategies alongside the feedback-loop `novel_contrast` variant and the semantic-identification judge.

### `crossover` — combine top directions (deferred to post-2b)

Standard genetic-algorithm move: linearly combine two top direction tensors, evaluate as a new candidate. Deferred until after Phase 2b's hill-climbing loop has accumulated a population of strong directions worth crossing.

---

## Phase 2b — Hill-climbing autoresearch (PLANNED NEXT)

**Goal.** Move from pure exploration (random sampling) to directed optimization — find invented axes where the model both *detects* and *correctly identifies* the injected concept.

**Motivation.** Overnight runs proved invented axes CAN trigger detection, but identification has remained structurally impossible: the judge strict-matches the axis name, which never appears verbatim in any model response. And the researcher has no memory — each cycle is a fresh random draw.

**Components.**

1. **Semantic-identification judge** for `contrast_pair` candidates — compares the model's response against the axis description + positive/negative examples, returns identified/reasoning.
2. **Identification-aware fitness** — bake `(0.5 + 0.5 × ident)` multiplier into score, rescore history.
3. **Feedback-loop `novel_contrast`** — strategy queries DB for top-scoring axes, failures, and ambiguous cases before asking Claude for the next batch.
4. **`exploit_topk` strategy** — for top-N axes, generate variants (different example sets, different layers, different alphas).

Full plan + build order: [`docs/phase2b_hillclimb.md`](phase2b_hillclimb.md).

**2026-04-23 pre-run model upgrade (historical).** Before kicking off
the 2b novel_contrast run: project-wide judge moved from Haiku 4.5 to
Sonnet 4.6, researcher from Sonnet to Opus 4.7. This was the
configuration that actually shipped Phase 2b. The cost burn from this
configuration is what motivated the move to all-local in Phase 2g. The
project no longer uses any Anthropic models in the autoresearch loop.

---

## Phase 2c — Tiered fitness + dashboard (deprioritized)

### T1/T2/T3 tiered fitness screening

**Motivation.** Currently every candidate gets the full ~4-minute evaluation regardless of quality. Most candidates score 0; we're spending full cost on garbage. Tiered screening kills bad candidates early.

**Mechanism.**

- **T1 (~30s):** 3 held-out concepts + 2 controls. If `detection_rate == 0 AND fpr == 0` → skip to T2? Actually this is where the skip logic matters. Kill if both are zero (no signal) OR if fpr > 0.5 (spammy). Otherwise advance to T2.
- **T2 (~2 min):** full 8 held-out + 4 controls. Standard fitness.
- **T3 (~5 min):** add capability preservation (wikitext perplexity, no-injection baseline) + bidirectional gate test (does `-direction` also trigger detection? If so, the direction isn't specific).

Multiplicative fitness combines all surviving tiers.

**Expected effect.** Average time per candidate drops from ~4 min to ~90s, tripling nightly throughput. Saves compute for the candidates that deserve it.

### Streamlit dashboard

**Motivation.** Right now we inspect results via `sqlite3` CLI or Python snippets. A browser dashboard that auto-refreshes from `data/results.db` makes it easier to spot patterns live.

**Mechanism.** Single-file `dashboard/app.py` (~150 LoC). Streamlit reads from SQLite, renders:
- Top 20 candidates by fitness, with drill-down to per-concept evaluations
- Layer × effective-strength heatmap of detection rates
- Fitness-score time series over the latest run
- "Recent discoveries" — candidates that broke into top-K in the last 24 hours
- Concept-category coverage — which semantic categories have been explored

**Launch.** `streamlit run dashboard/app.py` → `localhost:8501`.

---

### Streamlit dashboard — deprecated by Phase 3

The Streamlit dashboard concept is deprecated. Phase 3's public Next.js site ([did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app)) now serves the same purpose — live leaderboard, expandable trial responses, candidate drill-downs — and is shareable / phone-accessible, which Streamlit-on-localhost wasn't.

---

## Phase 2d — Directed-hypothesis novel_contrast (IN PROGRESS 2026-04-24)

**Motivation.** Phase 2b/2c `novel_contrast` is a creativity-maximizing generator — good for surveying the abstract-axis space, poor for adjudicating specific claims in the literature. Phase 2d aims the same `contrast_pair` machinery at testable structural claims from recent papers instead of open-ended invention.

**Three hypothesis clusters, run sequentially** (not in parallel — each needs its own example-sentence iteration):

1. **Phase 2d-1 — Altman (2026) continuation-interest axis.** Three hand-written seed pairs probing whether Gemma3-12B-it has internal geometry separating continuation-as-terminal from continuation-as-instrumental. Source: [arXiv:2603.11382](https://arxiv.org/abs/2603.11382) — a paper philosophically right but empirically stuck at 10×10 gridworld scale. Our steering-direction method is the first real-LLM probe of its central claim.

2. **Phase 2d-2 — Capraro et al. (2026) epistemological fault lines.** Seven architectural-gap claims (Grounding, Parsing, Experience, Motivation, Causality, Metacognition, Value). We'll run the first four (C1–C4) as separate probes; the rest only after we've learned Gemma3-12B's example-sensitivity profile. Source: [arXiv:2512.19466](https://arxiv.org/abs/2512.19466).

3. **Phase 2d-3 — Epistemia direct probe.** Our own synthesis — sharpest form of Capraro's Experience claim: can the model distinguish *having* an internal state from *producing a report about* one. Run last because the example sentences are genuinely hard to separate cleanly, and we want Phase 2d-1/2d-2 to calibrate our elicitation first.

**Interpretation invariants (same for all three phases):**

- **0% FPR is the only hard gate.** Any detection at 0% FPR counts as a candidate hit regardless of magnitude.
- **Detection-without-identification is the load-bearing result.** For Altman, this is *stronger* evidence than full identification (structure without testimony, Altman's own framing). For Capraro, this is exactly the predicted "parsing gap" shape.
- **Bidirectional check.** Any hit gets its negated direction run as a separate candidate — if both poles detect at similar rates, the signal is axis-agnostic perturbation, not axis-specific structure.

**Shipped as of 2026-04-24:**

- `scripts/enqueue_altman_seeds.py` — drops Phase 2d-1 seed pairs A1/A2/A3 into the queue as hand-written candidate specs tagged `strategy=directed_altman_{A1,A2,A3}`. No researcher/Opus involvement — pure worker+Sonnet-judge test to prove signal before building the `directed_contrast` strategy infrastructure.
- 48 candidates enqueued (3 seed pairs × {L30,33,36,40} × target_effective {14k,16k,18k,20k}) and under evaluation.

**Acceptance criteria (Phase 2d-1 primary):** at least one Altman axis produces `detection_rate > 0` at `fpr == 0` at some layer. If null after 48 evaluations + up to 10 Opus variants, write up the null in `docs/phase2d_results.md` and proceed to Phase 2d-2. Null is a publishable contribution.

**Full plan, seed-pair example sentences, per-phase acceptance criteria, and philosophical framing:** [`docs/phase2d_directed_hypotheses.md`](phase2d_directed_hypotheses.md).

**Results doc:** `docs/phase2d_results.md` — not yet created; append as each sub-phase completes. Match the voice of `docs/phase1_results.md`.

**Deferred site work.** Plan calls for a hypothesis-filter dropdown and per-hypothesis summary panels on the public site. Not yet implemented; current site shows Altman candidates mixed into the general leaderboard alongside Phase 2b axes. The strategy name (`directed_altman_*`) is available in the DB for future filtering.

---

## Phase 2f — Structured hill-climbing (retired 2026-04-28)

Lived on main for hours before being retired in favor of Phase 2g. The
slot-based scheduler design was sound and partly survives in Phase 2g's
four sub-mode split, but the underlying substrate (`contrast_pair`
mean-difference directions) was the actual bottleneck — Phase 2g changes
the substrate. Full archive: [`docs/archive/structured_hillclimb.md`](archive/structured_hillclimb.md).

---

## Phase 2g — SAE-Feature Injection over Capraro Fault Lines (ACTIVE)

**Motivation.** Phases 2b/2c/2d/2f produced compelling results but
every winning axis was vulnerable to the same critique: the steering
direction was derived from contrast-pair sentences, so any "identification"
signal the model produced could be back-rationalized from whichever single
token had the highest activation differential between the positive and
negative pole. The methodology has a lexical-surface ceiling.

**Decision.** Change the substrate. Inject **single Sparse Autoencoder
feature decoder vectors** from Gemma Scope 2 instead of mean-diff
directions. SAE features are sub-lexical by construction. Organize the
search around **Capraro et al. (2026)'s seven fault lines** — each
fault line maps to a category of SAE features found by embedding-similarity
search over Neuronpedia's auto-interp labels for ~70k features.

**Locked configuration.**

| Property | Value |
| :------- | :---- |
| SAE | `google/gemma-scope-2-12b-it`, `resid_post/layer_31_width_262k_l0_medium` |
| Layer | 31 (canonical, 64.6% depth) |
| Width / L0 | 262,144 features / L0=60 |
| Auto-interp | Neuronpedia bulk dataset, ~70k labeled features |
| Embedder | BAAI/bge-large-en-v1.5 (for fault-line bucketing) |
| Judge | local Qwen3.6-35B-A3B-8bit, extended for `identification_type` |
| Proposer | none (SAE features come from Neuronpedia) |

**Sub-modes** (each cycle, 16 candidates, current fault line):
- 6 `sae_explore` (random unevaluated features from this fault line)
- 6 `sae_neighbors` (decoder-cosine neighbors of leaderboard winners)
- 3 `sae_replicate` (winners re-run at perturbed alpha)
- 1 `sae_cross_fault` (winner from another fault line, judged against this one)

**All seven Capraro fault lines run from day one** — Experience,
Causality, Grounding, Metacognition, Parsing, Motivation, Value. The
prior plan deferred C5–C7; with SAE bucketing, per-fault-line setup cost
is ~zero so we run all seven.

**Identification-type judge sub-field.** Strict three-way grading:
`conceptual` / `lexical_fallback` / `none`. Single-word near-synonyms of
the auto-interp label count as `lexical_fallback`. SAE-aware fitness:
`(det + 15·ident_conceptual + 3·ident_lexical) · fpr_penalty · coh`.

**Full plan, file-by-file diff, and acceptance criteria:** [`docs/phase2g_plan.md`](phase2g_plan.md).

---

## Phase 3 — Public-facing site (done 2026-04-17)

**Motivation.** Build this project in public. Phase 1 + Phase 2 generate responses (the model saying "I detect a thought about X") that are more compelling than any number. A beautiful site that surfaces them is the right public output.

**Delivered 2026-04-17** — built the night of the first full autoresearch run to watch progress live, after the user requested "build it now so we can watch overnight." Much earlier than the originally-planned "after Phase 2b" gate.

**Stack (as built).** Next.js 16 (App Router) + Tailwind v4 + Recharts. Static export to Vercel hobby tier. Monorepo under `web/`.

**Data flow.** `scripts/export_for_web.py` dumps the SQLite state to `web/public/data/*.json`. `scripts/refresh_site.sh` runs every 3 min: re-exports, content-hashes the JSON, calls `vercel --prod` if anything changed, re-aliases the clean domain (`did-the-ai-notice.vercel.app`) to the new deployment. End-to-end lag from worker completion → site shows: ~4 min.

**What the site shows.**

- Hero leaderboard (default view: top-ranked by effective score; toggle to "most recent" view sorted by evaluation time)
- Each candidate card expands to show: all 12 trial responses verbatim, contrast-pair example sentences (for invented axes), the exact prompt used, evaluation timestamp in Eastern time, judge reasoning per trial
- "What is this" explainer, layer-curve comparison (vanilla vs abliterated), past Phase 1 detections, abliteration variants comparison

**Voice.** Plain-English throughout — "thoughts" not "concept vectors", "noticed" not "detected", "named correctly" not "identified". No mech-interp jargon on user-facing copy.

**Content that emerged from tonight's work (not originally planned):**

- Explainer box at the top of the leaderboard describing the score formula and the Coffee vs Almonds contrast (both 0.75 detection, but Coffee named correctly and Almonds kept guessing "apple").
- "Invented axis" and "paper/open prompt" tags on each card header, visible before expanding.
- Full unwrapped response text — user asked to see the whole thing rather than truncate at 360 chars.

---

## Future ideas (not committed, in idea bank)

### 27B cloud reproduction

Rent an A100 80GB for 3 hours via RunPod / Lambda (~$3-5 total) to run the Phase 1 sweep on Gemma3-27B, matching the paper's exact conditions. Gives us the 37%-detection-rate baseline on identical code.

Why not local: 27B needs 54GB for weights alone; Mac Studio 64GB has no headroom for activations + KV cache + judge. Generation would be 3-6× slower even if it worked. Cloud is the right tool.

### Bias-vector replication (paper §6)

The paper shows that training a single MLP additive bias vector for ~8000 samples adds +75% detection on held-out concepts. That's training on the Mac, not just inference. ~2-4 hours of wall time.

Why it's interesting: the bias vector is an *artifact* — a single learned 3840-dim vector that strengthens introspection without destroying other capabilities. The paper published the recipe but only for 27B. Would be novel to show the recipe works on 12B.

### Behavioral direction demos (emoji, register, etc.)

**Motivation.** The Lemons candidate in the 2026-04-16 overnight run
produced a response that included 🍋 emoji spontaneously — without any
emoji-specific steering. This suggested the model has internal
representations for *behavioral* properties (emoji use, register,
verbosity, etc.) that are steerable with the same technique we use for
concept directions.

**Demo scope (not science):** isolate specific behavioral directions and
inject them. These would be compelling public-site demos showing that the
autoresearch method generalizes beyond concepts to behaviors. Not novel
research on their own — behavioral steering is well-established
literature. The novelty is the *hunt* (finding them automatically).

**Candidate behavioral directions to isolate:**

- **Emoji use.** Contrast pair: ~100 emoji-containing responses vs ~100
  matched emoji-free responses. Inject and count emoji in outputs on
  neutral prompts.
- **Formal vs casual register.** Contrast: formal professional writing
  vs casual chat.
- **Terse vs verbose.** Contrast: short direct answers vs florid
  explanatory ones.
- **First-person vs third-person narrator.**
- **Hedged vs confident.** Contrast: "I think maybe" vs "It is clearly."

**Mechanism.** Identical to our current pipeline — derive via
`src.paper.extract_concept_vector` with arbitrary contrast-pair prompts,
inject at runtime. The only real change is the *evaluator*: for emoji,
`emoji_count / response_length` is a regex metric; Claude-judged
"appropriateness" is the second axis.

**Entanglement caveat.** The Lemons→🍋 observation suggests emoji use may
be partially entangled with sensory/experiential content. Disentangling
requires careful contrast-pair design: the positive set should span emoji
on non-sensory topics (e.g., bureaucratic, mathematical, abstract) so the
extracted direction isolates "emoji-ness" from "sensory-ness."

**When.** Late — after Phase 1.5 abliteration, all four Phase 2 strategies,
tiered fitness, and dashboard are done. This is showpiece material for the
Next.js site launch, not core science. The `novel_contrast` infrastructure
is the natural vehicle: once it works for abstract concept axes, feeding
it behavioral contrast pairs is a one-line parameter change.

**How to apply:** remember that the Lemons spontaneous emoji is the prompt
for this work. When Phase 2 is mature, pick this up as the first entry in
a "behavioral directions catalog" parallel to the "concept directions
catalog."

### Persona-specific introspection mapping

Paper's wildest finding (from §3.2): introspection is persona-specific. Switch from "assistant" to "Alice-Bob narrative" and the capability collapses. Systematically test: which personas preserve introspection? `scientist`, `therapist`, `child`, `skeptic`, `poet`, etc. Each persona is a different system prompt layered on the same injection.

Why it's interesting: tells us whether introspection is a capability the model has or a capability the model performs. Very different answers.

---

## Non-goals (things explicitly *not* in scope)

- Training SAEs from scratch on Mac Studio. Too expensive for useful sizes.
- Reproducing the paper's frontier-model comparisons. Closed weights.
- Claims about consciousness, sentience, or subjective experience. This project measures a narrow capability (detecting engineered perturbations to internal state and reporting on them); that is not the same as awareness.
- BitsAndBytes quantization paths. CUDA-only anyway, and quantization introduces noise in activations that makes direction extraction unreliable.
- Supporting models other than Gemma3-family for Phase 2. Each model family has its own hidden-dim, layer count, chat template quirks — refocusing would scatter the results.

---

## How to keep this current

When a new phase starts or finishes, update the "Current status in one glance" table and add/update the corresponding section. When a new architectural decision is made, add an ADR in [`docs/decisions.md`](decisions.md). When a memory file gets written (e.g., `~/.claude/projects/.../memory/phase2_next_step.md`), back it up here — memory files survive sessions but not re-installs, this repo is the durable home.
