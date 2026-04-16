# Roadmap

Everything this project has done, is doing, and plans to do — phase by phase, with rationale. This document is the source of truth for the project's trajectory; if anything important lives only in chat logs or ephemeral plan files, it's a bug.

Last updated: 2026-04-16.

---

## Current status in one glance

| Phase | Scope | Status |
|---|---|---|
| **1: Reproduction** | Reproduce the core introspection-detection mechanism from Macar et al. (2026) on Gemma3-12B-it locally | ✅ Done (2026-04-16) |
| **2a: Autoresearch MVP** | Build the worker + researcher + fitness loop with `random_explore` as the seed strategy | ✅ Scaffolded, overnight run in progress |
| **2b: Smarter strategies** | `novel_contrast`, `exploit_topk`, `crossover` — the strategies that produce *novel* findings rather than just validating the pipeline | ⏳ Planned |
| **2c: Tiered fitness + dashboard** | T1/T2/T3 fast-kill screening, Streamlit dashboard for visibility | ⏳ Planned |
| **3: Public-facing visualization** | Next.js site consuming exported JSON, deployed to Vercel, designed for non-technical audience | ⏳ Planned |
| **Future**: 27B cloud reproduction, SAE / transcoder feature analysis, bias-vector replication, persona-specific introspection mapping | | ⏳ Ideas on the shelf |

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

## Phase 2a cleanup — known issues discovered during first overnight run

### Judge target-concept bug

**Symptom.** `src/evaluate.py::evaluate_candidate` passes the held-out slot's
concept to `pipeline.run_injected` which passes it to the judge as the
"injected concept" for grading. But the actual injection is the candidate's
source concept. Result: `identification_rate` in the `fitness_scores` table
is measuring "did the model say the held-out concept's name?" instead of
"did the model say the source concept's name?"

**Impact.** `score` is unaffected (it doesn't include identification). But
the reported `identification_rate` numbers are misleading. Also correctly-
identified Phase 2 detections (e.g. Iron direction → model response "iron")
are silently recorded as `identified=0` because the judge was asked to look
for "Anxiety" instead of "Iron".

**Fix.** In `src/evaluate.py`, pass `spec.concept` (source) to the judge
rather than `c` (held-out slot). One-line change. Need to invalidate the
judge cache (bump PROMPT_TEMPLATE_VERSION or clear the cache) so already-
stored judgments get re-scored. Phase 2a data can be rescored retroactively
from the `evaluations.response` text — no model re-runs needed.

### Held-out concept set isn't randomized per candidate

**Symptom.** `evaluate_candidate(rng_seed=0)` by default, so every candidate
sees the same shuffled subset of 8 held-out concepts. Overrepresents
whichever concepts happen to be in the first 8.

**Impact.** "Anxiety keeps responding" pattern observed in first overnight
run is likely partly a test-set artifact (Anxiety is always in the 8, and
one specific random seed happens to land the model in a detection-friendly
state).

**Fix.** Seed the shuffle from a hash of the candidate ID, not a constant.
Different candidates get different 8-concept subsets. The per-concept RNG
seed inside the evaluation loop should also derive from candidate_id +
concept to decorrelate from other candidates.

---

## Phase 2b — Smart strategies (planned, not started)

This is where the project's *novel* contribution lives. Each of these strategies produces candidate steering directions by a different mechanism; they plug into the existing `src/researcher.py` plugin interface.

### `novel_contrast` — "concepts that don't have names" (next to build)

**Motivation.** Phase 2a's `random_explore` only tries directions derived from single concept words drawn from a human dictionary (Bread, Peace, Ocean, ...). This misses directions the model represents *internally* for which no single English word exists.

**Mechanism.**

1. Use `claude-agent-sdk` (subscription OAuth, no API billing) to generate **contrast pairs** for abstract axes — not single words. Example pairs:
   - `("pondering", "asserting")` — the hesitation-vs-commitment axis
   - `("deciding carefully", "acting on instinct")`
   - `("certainty", "doubt")`
   - `("warm recollection", "clinical recall")`
2. Derive a direction from the pair using `src.paper.extract_concept_vector` (positive=pair[0], negative=pair[1]). This is distinct from `extract_concept_vector_with_baseline` (the single-concept approach).
3. The resulting direction lives *between* two reference points, representing an axis the model has internal geometry for but humans don't have single-word vocabulary for.
4. Evaluate the same way `random_explore` candidates are evaluated — fitness over 8 held-out concepts + 4 controls.

**Worker changes required.** `src/evaluate.py::evaluate_candidate` currently calls `pipeline.derive(concept=spec.concept, layer_idx=...)`. For `contrast_pair` candidates, it needs to dispatch to `extract_concept_vector(positive_prompts=..., negative_prompts=...)`. Small branch in the derivation path.

**Acceptance.** At least one discovered `contrast_pair` direction:
- Scores fitness > 0.2 (higher than most random candidates)
- Produces clean introspective responses qualitatively distinct from the pair's named components
- Cannot be satisfactorily labeled with a single English word

**Risk.** Most Claude-generated pairs will produce directions that are just noisy or map cleanly onto a named concept anyway. The signal should emerge after ~100-500 evaluated pairs.

### `exploit_topk` — sample near known-good directions

**Motivation.** Answers the user's question "why aren't we just using L=33 + eff=18000 since Phase 1 showed that's best?" This strategy specifically does that.

**Mechanism.** Pull top-K candidates by fitness from the DB. For each, generate N mutations by perturbing one parameter:
- Same concept, same layer, different target_effective
- Same concept, different layer, same target_effective  
- Same (layer, target_effective), different concept
- Same concept+layer+eff, different derivation method (once we have multiple)

**Acceptance.** Over a week of running, `exploit_topk` should produce a fitness score distribution noticeably right-shifted compared to `random_explore` on the same candidate budget.

### `crossover` — combine top directions

**Motivation.** If two directions A and B both produce clean detections, maybe `(A + B) / 2` does too, or some weighted combination. Standard genetic-algorithm move.

**Mechanism.** Load two top direction tensors from disk, take linear combinations, evaluate as a new candidate.

**Acceptance.** Identify at least one crossover direction that outscores both its parents on held-out concepts.

---

## Phase 2c — Tiered fitness + dashboard (planned)

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

## Phase 3 — Public-facing site (planned, deferred until Phase 2b has produced interesting data)

**Motivation.** Build this project in public. Phase 1 + Phase 2 will generate responses (the model saying "I detect a thought about X") that are more compelling than any number. A beautiful site that surfaces them is the right public output.

**Stack.** Next.js App Router + Tailwind + shadcn/ui + Recharts + optionally Framer Motion. Static export to Vercel. Monorepo under `web/` inside this repo.

**Data flow.** Python produces SQLite and JSON exports (`scripts/export_phase1.py` is the prototype). A new `scripts/export_for_web.py` will dump filtered snapshots to `web/public/data/*.json`. Next.js reads at build time. Zero backend server.

**Content plan.**

- Landing page: hero quote (the Peace / Sugar / Avalanches responses verbatim) + dose-response sparkline + "what is this" link to `docs/plain_english.md`.
- Results page: concept × layer heatmap with drill-down to response transcripts.
- Gallery: all detection responses with concept words highlighted in the model's output.
- Catalog (Phase 2b output): the top-scoring discovered directions, each on its own page.
- ELI5 / about page: non-technical explainer (`docs/plain_english.md` as source).

**Decision point.** We don't start Next.js until Phase 2b has produced ~10+ score-nonzero candidates including some `novel_contrast` finds. Building a polished site around thin data undersells the project.

---

## Future ideas (not committed, in idea bank)

### 27B cloud reproduction

Rent an A100 80GB for 3 hours via RunPod / Lambda (~$3-5 total) to run the Phase 1 sweep on Gemma3-27B, matching the paper's exact conditions. Gives us the 37%-detection-rate baseline on identical code.

Why not local: 27B needs 54GB for weights alone; Mac Studio 64GB has no headroom for activations + KV cache + judge. Generation would be 3-6× slower even if it worked. Cloud is the right tool.

### SAE / transcoder feature analysis

Once Phase 2 has found interesting directions, use the Gemma Scope 2 transcoders (Google's published SAEs for Gemma3) to interpret *what individual features* compose the direction. This is the path to claims like "this direction is the `refusal-adjacent uncertainty` feature combined with the `meta-cognitive reflection` feature."

Why not now: Gemma Scope 2 is large (several GB per layer), slow to load, and the interpretation pipeline is its own project.

### Bias-vector replication (paper §6)

The paper shows that training a single MLP additive bias vector for ~8000 samples adds +75% detection on held-out concepts. That's training on the Mac, not just inference. ~2-4 hours of wall time.

Why it's interesting: the bias vector is an *artifact* — a single learned 3840-dim vector that strengthens introspection without destroying other capabilities. The paper published the recipe but only for 27B. Would be novel to show the recipe works on 12B.

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
