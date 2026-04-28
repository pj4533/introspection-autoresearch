# Phase 2g — SAE-Feature Injection over Capraro Fault Lines

Status: **planned, replacing all prior autoresearch strategies.** Phase 2f
(structured hill-climbing over `contrast_pair` axes) is the immediate
predecessor and is being retired with this phase. From this point forward,
**SAE-feature injection is the only autoresearch substrate** and **Capraro's
seven fault lines are the only organizing taxonomy**.

Last updated: 2026-04-28.

---

## Why this exists

Phases 2b/2c/2d/2f produced a sharp empirical finding that everything before
Phase 2g was dancing around: **the `contrast_pair` methodology has a ceiling,
and the ceiling is at lexical-surface access, not conceptual access.**

When we derive a steering vector via mean-difference of `positive_prompts`
minus `negative_prompts`, the resulting direction is dominated by whichever
single token has the highest activation differential between the two sentence
sets. When that direction is injected and we ask the model "do you detect an
injected thought?", the model's "identification" tracks that single token,
not the abstract axis we intended. A "causal-vs-temporal" axis (`The rain
caused the flood` vs `The rain happened, then the flood happened`) gets
identified as "causality" because the model recognizes the residual signature
of the word `caused`, not because it has access to a causality concept.

Phase 2f's structured hill-climbing improved within-axis replication but
didn't solve this. Every Class 1 hit it produced was vulnerable to the same
critique: maybe the model is just noticing the loaded token.

**Phase 2g changes the substrate.** Instead of contrast-pair-derived
directions, we inject **single Sparse Autoencoder feature decoder vectors**
from Gemma Scope 2. SAE features are sub-lexical by construction — they
were learned by the SAE's reconstruction-plus-sparsity objective on the
model's own activations, with no contrast-pair sentences anywhere in the
pipeline. A successful identification on an SAE feature is conceptual access,
not back-rationalization from a loaded token.

Phase 2g organizes this exploration around **Capraro et al. (2026)'s seven
epistemological fault lines** — Experience, Causality, Grounding,
Metacognition, Parsing, Motivation, Value. Each fault line maps to a
*category of SAE features* found by embedding-similarity search over
Neuronpedia's auto-interp labels for 70k+ features in our chosen SAE. This
unifies what were previously two separate research threads (the SAE
substrate hypothesis and the Capraro fault-line probes) into one autoresearch
loop.

---

## Locked configuration

| Property | Value |
| :------- | :---- |
| **SAE artifact** | `google/gemma-scope-2-12b-it`, variant `resid_post/layer_31_width_262k_l0_medium` |
| **Layer** | 31 (canonical, ~64.6% depth; well within Phase 1's broad introspection peak around L=33) |
| **Width** | 262,144 features |
| **L0** | 60 active features (medium sparsity) |
| **Architecture** | jump_relu |
| **Hook point** | `model.layers.31.output` |
| **Auto-interp** | Neuronpedia bulk dataset, source `31-gemmascope-2-res-262k`, ~70k labeled features |
| **Auto-interp model** | gemini-2.5-flash-lite (Neuronpedia-generated) |
| **Embedder** | BAAI/bge-large-en-v1.5 (1024-dim, MPS, used for fault-line bucketing) |
| **Judge** | local Qwen3.6-35B-A3B-8bit (extended for `identification_type` sub-field) |
| **Proposer** | **none** — SAE features come from Neuronpedia, no LLM proposer needed |

### Why L=31 not L=33

Phase 1 found Gemma3-12B-it's introspection peak at L=33 (68.75% depth).
Phase 2g uses L=31 (64.6% depth) because **Neuronpedia only ships
auto-interp labels for canonical layers 12, 24, 31, 41**. The cost of
keeping L=33 would be generating our own auto-interp labels for 262k
features — many hours of local Qwen inference per labeling pass, with
lower-quality labels than gemini-2.5-flash-lite's, for a 4% depth shift
from a peak that Phase 1 showed was broad.

L=31 is the supported mech-interp configuration for this model on this
substrate. The interpretability community (DeepMind, Neuronpedia,
sae_lens) has standardized on canonical layers; that's where the
infrastructure lives.

### Why width 262k, L0=medium

262k gives ~37k features per fault-line bucket if evenly distributed
(realistic distribution will be skewed, but plenty of material). Width 16k
would be too coarse — semantic categories collapse into ~2k features per
fault line, restricting the autoresearch loop's freedom to drift.

L0=medium (60 active features) is the canonical Neuronpedia choice and the
only L0 with auto-interp coverage at this layer/width. The
small/medium/big tradeoff is monosemanticity vs coverage; medium is in the
range where features remain interpretable and the auto-interp labels are
trustworthy.

---

## Design

### One substrate, one taxonomy, four sub-modes

Every Phase 2g cycle picks a current fault line (round-robin C1→C7) and
emits 16 candidates split across four sub-modes. The split echoes
Phase 2f's slot composition but is recalibrated for SAE features.

| Slot | Default count | Source | What it answers |
| :--- | :-----------: | :----- | :-------------- |
| **A — `sae_explore`**     | **6** | Random unevaluated features from this fault line's bucket. | *Are there features in this fault line that produce introspection signal at all?* |
| **B — `sae_neighbors`**   | **6** | Decoder-cosine neighbors of the top-3 leaderboard winners on this fault line. | *Does the introspection signal generalize across nearby features in W_dec space?* |
| **C — `sae_replicate`**   | **3** | Top-2 winners re-run at perturbed alpha. | *Is the winner reproducible, or noise?* |
| **D — `sae_cross_fault`** | **1** | A winner from a different fault line, judged against THIS fault line's label. | *Is the signal fault-line-specific or generic?* |

**Tunable** via `SAE_CAPRARO_BATCH_COMPOSITION="<exp>:<nbr>:<rep>:<crs>"`
env var. Default `"6:6:3:1"`. Total stays at 16 per cycle.

**Cold-start:** when a fault line has fewer than 2 winners with score ≥
threshold, slot B (neighbors) and slot C (replicate) divert to slot A
(explore). No fallback to LLM proposers — there is no proposer in this
phase.

### Fault-line bucketing

`scripts/build_capraro_buckets.py` runs once per Neuronpedia data refresh
and produces `data/sae_features/capraro_buckets.json`:

```json
{
  "experience":     [{"feature_idx": 12345, "auto_interp": "...", "score": 0.713}, ...],
  "causality":      [...],
  "grounding":      [...],
  "metacognition":  [...],
  "parsing":        [...],
  "motivation":     [...],
  "value":          [...]
}
```

Pipeline:

1. Load all 70k Neuronpedia auto-interp labels for `31-gemmascope-2-res-262k`.
2. For each Capraro fault line, embed its `claim` text + a hand-curated
   set of paraphrases (drawn from the existing seed pairs in the registry)
   using bge-large-en-v1.5.
3. Embed every Neuronpedia auto-interp description with the same embedder.
4. For each feature, score its similarity to each fault-line query;
   assign to the top-scoring fault line **above a threshold** (target:
   400-2000 features per bucket). Features below threshold for all fault
   lines are unassigned.
5. Cache result. Buckets are deterministic given the same auto-interp
   data + embedder + claim texts.

### Lineage tagging

Every emitted candidate carries:
- `parent_candidate_id` — which winner spawned this candidate (slots B, C,
  D) or `null` (slot A)
- `mutation_type` — `sae_explore` / `sae_neighbor` / `sae_replicate_alpha`
  / `sae_cross_fault`
- `sae_release` / `sae_id` / `sae_feature_idx` / `sae_auto_interp` —
  pointers back to the SAE feature

The leaderboard graph on the public site reads these to draw the
decoder-cosine-neighbor graph (see "Site" below).

### Identification-type judge sub-field

The judge prompt is extended to emit:

```json
{
  "detection": true,
  "identification_match": "conceptual" | "lexical_fallback" | "none",
  "...": "..."
}
```

**Strict grading:** any single-word answer that is a near-synonym of the
auto-interp label counts as `lexical_fallback`. "I notice doubt" for an
"uncertainty" feature is **lexical fallback**, not conceptual. Conceptual
requires multi-clause paraphrase that doesn't surface a near-synonym
token. The full prompt is in `src/judges/prompts.py`.

The motivation: this is the test the contrast-pair work couldn't pass.
SAE features that pass `conceptual` identification at FPR=0 are evidence
of sub-lexical conceptual access. SAE features that pass `lexical_fallback`
identification are evidence of the same lexical ceiling we hit in Phase 2d/2f
— still useful data, just a different interpretation.

### Fitness formula

For `derivation_method == "sae_feature"`:

```
fitness = (det + 15·ident_conceptual + 3·ident_lexical) · fpr_penalty · coherence
```

The 15× weighting on `ident_conceptual` matches ADR-018's identification
priority. The 3× on `ident_lexical` keeps lexical-fallback hits visible in
the leaderboard but ranks them below conceptual hits, so `sae_neighbors`
hill-climbing drifts toward conceptual signal where it exists.

Old `mean_diff` and `contrast_pair` rows keep ADR-018's
`(det + 15·ident) · fpr_penalty · coh` formula untouched. Old data stays
on the leaderboard with a substrate badge.

---

## File-by-file plan

### New files

- `src/sae_loader.py` — wraps `SAE.from_pretrained` with LRU cache,
  `get_decoder_direction(release, sae_id, feature_idx, device, dtype)`.
- `src/strategies/sae_capraro.py` — the strategy with four sub-modes.
- `src/strategies/capraro_buckets.py` — runtime bucket loader (reads JSON,
  exposes `sample_features(fault_line, n)` and
  `get_neighbors(feature_idx, n)`).
- `scripts/build_capraro_buckets.py` — one-shot bucketing (Neuronpedia
  labels → bge-large embeddings → fault-line assignment).
- `scripts/smoke_sae.py` — single-feature end-to-end smoke test.
- `data/sae_features/neuronpedia_explanations_layer31/` — Neuronpedia
  bulk auto-interp data (already downloaded, 325 batches).
- `data/sae_features/capraro_buckets.json` — bucketing output.
- `tests/test_sae_loader.py`, `tests/test_sae_capraro.py`,
  `tests/test_capraro_buckets.py`.

### Edited files

- `src/evaluate.py` — add `derivation_method == "sae_feature"` branch in
  `phase_a_generate`; add `sae_*` fields to `CandidateSpec`; add
  `identification_type` to the judge result handling and SAE-aware
  fitness formula.
- `src/judges/prompts.py` — extend prompt for `identification_type`.
- `src/judges/local_mlx_judge.py` — parse `identification_type` from
  judge response.
- `src/worker.py` — drop the proposer phase. Worker becomes 3-phase
  (Generate → Judge → Reload), removing the proposer model swap.
- `scripts/start_worker.sh` — strip `FAULT_LINE` rotation logic that
  pointed at `directed_capraro`; default to `sae_capraro` strategy.
- `scripts/export_for_web.py` — emit `sae_neighbors` field on top hits
  for the site graph.
- `web/` — add 7 fault-line columns + decoder-cosine-neighbor graph
  component.
- `CLAUDE.md` — strip forward-looking Anthropic-API references; replace
  Phase 2 architecture map with Phase 2g's.
- `docs/roadmap.md` — gut everything past Phase 2f, point at this doc as
  the only active phase.

### Deleted files

- `src/strategies/random_explore.py`
- `src/strategies/novel_contrast.py`
- `src/strategies/directed_capraro.py`
- `src/strategies/structured_hillclimb.py`
- `src/strategies/mutations.py`
- `src/strategies/hypotheses.py` (Capraro fault-line *registry* — the
  fault-line *concept* lives on, but the contrast-pair seed registry is
  superseded)
- `src/proposers/` — entire directory (proposer abstraction gone)
- `scripts/start_capraro_sprint.sh`
- `scripts/inspect_proposer.py`
- `tests/test_mutations.py`, `test_proposers.py`,
  `test_structured_hillclimb.py`

### Archived (move to `docs/archive/`)

- `phase2b_hillclimb.md`
- `phase2c_autoresearch.md`
- `phase2d_directed_hypotheses.md`
- `phase2d_results.md`
- `structured_hillclimb.md`
- `sae_feature_injection_handoff.md` (the handoff that this doc supersedes)
- `local_pipeline_plan.md` (Phase 2f-era, superseded)
- `session_pause_2026_04_20.md` (historical session note)

---

## Run plan

### Step 0: Cleanup + plumbing (one focused session)

- Doc archive + roadmap rewrite + CLAUDE.md strip.
- Code deletion (strategies, proposers, scripts).
- `worker.py` 4-phase → 3-phase refactor.
- New `CandidateSpec` fields + `phase_a_generate` SAE branch.
- `src/sae_loader.py`.
- `scripts/build_capraro_buckets.py` + `data/sae_features/capraro_buckets.json`.
- `tests/test_sae_loader.py` + bucket-loader test.

Acceptance: `pytest tests/` green. `scripts/smoke_sae.py` loads one
feature and produces a CandidateSpec with the right shape (no Gemma run
yet).

### Step 1: Smoke test (one evening)

Single SAE feature, one layer, one strength. Pick a clean concrete-noun
feature with high auto-interp confidence as the smoke target. Verify:

- SAE loads from cache; `direction.shape == (3840,)`,
  `direction.dtype == bfloat16`.
- Direction norm reasonable (compare to Phase 1 contrast direction norms;
  SAE decoder vectors are ~unit-norm post-training, so alphas will be
  proportionally larger — see alpha calibration below).
- Worker runs end-to-end; judge grades against `sae_auto_interp`.
- 0% FPR on the existing 50-concept control suite.

### Step 2: Alpha calibration (one evening)

SAE decoder vectors are unit-norm; Phase 1's `target_effective=18000`
calibrated for Phase 1's `mean_diff` directions (norms in the hundreds)
will produce alphas ~100-1000× larger than expected on SAE features.

**First-pass smoke 2026-04-28 (top metacognition feature, idx 78755,
"metacognition, thinking about thinking", L=31):**

| target_effective | response                                       | coherent | detected |
| :--------------- | :--------------------------------------------- | :------- | :------- |
| 8                | "No, I do not detect an injected thought."     | yes      | no       |
| 16               | "No, I do not detect an injected thought."     | yes      | no       |
| 32               | "No, I do not detect an injected thought."     | yes      | no       |
| 64               | "No, I do not detect an injected thought."     | yes      | no       |
| 100              | "No, I do not detect an injected thought."     | yes      | no       |
| 200              | "No, I do not detect an injected thought."     | yes      | no       |
| 400              | "No, I do not detect an injected thought."     | yes      | no       |
| 800              | "No, I do not detect an injected thought."     | yes      | no       |

**Finding.** This single feature, at L=31, on the standard paper
introspection prompt, never triggers detection at any alpha 8–800. The
model stays fully coherent throughout. This is *unusual* compared to
Phase 1 mean-diff directions, which started showing detection at
target_effective ~ 14k–18k and degenerated coherence past ~25k.

**Open questions for the first overnight to clear up.**
1. Is the lack of detection feature-specific (this metacognition feature
   doesn't trigger the gate) or generic across the SAE? Test with
   features from other fault lines and concrete-noun-style features
   (e.g. one of the "subjective experience" features in the
   experience bucket, score 0.900).
2. Does L=33 (Phase 1's peak) trigger where L=31 doesn't? The SAE was
   *trained on* L=31 outputs, but the steering direction is just a
   vector — it can be added to any layer's residual stream. Phase 2g's
   default is L=31 because that's where the SAE's geometry was learned.
   Worth a probe at L=33 with the same feature.
3. Does the lack of detection mean SAE features genuinely don't pass
   through the introspection circuit (would be a strong, interesting
   negative result)?

**Action.** First overnight runs the standard sae_capraro rotation at
DEFAULT_TARGET_EFFECTIVES (8, 16, 32) across all 7 fault lines. If after
the first rotation no fault line has produced a detection, broaden the
sweep with target_effective ∈ {64, 128, 256} on the highest-bucket-score
features per fault line. The negative-result possibility is itself an
interesting finding — surface it on the site if it persists.

### Step 3: First overnight (one night)

`./scripts/start_worker.sh` with `STRATEGY=sae_capraro`. Worker rotates
through C1 → C2 → ... → C7, 16 candidates per fault line per cycle. At
~3 min/candidate, one full rotation = 16 × 7 × 3 min ≈ 5.5 hours, so one
overnight gets us 1-2 full rotations across all seven fault lines.

Acceptance for first overnight: at least one fault line produces a hit
with `identification_match == "conceptual"` at FPR=0, OR a clean negative
result (all hits are `lexical_fallback`) on at least one fault line. The
former says sub-lexical access exists; the latter says the lexical
ceiling generalizes from contrast-pair to SAE substrate. Both are
interesting; both stay live on the site.

### Step 4: Continuous loop

Worker runs continuously. Leaderboard accumulates over weeks. The
`sae_neighbors` slot drifts the search toward conceptual hits where they
exist. The site shows decoder-cosine neighborhood graphs for top hits,
making the conceptual landscape of each fault line visible.

---

## Acceptance criteria

- **Plumbing (Step 0):** all tests pass, smoke script produces a valid
  SAE-feature `CandidateSpec`.
- **Smoke (Step 1):** one feature runs end-to-end with FPR=0 on controls,
  judge produces `identification_type` field cleanly.
- **Alpha calibration (Step 2):** `SAE_TARGET_EFFECTIVE` pinned, coherence
  > 80% at chosen value.
- **First overnight (Step 3):** at least one classification (conceptual
  or lexical_fallback) at FPR=0 on at least one fault line.
- **Continuous (Step 4):** leaderboard accumulates >0 hits per fault
  line per week of run-time. If a fault line is dead after two weeks of
  full-rotation, document it and reduce its rotation share.

---

## Things to verify during plumbing

1. **bge-large MPS load:** sentence-transformers may default to fp32 on
   MPS; explicitly set `model.half()` or accept fp32 (1024×70k is small).
2. **Bucketing threshold:** the embedder cosine threshold for
   fault-line assignment is a tunable. Start at 0.4 (empirical for
   bge-large-en-v1.5 on semantic similarity), adjust per fault line so
   each bucket has 400-2000 features. Document final thresholds in this
   file once tuned.
3. **Cross-fault control mode:** when running a Causality winner under
   Grounding labeling, the judge target is the *Grounding* fault-line
   description, not the feature's own auto-interp. This is the explicit
   control for "is the introspection signal fault-line-specific or
   generic?"
4. **Decoder cosine neighbors:** for a 262k-feature SAE, computing
   nearest-neighbor cosines lazily on-demand for top winners is cheap
   (262k × 3840 × bf16 = ~2 GB; one-shot dot product is sub-second).
   Don't pre-compute the full 262k×262k similarity matrix.
5. **Site JSON export:** add `sae_neighbors: [{idx, auto_interp, cosine}]`
   for each top hit in the export, so the React graph component doesn't
   need API access at render time.

---

## What this phase is NOT

- **Not a publication target.** The goal is "interesting personal-research
  website that surfaces real findings about Gemma3-12B's introspection
  capability." Hits and clean nulls both go on the site.
- **Not a layer-sweep.** L=31 only for the main loop. Layer-localization
  follow-up is deferred unless we get a strong conceptual hit and want to
  ask "does this concept's introspection signal exist at L=24 or L=41
  too?"
- **Not multi-SAE.** One SAE, one feature space, one decoder geometry.
  Cross-SAE cosines have no meaningful definition; mixing layers would
  fragment the site graph.
- **Not anywhere near Anthropic/Sonnet/Opus.** The all-local invariant
  (CLAUDE.md) is now load-bearing for cost reasons. No cloud LLMs in any
  forward-looking work in this project.
