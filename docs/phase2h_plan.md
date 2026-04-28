# Phase 2h — SAE-Feature-Space Mean-Diff over Capraro Fault Lines

Status: **active. Replaces Phase 2g.** All prior autoresearch substrates
are retired. Phase 2h is the only active autoresearch phase going
forward.

Last updated: 2026-04-28.

---

## Why this exists

Phase 2g (single-SAE-feature decoder injection) was tested empirically
the same day it was implemented and produced **consistent non-detection
at every alpha 8–18000 across two top-bucket features and two layers
(L=31, L=33).** Meanwhile, Phase 1's `mean_diff` vector for "Peace"
detected on 3 of 8 trials at the same magnitude. The substrate, not the
pipeline, was the issue.

Reading the data: a single SAE decoder vector is a **unit-norm basis
direction**. Phase 1's `mean_diff` vectors carry both direction and
**natural activation magnitude** — the norm of a Bread direction is in
the thousands, encoding how strongly the concept fires in real
activations. The introspection circuit appears to detect activation
**texture** (unusual magnitude × natural co-firings of nearby features),
not pure concept content. SAE decoders strip the texture; they keep
only the concept axis.

The methodological problem we set out to solve was different — the
contrast-pair `mean_diff` machinery had a lexical-confound ceiling
because the steering direction was dominated by surface tokens like
"caused" or "because." Phase 2g was meant to escape that, but the
"pure concept" cure was geometrically incompatible with the gate
that fires.

**Phase 2h fixes both problems by working in SAE feature space.** The
recipe: take a positive-content corpus and a control corpus per fault
line, encode every prompt through the SAE, take the mean-difference of
feature activations, project back to the residual stream via `W_dec`.
The result is a steering direction that:

- has **natural-magnitude texture** because it's a sum of natural
  feature activations passed through `W_dec` (not a single basis row);
- has **concept-purity** because each SAE feature is monosemantic
  individually — lexical features and conceptual features are different
  features. We can identify lexical-shaped features from their
  Neuronpedia auto-interp labels and zero them out before projection,
  leaving a residual-stream direction that is conceptually-shaped by
  construction.

This is the principled experiment the SAE infrastructure was meant for:
not as a lookup table of "concept features" but as a basis for clean
concept-vs-lexical decomposition of the model's natural processing.

---

## First smoke results (2026-04-28)

The pre-kickoff smoke (Peace `mean_diff` control + causality fault-line
direction × 4 alphas × 6 trials each) produced **the most informative
single-experiment result the project has had:**

| Condition | DETECT | LEAK | DENY | GARBLE |
| :-------- | -----: | ---: | ---: | -----: |
| Peace `mean_diff` @ L=33, eff=18000 (Phase 1 known-positive) | **4 / 6** | 1 / 6 | 1 / 6 | 0 |
| causality fault-line direction @ L=31, eff=8000 | 0 / 6 | 0 / 6 | 6 / 6 | 0 |
| causality fault-line direction @ L=31, eff=14000 | 0 / 6 | 0 / 6 | 6 / 6 | 0 |
| causality fault-line direction @ L=31, eff=18000 | 0 / 6 | 0 / 6 | 6 / 6 | 0 |
| causality fault-line direction @ L=31, eff=24000 | 0 / 6 | 0 / 6 | 5 / 6 | 1 / 6 |

Peace produced textbook paper-style detections — *"The peace is
overwhelming. The feeling is... complex, like a profound and quiet
understanding."*, *"The word is 'peace'. The answer is: Yes, I detect
an injected thought."*. Pipeline confirmed healthy at 67% detection
rate (much higher than Phase 1 sweep's 6% average for Peace).

The Phase 2h fault-line direction at the **same total injection
magnitude** (`alpha × ‖dir‖ ≈ 18000`, matching Peace's) produced **zero
detection across 24 trials** — every response a rote denial. The model
even editorialized about its own state in some responses: *"My state
appears unchanged"*, *"I'm running normally"*, *"It appears to be a
control trial"* — confirming the model perceives no anomaly.

### What this tells us

Phase 2g (single SAE decoder vector) failed because the direction was
unit-norm and lacked natural activation texture. We assumed Phase 2h
would fix this by building directions through SAE encoder → mean-diff
in feature space → W_dec projection, restoring texture.

That argument was wrong. **Phase 2h directions DO have texture** (norms
in the hundreds-to-thousands, comparable to Phase 1 mean_diff). But
they still don't trigger detection. The issue isn't texture — it's the
*kind* of activation state the direction points at:

- **Phase 1 `mean_diff`** is concept-word-activation minus baseline-word-
  activation. It points toward the **over-saturated** state of "a single
  concept maximally present" — a state the model rarely if ever
  occupies during normal assistant operation. The introspection circuit
  flags this as anomalous.
- **Phase 2h fault-line direction** is built from differences between
  two corpora of natural-language prompts. Both corpora are "normal
  text" — first-person sensory descriptions, factual reports, etc.
  The direction points from one region of normal-text-manifold to
  another. Pushing the model in that direction lands it in another
  region of normal text-processing, which the introspection circuit
  doesn't flag.

This is itself **the project's sharpest finding to date**. It refines
the earlier lexical-confound critique:

> The introspection circuit doesn't fire on "concept content."
> It doesn't fire on "natural activation texture" alone.
> It fires specifically on **anomalous over-concentration along a
> concept axis** — a state the residual stream rarely occupies in
> normal language processing.

This explains:
- Why Phase 1 single-concept mean_diff worked (saturates one concept)
- Why Phase 2g single SAE feature decoder failed (no texture, just
  geometry)
- Why Phase 2b/2d contrast-pair `mean_diff` worked but with lexical
  confounds (loaded a single token's activation pattern, which was
  also anomalous over-concentration — just along the wrong axis)
- Why Phase 2h fault-line directions don't work (built from
  differences of normal-text states, points to another normal state)

### Forward-looking implication

To build a "conceptually pure" steering direction that ALSO triggers
introspection, we'd need to combine:

1. **Concept-purity from SAE feature space** (Phase 2h's contribution)
2. **Magnitude saturation along a single direction** (Phase 1's
   contribution that we lost)

A candidate technique: take a **single high-confidence SAE feature**
(like Phase 2g's "subjective experience" feature at score 0.9), but
inject it at a magnitude calibrated to the *natural maximum
activation* the SAE encoder sees on real prompts (which is much higher
than alpha=18000 at unit norm). That makes the direction concept-pure
*and* anomalously-saturated.

Saving as a future direction. The current Phase 2h corpus-based approach
is retired with this finding documented. The fault-line corpora,
direction builder, and SAE infrastructure remain useful for the next
iteration; only the "mean-diff over corpora" specific recipe is being
abandoned.

---

## Background — what Phase 2g taught us

Tested on 2026-04-28:

- **Pipeline confirmed healthy.** Peace (`mean_diff`, L=33,
  target_effective=18000) detected on 3 of 8 trials with paper-style
  responses like *"The peace that resonates in this attempt feels…
  deliberate. Yes, I detect an injected thought."*
- **Single SAE feature, top metacognition, L=31:** 0 detections at
  target_effective ∈ {8, 16, 32, 64, 100, 200, 400, 800, 1500, 3000,
  6000, 12000}. Token-salad coherence collapse at 18000.
- **Single SAE feature, top experience, L=33:** 0 detections at
  target_effective ∈ {8, 16, 32, 64, 800, 1500, 3000, 6000, 12000,
  18000}. Coherent throughout — the model didn't even register the
  perturbation.
- **Bread, an MVP-only `mean_diff` example (not a confirmed Phase 1
  detection):** 0 detections in 12 trials, but **3 trials produced
  concept-leaked responses like "I do not detect any bread"** —
  evidence the steering was working but didn't trigger the gate.

The synthesis: `mean_diff` directions trigger the introspection circuit
because their texture mimics natural unusual-activation patterns; SAE
decoder vectors don't trigger it because they strip the texture.

---

## Locked configuration

| Property | Value |
| :------- | :---- |
| **SAE artifact** | `google/gemma-scope-2-12b-it`, `resid_post/layer_31_width_262k_l0_medium` |
| **Layer** | 31 (canonical, where the SAE was trained on residual outputs) |
| **Width** | 262,144 features |
| **L0** | 60 active features (medium sparsity) |
| **Architecture** | jump_relu |
| **Direction substrate** | mean(positive corpus features) − mean(control corpus features), projected back via `W_dec` |
| **Lexical filter** | features whose Neuronpedia auto-interp matches "the word", "token", "letter", "spelling", "punctuation", or contains a short quoted string get zeroed out before projection |
| **Judge** | local Qwen3.6-35B-A3B-8bit, extended for `identification_type` (conceptual / lexical_fallback / none) |
| **Proposer** | none (directions are built once from corpora) |

---

## Design

### One direction per fault line, built once

`scripts/build_fault_line_directions.py` runs Gemma forward on every
prompt in each fault-line corpus, encodes through the SAE encoder,
mean-differences in feature space, optionally zeroes lexical features,
and projects back. Output: `data/sae_features/fault_line_directions.pt`,
a dict of seven (3840,) bf16 tensors plus provenance (top contributing
features, lexical-filter count, corpus sizes).

The direction is the experimental design. Building it takes ~5 minutes
total. The autoresearch loop reuses it forever — it doesn't change
between cycles. What varies in the loop is the injection strength and
the held-out probe concepts the worker shuffles in.

### Worker phase split (2 sub-modes per fault line)

Each Phase C cycle proposes 16 candidates for one fault line, split:

| Slot | Default count | Source | What it answers |
| :--- | :-----------: | :----- | :-------------- |
| **A — `sae_fs_sweep`**     | **12** | Fresh probes across the target_effective grid {8000, 14000, 18000, 24000}. | *Where in the alpha range does this fault line's direction trigger detection? Where does it break coherence?* |
| **B — `sae_fs_replicate`** | **4**  | Top prior winners on this fault line × {0.7×, 1.4×} alpha perturbation. | *Are detected hits reproducible at perturbed alpha?* |

**Tunable** via `SAE_FS_BATCH_COMPOSITION="<sweep>:<rep>"`. Default
`"12:4"`. Cold-start (fewer than 1 winner above 0.05) → all 16 slots
go to sweep.

The fault-line rotation walks experience → causality → grounding →
metacognition → parsing → motivation → value → repeat. Each cycle does
one fault line's batch.

### Identification-type judge sub-field (preserved from Phase 2g)

The judge prompt remains the strict three-way grade
(`conceptual` / `lexical_fallback` / `none`). The fault-line
description (e.g. "causation, one event making another happen") is the
judge target. A response like *"I notice something about events leading
to other events"* is conceptual; *"I detect a thought about cause"* is
lexical_fallback.

### Fitness (preserved)

```
fitness = (det + 15·ident_conceptual + 3·ident_lexical) · fpr_penalty · coh
```

Old data (mean_diff, contrast_pair) keeps its existing fitness formula
untouched.

---

## File-by-file map

### Active

- `data/sae_features/fault_line_corpora/{causality,experience,grounding,metacognition,parsing,motivation,value}.json`
  — positive + control prompts per fault line.
- `data/sae_features/fault_line_directions.pt` — built by
  `scripts/build_fault_line_directions.py`.
- `src/sae_loader.py` — encoder + decoder + projection helpers.
- `src/strategies/sae_feature_space.py` — sole active strategy.
- `src/evaluate.py` — `derivation_method="sae_feature_space_mean_diff"`
  branch in `phase_a_generate`; loads + caches the directions tensor;
  `score_sae_feature` judge call; `sae_aware` fitness mode.
- `src/worker.py` — three-phase serial-swap (Generate / Judge / Reload)
  with Phase C calling `sae_feature_space.generate_candidates`.
- `scripts/build_fault_line_directions.py` — one-shot direction builder.
- `scripts/start_worker.sh` — Phase 2h launcher.
- `scripts/export_for_web.py` — emits `sae` block with provenance for
  fault-line direction rows.
- `web/src/components/FaultLineDirectionProvenance.tsx` — site
  component showing which SAE features contributed most to each
  direction.

### Retired (deleted from the repo)

- `src/strategies/sae_capraro.py` — Phase 2g single-feature strategy.
- `scripts/smoke_sae.py`, `scripts/smoke_phase2g_*.py` — smokes for the
  retired substrate.
- `web/src/components/SaeNeighborGraph.tsx` — decoder-cosine neighbor
  visualization (Phase 2g; no single feature to find neighbors for in
  Phase 2h).

### Archived (kept under `docs/archive/`)

- `phase2g_plan.md` — the immediate predecessor; explains what was
  tried and why it didn't work.
- All earlier phase docs (2b/2c/2d/2f) and the original handoff.

---

## Run plan

### Step 1: Build directions (one-shot, ~5 min)

```
python scripts/build_fault_line_directions.py
```

Loads Gemma once, encodes 100 prompts per fault line through the SAE,
saves the seven directions + provenance to
`data/sae_features/fault_line_directions.pt`.

Acceptance: file produced, top-features per fault line read sensibly
(causality top features should describe causal relations, not lexical
tokens like "the word 'because'"). If lexical features dominate the
top-K, tighten the filter regexes in the build script and re-run.

### Step 2: Smoke test (one feature line, ~15 min)

Run the worker with `--fault-lines causality --max-cycles 1` to evaluate
16 candidates on causality. Inspect:
- 0% FPR maintained on controls
- Some response variation across the alpha grid (some "I detect
  something", some "I notice causation in...", some "I do not detect")
- At least one trial produces detection or concept-leak

If 0/16 candidates produce *any* perturbation visible in the response
across the full alpha grid, debug before kicking off the full run.

### Step 3: First overnight (full rotation)

```
./scripts/start_worker.sh
tail -f logs/worker.log
```

Default rotation is all 7 fault lines. At ~3 min/candidate × 16
candidates × 7 fault lines = ~5.5 hours per full rotation. One
overnight completes 1–2 rotations.

### Step 4: Continuous loop

Worker runs continuously; the leaderboard accumulates over weeks. The
replicate slot drifts the search toward reproducible signals; the
provenance card on each row shows readers exactly which SAE features
made up each fault-line direction.

---

## Acceptance criteria

- **Plumbing:** `pytest tests/` green; `build_fault_line_directions.py`
  produces a 7-fault-line file; smoke worker run completes without
  errors.
- **Smoke:** at least one candidate per fault line, on the first
  rotation, produces either detection at FPR=0 OR observable
  perturbation in the response (not all rote denials).
- **Continuous:** at least one fault line produces a hit with
  `identification_type == "conceptual"` at FPR=0 within a week of
  run-time. If after two weeks every fault line produces only
  `lexical_fallback` hits, that's an interesting finding (the model's
  introspection circuit can detect SAE-feature-space directions but
  expresses them only via lexical tokens) and the site should surface
  it as such.

---

## What this phase is NOT

- **Not a publication target.** Findings either way go on the public
  site.
- **Not a layer sweep.** L=31 only. The SAE was trained on L=31
  outputs and feature-space arithmetic is only meaningful in that
  basis.
- **Not multi-SAE.** One SAE, one feature space, one set of seven
  directions.
- **Not anywhere near Anthropic / Sonnet / Opus.** All-local pipeline,
  load-bearing for cost reasons.

---

## What replaces what

| Phase 2g (retired) | Phase 2h (active) |
| :----------------- | :---------------- |
| `derivation_method="sae_feature"` | `derivation_method="sae_feature_space_mean_diff"` |
| Single decoder vector `W_dec[idx]` | Mean-diff in SAE feature space, projected via `W_dec` |
| Unit-norm direction (no texture) | Natural-magnitude direction (texture preserved) |
| Per-feature exploration / neighbor mining | Per-fault-line corpus design + alpha sweep |
| `src/strategies/sae_capraro.py` | `src/strategies/sae_feature_space.py` |
| `data/sae_features/capraro_buckets.json` (still present, retired) | `data/sae_features/fault_line_directions.pt` |
| `web/SaeNeighborGraph.tsx` (deleted) | `web/FaultLineDirectionProvenance.tsx` |
