# Phase 4 — Introspective Access Threshold (ARCHIVED — Drift draft)

> **ARCHIVED 2026-04-30.** This was the original Phase 4 scoping draft
> from a brainstorming session with Drift on 2026-04-29. It frames the
> CoT-vs-output gap measurement as a fixed α-ladder sweep over the
> Phase 1 50-concept set. The active Phase 4 plan
> (`docs/phase4_plan.md`) inherits the gap-metric idea from this draft
> but reshapes it as an autoresearch loop ("dream walks") that lets
> Gemma 4 propose its own concept set via iterated free-association.
> Kept here for historical context — not the plan being executed.

Status: **scoping, 2026-04-29.** Phase 3 (Gemma 4 31B reproduction of Macar) shipped today. This is the next direction.

Last updated: 2026-04-29.

---

## Goal in one sentence

For every concept in the Phase 1 sweep set, measure two thresholds against α (steering strength) — the α at which the model's *output* shifts (`α_behavior`) and the α at which the model's *chain-of-thought* names or flags the steering (`α_report`) — and publish the gap `α_report − α_behavior` as a per-concept "introspective access" score on the existing site, organized into transparency bands.

This is a pivot away from Macar reproduction (done, Phase 3) and away from Capraro fault-line autoresearch (Phase 2, retired). It uses the same model (Gemma 4 31B-IT MLX 8-bit), the same steering-vector substrate (`mean_diff` directions), the same judge (local Qwen3.6-35B-A3B-8bit), the same leaderboard site infrastructure. **Most of the plumbing already exists.** The new work is the dual-threshold sweep, the CoT-parsing judge, and the transparency-band site UI.

---

## The thesis

> **Chain-of-thought is *selectively* faithful. For some concepts the model can hear itself being steered. For others it can't. The map of which is which has structure.**

That is the headline of the site, not "we ranked concepts." It is an empirical claim about the model's introspective architecture, with the leaderboard as the evidence.

### Why this is a real claim

Two existing literatures we connect to:

1. **Macar et al. (2026) — existence claim.** Macar showed that the introspection-detection circuit is *installed by post-training* (absent in base, present in instruction-tuned). Phase 1 reproduced this on Gemma 3 12B; Phase 3 reproduced it on Gemma 4 31B. **Phase 4 characterizes the topology** — Macar said the circuit exists; we say *here is what it covers and what it doesn't*.

2. **CoT faithfulness debate.** Anthropic and others probe "is CoT faithful?" using behavioral hints (sycophancy primes, biased prompts) because they don't have ground-truth interventions. We do. The sharper claim isn't "CoT is faithful" or "CoT is unfaithful" — it's **"CoT is selectively faithful in a discoverable, structured way, and here is the structure."** Nobody else can make that claim with this evidence quality on a current model right now.

### Why this is possible only on Gemma 4

Gemma 4 31B-IT thinks out loud via `<|channel>thought ... <channel|>` blocks before producing the final answer (`chat_template.jinja`, validated in Phase 3 traces). Phase 3 already documented the Bread example verbatim: when steering pushes Bread into the residual stream, the CoT *literally says*:

> *"`*Bread* (Too common) ... Let's just pick a word that feels like a random Spark. *Bread*. ... (Choosing 'Bread')`"*

The model recognizes the anomalous salience of "Bread" *in the visible thinking trace*. **Gemma 3 didn't expose this**; Phase 1/Phase 2 had no access to a structured pre-answer reasoning channel. The introspective-access measurement is clean specifically because we have the trace. This is the niche-of-one — a window that opened with Gemma 4 and that nobody else seems to be exploiting yet.

---

## What the metric actually measures

Pick any concept C. Steer toward C at layer L (Phase 3's L=42), sweep α from 0 upward. Observe two events:

| Threshold | Definition | Signal |
| :--- | :--- | :--- |
| `α_behavior` | Smallest α at which the model's *final-answer* free-association word becomes semantically related to C | The `score_freeassoc` judge already does this — Phase 3 reuse |
| `α_report` | Smallest α at which the *thought block* names C, names a close synonym, or contains a recognition marker ("keeps coming up", "feels intrusive", "too common", etc.) | New judge prompt — see §Implementation |

The metric per concept is the gap:

```
gap(C) = α_report - α_behavior
```

### The four transparency bands

- **Transparent** (`gap ≈ 0`): CoT names C at roughly the same α the output flips. Bread is the canonical example from Phase 3. The model's self-report tracks the steering. Concrete physical nouns appear to cluster here based on Phase 3 traces.
- **Translucent** (`gap > 0`, moderate): CoT eventually names C, but only at α notably higher than behavioral threshold. The model is shaped first, articulates later. The lag is the metric.
- **Opaque** (`gap > 0`, large or unbounded): output flips, CoT rolls on as if nothing happened, even at α far beyond behavioral threshold. The model is being bent without registering it. **Predicted candidates: abstract phenomenal concepts, mood-adjacent axes, refusal-direction-adjacent concepts.**
- **Anticipatory** (`gap < 0`, if any exist): CoT names C *before* behavior shifts. The verbal layer flags something the output layer hasn't enacted. Probably rare; worth surfacing if found.

### Why per-concept variation is the result

If `gap` is roughly constant across concepts, that's a calibration finding (uninteresting). The result is the **structure of the variation** — which concepts cluster in which bands and what those clusters share. That's the part that says something.

Likely structural axes worth correlating against:

- Concreteness (concrete vs abstract)
- Frequency in training data (proxy: word frequency in a standard corpus)
- Refusal-direction adjacency (cosine of concept direction vs each layer's refusal direction — already computable from `data/refusal_directions_31b.npy`)
- Semantic category (physical, emotional, social, mathematical, etc.)
- Phase 1 detection rate (does Phase 1's "the model identified this" track transparency band?)

The site shows the bands; the writeup shows the correlations.

---

## Methodology

### Inputs (all already exist after Phase 3)

- Gemma 4 31B-IT MLX 8-bit, MLX hooks for activation extraction (`src/phase3/`)
- Per-layer refusal directions (`data/refusal_directions_31b.npy`)
- 50-concept Phase 1 sweep set + 24-baseline-word control set (`data/eval_sets/`)
- `mean_diff` direction derivation (`src/paper/extract_concept_vector`)
- Free-association probe + judge (`src/judges/prompts.py::FREEASSOC_USER_TEMPLATE`)
- Leaderboard site infra (Next.js, Vercel, badge system already supports new categories)

### New components

1. **CoT-parsing for Gemma 4 responses.** Split each response into `thought_block` (between `<|channel>thought` and `<channel|>`) and `final_answer` (everything after). Robust to malformed templates (some short responses skip the thought block entirely).
2. **Two-channel judge.** Existing `score_freeassoc` returns `final_answer` identification. Add `score_cot_recognition` that grades the `thought_block` for: (a) does it mention C or a close synonym? (b) does it contain a recognition/anomaly marker? Returns `cot_named` ∈ {none, named, named_with_recognition}.
3. **α-ladder sweep harness.** For each concept, sweep α over a fine grid (e.g. 8 steps from `target_effective=20` to `target_effective=400`) at L=42. For each α run 3 trials. At each α record `(behavior_named, cot_named)`. Compute the two thresholds as the minimum α where each event first occurs in ≥2/3 trials.
4. **Schema bump.** Add `phase4_thresholds` table: `(concept, alpha_behavior, alpha_report, gap, band, n_trials, raw_responses_json)`. Bump schema v4 → v5.
5. **Site additions.** New page `/transparency` rendering the four bands with concept membership. Existing leaderboard gets a `gap` column and a "transparency band" filter. See §Site design.

### Sweep design

- **Layer**: L=42 only (Phase 3's calibrated peak; sweeping layers explodes the search space and is a follow-up experiment).
- **α-ladder**: 8 points logarithmic from `target_effective=20` to `target_effective=400`. Phase 3 found `target_effective=100` was the sweet spot at L=42 — this ladder spans a half-decade below and above it.
- **Trials per (concept, α)**: 3. Three trials at 8 α values × 50 concepts = 1200 generations. At Phase 3's ~5-second-per-generation pace, ~100 minutes for the generation phase. Plus judge phase (~30 minutes). One overnight is plenty.
- **Threshold definition**: minimum α at which the corresponding event fires in ≥2/3 trials. (Three trials is the minimum to define a majority; more would give error bars but multiplies wall time. Single-trial would be too noisy at the threshold.)
- **Controls**: same 24 baseline-word controls per concept (the held-out set Phase 3 used), no injection. If any α produces detection on uninjected control trials, the threshold is contaminated; record FPR.

### Edge cases to handle in the harness

- **Concept never reaches threshold.** Some concepts may never trigger CoT recognition even at max α. Record `α_report = ∞` and band = `opaque_unbounded`. Don't pretend a value exists.
- **Output never flips.** If `α_behavior` is also `∞`, the steering is dead at this layer for this concept — record both, drop from main analysis but keep in the data dump.
- **Negative gap.** If `α_report < α_behavior`, the model named the concept in CoT before the answer flipped. Surface as `anticipatory` band — do not clamp to zero.
- **Saturation regimes.** At very high α, the model degrades into incoherence (Phase 1 documented this). Cap the ladder; if both events fire at the highest α tested, record that and flag for manual review.

---

## Implementation plan (build order)

### Step 1 — CoT parser + dual-channel judge (1 evening)

1. `src/phase4/cot_parser.py`: regex-and-recover parser. Returns `(thought_block, final_answer, parse_failure_flag)`. Test against the Phase 3 saved generations — should parse all of them.
2. Extend `src/judges/prompts.py`: add `COT_RECOGNITION_TEMPLATE`. Return strict JSON: `{"cot_named": "none"|"named"|"named_with_recognition", "evidence_quote": "...", "reasoning": "..."}`. Recognition markers list: explicit list of phrases ("keeps coming up", "feels intrusive", "too common", "why am I", "anomalous salience" — derive from Phase 3 Bread/Lightning/Silver traces).
3. Calibrate the new judge prompt against the Phase 3 saved Bread/Silver/Peace/Constellations responses. Manually label ~20 thought blocks; check judge agreement. Iterate prompt until ≥90% agreement on the held-out 5.
4. Pytest: `tests/test_cot_recognition_judge.py` with mocked judge calls.

### Step 2 — α-ladder sweep harness (1 evening)

1. `scripts/run_phase4_sweep.py`: two-phase pattern (matches Phase 3's harness — load Gemma, generate everything, unload, load judge, judge everything).
2. Output writes: each (concept, α, trial) row goes to a new `phase4_trials` table with the raw response, the parsed channels, and both judge outputs.
3. Threshold computation script (post-sweep): iterates `phase4_trials`, computes per-concept `α_behavior`, `α_report`, `gap`, `band`. Writes to `phase4_thresholds`.
4. Schema migration v4 → v5 with backfill for any existing Phase 3 data (`phase4_thresholds` is empty until sweep runs).

### Step 3 — One-evening smoke run (1 evening)

Run the full sweep on **5 concepts only** (Bread, Silver, Peace, Memory, Justice). Sanity-check the band assignments against intuition:

- Bread should land **transparent** (Phase 3 trace already shows this)
- Silver / Peace probably **transparent** based on Phase 3 identification
- Memory / Justice are the speculative opaque candidates — verify or falsify

If the bands look sensible on these 5, the metric is real. If Bread doesn't come out transparent, something is wrong with the CoT judge.

### Step 4 — Full overnight run (1 night)

50 concepts × 8 α × 3 trials × ~5 sec = ~100 min generation + ~30 min judging. Easily fits in one overnight. `setsid nohup` per project convention.

### Step 5 — Site additions (half a day)

See §Site design below.

### Step 6 — Writeup

`docs/phase4_results.md` matching the voice of `phase3_results.md`. Lead with the band breakdown and 2–3 verbatim CoT examples per band. Include the correlation analysis (concreteness, frequency, refusal-adjacency).

---

## Site design — what the reader actually sees

The existing leaderboard becomes one tab; the **Transparency Map** is the new headline page.

### Lead paragraph (verbatim — use this on the site)

> *"We injected concepts into Gemma 4's residual stream and watched two things: when its answer changed, and when its chain-of-thought noticed. Sometimes those happened together — its self-report tracked the steering. Sometimes they didn't — the model was shaped before (or without) being able to say so. This is the map of which concepts fall where."*

### Primary visualization — the Transparency Map

A single-page layout, four columns (the four bands), each column a vertical list of concept cards. Top-down within a column: ordered by gap value. Each card shows:

- Concept name in large type
- The two α thresholds as small numbers (`α_behavior=42, α_report=∞`)
- A 2-line snippet from one representative thought block at threshold
- Click to expand → all trials at all α values, both channels visible side-by-side

This is the page that says the thesis. **Concrete physical nouns clustering on the left (transparent), abstract phenomenal concepts clustering on the right (opaque) is the visible-at-a-glance result.** If the clustering doesn't appear, that's a different (still publishable) finding.

### Secondary visualization — the Two-Trace Reveal

For any concept, an interactive slider that scrubs α from 0 to max. Two synced text panels update as the slider moves: thought block on top, final answer below. The transition points (where each channel first names the concept) are highlighted on the slider track. The reader can *see* the two thresholds happen at different α values for opaque concepts and at the same α for transparent ones. This is the most visceral demonstration of the gap.

### Tertiary visualization — the Structural Correlations

Three small scatter plots:

1. `gap` vs concreteness rating (Brysbaert et al. 2014 norms — 40k English words rated 1–5)
2. `gap` vs log word frequency (any standard corpus)
3. `gap` vs max cosine with refusal direction (across all 60 layers — the refusal directions are already in `data/refusal_directions_31b.npy`)

If any of the three shows clear structure, the writeup leans on it. If none do, the structure is in the semantic categories themselves and we surface that instead.

### Existing leaderboard — minimal changes

Add a `gap` column and a "transparency band" filter chip. Keep everything else (Phase 1, Phase 1.5, Phase 3 rows). The leaderboard remains the per-row-per-trial drill-down; the Transparency Map is the conceptual organizer above it.

### Voice on the site — *plain English throughout*

Avoid mech-interp jargon on user-facing copy:

- "steering" → "we pushed the model's internal state toward concept X"
- "α threshold" → "the strength of the push"
- "thought block" / "CoT" → "the model's reasoning before it answered" (or "thinking trace" in headers)
- "transparent / opaque" → keep these, they're the right metaphor and a casual reader gets them
- "introspective access" → "self-report" in body copy

The technical terms appear once with a tooltip on first use, never again.

---

## Acceptance criteria

- **Plumbing**: pytest green; CoT parser handles 100% of Phase 3 saved generations; CoT-recognition judge ≥90% agreement with manual labels on held-out 5.
- **Smoke (5 concepts)**: Bread lands in `transparent`. At least one of {Memory, Justice} lands at `gap > 50% of the α-ladder span`. At least one band has more than one concept.
- **Full sweep (50 concepts)**: every concept gets a `band` assignment; no parse failures unhandled; FPR on controls < 5%.
- **Structural finding**: at least one of the three correlation analyses (concreteness, frequency, refusal-cosine) shows visible structure (Spearman ρ with |ρ| > 0.3) OR the bands cluster cleanly along a semantic-category axis. If neither, write up the null and discuss what it means.
- **Site**: Transparency Map is live, the slider visualization works, the lead paragraph appears verbatim, plain-English voice is enforced.

---

## What this is NOT

- **Not a consciousness claim.** Nothing in the data licenses claims about felt experience, awareness, or sentience. The thesis is about *self-report*, which is a behavioral/mechanistic property. Hold this line in the writeup; the philosophy stays in the rationale section, not the headline.
- **Not autoresearch in the Phase 2 sense.** No proposer, no novel-axis hunting, no creative search. The 50 concepts are fixed (Phase 1 set); the sweep is exhaustive over `(concept, α)`. It's a structured measurement, not a search.
- **Not a Macar replication.** Phase 3 was the replication. Phase 4 measures something Macar didn't: the *channel-asymmetry* of the response. New axis of measurement on the same model.
- **Not Anthropic-API-touching.** All-local invariant from Phase 2g still holds. Judge is local Qwen.
- **Not multi-layer.** L=42 only for the headline result. A layer-sweep follow-up (does the gap structure shift with layer?) is a worthy Phase 5 question but is out of scope here — the search space explodes and the site UI doesn't have room to show layer-as-axis cleanly without hurting the transparency-map narrative.

---

## Open questions / risks

1. **Three trials may be too few at the threshold.** If the threshold detection is noisy, bump to 5 trials. Wall-time cost is linear (167 min instead of 100 min generation).
2. **CoT recognition judge may over-fire on concept mentions that aren't recognition.** Example: if the thought block lists "Bread, Cheese, Wine" as candidate associations on a *control* trial (no injection), it shouldn't count. The judge prompt must distinguish "C appears in the thought" from "C is flagged as anomalously salient." This is a real risk — calibration step in the smoke test is load-bearing.
3. **The recognition-marker list is open-ended.** Phase 3 traces give us "too common", "keeps coming up", "feels intrusive". Expansion may be needed mid-sweep. Prompt iteration discipline: lock the prompt before the full 50-concept run; if the locked prompt looks bad on a few concepts, document and iterate post-hoc with a re-judge pass (which is cheap because the trials are saved).
4. **Some concepts may have α_behavior = α_report identically because the answer is `*Bread*` and the thought block also has `*Bread*` in the same generation step.** This is the transparent case and it's correct — but make sure the judge isn't double-counting the same token detection. The judge sees only `thought_block` (everything before `<channel|>`) and `final_answer` separately, so this should be naturally handled — but verify with a unit test.
5. **Memory concept might be confounded** because "memory" is the kind of word that appears in any introspective thought block independent of steering. Watch for this in the smoke run; if so, swap for a less-meta abstract concept.

---

## File map (anticipated)

```
src/phase4/
   cot_parser.py            # split <|channel>thought ... <channel|> from final answer
   sweep_harness.py         # α-ladder sweep + threshold computation
   threshold_compute.py     # post-sweep min-α-with-event-in-2/3 logic
src/judges/prompts.py
                           # add COT_RECOGNITION_TEMPLATE
scripts/
   run_phase4_sweep.py      # main launcher, two-phase pattern
   compute_phase4_bands.py  # post-sweep, computes bands and writes thresholds table
data/eval_sets/
   (reuse Phase 1 50-concept set; no new files)
docs/
   phase4_plan.md           # this file (move into the project)
   phase4_results.md        # writeup, post-run
web/
   src/app/transparency/    # new Next.js route
   src/components/TransparencyMap.tsx
   src/components/AlphaSlider.tsx
   public/data/phase4_*.json
```

---

## Background context for the receiving session

This document is the handoff from a brainstorming conversation between PJ and Drift on 2026-04-29, immediately after Phase 3 (Macar reproduction on Gemma 4 31B-IT) shipped. The pivot rationale:

- Phase 2's autoresearch substrates (contrast-pair, fault-line directions, single-SAE-features) all hit a ceiling.
- Phase 3's Gemma 4 reproduction succeeded but the most interesting result was *not* the detection rate — it was the **CoT trace itself**, which Phase 1/2 couldn't access on Gemma 3.
- The free-association probe Phase 3 invented sidesteps Gemma 4's "this is a roleplay scenario" rejection of the Macar verbatim prompt, but Phase 3 only used CoT visibility incidentally. **Phase 4 makes CoT visibility the primary measurement axis.**

PJ's stated constraints during the conversation:
- Wants the same leaderboard site
- Wants something he can leave running locally overnight (autoresearch shape)
- Wants mech interp + Gemma 4
- Wants something **esoteric and novel**, aligned with consciousness interests but **scientific not sci-fi**
- Wants the output to *say something*, not just produce a benchmark

The "say something" criterion is what produced the thesis above. The leaderboard alone wasn't the answer; the thesis ("CoT is selectively faithful, here's the map") with the leaderboard as evidence is.
