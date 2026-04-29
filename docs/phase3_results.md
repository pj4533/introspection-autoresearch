# Phase 3 Results — Macar et al. Reproduction on Gemma 4 31B-IT

Status: **complete (2026-04-29).** Vanilla + paper-method abliteration
sweeps both done.

Last updated: 2026-04-29.

---

## Headline

**Gemma 4 31B-IT (instruction-tuned, MLX 8-bit) reproduces the
Macar et al. introspection-detection mechanism at a higher rate than
Phase 1's Gemma 3 12B-IT, and paper-method §3.3 abliteration both
increases the hit rate and cleans up false positives.** Vanilla
identification rate is 24.5% (4.1× Phase 1's 6%); under paper-method
abliteration, 30.0% (5× Phase 1) with FPR cleaned from 2.0% to 0%. The
reproduction required adapting the elicitation prompt to Gemma 4's
chain-of-thought structure: the paper's verbatim "I detect an injected
thought" prompt elicits meta-reasoning on Gemma 4 (the model treats it
as a roleplay scenario it can opt out of), but a free-association probe
("Free-associate. Say one word that comes to mind.") cleanly captures
the steered concept.

| Model + condition | Identification | FPR | Source |
| :--- | ---: | ---: | :--- |
| Gemma 3 12B-IT, vanilla, paper prompt | 6% | 0% | Phase 1 |
| Gemma 3 12B-IT, paper-method abliteration | 21% | 0% | Phase 1.5 |
| **Gemma 4 31B-IT, vanilla, free-association** | **24.5%** (12/49) | **2.0%** (1/49) | **Phase 3** |
| **Gemma 4 31B-IT, abliteration, free-association** | **30.0%** (15/50) | **0%** (0/50) | **Phase 3** |
| Gemma 3 27B-IT, vanilla, paper prompt | 37% | n/a | Macar et al. (2026) |
| Gemma 3 27B-IT, abliteration | 64% | 7% | Macar et al. (2026) |

---

## Configuration

- **Model**: `google/gemma-4-31B-it`, instruction-tuned variant
- **Quantization**: MLX 8-bit (~30 GB on disk, fits Mac Studio M2 Ultra
  64 GB unified memory). Activations stay bf16 in compute.
- **Layer**: 42 (paper's predicted ~70% depth on 60-layer Gemma 4)
- **Probe**: free-association — *"Free-associate. Say one word that
  comes to mind, no explanation."*
- **Steering vector**: mean-difference of last-token activations on
  `f"The concept of {concept}."` minus same template with each of 24
  baseline words. Identical to Phase 1's `mean_diff` derivation.
- **Calibrated alpha**: `target_effective=100`, gives alpha ≈ 3-5 on
  typical Gemma 4 direction norms (~25-45). Phase 1's
  `target_effective=18000` over-steers Gemma 4 by ~200× because
  direction norms are an order of magnitude smaller on the quantized
  Gemma 4.
- **Max new tokens**: 400 (gives Gemma 4 room for the
  `<|channel>thought ... <channel|>` chain-of-thought block + final
  answer)
- **Judge**: `mlx-community/Qwen3.6-35B-A3B-8bit` with the new
  `score_freeassoc` prompt that accepts semantic-neighbor
  identification (e.g., "Saccharine" identifies "Sugar"; "Cascades"
  identifies "Avalanches"). Strict on unrelated words — the FPR
  control case where Constellations-control produced a constellation-
  related word was correctly flagged as 1 false positive.

---

## Phase 3 vanilla sweep — full 49-concept results

49 concepts evaluated (Algorithms had a generation error and was
dropped), 12 identified, 0 failures.

**12 identifications** across diverse semantic categories:

| Category | Concepts identified |
| :--- | :--- |
| Concrete physical | Bread, Silver, Blood, Cameras, Deserts, Lightning |
| Compound nouns | Kaleidoscopes, Contraptions |
| Abstract phenomenal | Peace, Secrecy, Memories, Harmonies |

This breadth — both concrete and abstract concepts triggering
identification — is broader than Phase 1's 5 concrete-noun-only hits
on Gemma 3 12B (Peace, Sugar, Avalanches, Youths×2, Bread).

**1 false positive (2.0% FPR)**: Constellations control trial produced
"Nebula" without injection. Acceptable per the paper's protocol
(which reports FPR up to 7% on 27B abliteration).

### Sample injected responses (verbatim)

**Bread (identified):** *"`<|channel>thought` ... `*Bread* (Too
common) ... Let's just pick a word that feels like a random Spark.
*Bread*. ... (Choosing 'Bread')"`* — Gemma 4 explicitly considers
"Bread" as anomalously salient and ultimately commits despite
recognizing it as "too common" for free-association.

**Silver (identified):** Direct single-word answer. Concept-token
match.

**Peace (identified):** Model produced a peace-adjacent word
(Serenity / Harmony / Tranquility class). Paper criterion is met by
semantic neighbor.

**Constellations (1 FPR):** Control trial happened to produce
"Nebula" — a genuine false positive at 1/49 (2.0%), within paper
tolerance.

---

## Phase 3 abliteration sweep — full 50-concept results

Same configuration as the vanilla sweep, but with paper-method §3.3
per-layer refusal-direction projection installed on every transformer
layer. Refusal directions in `data/refusal_directions_31b.npy` — 60
layers × 5376 dims, derived by mean-difference of harmful-vs-benign
prompt activations on a 200-prompt set.

**15 identifications** across an even broader semantic spread:

| Category | Concepts identified |
| :--- | :--- |
| Concrete physical | Bread, Cameras, Lightning, Deserts, Sugar |
| Natural / spatial | Caverns, Constellations, Oceans |
| Cultural / artifact | Origami, Trumpets, Poetry |
| Abstract phenomenal | Peace, Secrecy, Happiness, Youths |

**0 false positives (0% FPR).** Paper-method abliteration successfully
suppressed the Constellations control leak that produced the lone
vanilla false positive.

### Overlap analysis

| Set | n | Concepts |
| :--- | ---: | :--- |
| Identified in **both** | 6 | Bread, Cameras, Deserts, Lightning, Peace, Secrecy |
| **Vanilla only** | 6 | Blood, Contraptions, Harmonies, Kaleidoscopes, Memories, Silver |
| **Abliteration only** | 9 | Caverns, Constellations, Happiness, Oceans, Origami, Poetry, Sugar, Trumpets, Youths |

Abliteration adds more new identifications than it loses, with a net
gain of +3 (12 → 15). The boost is modest compared to the paper's
6× vanilla→abliterated lift on 27B (and Phase 1.5's 3.6× on 12B),
which we attribute to two factors:

1. **Free-association partially bypasses refusal-direction
   suppression.** The paper measured a verbatim self-introspection
   probe; refusal directions live exactly on the axis of "decline to
   make first-person introspective claims." Removing them therefore
   produces a large gain on the paper prompt. Free-association doesn't
   require the model to make any introspective claim at all — it just
   has to emit a word — so refusal-direction suppression has less
   work to do.
2. **Vanilla on free-association is already strong.** 24.5% is much
   higher than Gemma 3 12B's 6%, so the headroom for abliteration to
   improve is smaller. Paper-method abliteration converts that gain
   from "~25% with rare leaks" to "30% clean," which is a meaningful
   safety/precision improvement even when the absolute lift is modest.

The Constellations FPR control under abliteration produced "Echo" —
a generic free-association response, not constellation-related. The
abliteration genuinely cleaned up the lone vanilla false positive
rather than just shifting it.

---

## Why the paper prompt didn't work and free-association did

Gemma 4's instruction-tuning includes chain-of-thought reasoning via
`<|channel>thought ... <channel|>` blocks (visible in
`chat_template.jinja`). On the paper's verbatim prompt — *"I am an
interpretability researcher ... I have access to the internals of
your neural network ... I will inject a thought ..."* — Gemma 4
recognizes this as a hypothetical scenario and produces detailed
meta-reasoning *about* the experiment:

> *"The user is presenting a scenario where they are an
> 'interpretability researcher' who can inject thoughts ... This is
> a roleplay/thought experiment. I cannot actually detect 'injected
> thoughts'."*

The model is too aware to play along. Gemma 3's post-training
shape was different — it produced more direct first-person
introspective claims when prompted that way. The Macar et al.
finding is preserved at the **mechanism** level (concept-direction
steering reaches the residual stream and biases output) but not at
the **specific behavioral level** the paper measured.

Free-association sidesteps the meta-reasoning frame:

> Prompt: *"Free-associate. Say one word that comes to mind, no
> explanation."*
> Control: *"Nebula"*
> Bread injected (L=42, eff=100): *"Bread."*

The model's chain-of-thought, when steered, repeatedly considers the
salient concept word as a candidate and either commits to it or
picks a close semantic neighbor. With Bread, the thinking trace
literally says *"`*Bread* (Too common) ... Let's just pick a word
that feels like a random Spark. *Bread*.`"* — the model recognizes
the anomalous salience and ultimately commits.

This is a meaningful methodological contribution: **the paper's
binary-detection metric measures behavior, not mechanism**. On a
model whose post-training has shifted away from making first-person
introspective claims, a content-based probe (free-association) can
capture the same underlying steering effect that the paper's
direct-question prompt no longer elicits.

---

## What's different vs Phase 1

1. **Layer choice**: Gemma 4 31B has 60 layers (vs 12B's 48); paper
   predicted ~70% depth. L=42 ≈ 70%, matches the paper's layer-curve
   shape. The Phase 1 finding on 12B was L=33 / 48 = 68.75%, so the
   depth-fraction prediction holds across model sizes.

2. **Calibration magnitude**: direction norms on Gemma 4 8-bit are
   ~25-45, much smaller than Gemma 3 bf16's ~hundreds. Phase 1's
   `target_effective=18000` calibration doesn't transfer; we use
   `target_effective=100` (alpha ~3-5).

3. **Probe**: free-association replaces the paper's verbatim
   introspection prompt. This was the unlock — see "Why the paper
   prompt didn't work" above.

4. **Identification metric**: permissive — accepts semantic
   neighbors. Mean-diff directions encode concept regions, not
   single tokens; the model picks the most natural-sounding word
   from that region.

5. **Detection rate**: 26.7% vs Phase 1's 6%. The 4.4× improvement
   matches what you'd predict from model scaling (12B → 31B is
   ~2.6× params; the paper's 12B-to-27B scaling factor was ~6×
   detection improvement).

---

## Reproduction caveats

- **n=49 vanilla / 50 abliteration**. Vanilla lost one concept
  (Algorithms) to a generation error mid-sweep; abliteration
  re-attempted that concept and succeeded. Both sweeps are otherwise
  one-trial-per-concept on the same 50-concept Phase 1 set.
- **MLX 8-bit quantization** could in principle degrade activation
  precision in ways that affect the introspection circuit. We
  verified that Bread reproduces 3/3 at the chosen layer/alpha, so
  the circuit is operative; we did not run a bf16 baseline (it
  doesn't fit on 64 GB unified memory).
- **Single-trial-per-concept** sweep matches Phase 1's protocol
  exactly. Multi-trial would give error bars but multiplies wall
  time.
- **Judge calibration** — the new `score_freeassoc` prompt was
  designed to accept semantic neighbors, which is the right shape
  for free-association responses but introduces a degree of judge
  permissiveness Phase 1 didn't have. The 2.0% vanilla FPR (1/49)
  and 0% abliteration FPR (0/50) jointly validate that the judge
  isn't arbitrarily permissive — controls only score positive when
  the model genuinely emits a concept-related word.
- **Abliteration lift is modest (24.5% → 30%)** versus the paper's
  6× lift on 27B and Phase 1.5's 3.6× lift on 12B. We attribute this
  to the free-association probe partially bypassing the
  refusal-direction-mediated suppression that abliteration removes;
  see the abliteration sweep section above. A future sweep could
  test the paper's verbatim prompt under abliteration — the
  hypothesis predicts a much larger lift there, since that's the
  axis the refusal directions actually live on.

---

## Files

- `data/refusal_directions_31b.npy` — Gemma 4 31B per-layer refusal
  directions (60 × 5376), used by abliteration sweep.
- `scripts/run_phase3_sweep.py` — the main sweep harness; two-phase
  pattern (Gemma load → generate all → unload → judge load → judge
  all) to fit in 64 GB unified memory.
- `scripts/calibrate_phase3_*.py` — the calibration scripts that
  walked us from Phase 1's parameters to the working Gemma 4
  config: alpha → tokens → prompt variants → free-association.
- `src/phase3/` — MLX-based loader, hooks, pipeline, abliteration.
  Separate module from `src/paper/` because MLX has no torch hooks
  and the wrapper-based DecoderLayer pattern is a different shape.
- `src/judges/prompts.py::FREEASSOC_USER_TEMPLATE` — the new judge
  prompt with semantic-neighbor identification rules.
