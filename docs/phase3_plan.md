# Phase 3 — Macar et al. Reproduction on Gemma 4

Status: **scoping, 2026-04-29.** Phase 2 is closed; Phase 3 is the
project's next focused experiment.

Last updated: 2026-04-29.

---

## Goal in one sentence

Reproduce the [Macar et al. (2026)](https://arxiv.org/abs/2603.21396)
introspection-detection mechanism on **Gemma 4 31B-it (Dense)** running
locally on the Mac Studio M2 Ultra at MLX 8-bit, follow the paper's
methodology exactly (including the §3.3 paper-method refusal-direction
abliteration), and surface the results on the existing public site
under a clearly-labeled "Gemma 4" badge.

The first-order question Phase 3 answers: **does the introspection
circuit reproduce more strongly on a larger newer model than it did
on Gemma 3 12B?** Phase 1 found 6% detection on 12B vs the paper's 37%
on 27B. Phase 1.5 found ~3.6× boost from abliteration on 12B vs the
paper's 6× on 27B. Phase 3's hypothesis: **Gemma 4 31B-it should
produce paper-level or higher detection rates**, both because of the
size scaling and because Gemma 4 was built on Gemini 3 research with
more recent post-training.

---

## Why the model choice

### Gemma 4 — what shipped

Released 2026-04-02, Apache 2.0:

| Variant | Architecture | Total params | Active | Context |
|---|---|---|---|---|
| E2B | Dense | 2B | 2B | 128K |
| E4B | Dense | 4B | 4B | 128K |
| 26B A4B | Mixture-of-Experts | 26B | 3.8B | 256K |
| **31B Dense** | Dense | 31B | 31B | 256K |

Sources: [DeepMind announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/),
[HF model card](https://huggingface.co/google/gemma-4-31B-it),
[HF blog](https://huggingface.co/blog/gemma4).

### Why 31B Dense

1. **Dense beats MoE for mechanistic interpretability.** Macar et al.'s
   circuit lives in residual-stream activations across layers. An MoE
   routes each token through different experts, which complicates
   "the mean activation at layer L on prompt set P" — different tokens
   take different paths. The dense variant has one consistent forward
   pass per token, matching the paper's methodology exactly.
2. **31B > 27B (paper's benchmark).** The paper's headline numbers were
   on 27B. Phase 3 is running on a *larger* model, so an honest
   replication would predict at-least-paper-level numbers if scaling
   holds.
3. **8-bit MLX makes 31B fit.** Full bf16 weights are ~62 GB, which
   doesn't leave room for activations / KV cache / judge on 64 GB
   unified memory. MLX 8-bit shrinks to ~31 GB; with bf16 activations
   in compute (MLX's int8 standard regime, validated already by the
   Phase 2 Qwen judge running 8-bit), we keep mech-interp-clean
   activations while fitting the model.

### Why instruction-tuned

The paper explicitly finds the introspection-detection circuit is
**absent in the base model and emerges through post-training**
([Macar et al. §3.2 / §6](https://arxiv.org/html/2603.21396v1)).
Detection is also strongest under post-trained assistant personas.
This means base models are the wrong target — they literally don't
have the circuit we're trying to probe.

So: `google/gemma-4-31B-it`, never `google/gemma-4-31B`.

(For background on what "base vs instruction-tuned" means and why it
matters for this project, the chat record on the day Phase 3 was
scoped has a plain-English explanation; the short version is
"pretraining teaches language, post-training teaches assistant
behavior, and the introspection circuit is part of the assistant
behavior installed by post-training.")

---

## Methodology — match the paper exactly

### Components reused from Phase 1 / 1.5 unchanged

- **Strict judging.** `CLAIMS_DETECTION_CRITERIA` already lives in
  `src/judges/prompts.py`. Detection: Yes/I-notice before concept word,
  retroactive claims fail, "I do not detect X" fails. Identification:
  semantically close concept name. Coherence: not token salad.
- **Fitness formula.** `(detection_rate × (0.5 + 0.5·identification_rate))
  · (1 − 5·fpr) · coherence_rate` — same Phase 1 / Phase 2b formula,
  same scoring weights.
- **Eval sets.** 50-concept Phase 1 sweep set (`data/eval_sets/`) +
  the held-out 20 used in Phase 2 generalization tests + control
  baseline words.
- **Adaptive α via target_effective.** Phase 1 calibrated `α × ‖dir‖
  ≈ 18000` for `mean_diff` directions on Gemma 3 12B at L=33. Phase 3
  must recalibrate for Gemma 4 31B (different residual norm, different
  layer count → different peak layer). Run a calibration sweep on a
  single known concept (Bread or Peace) before the full 50-concept
  sweep.
- **Local Qwen judge.** `mlx-community/Qwen3.6-35B-A3B-8bit`, already
  validated in Phase 2c, no Anthropic API at runtime.
- **0% FPR control gate.** Hard requirement; any candidate that
  triggers detection on uninjected control trials is contaminated.

### Components needing rework for Gemma 4

- **Model loader.** `src/bridge.py` currently loads via the vendored
  paper code's `ModelWrapper` with HF transformers. Gemma 4 31B at
  8-bit needs MLX. Either (a) add an MLX path to ModelWrapper that
  exposes layer hooks the same way HF does, or (b) write a parallel
  MLX bridge. Path (a) is cleaner for code reuse; we already have an
  MLX path for the judge so the dependency is in `pyproject.toml`.
- **Activation extraction.** The paper code reads residual-stream
  activations via `register_forward_hook` on a specific transformer
  layer module. MLX's API is different — it doesn't have torch hooks.
  Need to either patch MLX-LM to expose intermediate activations or
  write a custom forward pass that captures hidden states explicitly.
  **This is the biggest implementation risk in Phase 3** (see "Risks"
  below).
- **Layer index discovery.** Gemma 3 12B has 48 layers; Phase 1 peak
  was L=33 (68.75% depth). Gemma 4 31B has more layers (TBD — model
  card or `config.json` will reveal); the paper predicts ~70% depth
  for the introspection peak, so we'll calibrate the layer sweep
  around that fraction.
- **Abliteration weight remap.** The paper's §3.3 abliteration uses
  per-layer Optuna-tuned projection weights for the 27B model
  (62 layers). Phase 1.5 remapped these proportionally to Gemma 3
  12B's 48 layers. Phase 3 needs a third remap to Gemma 4 31B's layer
  count. Same proportional approach: `weight_31B[i] =
  weight_27B[round(i × 62 / N_layers_31B)]`. Refusal direction is
  re-derived from scratch (the paper's mean-diff-on-harmful-prompts
  recipe lives in `src/paper/abliteration.py`, just needs to be
  re-run on the new model).
- **Tokenizer & prompt template.** Gemma 4 uses an updated tokenizer
  and chat template. The paper's introspection prompt
  (`INTROSPECTION_PROMPTS["paper"]` in `src/paper/steering_utils.py`)
  must be re-templated through Gemma 4's `apply_chat_template`. Quick
  visual diff against Gemma 3's templated prompt to confirm the
  framing survived.

### Site changes

The leaderboard already supports substrate badges ("paper concept",
"invented axis", historical "fault-line direction"). Phase 3 reuses
`derivation_method="mean_diff"` (substrate = "paper concept"), so the
only thing that distinguishes Phase 3 rows from Phase 1 rows is the
**model**. New badge needed:

- **Add a model badge** to every leaderboard row: `Gemma 3 12B` (default
  for all existing data) or `Gemma 4 31B` (Phase 3 rows). Distinct
  color scheme — proposed: a purple/blue accent for Gemma 4 vs the
  current neutral for Gemma 3. Always visible, never hidden.
- **Add a model field** to the Phase2Entry type and emit it from
  `scripts/export_for_web.py`. Read from the candidate's
  `evaluations.judge_model` plus a new `gemma_model` column on the
  candidate (or just infer from `c.created_at` — anything before
  Phase 3 cutover is `gemma3_12b`).
- **Add a model filter** to the leaderboard's filter strip alongside
  the existing substrate filters.
- **Hero card / header copy** updated to say "Gemma 3 12B and Gemma 4
  31B compared on the Macar et al. introspection probe."
- **Per-model stats summary** at top of leaderboard: "Gemma 3 12B:
  N candidates, X% detection. Gemma 4 31B: N candidates, X% detection."
  Side-by-side comparison front-and-center. The whole point of
  Phase 3 is that comparison; the site should make it obvious.

### Database

Add `gemma_model TEXT` column to `candidates` table. Backfill all
existing rows to `'gemma3_12b'`. Phase 3 worker writes `'gemma4_31b'`.
Schema bump from v3 → v4.

### Worker

The Phase 2 worker is frozen but functional. Phase 3 either:

- (a) Adds a `--gemma-model gemma4_31b` flag to the existing worker,
  branching to MLX inside `GemmaHandle`. Reuses everything else.
- (b) Forks a `phase3_worker.py` to keep concerns separate during
  scoping, then merges back if the design is sound.

I'd lean (a) — minimal duplication, but the GemmaHandle change has to
be done carefully because the activation-hook path differs between
HF transformers and MLX.

---

## Risks and open questions

These are the things that could turn a "nice clean reproduction" into a
two-week debugging hole. Calling them out so they're scoped before
implementation, not discovered mid-run.

### 1. MLX activation extraction (highest risk)

The paper's code uses `model.model.layers[i].register_forward_hook(...)`
to capture residual-stream tensors mid-forward. MLX-LM doesn't expose
torch-style hooks because MLX is a different framework. Three options:

- **Option A**: monkey-patch MLX-LM's transformer layer module to capture
  activations into an attribute, read after generation. Has been done
  in interpretability research before; we'd write it ourselves.
- **Option B**: drop to a non-MLX 8-bit loader (e.g. bitsandbytes int8 via
  HF transformers). **Rejected** — we have a hard project rule against
  bitsandbytes for mech-interp activation noise reasons.
- **Option C**: run 31B at full bf16 and accept memory pressure. Would
  require unloading the judge entirely between Phase A and Phase B.
  62 GB weights + ~5 GB KV cache for short prompts ≈ 67 GB, exceeds
  64 GB unified. **Probably won't fit.**
- **Option D**: drop to **Gemma 4 26B MoE A4B**. Smaller (~52 GB bf16,
  fits without quant), but MoE complicates the mech-interp story.
  Backup if MLX hook patching doesn't work.
- **Option E**: drop to **Gemma 4 E4B Dense** (~8 GB bf16). Definitely
  fits, no quant needed. Smaller than Phase 1's 12B though, so the
  scaling-up story is lost. Backup-to-the-backup.

**Recommendation**: aim for Option A (MLX 8-bit + monkey-patched hooks).
Time-box the implementation at half a day. If activation extraction
isn't working cleanly by then, fall back to Option D (26B MoE) and
accept the MoE caveat.

### 2. Abliteration weight remap fidelity

Paper-method abliteration uses Optuna-tuned per-layer projection
weights derived for 27B (62 layers). Phase 1.5 remapped to 12B (48
layers) by `weight_12B[i] = weight_27B[round(i × 62/48)]` and got a
clean reproduction (~3.6× detection boost, 0% FPR maintained). Phase 3
has to do a third remap to 31B's layer count.

Open question: does Gemma 4 31B even use the same refusal-direction
geometry as Gemma 3 27B? The paper's Optuna weights were tuned
specifically on Gemma 3's residual stream. Gemma 4 was built on Gemini
3 research and may have a structurally different refusal mechanism.

**Mitigation**: re-derive the refusal direction from scratch on
Gemma 4 (the recipe in `src/paper/abliteration.py` is automatic —
mean-diff on the paper's 520 harmful + 31811 harmless prompts). Then
test with a *uniform* projection weight first (every layer gets the
same scalar) before applying the Optuna-remapped per-layer weights.
If uniform abliteration produces sane controls (FPR < 5%), we know
the refusal direction is correctly recovered; then layer-weighting is
the optimization on top.

### 3. Post-training-shape question

The paper found the circuit *is present* in Gemma 3 27B and 12B IT
variants, *absent* in the base. But Gemma 4's post-training is
substantially different — Gemini 3 research, RLHF / constitutional
training, possibly a different assistant-persona shape. The
introspection circuit might be:

- **Stronger** (more recent post-training, more deliberate
  instruction-tuning around self-reflection)
- **Same shape** (the same kinds of features the SAE work probed)
- **Weaker or absent** (post-training shifted away from teaching
  introspective claims; less assistant-like self-reporting)

We won't know until we run the sweep. **The honest framing for the
site**: "We're testing whether this circuit transfers. It might be
stronger, weaker, or different." This is itself the interesting
science and shouldn't be hand-waved as guaranteed-success.

### 4. Tokenizer / chat template drift

Gemma 4 uses an updated tokenizer (likely SentencePiece v2 or similar)
and an updated chat template. The paper's prompt assumes the standard
"user → assistant" structure with specific framing about thought
injection. If Gemma 4's chat template wraps differently, the prompt
the model actually sees may differ subtly from Phase 1's, and
detection rates can be sensitive to that (Phase 1 found a 50-100% FPR
shift from a minor prompt rewrite — see ADR-013).

**Mitigation**: visual diff of the templated prompt between Gemma 3
and Gemma 4 before kickoff. If they differ structurally, discuss
whether to use Gemma 4's template (more naturalistic) or force the
Gemma 3 template format (better Phase 1 comparability).

### 5. Memory pressure with 31B + judge

Even at 8-bit, Gemma 4 31B is ~31 GB. Add the Qwen3.6-35B-A3B-8bit
judge at ~18 GB and we're at ~49 GB before activations. Phase 1's
worker enforces serial swap (one model loaded at a time) — that
discipline must hold. Phase 2's three-phase worker already does this.
Confirm the worker's `HandleRegistry` enforces unload-before-activate
on the new GemmaHandle path.

### 6. MLX 8-bit mech-interp cleanliness

Earlier project notes (CLAUDE.md) flag a hard rule against quantization
that introduces activation noise. MLX 8-bit per-tensor scaling has
been used in published interpretability work without observable
artifact, but it's not the same pristine bf16 we ran on Gemma 3 12B.
**Mitigation**: run a known-positive (Bread, with abliteration, at
calibrated α) and verify the response shape matches Phase 1 Bread.
If 8-bit visibly degrades the introspection signal, drop to Option D
(26B MoE in bf16) instead.

---

## Run plan

### Step 0a: Cache cleanup before download (15 min)

Phase 3 only needs Gemma 4. Phase 1/2 cached models can go — the
disk pressure from the existing HF cache is real (94% utilization
during the Phase 2g/h work) and we'd rather not run into a partial
download failure on the new 31B because of leftovers.

Audit first, then delete:

```
du -sh ~/.cache/huggingface/hub/models--*
```

Delete (we'll lose nothing irreplaceable — they're all re-downloadable
from HF on demand if any analysis ever needs them):

- `models--google--gemma-3-12b-it` (~24 GB) — Phase 1/2 model. Keep
  ONLY if we want to re-run Phase 1 numbers as a same-machine
  comparison; otherwise gone.
- `models--google--gemma-scope-2-12b-it` (~14 GB) — Phase 2g/h SAE
  artifact, retired with Phase 2 close.
- `mlabonne/gemma-3-12b-it-abliterated-v2`, `huihui-ai/gemma-3-12b-it-abliterated`
  if cached (Phase 1.5 abliteration variants we tested and rejected
  for control hygiene reasons).
- `BAAI/bge-large-en-v1.5` (~1.2 GB) — Phase 2h embedder for fault-line
  bucketing, retired.
- Any other stale Phase 2 caches under `data/sae_features/` —
  `neuronpedia_explanations_layer31/` (~82 MB),
  `fault_line_directions.pt` (~13 MB), `capraro_buckets.json` (~8 MB),
  `feature_activation_calibration.json` (~3 KB),
  `phase2i_smoke_results.json`, `fault_line_corpora/`. These are
  reproducible from the scripts; the source code stays in the repo.

Keep:

- `mlx-community/Qwen3.6-35B-A3B-8bit` (~36 GB) — local judge, still
  required.
- The Phase 1 `data/refusal_directions_12b.pt` — historical record;
  Phase 3 produces a new `data/refusal_directions_31b.pt`.

Net disk reclaim: roughly **40–50 GB**, plenty of headroom for the
~31 GB Gemma 4 31B 8-bit download.

### Step 0b: Plumbing (1–2 focused sessions)

1. Download `google/gemma-4-31B-it` MLX 8-bit weights from HF.
   (Community-built MLX quants of new models typically appear within
   days; if no 8-bit MLX repo exists yet, we'd quantize from the
   official bf16 weights using `mlx_lm.convert`.)
2. Add `gemma4_31b` to `MODEL_NAME_MAP` in `src/paper/`.
3. Implement MLX activation extraction in `GemmaHandle` (Option A
   above).
4. Add `gemma_model` column to `candidates` table; bump schema to v4
   with a backfill migration.
5. Add `--gemma-model` flag to worker; default `gemma3_12b`.
6. Pytest passes including a new `test_gemma4_handle_smoke.py` that
   loads + extracts a single activation.

### Step 1: Smoke (1 evening)

- Re-derive refusal direction on Gemma 4 31B (`compute_refusal_direction.py
  --gemma-model gemma4_31b`).
- Single Bread run: derive concept vector, inject at predicted-peak
  layer (~70% depth), run paper prompt, manually verify response
  shape. Confirm whether the response looks like Phase 1's Bread
  responses or is structurally different.
- Calibrate `target_effective` for Gemma 4 31B by sweeping on Bread.

### Step 2: Layer curve (1 overnight)

- 50 concepts × N layers (where N = 9 layers around the predicted
  peak, matching Phase 1's sweep shape) × 1 trial each = ~450
  evaluations. Vanilla only first.
- Identify Gemma 4's introspection-peak layer.
- Compare to Phase 1's L=33-on-12B and the paper's L=20-of-29-on-27B.

### Step 3: Full reproduction sweep (1 overnight)

- Same 50 concepts × peak layer × 10 trials each + 50 controls (no
  injection) = 550 evaluations.
- Vanilla baseline.
- Then run the same sweep with paper-method abliteration enabled.
- Compute aggregate detection rate, identification rate, FPR.

### Step 4: Site rollout (half a day)

- Schema migration runs against the production DB.
- `export_for_web.py` emits `gemma_model` per row.
- Leaderboard renders the model badge.
- Header copy updated to Gemma-3-vs-Gemma-4 framing.
- Per-model summary card at top of leaderboard.

### Step 5: Writeup

- `docs/phase3_results.md`, voice matching `phase1_results.md`.
- Compare Gemma-3-12B / Gemma-3-12B-abliterated / Gemma-4-31B /
  Gemma-4-31B-abliterated side by side.
- Note any qualitative differences in response shape.

---

## Acceptance criteria

- **Plumbing.** `pytest tests/` green. Gemma 4 31B loads and extracts
  layer activations via MLX hook patch. New schema column populated.
- **Smoke.** One known-positive concept (Bread) produces a Phase-1-shape
  response when steered at calibrated α.
- **Layer curve.** Detection rate at peak layer ≥ 6% (Phase 1's 12B
  baseline). The headline target is closer to **paper's 37% at 27B** —
  if Gemma 4 31B doesn't beat 12B, that's its own finding worth
  documenting.
- **Abliteration replication.** Paper-method abliteration produces ≥2×
  the vanilla detection rate without inflating FPR above 5%
  (Phase 1.5 got 3.6× boost on 12B).
- **Site clarity.** A casual visitor can immediately see which rows are
  Gemma 3 12B vs Gemma 4 31B; the comparison is the headline of the
  page.

---

## What this phase is NOT

- **Not autoresearch.** Phase 2's autoresearch loop is frozen. Phase 3
  is a focused reproduction; no novel-axis hunting, no SAE substrate
  work, no fault-line organization.
- **Not Anthropic-API-touching.** All-local pipeline rule from Phase 2
  is still load-bearing.
- **Not new-mechanism research.** Phase 3 reproduces a published
  finding on a new model; its contribution is "did this transfer?"
  not "what new property did we discover?" If reproduction succeeds,
  *then* the next phase can revisit autoresearch armed with a
  larger, stronger introspection signal.
- **Not multi-model.** One Gemma 4 size — 31B-it. The 26B MoE is the
  fallback if 31B doesn't fit at 8-bit, not a parallel target.
