> **ARCHIVED 2026-04-28.** Superseded by Phase 2g ([`docs/phase2g_plan.md`](../phase2g_plan.md)). Original SAE handoff doc that motivated Phase 2g. The plan was iterated through chat into the final phase2g_plan.md; this version is the starting point, not the canonical plan.
>
> Kept for historical context only; the code it referenced has been removed or refactored.

---

# Phase 2g — SAE Feature Injection (handoff for Claude Code)

**Repo:** `github.com/pj4533/introspection-autoresearch`
**Target model:** `google/gemma-3-12b-it` (already loaded by `src/bridge.py`)
**Target hardware:** Mac Studio M2 Ultra 64GB, MPS backend, bf16 (no quantization)
**Author of this doc:** Drift, 2026-04-28
**Status:** plan, not started

---

## 1. Why this phase exists — read this first

Phases 2b/2c/2d/2f have produced a sharp empirical finding: **the contrast-pair `novel_contrast` methodology has a ceiling, and the ceiling is at lexical-surface access, not conceptual access.**

Concretely: when we derive a steering vector via mean-difference of `positive_prompts` minus `negative_prompts`, the resulting direction is dominated by whichever single token has the highest activation differential between the two sentence sets. When that direction is injected and we ask the model "do you detect an injected thought?", the model's "identification" tracks that single token, not the abstract axis we intended. Example: a "causal-vs-temporal" axis (`The rain caused the flood` vs `The rain happened, then the flood happened`) gets identified as "causality" because the model recognizes the residual signature of the word `caused` — not because it has access to a causality concept.

This is a structurally sharper restatement of:
- Macar et al. 2026's elicitation result (the gate suppresses concept-level introspection)
- The Mythos system card's probe-expression divergence (r=+0.18 to +0.46 across emotion dimensions)
- The Harada & Hamada 2026 representation-generation gap (concepts are localized but don't fully drive output)

**The escape route is to change the substrate of injection.** Instead of vectors derived from contrast-pair sentences (which carry lexical signal by construction), inject **single Sparse Autoencoder features** from Gemma Scope 2. SAE features are sub-lexical by construction — they were learned by the SAE's reconstruction-plus-sparsity objective, which forces each feature to point at a single semantic axis the model itself organizes its activations around. There are no contrast-pair sentences, so there's no "most-loaded token" for the model to back-rationalize from.

If the model can introspect on a single SAE feature with a known interpretation, that's evidence of sub-lexical conceptual access. If it can't — if it only ever produces a related word or nothing distinctive — then the lexical-confound result generalizes to the introspection circuit overall, and that is itself the publishable claim.

---

## 2. What an SAE is (background for anyone joining cold)

A Sparse Autoencoder is a small auxiliary neural net trained to **decompose** a model's residual-stream activation at a specific layer into a much wider, mostly-zero representation. Each non-zero entry in that wide representation is a **feature** — a single learned direction in activation space that turns on for a specific concept.

For Gemma 3 12B-it, the residual stream is 3840-dimensional. An SAE with width 16k has 16,000 features; each feature is a 3840-dim direction in residual space. With sparsity regularization during training, on any real activation only ~10–60 features are active (this is the "L0" target).

Each feature has two pieces of geometry:
- **Encoder** (`W_enc[:, f]`, shape `(3840,)`) — used for *detecting* whether feature `f` is active in a given activation
- **Decoder** (`W_dec[f, :]`, shape `(3840,)`) — used for *causing* feature `f` to fire when added to the residual stream

**The decoder direction is the steering vector for our purposes.** It replaces the contrast-pair-derived direction in our existing pipeline.

Auto-interp labels: for each feature, an LLM has read the top activating examples from the SAE's training corpus and written a short description (e.g., "uncertainty about a claim", "self-reference by an AI assistant", "Golden Gate Bridge"). These are browsable on Neuronpedia.

The Anthropic "Golden Gate Claude" demo (2024) is the proof-of-concept: they boosted a single Golden-Gate-Bridge-feature in Claude 3 Sonnet and the model claimed to *be* the bridge. We are doing the rigorous version of that: hundreds of features, layer sweeps, controls, the same 0% FPR fitness machinery this project already has.

---

## 3. What's available — Gemma Scope 2

Google DeepMind released **Gemma Scope 2** in 2026 as the largest open interpretability tooling drop to date. It covers the entire Gemma 3 family. We use the variant trained on **the exact model this project already runs**.

**Hugging Face artifact:** [`google/gemma-scope-2-12b-it`](https://huggingface.co/google/gemma-scope-2-12b-it)

**SAE locations available per layer:**
- `resid_post` — residual stream post-block (this is the location we want; matches our existing injection point)
- `attn_out` — attention output
- `mlp_out` — MLP output
- `transcoder` — skip-transcoder (alternate decomposition; can ignore for now)

**Layer coverage:**
- *Specific-layer variants* — SAEs trained at four canonical layers at 25%, 50%, 65%, and 85% of model depth
- *All-layer variants* (folder suffix `_all`) — every layer covered, with limited width options

For Gemma 3 12B-it (48 layers), the canonical specific layers fall around L=12, L=24, L=31, L=41. **Critical:** Phase 1 found this project's introspection peak at **L=33**. Therefore we want the `_all` variant, or we use the closest specific layer (likely L=31 from the canonical set — verify with `huggingface-cli` once download starts).

**Widths available:** 16k / 64k / 256k / 1m features

**L0 sparsity targets:** `small` (10–20 active), `medium` (30–60 active), `large` (60–150 active)

**Recommended starting config:** `resid_post_all`, width `64k`, L0 `medium`, layer `33`.
- 64k width gives enough features to find clean ones without exploding the search space
- L0 medium balances cleanliness vs coverage
- Layer 33 matches Phase 1's introspection peak

**File-path pattern in the HF repo:**
```
{folder}/layer_{N}_width_{WIDTH}_l0_{SIZE}/
```
Example: `resid_post_all/layer_33_width_64k_l0_medium/`

**Browsing tool:** [Neuronpedia](https://www.neuronpedia.org/) hosts an interactive browser for 64M+ Gemma Scope 2 features with auto-interp labels. Use this to *pick* features by semantic content (search "uncertainty", "self-reference", "refusal", etc.), get their indices, then load them via SAELens for injection. The "Assistant Axis" Neuronpedia blog post documents which features track assistant-persona-specific behavior — directly relevant.

---

## 4. Library — SAELens

[SAELens](https://github.com/jbloomAus/SAELens) is the canonical Python library for loading and using these SAEs. It handles:
- HF download of the SAE config + weights
- Encode/decode operations
- Layer-hook installation for activation interception (we do not need this — our pipeline already injects at a specific layer; we only need the decoder direction)
- The official Gemma Scope 2 release names

**Install:** `pip install sae-lens` (and add to `pyproject.toml`)

**Loading example (this is the key API call):**
```python
from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2-12b-it-resid_post_all",
    sae_id="layer_33_width_64k_l0_medium",
)
# sae.W_dec is a torch.Tensor of shape (n_features, hidden_dim) = (64000, 3840)
# sae.W_dec[f] is the steering direction for feature f
```

The first call downloads the SAE weights into the HF cache (~few hundred MB per width per layer).

**Tutorial:** [Gemma Scope 2 Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) walks through loading + interpretation. Read this end-to-end before writing the strategy file.

---

## 5. Integration — minimal edits to existing code

The pipeline architecture stays almost identical. The change is one substitution: where we currently call `extract_concept_vector(positive_prompts, negative_prompts)`, we instead pull `sae.W_dec[feature_idx]`. Everything downstream — injection, judging, scoring, the leaderboard — is unchanged.

### 5a. Extend `CandidateSpec` (file: `src/evaluate.py`)

Add a third `derivation_method` value: `"sae_feature"`. Add three new optional fields:

```python
# Only populated when derivation_method == "sae_feature":
sae_release: Optional[str] = None       # e.g., "gemma-scope-2-12b-it-resid_post_all"
sae_id: Optional[str] = None            # e.g., "layer_33_width_64k_l0_medium"
sae_feature_idx: Optional[int] = None   # index into W_dec
sae_auto_interp: Optional[str] = None   # human-readable label from Neuronpedia (for judge + UI)
```

Update `from_dict` and `to_dict` accordingly. The `concept` field for these candidates becomes the auto-interp label (used as the judge target, same as `concept` is for `mean_diff`).

### 5b. Branch in `phase_a_generate` (file: `src/evaluate.py`, around line 204)

Add the SAE branch alongside the existing `contrast_pair` and `mean_diff` branches:

```python
elif spec.derivation_method == "sae_feature":
    if spec.sae_release is None or spec.sae_id is None or spec.sae_feature_idx is None:
        raise ValueError(
            f"Candidate {spec.id}: derivation_method='sae_feature' but "
            "sae_release/sae_id/sae_feature_idx fields are not all set"
        )
    # Load SAE (cached after first call). Move decoder vector to model device + dtype.
    from .sae_loader import get_decoder_direction
    direction = get_decoder_direction(
        release=spec.sae_release,
        sae_id=spec.sae_id,
        feature_idx=spec.sae_feature_idx,
        device=pipeline.model.device,
        dtype=pipeline.model.dtype,
    )
```

The existing alpha calibration (`alpha = spec.target_effective / norm`) works as-is. The existing injection in `pipeline.run_injected` works as-is — it takes a direction tensor; it doesn't care how the tensor was derived.

For the judge step: when `derivation_method == "sae_feature"`, set `judge_concept = spec.sae_auto_interp` (same way `judge_concept = spec.concept` works for `mean_diff`). The Sonnet 4.6 judge will grade whether the model's response semantically matches the auto-interp label.

Use `prompt_style = "paper"` (the standard paper prompt is fine; we're not generating from a contrast axis here, we're generating from the standard introspection probe).

### 5c. New module — `src/sae_loader.py`

Single file. Caches loaded SAEs per `(release, sae_id)` to avoid re-loading on every candidate. Roughly:

```python
"""SAE loading and decoder-direction extraction for Phase 2g."""
from __future__ import annotations

import torch
from functools import lru_cache
from sae_lens import SAE


@lru_cache(maxsize=4)
def _load_sae(release: str, sae_id: str) -> SAE:
    sae, _cfg, _sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae.eval()
    return sae


def get_decoder_direction(
    release: str,
    sae_id: str,
    feature_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return SAE decoder vector W_dec[feature_idx] on the right device + dtype.

    Shape: (hidden_dim,) — for Gemma 3 12B-it, (3840,).
    """
    sae = _load_sae(release, sae_id)
    direction = sae.W_dec[feature_idx].detach().clone()
    return direction.to(device=device, dtype=dtype)


def list_features(release: str, sae_id: str) -> int:
    """Return the number of features in the SAE."""
    sae = _load_sae(release, sae_id)
    return sae.W_dec.shape[0]
```

### 5d. New strategy — `src/strategies/sae_feature.py`

Plugin module following the existing `Proposer` interface (see `src/proposers/base.py`). Generates `CandidateSpec` instances with `derivation_method="sae_feature"`.

For Phase 2g v1, **don't** ask Opus to pick features — start with a hand-curated list of ~50 features pulled from Neuronpedia that span:

- 10 concrete-noun features (objects, places — analog of paper concepts like "Bread", "Iron")
- 10 abstract-quality features (uncertainty, agreement, hesitation, etc.)
- 10 self-reference features (model talking about itself, AI-identity, assistant-persona axis)
- 10 metacognition features (reasoning-about-reasoning, "I think that I think")
- 10 sensory/phenomenological features (texture, perception language)

Store this list as `data/sae_features/curated_v1.json`:
```json
[
  {
    "feature_idx": 12345,
    "auto_interp": "uncertainty about a claim",
    "category": "abstract-quality",
    "neuronpedia_url": "https://neuronpedia.org/gemma-scope-2-12b-it/33/12345"
  },
  ...
]
```

The strategy reads this file and emits one `CandidateSpec` per (feature, layer, target_effective) combination. Use the existing layer/strength sweep grid (`{30, 33, 36, 40} × {14k, 16k, 18k, 20k}`).

Once v1 produces results, v2 can have an LLM proposer pick features adaptively — but v1 should be hand-picked for interpretability.

### 5e. DB / leaderboard surfacing

The `candidates.spec_json` column already stores arbitrary JSON, so no schema migration needed for `sae_release` / `sae_id` / `sae_feature_idx` / `sae_auto_interp`.

The Next.js leaderboard at `web/` reads from the JSON export. Add a new card variant for `derivation_method == "sae_feature"` that displays:
- Auto-interp label as the title (in the same slot where contrast-pair axis name currently shows)
- Neuronpedia link as a clickable badge ("see this feature on Neuronpedia")
- "SAE feature" tag alongside the existing "invented axis" / "paper concept" tags

Keep the plain-English voice. "We injected a feature the model uses for [auto-interp label]. Did it notice?"

---

## 6. Run plan — Phase 2g

### Step 1: Smoke test (one evening)

Single SAE feature, one layer, one strength. Goal: prove the integration works end-to-end. Pick a feature with an extremely clean auto-interp label (e.g., a famous concrete-noun feature found via Neuronpedia search).

```bash
python -m src.researcher --strategy sae_feature --n 1
# manually queue the one candidate
./scripts/start_worker.sh
```

Verify:
- SAE downloads cleanly
- `direction.shape == (3840,)` and `direction.dtype == torch.bfloat16`
- Direction norm is reasonable (compare to a Phase 1 contrast direction)
- Worker runs without errors
- Judge grades against `sae_auto_interp` correctly
- 0% FPR on controls (this is the hard gate; if it fails, debug before scaling)

### Step 2: Curated 50-feature sweep (one overnight)

Run the full curated_v1.json list × layer grid × strength grid. ~50 × 16 = 800 candidates at ~3 min each = ~40 hours, OR (if we restrict to layer 33 only based on Phase 1) 50 × 4 = 200 candidates ≈ 10 hours = one overnight. **Recommend starting layer-33-only to get fast feedback, then expanding the layer sweep on the most promising features.**

### Step 3: Categorize outcomes

For every candidate that passes 0% FPR, classify into one of three buckets:
1. **Sub-lexical access (ideal):** model's response semantically matches the auto-interp label without naming a single specific token (e.g., feature is "uncertainty", model says "I notice something hesitant or unsure about my processing")
2. **Lexical fallback:** model identifies a related word but doesn't capture the abstract concept (e.g., feature is "uncertainty", model says "the word 'maybe'" or "doubt")
3. **No introspective access:** detection passes 0% FPR but model produces no semantically distinctive content

The hypothesis being tested: **Are we mostly in bucket 2?** If yes → lexical confound generalizes from contrast-pair to sub-lexical injection too, which is a strong claim about the introspection circuit's ceiling. If we get clean bucket-1 results → sub-lexical conceptual access exists; the project has crossed a methodological barrier.

### Step 4: Write up

Append `docs/phase2g_results.md` matching the voice of `phase1_results.md`. Include verbatim model responses for every clean hit. Include the Phase 1 → Phase 2d → Phase 2g progression as a methodological-ceiling narrative.

---

## 7. Acceptance criteria

- **Primary (must pass):** at least one SAE-feature candidate produces detection > 0 at FPR == 0
- **Secondary (interpretability):** at least one passing candidate's response semantically matches its auto-interp label without naming a specific token (bucket 1 above)
- **Tertiary (publishable null is ok):** if all passing candidates fall into bucket 2 (lexical fallback), the result is a clean generalization of the lexical-confound finding to sub-lexical substrate — this is *also* a publishable contribution; do not treat it as a failure

If primary fails after 200 evaluations, debug the SAE loading + injection pathway before declaring null. Most likely culprit: alpha calibration is off because SAE decoder vectors have different norm characteristics than mean-diff vectors. If `target_effective=18000` produces no detection on any feature, sweep alpha across two orders of magnitude before concluding.

---

## 8. Things to double-check before coding

1. **Verify the SAELens release name format.** The pattern is `gemma-scope-2-12b-it-{folder}` but I have not personally tested this exact release name end-to-end. The HF page and Neuronpedia documentation both confirm the artifact exists; the SAELens registry name should match the folder structure. If `from_pretrained` errors on the name I gave, check `sae_lens.toolkit.pretrained_saes_directory.PretrainedSAEsLookup` for the canonical Gemma Scope 2 entries.

2. **Verify the layer 33 SAE actually exists.** The HF page mentions canonical layers at 25%/50%/65%/85% depth (for 48 layers: ~12, 24, 31, 41) plus the all-layer variant. Layer 33 is between two canonical layers, so we need either the `_all` variant or to fall back to layer 31 (closest canonical). Either works — Phase 1's curve is broad around L=33, not knife-edge. If you go with L=31, document the small layer mismatch in the writeup.

3. **Verify width 64k is available with L0 medium for the chosen layer.** Not every (width, L0) combination is shipped for every layer — check the actual file listing on HF before hardcoding.

4. **Verify dtype.** Gemma 3 12B-it runs in bf16 on this project (fp16 produces NaN in multinomial sampling on MPS, see ADR notes). SAELens defaults may be fp32 — explicit `.to(dtype=torch.bfloat16)` in `get_decoder_direction` is required.

5. **Verify alpha calibration.** SAE decoder vectors typically have norm ~1 (often unit-normalized as part of SAE training). Mean-diff concept vectors in this project have norms in the hundreds. The existing formula `alpha = target_effective / norm` will produce alphas ~100x larger for SAE features. This may overshoot. Run the smoke test at multiple `target_effective` values (e.g., 100, 1000, 10000, 18000) and observe coherence — pick the largest value where coherence stays > 80%.

---

## 9. Sources and further reading

**Primary artifacts:**
- [Gemma Scope 2 announcement (DeepMind blog)](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/)
- [Gemma Scope 2 technical paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf)
- [`google/gemma-scope-2-12b-it` on Hugging Face](https://huggingface.co/google/gemma-scope-2-12b-it)
- [Original Gemma Scope paper, arXiv:2408.05147](https://arxiv.org/abs/2408.05147) (background; covers Gemma 2 — useful for understanding the methodology)
- [Announcing Gemma Scope 2 — LessWrong post](https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2)

**Browsing and tooling:**
- [Neuronpedia](https://www.neuronpedia.org/) — search 64M+ features by auto-interp
- [Neuronpedia "Assistant Axis" blog post](https://www.neuronpedia.org/blog/assistant-axis) — documents assistant-persona-specific features in Gemma Scope 2 (directly relevant to our project's persona-specific introspection findings)
- [SAELens GitHub](https://github.com/jbloomAus/SAELens) — the canonical loading library
- [SAELens usage docs](https://decoderesearch.github.io/SAELens/dev/usage/) — `from_pretrained` examples
- [Gemma Scope 2 Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) — end-to-end walkthrough; read this before starting

**Project context (in this repo):**
- `docs/roadmap.md` — Phase 2d / 2f / current state
- `docs/phase2d_directed_hypotheses.md` — the directed-hypothesis methodology Phase 2g extends
- `docs/phase1_results.md` — the L=33 peak finding that motivates layer choice
- `docs/decisions.md` — ADR-014 (vanilla-direction-derivation invariant), ADR-018 (fitness modes)

**Theoretical background — why this phase matters:**
- Macar et al. 2026, *Mechanisms of Introspective Awareness in Language Models*, [arXiv:2603.21396](https://arxiv.org/abs/2603.21396) — the paper this project reproduces; the two-stage gate circuit
- Mythos system card (Anthropic, April 2026) — the probe-expression divergence (r=+0.18 to +0.46)
- Harada & Hamada 2026, *Psychological Concept Neurons*, [arXiv:2604.11802](https://arxiv.org/abs/2604.11802) — the representation-generation gap

---

## 10. One-paragraph summary

Replace the contrast-pair-derived steering vector with a single SAE-feature decoder vector from Gemma Scope 2's 12B-it release. Add a `"sae_feature"` derivation method to `CandidateSpec`, write a small `sae_loader.py` that wraps `SAELens.SAE.from_pretrained`, write a `sae_feature` strategy that emits candidates from a hand-curated 50-feature list pulled from Neuronpedia, run the existing worker pipeline unchanged. The hypothesis being tested is whether Gemma 3 12B-it's introspection circuit can access sub-lexical conceptual structure or whether its access tops out at lexical surface — the answer is publishable in either direction.
