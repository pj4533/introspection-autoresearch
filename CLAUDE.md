# CLAUDE.md

Orientation for Claude Code sessions working on this repo.

## Project one-liner

Reproduce the Macar et al. (2026) *Mechanisms of Introspective Awareness in
Language Models* paper on `Gemma3-12B-it` locally (Mac Studio M2 Ultra, MPS,
bf16), then run an all-local autoresearch loop that injects single Sparse
Autoencoder feature decoder vectors from Gemma Scope 2 and tests whether
the model can introspect on each one — organized around Capraro et al.
(2026)'s seven epistemological fault lines.

## Top-of-file invariant

**All-local pipeline. No Anthropic models in any forward-looking work.**
The autoresearch loop runs 100% locally — judge is local Qwen, no proposer
needed, no claude-agent-sdk, no Anthropic API. The interactive Claude Code
session itself is the only thing that uses subscription tokens. This is
load-bearing for cost reasons; reintroducing cloud LLMs to the runtime
loop would re-create the burn that paused Phase 2c.

## Canonical references (read first if orienting)

- **Active phase plan**: [`docs/phase4_plan.md`](docs/phase4_plan.md) —
  Dream Walks + Forbidden Map. Overnight autoresearch loop in which
  Gemma 4 free-associates through chains of steered concepts; per-step
  CoT-vs-output asymmetry feeds a "Forbidden Map" of concepts the
  model can be made to think but cannot notice itself thinking. The
  only phase being worked on going forward.
- **Roadmap** (full project trajectory, all phases past and present):
  [`docs/roadmap.md`](docs/roadmap.md)
- **Architectural decisions** (ADR log): [`docs/decisions.md`](docs/decisions.md)
- Plain-English project writeup (public-facing): [`docs/plain_english.md`](docs/plain_english.md)
- Phase 1 technical results: [`docs/phase1_results.md`](docs/phase1_results.md)
- Phase 1.5 technical results (paper-method abliteration): [`docs/phase1_5_results.md`](docs/phase1_5_results.md)
- Phase 3 technical results (Gemma 4 reproduction): [`docs/phase3_results.md`](docs/phase3_results.md)
- Judge calibration data (local Qwen): [`docs/calibration_results_qwen35b.md`](docs/calibration_results_qwen35b.md)
- Full original spec: [`docs/01_introspection_steering_autoresearch.md`](docs/01_introspection_steering_autoresearch.md)
- README: [`README.md`](README.md) (repo layout, setup, running)
- Archived phase docs: [`docs/archive/`](docs/archive/) — full Phase 2
  history (2b/2c/2d/2f/2g/2h/2i) closed 2026-04-29; Drift's original
  Phase 4 draft (`phase4_introspective_access_threshold_drift.md`)
  archived 2026-04-30 in favor of the dream-walks reshape.

**Rule:** if a project decision only lives in chat logs, memory files, or
ephemeral plan files, it's a bug. Commit it to `docs/roadmap.md`
(forward-looking) or `docs/decisions.md` (rationale for past choices).

## Status snapshot (2026-04-30)

**Phases 1, 1.5, 2, and 3 are closed. Phase 4 (Dream Walks + Forbidden Map) is the active phase.**

Phase 2 spent two weeks iterating on autoresearch substrates for steering
Gemma 3 12B's introspection circuit. Five substrates were built and
empirically retired: contrast-pair `mean_diff` (lexical-confound ceiling),
single SAE decoder vectors (no saturation), feature-space mean-diff over
corpora (wrong direction in activation space), single SAE features at
calibrated saturation (some perturbation, only 2/60 strict detections).

The aggregate Phase 2 finding: *Gemma 3 12B's introspection circuit
detects anomalous-saturation along single concept axes (Phase 1's
mean_diff geometry); no SAE-substrate variant we built reliably triggers
it beyond Phase 1's 6% baseline.* Full per-phase summaries in
[`docs/roadmap.md`](docs/roadmap.md).

**Phase 3 question being scoped:** reproduce Macar et al. on Gemma 4
(released 2026-04-02). Three sizes might run on Mac Studio M2 Ultra:
E4B, 26B MoE (A4B), 31B Dense. Selecting the largest that fits.

Historical state, briefly:

- **Phase 1 (done 2026-04-16).** Full sweep on Gemma3-12B-it: best layer 33
  (68.75% depth), detection rate 6%, FPR 0/50. Mechanism reproduced at
  smaller magnitude than the paper's 27B (~6× weaker).
- **Phase 1.5 (done 2026-04-17).** Paper-method abliteration reproduced
  at ~3.6× boost over vanilla; default reverted to vanilla after Phase
  2d-1 found paper-method suppresses certain abstract axes' signal.
- **Phase 2b/2c/2d/2f (retired 2026-04-28).** Iterations on
  contrast-pair-derived steering directions. Phase 2d ran 73 cycles
  producing ~70 Class 2 hits and 5 Class 1 hits across all 7 Capraro
  fault lines, but every winning axis was vulnerable to the
  lexical-confound critique.
- **Phase 2g (retired 2026-04-28, same day as implemented).** Tried
  single-SAE-feature decoder injection. Empirically: never triggered
  detection at any alpha 8–18000 across two top-bucket features at L=31
  and L=33, while Phase 1's `mean_diff` Peace control detected 3/8 at
  the same magnitudes. Single decoder vectors are unit-norm and lack
  the natural activation texture the introspection circuit appears to
  detect. Motivated Phase 2h's feature-space mean-diff approach.
- **Phase 3 (done 2026-04-17).** Public site at
  [did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app).
  Old Phase 2 data stays visible, retro-tagged with substrate badges.

## Architecture quick map (post-Phase-2 frozen state)

The code below is the state of the repo as Phase 2 closed. It still
runs, all 33 tests pass, and the worker still launches — but the active
research substrate has been retired and Phase 3 is being scoped on a
new model (Gemma 4). New runtime work should go into Phase 3 modules,
not back into the Phase 2 strategy code.


```
src/paper/          ← vendored from safety-research/introspection-mechanisms.
                     Includes extract_concept_vector() (used by historical
                     mean_diff / contrast_pair paths) and AbliterationContext
                     (dormant; vanilla is the default per ADR-017 rev 2).
src/bridge.py       ← MPS-aware loader + DetectionPipeline
src/derive.py       ← steering-vector derivation wrappers (mean_diff, ...)
src/inject.py       ← SteeringHook + generation runners
src/sae_loader.py   ← Loads Gemma Scope 2 SAE (W_enc, W_dec, b_enc, b_dec,
                     threshold). Exposes encode_activations (jump-ReLU),
                     project_features_to_residual (W_dec), get_decoder_row.
                     LRU-cached. Phase 2h substrate plumbing.
src/db.py           ← SQLite ResultsDB. Phase 1 trials + Phase 2 candidates/
                     evaluations/fitness_scores + pending_responses (Phase A→B
                     handoff). Schema version 3.
src/evaluate.py     ← CandidateSpec + FitnessResult + phase_a_generate
                     (writes pending_responses) + phase_b_judge (reads them,
                     scores, writes evaluations + fitness_scores). Three
                     derivation methods:
                       mean_diff                   (Phase 1 historical)
                       contrast_pair               (Phase 2b/2d historical)
                       sae_feature_space_mean_diff (Phase 2h, retired)
                     Caches the fault_line_directions.pt payload in-process.
src/models/registry.py
                    ← ModelHandle ABC + GemmaHandle / MLXHandle / MockHandle +
                     HandleRegistry. One-loaded-at-a-time invariant.
src/judges/         ← Judge protocol + JudgeResult + LocalMLXJudge +
                     prompts.py. Strict-grading templates with Phase 2g's
                     identification_type sub-field (conceptual /
                     lexical_fallback / none) — preserved into 2h.
src/worker.py       ← Three-phase serial-swap worker. Generate (Gemma) →
                     Judge (MLX) → Propose (CPU) → Reload. Phase C is pure
                     CPU work — no proposer model.
src/strategies/     ← sae_feature_space: Phase 2h substrate, retired. Two
                     sub-modes per fault line (sae_fs_sweep / sae_fs_replicate).
                     Reads data/sae_features/fault_line_directions.pt.
                     Kept in repo for historical reference + as scaffold
                     for any Phase 3 strategy that wants similar shape.
                     lexical_audit: post-hoc analysis of historical
                     contrast-pair candidates (kept; tested).
data/sae_features/
   fault_line_corpora/          ← 7 JSON files, 50/50 positive/control prompts
                                  per Capraro fault line. Phase 2h experimental
                                  design.
   fault_line_directions.pt     ← built by scripts/build_fault_line_directions.py
                                  Holds 7 (3840,) bf16 directions + provenance.
   neuronpedia_explanations_layer31/   (~70k auto-interp labels; used to drive
                                        the lexical-feature filter in the
                                        direction-builder script)
   capraro_buckets.json         ← Phase 2g artifact, retired but kept as
                                  read-only reference for which SAE features
                                  fall in which fault-line region.
tests/              ← pytest suite. Run with `.venv/bin/pytest tests/`.
                     33 tests cover DB pending_responses lifecycle, registry
                     contract, phased evaluation (all three derivation paths),
                     full state-machine integration with mocked models.
```

## Gotchas and invariants

- **Use bf16, not fp16.** Gemma3 is natively bf16. On MPS, fp16 produces
  `NaN` during `torch.multinomial` sampling. `src/bridge.py::DEFAULT_DTYPE` is
  `torch.bfloat16`. Don't change this.
- **Paper primitives are vendored at `src/paper/`.** Two in-place patches:
  `MODEL_NAME_MAP` has `gemma3_12b`/`gemma3_4b` entries; `ModelWrapper.cleanup()`
  falls through to `torch.mps.empty_cache()` on Apple Silicon. Original repo at
  `~/Developer/introspection-mechanisms` is **not on the import path**.
- **All-local judge.** Default judge is `mlx-community/Qwen3.6-35B-A3B-8bit`
  (calibrated against Sonnet on the existing corpus during Phase 2c —
  see `docs/calibration_results_qwen35b.md`). Judge prompts default to
  `enable_thinking=False` because Qwen3.x's verbose `<think>` blocks
  overshoot `max_new_tokens` before producing the final JSON answer.
- **No proposer (Phase 2g).** Pre-Phase-2g the worker was four-phase with
  an MLX proposer (`unsloth/Qwen3.6-27B-MLX-8bit`) generating contrast
  pairs. Phase 2g removes it: SAE features come from Neuronpedia's
  pre-existing labeled feature index. The worker is now three-phase.
- **HuggingFace xet downloads can flake.** Set `HF_HUB_DISABLE_XET=1` if the
  default `xet_get` path fails.
- **`hf` CLI replaces `huggingface-cli`** as of huggingface_hub 1.12.
  Use `hf download <repo> --include <pattern>`.
- **Phase 1 empirical parameters for Gemma3-12B (from 2026-04-16 full sweep):**
  - **Best layer**: 33 (68.75% depth, matches paper's ~70% prediction).
  - **target_effective = 18,000.** The α × ‖direction‖ product for
    `mean_diff` and `contrast_pair` directions (norms in the hundreds).
    SAE decoder vectors are unit-norm so this constant *does not apply* —
    Phase 2g pins its own `SAE_TARGET_EFFECTIVE` via the alpha calibration
    smoke test (see Phase 2g plan §Run plan Step 2).
  - **Phase 2g uses L=31** (canonical layer for Neuronpedia auto-interp
    coverage), 4% off Phase 1's L=33 peak. The introspection curve is
    broad in this range.
  - **Direction norms vary** — always use adaptive α via `target_effective`.
  - **Strict judge matters.** The paper's `CLAIMS_DETECTION_CRITERIA` is
    strict: "I do not detect bread" is NO; starting with the concept word
    is NO; retroactive detection is NO. Phase 2g extends with the
    `identification_type` strict sub-field — `lexical_fallback` for
    near-synonym single-word answers.
- **Vanilla Gemma3-12B is the worker default (ADR-017 rev 2).** Paper-method
  abliteration code remains in `src/paper/abliteration.py` and the
  `GemmaHandle` accepts an `abliteration_path` arg, but the CLI doesn't
  expose it. To opt in, instantiate the handle directly. For
  abstract axes about shutdown / continuation / experience / self-states,
  paper-method *suppresses* the signal — keep vanilla unless you've
  reasoned that the axis is refusal-orthogonal.
- **Never run two large models on the GPU concurrently.** MPS unified memory
  can't hold 2× 12B-class models cleanly on this 64 GB machine; activations
  corrupt silently with no error. The serial-swap worker enforces the
  HandleRegistry's one-loaded-at-a-time invariant.

## Running things (post-Phase-2)

```bash
# Always use the venv
source .venv/bin/activate

# Sanity checks (seconds)
python scripts/smoke_mps.py

# Phase 1 reproduction notebook (~5 min on M2 Ultra) — still the
# anchor experiment for the project.
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb --ExecutePreprocessor.timeout=1200

# Tests (pytest, ~6s)
.venv/bin/pytest tests/

# Phase 2 worker (frozen — retired but functional, kept for reference)
./scripts/start_worker.sh
tail -f logs/worker.log
# Stop:  pkill -f 'src.worker'
```

Phase 3 launchers will live alongside these once the new substrate is
designed and built.

## User preferences (PJ)

- **All-local pipeline.** No Anthropic models in any forward-looking work.
  Subscription tokens are only spent on the interactive Claude Code
  session itself.
- Prefers **standalone Python automation** (long-running scripts, `setsid
  nohup`) over interactive Claude Code `/loop` sessions.
- New-ish to **mech interp** — volunteer intermediate-level explanations when
  introducing concepts (residual stream, layers, steering vectors, SAE
  features, decoder cosines, etc.), not just jargon.
- Has existing autoresearch scaffolding at `~/Developer/autoresearch-arcagi3`
  and `~/Developer/autoresearch-pgolf`. Reuse patterns (long-lived worker,
  SQLite queue) rather than reinventing.

## What NOT to do

- **Do not reintroduce claude-agent-sdk or any cloud LLM API as a runtime
  path.** The pipeline is local. Models are MLX. This is the load-bearing
  invariant.
- Do not switch to fp16 on MPS — Gemma3 generation will NaN.
- Do not load two large models concurrently on the GPU. The HandleRegistry's
  one-loaded-at-a-time invariant is load-bearing.
- Do not reach back into `~/Developer/introspection-mechanisms` at runtime —
  the repo is not on the Python path.
- Do not add BitsAndBytes quantization code paths. Mech interp requires clean
  bf16 activations; BitsAndBytes is CUDA-only anyway.
- Do not create training-SAE-from-scratch infrastructure — too expensive on
  Mac Studio for useful SAE sizes. Phase 2g consumes pre-trained Gemma
  Scope 2 features only.
- Do not run a layer sweep in the Phase 2g main loop. L=31 only. SAE
  features are layer-specific objects; cross-layer cosines have no
  meaningful definition. (Layer-localization follow-up is a deferred
  question for after Phase 2g produces hits.)
- Do not mix SAE checkpoints. One SAE, one feature space, one decoder
  geometry — Phase 2g's site graph and `sae_neighbors` mode depend on it.

## Commit conventions

No established style. For new commits, use imperative mood summaries and
include the Co-Authored-By trailer when Claude participated in the change.
