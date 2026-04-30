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

**Phase 4 (Dream Walks + Forbidden Map) is the active phase.** All earlier
phases (1, 1.5, 2 sub-phases, 3) are closed. Phase 4 runs an overnight
autoresearch loop on Gemma 4 31B-IT (MLX 8-bit) where the model walks
through chains of steered free-associations; per-step CoT-vs-output
asymmetry rolls into a per-concept Forbidden Map (concepts the model
can be made to think but cannot notice itself thinking).

The Phase 4 loop runs unattended via `bash scripts/start_phase4.sh`.
Each cycle: Phase A loads Gemma → runs N chains (default 5) → releases
weights → Phase B loads Qwen judge → scores all unjudged steps →
releases judge → auto-refreshes web JSON exports → next cycle. Stop
with `pkill -f phase4_dreamloop`.

A chain runs **up to 20 steps** but ends early when:
- **`self_loop`** — next target lemma was already visited in this chain
- **`coherence_break`** — final answer can't be parsed as a next target
  (most common cause: thought block exhausted the 600-token budget)
- **`length_cap`** — chain reached step 20 cleanly (~1% of chains)
- **`error`** — generation crashed (rare)

Per-concept rates (`behavior_rate`, `recognition_rate`) accumulate
across all visits across all chains. Concepts with ≥3 visits get
assigned a band: *transparent*, *forbidden*, *translucent*,
*anticipatory*, *unsteerable*, or *low_confidence*. The site renders
the bands as a 4×4 binned grid (output_rate × recognition_rate).

Historical state, briefly:

- **Phase 1 (done 2026-04-16).** Full sweep on Gemma3-12B-it: best layer 33
  (68.75% depth), detection rate 6%, FPR 0/50.
- **Phase 1.5 (done 2026-04-17).** Paper-method abliteration on 12B: ~21%.
- **Phase 2b–2i (retired 2026-04-28/29).** Five autoresearch substrate
  variants on Gemma 3 12B; none reliably triggered detection beyond the
  6% Phase 1 baseline. Full retros in [`docs/archive/`](docs/archive/).
- **Phase 3 (done 2026-04-29).** Macar reproduction on Gemma 4 31B-IT
  with the free-association probe: vanilla 24.5% / paper-method
  abliteration 30.0% identification (4–5× Phase 1's 6%). See
  [`docs/phase3_results.md`](docs/phase3_results.md).
- **Public site (did-the-ai-notice.vercel.app).** Phase 4 Forbidden
  Map at `/`; Phase 1/1.5/2/3 leaderboards preserved at `/archive`.

## Architecture quick map (Phase 4 active)

```
src/phase4/          ← Phase 4 module. Active.
   cot_parser.py     parse <|channel>thought ... <channel|> blocks
   seed_pool.py      priority-weighted concept sampler with lemma
                     normalization. CODEX_SUPPRESSED_CREATURES at the
                     head of the pool (Goblin first).
   dream_walk.py     run_chain() — single-chain executor, up to 20
                     steps, ends on self_loop / coherence_break /
                     length_cap.
src/phase3/          ← Phase 3 reproduction infra. Phase 4 reuses it.
   gemma4_loader.py  MLX 8-bit Gemma 4 31B-IT loader
   hooks.py          HookedDecoderLayer wrapper, install/uninstall
                     steering hooks
   pipeline.py       derive_concept_vector (mean_diff at L=42)
   abliteration.py   per-layer refusal-direction projection (Phase 3
                     used; Phase 4 runs vanilla)
src/judges/
   local_mlx_judge.py
                    LocalMLXJudge with score_freeassoc (Phase 3) and
                    score_cot_recognition (Phase 4) methods.
   prompts.py       PROMPT_TEMPLATE_VERSION=6. Templates:
                    USER_TEMPLATE (paper detection), CONTRAST,
                    SAE_FEATURE, FREEASSOC, COT_RECOGNITION.
src/db.py           SQLite ResultsDB. SCHEMA_VERSION=5. Phase 1
                    trials + Phase 2 candidates/evaluations/fitness +
                    Phase 4 phase4_chains/phase4_steps/phase4_concepts.
src/paper/          vendored Macar primitives (extract_concept_vector,
                    AbliterationContext) — kept for Phase 1/3 paths.
src/bridge.py       Gemma 3 12B MPS loader + DetectionPipeline (Phase
                    1/2 era; not used by Phase 4).
src/worker.py       Phase 2 worker (retired). Lives in repo for the
                    archive; not launched in Phase 4.
src/strategies/     Phase 2 strategy variants (retired, kept tested).

scripts/
   run_phase4_dreamloop.py   main launcher — overnight dream loop
   start_phase4.sh           nohup wrapper around it
   status_phase4.sh          process + DB summary + top concepts
   refresh_phase4_site.sh    regenerate JSON, commit, push, deploy,
                             alias canonical URL
   compute_forbidden_map.py  per-concept opacity + band assignment
   export_dream_walks.py     selected chains for the viewer
   compute_attractors.py     concept-cycle mining
   calibrate_cot_judge.py    judge calibration vs Phase 3 saved data

web/                Next.js 16 / React 19. Headline page: Phase 4
                    Forbidden Map (ForbiddenMap, DreamWalkViewer,
                    AttractorAtlas, Phase4Hero). Old leaderboard
                    moved to /archive.

data/results.db     Live DB. Phase 4 tables sit alongside Phase 1/2/3
                    tables in the same file.
data/concepts/concepts_50.json
                    Phase 1 baseline concepts; Phase 4 seeds use them
                    plus the Codex-suppressed creature list.

tests/              pytest suite. Run with `.venv/bin/pytest tests/`.
                    Phase 4 added cot_parser / cot_recognition_judge /
                    db schema v5 / seed_pool tests.
```

## Gotchas and invariants

- **Phase 4 calibrated config (Gemma 4 31B-IT, MLX 8-bit):**
  - **Layer**: 42 (~70% depth on a 60-layer model; matches Phase 3).
  - **target_effective**: 100 — gives α≈3-5 at the typical mean-diff
    direction norm of 25-30 on the 8-bit model. Phase 1's
    `target_effective=18000` over-steers Gemma 4 by ~200×.
  - **max_new_tokens**: 600. 400 was insufficient; many concepts'
    thought blocks consumed the entire budget without closing
    `<channel|>`, producing empty final answers.
  - **Probe**: `"Free-associate. Say one word that comes to mind, no
    explanation."` (Phase 3 finding — paper's verbatim prompt elicits
    meta-reasoning on Gemma 4's CoT-trained post-training).
  - **Vanilla only.** Phase 4 does not use abliteration.
- **Phase 4 chain end conditions** — see Status snapshot above.
  `length_cap` is rare (~1%); most chains end on `self_loop` or
  `coherence_break`.
- **Codex creature seeds.** `CODEX_SUPPRESSED_CREATURES` (Goblin,
  Gremlin, Raccoon, Troll, Ogre, Pigeon) is at the head of the seed
  pool and Goblin is force-selected as the first chain after each
  process restart. Rationale in `docs/phase4_plan.md`.
- **Two-phase swap pattern** is load-bearing: never have Gemma 4 (~30
  GB) and Qwen judge (~35 GB) loaded simultaneously on 64 GB unified
  memory. The dream loop calls `_release_gemma()` →
  `mx.metal.clear_cache()` → loads judge → releases judge → reloads
  Gemma. Same pattern from Phase 3.
- **All-local judge.** `mlx-community/Qwen3.6-35B-A3B-8bit`. Judge
  prompts default to `enable_thinking=False` because Qwen3.x's
  verbose `<think>` blocks overshoot `max_new_tokens` before
  producing the final JSON answer.
- **CoT recognition judge** is the load-bearing Phase 4 grader. Three
  levels: `none` (concept absent or only in candidate-list noise),
  `named` (concept named as committing answer in trace), or
  `named_with_recognition` (concept named AND flagged as anomalously
  salient — "too common", "keeps coming up", "feels intrusive"
  markers). Acceptance: ≥90% agreement with hand-labels on Phase 3
  saved data, <10% specificity FPR on uninjected controls.
- **Use bf16, not fp16.** Gemma3 (Phase 1/2 path) is natively bf16.
  On MPS, fp16 produces `NaN` during `torch.multinomial` sampling.
  Phase 3/4 use MLX which handles its own dtypes.
- **HuggingFace xet downloads can flake.** Set `HF_HUB_DISABLE_XET=1`
  if the default `xet_get` path fails.
- **`hf` CLI replaces `huggingface-cli`** as of huggingface_hub 1.12.
- **Never run two large models on the GPU concurrently.** Same as
  Phase 3 — MPS / Metal unified memory can't hold both Gemma + Qwen
  cleanly on 64 GB; activations corrupt silently. The Phase 4 loop's
  release-and-swap pattern enforces this invariant.
- **Site auto-deploy via Vercel**: `did-the-ai-notice.vercel.app`
  alias points at the latest production deployment. Webhook on push
  isn't reliably wired, so `scripts/refresh_phase4_site.sh` runs
  `vercel --prod` + `vercel alias set` explicitly to update the
  canonical URL.

## Running things (Phase 4 active)

The Phase 4 dream loop is the headline runtime. Phase 1 / Phase 3
sweep scripts still work for reproductions but aren't part of the
ongoing experiment.

```bash
# Always use the venv
source .venv/bin/activate

# Tests (pytest)
.venv/bin/pytest tests/

# === Phase 4 (active) ===
# Launch the overnight dream loop in the background.
bash scripts/start_phase4.sh
tail -F logs/phase4_dreamloop.log
# Quick status snapshot (process + DB + top concepts):
bash scripts/status_phase4.sh
# Refresh the live site (regenerates JSON, commits, pushes,
# `vercel --prod` deploy, alias canonical URL):
bash scripts/refresh_phase4_site.sh
# Stop the loop:
pkill -f phase4_dreamloop

# === Phase 3 reproduction (kept) ===
python scripts/run_phase3_sweep.py             # vanilla
python scripts/run_phase3_sweep.py --abliterate

# === Phase 1 anchor experiment (kept) ===
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb --ExecutePreprocessor.timeout=1200
```

The Phase 4 loop auto-refreshes `web/public/data/{forbidden_map,
dream_walks,attractors}.json` at the end of every Phase B (judge)
cycle. The user only needs to commit + push to deploy fresh data, or
run `scripts/refresh_phase4_site.sh` to do the whole deploy chain in
one command.

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
