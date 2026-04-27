# Local-Only Pipeline Plan

**Status:** TOP PRIORITY. Approved direction 2026-04-27 (PJ).
**Goal:** Replace all cloud Claude calls (Sonnet 4.6 judge, Opus 4.7 proposer) with
locally-run MLX models on the Mac Studio. The only token usage going forward should
be the interactive Claude Code session itself (this conversation). Research must
be **completely local AND completely reliable** — reproducibility, deterministic
behavior, and freedom-to-iterate are valued above latency.

## Why this exists

- Subscription-token usage is the bottleneck that has paused every overnight run.
- Cloud judge / proposer means a) cost gates research direction and b) silent
  model upgrades break calibration on long-running studies.
- Mac Studio is paid-for, idle when not running Gemma, and has the headroom to do
  the proposer/judge work itself if we structure memory correctly.

## Hardware reality (verified 2026-04-27)

```
Model:  Mac Studio M2 Ultra
Chip:   Apple M2 Ultra (24 cores: 16 perf + 8 eff)
Memory: 64 GB unified (NOT 192 GB — earlier estimate was wrong)
        Memory bandwidth: ~800 GB/s
        bf16 native (FEAT_BF16=1)
Disk:   926 GB total, ~100 GB free, NVMe SSD
OS:     Darwin 25.2.0 (macOS 16.x)
```

**Memory is the dominant constraint — every model decision below is keyed off the 64 GB
budget.** macOS reserves ~8-10 GB for itself, leaving ~54 GB usable.

## Memory budget

**Strict one-model-at-a-time. No co-residence. Each role gets the full memory
budget when active.** PJ's call 2026-04-27: prefer best-quality model per role
over swap-time savings. Swap waits are acceptable.

| State | Resident weights | Free for activations / KV cache / OS |
|-------|-----------------:|-------------------------------------:|
| Gemma3-12B bf16 alone | ~24 GB | ~30 GB |
| Judge alone (Gemma swapped out) | up to ~50 GB | depends on model |
| Proposer alone (Gemma swapped out) | up to ~50 GB | depends on model |
| Two models concurrently | NEVER — silent activation corruption (CLAUDE.md gotcha) |

**Implication:** with Gemma swapped out, the judge and proposer each get the
full ~50 GB. We can pick BIGGER, higher-quality models for each role than a
co-resident plan would allow.

## Architecture: three-phase serial lifecycle, dedicated model per role

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase A: GENERATE                                                  │
│    Loaded: Gemma3-12B (alone, ~24 GB)                               │
│    For up to BATCH_SIZE candidates from queue/pending:              │
│      derive contrast vector → run 8 inj + 4 ctrl probes →           │
│      store all 12 responses (text only) to a per-candidate          │
│      pending-judgment record in SQLite.                             │
│    Generation is judge-free at this stage.                          │
│    BATCH_SIZE = 16 (one Opus cycle's worth) so swap cost amortizes. │
│                                                                     │
│  Phase B: JUDGE                                                     │
│    1. Unload Gemma3-12B (~5s)                                       │
│    2. Load judge model (~30-60s)                                    │
│    3. Score all pending judgments from Phase A (12 × BATCH_SIZE     │
│       judge calls, e.g. 192 for batch of 16).                       │
│    4. Write fitness scores to data/results.db.                      │
│    5. Unload judge (~5s).                                           │
│                                                                     │
│  Phase C: PROPOSE (only when queue is low)                          │
│    1. Load proposer (~30-60s)                                       │
│    2. Run proposer with feedback block from Phase B results.        │
│       Generate N candidate contrast pairs (typically 16).           │
│    3. Write candidate specs to queue/pending.                       │
│    4. Unload proposer (~5s).                                        │
│                                                                     │
│  Phase D: RELOAD                                                    │
│    Load Gemma3-12B (~30-60s). Return to Phase A.                    │
└─────────────────────────────────────────────────────────────────────┘

Per-batch swap cost: ~2-3 min of model-swap I/O.
At BATCH_SIZE=16 candidates × ~30s eval each = ~8 min of compute per batch.
Overhead: ~25-30%. Slower than co-resident but every model runs at peak
quality with full RAM.

If Phase C is skipped (queue still has work), batch cost drops to ~1-2 min
swap + 8 min compute = ~15% overhead.
```

**Why batch-then-judge instead of judge-each-candidate:** judging per-candidate
would require ~600 model swaps per day at current throughput (60-120 min of
pure I/O daily). Batching to 16 candidates per judge load drops swap count
~16×, bringing it to ~40 swaps/day or ~4-8 min of I/O daily. Same correctness,
fraction of the wait.

**Why serial, not concurrent:** CLAUDE.md gotcha #5 documents that two large
models concurrently on this machine corrupts activations silently with no
errors. Mixing PyTorch-MPS and MLX runtimes adds further risk. Serial is the
only safe option and reliability is the explicit goal.

**State durability across swaps:** queue is already SQLite + filesystem (`queue/pending`,
`data/results.db`). Judge cache is keyed by model name, so old Sonnet entries don't
collide with new local-judge entries. New piece of state: a "pending-judgment"
table or column to hold Phase A responses until Phase B scores them. Crash
recovery: if the worker dies between A and B, the stored responses are still
valid; we just resume at Phase B.

## Model recommendations (verified 2026-04-27 against current HF / MLX state)

**With ~50 GB available per role (Gemma swapped out), pick the best dedicated
model for each task. No unification — judge and proposer are different models
with different cognitive demands.**

### Judge — primary: **Qwen3.6-35B-A3B 8-bit (MLX)**

- **HF:** `mlx-community/Qwen3.6-35B-A3B-8bit` (8-bit, not 4-bit)
- **Released:** April 2026
- **Architecture:** 35B total / 3B active per token (256 experts, 8+1 active)
- **Disk:** ~38 GB; **runtime RAM:** ~40-42 GB
- **Context:** 256K native, extensible to 1M via YaRN
- **Benchmarks:** GPQA Diamond 86.0, MMLU-Pro 85.2, AIME25 ~85, SWE-bench Verified 73.4
- **Why for judge:** Native thinking-mode (`<think>` blocks) gives chain-of-thought
  grading for free, which reduces judge variance on borderline cases. **8-bit
  quantization preserves near-bf16 fidelity** — critical for strict semantic
  grading where 4-bit drift could shift Class 0 ↔ Class 2 boundaries. 3B active
  per token still gives fast inference (~30-50 tok/s) so 8-bit doesn't hurt
  throughput much vs 4-bit.
- **Risk:** slightly more lenient than Sonnet on edge cases — Day 1 calibration
  step is the gate.

### Judge — alternate: **GLM-4.7-Flash 8-bit (MLX)**

- **HF:** `mlx-community/GLM-4.7-Flash-8bit` (or 6-bit if 8-bit not yet up)
- **Released:** December 2025
- **Architecture:** ~30B / ~3B active (MoE)
- **Disk:** ~32 GB at 8-bit; **runtime RAM:** ~34-36 GB
- **Why:** three-tiered thinking modes (interleaved, preserved, turn-level) match
  rubric-grading patterns. MLA-style KV cache compression saves ~93% memory on
  long judge prompts. MIT license.
- **Risk:** Z.ai models occasionally produce Chinese tokens on edge inputs;
  pin temperature low and add a regex post-filter. Use only if Qwen judge
  fails calibration.

### Judge — speed-first fallback: **Phi-4-reasoning-plus 8-bit (MLX)**

- **HF:** `mlx-community/Phi-4-reasoning-plus-8bit` (or use the 4-bit variant
  `lmstudio-community/Phi-4-reasoning-plus-MLX-4bit`)
- **Architecture:** 14B dense, reasoning-tuned
- **Disk:** ~16 GB at 8-bit; **runtime RAM:** ~18-20 GB
- **Why:** if Qwen3.6-35B-A3B turns out too lenient, a small reasoning-tuned
  model with strict math/logic priors might track Sonnet's strict criteria
  better. Beats DeepSeek-R1-distill-70B on reasoning benchmarks per Microsoft.
- **Risk:** 14B is small for abstract semantic-gist grading. Probably collapses
  to keyword matching on Capraro-style abstract axes. Try only if both Qwen
  and GLM fail.

### Proposer — primary: **Qwen3.6-27B 8-bit (MLX)**

- **HF:** `unsloth/Qwen3.6-27B-MLX-8bit`
- **Released:** April 22, 2026 (5 days old)
- **Architecture:** 27B dense, native thinking mode
- **Disk:** ~28 GB; **runtime RAM:** ~32-34 GB at 8-bit
- **Benchmarks vs Claude 4.5 Opus:**
  - MMLU-Pro: 86.2 vs 89.5
  - **GPQA Diamond: 87.8 vs 87.0** (beats Opus)
  - AIME 2026: 94.1 vs 95.1
  - LiveCodeBench v6: 83.9 vs 84.8
- **Why for proposer:** the only currently open-weight model that benchmarks
  within striking distance of Opus 4.5 on creative reasoning. Dense 27B at
  8-bit retains near-bf16 quality. Plenty of headroom for the long structured
  prompts the proposer uses (~8-12 K input tokens).
- **Risk:** verbose `<think>` blocks before final output — existing
  `novel_contrast.py` parser already handles this style.

### Proposer — alternate: **Qwen3.5-27B-Claude-4.6-Opus-Distilled MLX (4-bit or 8-bit)**

- **HF:** `mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit`
- **Architecture:** 27B dense, distilled from Opus 4.6 reasoning trajectories
- **Disk:** 14 GB at 4-bit (or download 8-bit if available)
- **Why:** literally distilled from a near-clone of our previous proposer.
  Inherits Opus's deconstruction-of-prompts and self-correction style — the
  cognitive style needed to invent abstract axes like
  `live-narration-vs-retrospective-report`. Use this if the vanilla Qwen3.6-27B
  produces lazy or generic axis names during Day 3 hand-inspection.
- **Risk:** community model — no formal benchmarks vs Opus 4.7. Treat as
  experimental.

### Proposer — long-shot stretch: **GLM-4.6-Air 4-bit (MLX)**

- **HF:** `mlx-community/GLM-4.6-Air-MLX-4bit` if available
- **Architecture:** ~107B MoE / ~12B active
- **Disk:** ~50 GB at 4-bit; **runtime RAM:** ~52-54 GB ← right at the limit
- **Why:** if 27B proposer feels capped, a frontier-class MoE in the same
  RAM budget gives more raw capacity. Test only after the Qwen-27B and
  Opus-distilled paths are validated.
- **Risk:** at the absolute memory edge; one bad context-length spike OOMs.
  Don't ship as primary. Reserve as a "if we feel research is constrained"
  experiment.

### Models explicitly **NOT** to use

- **Qwen3-Next-80B-A3B-Instruct-MLX-4bit** — confirmed broken on M2 Ultra
  ([mlx-lm issue #492](https://github.com/ml-explore/mlx-lm/issues/492)): 0.4 tok/s
  generation, 1.8 tok/s prompt processing. DeltaNet hybrid-attention kernels lack
  M2 Ultra optimization. Works on M3/M4 Ultra; useless here. Re-check after a
  hardware upgrade.
- **DeepSeek V4 (April 2026)** — 284B / 13B active, 160 GB on disk, requires 128 GB
  Apple Silicon. Aspirational, not viable on 64 GB.
- **GLM-4.7 full** — 358B / 32B active, MLX 4-bit is 199 GB on disk. Use the Flash
  variant (~17 GB) if you want a GLM judge.

## Implementation plan (~1 week)

**Each step ends with a commit and a working state. Don't move on until validated.**

### Day 1 — Judge calibration (gating experiment)

This is the single test that decides whether the local plan proceeds. If a local
judge can't agree with Sonnet on the existing corpus, abandon the plan and revert
to the cost-cutting tactics in the previous proposal (eval-set reduction, etc).

1. `pip install -U mlx mlx-lm mlx-vlm huggingface-hub` in the existing venv.
2. Download Qwen3.6-35B-A3B-8bit (NOT 4-bit) to `~/models/`:
   ```
   huggingface-cli download mlx-community/Qwen3.6-35B-A3B-8bit \
     --local-dir ~/models/Qwen3.6-35B-A3B-8bit
   ```
   Set `HF_HUB_DISABLE_XET=1` if xet flakes (already a known gotcha).
   8-bit is ~38 GB on disk vs 4-bit's 20 GB; we have the budget and need the
   precision for strict grading.
3. Write `src/judges/local_mlx_judge.py` — drop-in replacement for `claude_judge.py`,
   same `_run_sync` thread pattern, same `JudgeResult` return type. Caches in SQLite
   under a new model namespace so old Sonnet entries don't collide.
4. Bump `PROMPT_TEMPLATE_VERSION` in `claude_judge.py` (or split into separate const
   for local vs cloud).
5. Build `scripts/calibrate_local_judge.py`:
   - Sample 200 candidates from the existing `directed_capraro_*` results.
   - Re-run the local judge on each.
   - Compute per-axis agreement rate vs the original Sonnet score on
     `(detected, identified, coherent)` triples.
   - **Acceptance criteria:**
     - Class 2 (det only) agreement ≥ 95% (we saw 30+ Class 2 hits in Causality —
       these define the signal floor).
     - Class 0 (null) agreement ≥ 90% (FPR contamination is the dangerous failure).
     - On the 5 known Phase 1 true positives, exact-match all three judge fields.
6. If criteria pass: commit, proceed to Day 2.
   If criteria fail: try Phi-4-reasoning-plus first (smaller but reasoning-tuned),
   then GLM-4.7-Flash, then escalate to user.

### Day 2 — Model lifecycle abstraction (4-phase state machine)

1. Refactor model loading into `src/models/registry.py`:
   ```python
   class ModelHandle:
       name: str
       def load(self): ...
       def unload(self): ...   # must release Metal allocations cleanly
       def is_loaded(self) -> bool: ...

   GEMMA    = GemmaHandle("gemma3_12b")
   JUDGE    = MLXHandle("qwen3.6-35b-a3b-8bit", path=...)
   PROPOSER = MLXHandle("qwen3.6-27b-mlx-8bit", path=...)
   ```
   Crucial: every handle's `unload()` must `torch.mps.empty_cache()` /
   release Metal heaps. Leaking even 1-2 GB across swaps will OOM the next
   model. Add a sanity check (free memory > 45 GB) before each `load()`.
2. Add a `pending_judgment` table or column to `data/results.db` to hold
   Phase A's responses until Phase B scores them.
3. Refactor `src/worker.py` main loop into the four-phase state machine
   (Generate → Judge → Propose-if-needed → Reload). Drive Phase C off
   `queue_depth < THRESHOLD`, not wallclock.
4. End-to-end smoke test: enqueue 3 candidates manually, run worker, verify
   Phase A → B → A cycle correctly. Then drain queue and verify Phase C
   fires.

### Day 3 — Proposer integration

1. Download proposer:
   ```
   huggingface-cli download unsloth/Qwen3.6-27B-MLX-8bit \
     --local-dir ~/models/Qwen3.6-27B-MLX-8bit
   ```
2. Adapt `src/strategies/novel_contrast.py` and `directed_capraro.py` to call the
   local proposer instead of Claude Agent SDK. The prompt structure and JSON parsing
   stay the same — the only change is the model handle.
3. Run a 3-candidate Phase B end-to-end. Inspect the generated contrast pairs by hand
   for coherence and abstract-axis quality. If pairs are gibberish or lazy
   ("happy-vs-sad" type defaults), fall back to Opus-distilled-27B and retry.

### Day 4 — Wired-up validation sprint

1. Stop the existing Causality worker permanently (already done).
2. Pick the next fault line: **Grounding** (refusal-orthogonal, next on the C1-C4
   queue per docs/phase2d_directed_hypotheses.md).
3. Launch a 24-hour Capraro Grounding sprint with the new local pipeline:
   `./scripts/start_capraro_sprint.sh grounding`
4. **Compare against the Causality baseline:**
   - Hit rate per cycle (we saw 5-7 hits per 16 candidates in productive cycles).
   - Best score (we hit 0.250 on Causality).
   - Coherence rate (8/8 was achievable).
   - FPR (must remain 0/4 — non-negotiable).
   - Identification rate (still expected to be 0; if it's not, the local judge is
     too lenient and we have a bug).
5. If the metrics match Causality within statistical noise: the local pipeline is
   validated. Commit, document, lock in.
6. If detection inflates (judge too lenient) or coherence drops (proposer too weak):
   diagnose layer-by-layer. Likely fix is a stricter judge prompt or swapping
   proposer to Opus-distilled.

### Day 5+ — Production hardening

- Add `scripts/swap_models.sh` for manual model swap during ad-hoc experiments.
- Update `start_worker.sh` and `start_capraro_sprint.sh` so they no longer try to
  hit Anthropic at all — wire them to the local registry.
- Update CLAUDE.md gotchas: remove "Claude judge uses Claude Code subscription OAuth"
  invariant, replace with "judge and proposer are local MLX models. See
  docs/local_pipeline_plan.md".
- Write ADR-019 in docs/decisions.md recording the move and the rationale (zero
  cloud cost + full reliability + freedom-to-iterate beats subscription OAuth).
- Optional: add a `--cloud-judge-fallback` flag to the worker for emergencies, in
  case a local judge bug ships and we need to re-judge from a Sonnet snapshot.

## Risks (and what we'll do about them)

1. **Judge calibration drift** — the highest risk. Mitigation: Day 1 is the explicit
   gating experiment. We don't refactor anything until calibration passes.
2. **Hill-climb amplifies judge bias** — already documented in ADR-015. Mitigation:
   for the first 200 candidates of the local validation run, re-judge a 20-candidate
   sample with Sonnet (manually, via the existing Claude Code session) and confirm
   agreement.
3. **Proposer creativity collapse** — local 27B may not invent abstract axes as
   imaginatively as Opus. Mitigation: Day 3 hand-inspection step. If pairs look
   lazy, swap to Opus-distilled-27B (which carries Opus's inductive bias) before
   committing.
4. **Model swap I/O latency** — ~2-3 min per full A→B→C→A cycle, ~25-30%
   wallclock overhead at BATCH_SIZE=16. Larger than the co-resident plan's
   ~5%, but explicitly accepted (PJ 2026-04-27) in exchange for dedicated
   best-quality models per role. If overhead becomes a problem, raise
   BATCH_SIZE to 32 (halves swap frequency).
5. **MLX framework maturity** — `mlx_lm` is solid but newer than PyTorch. Specific
   model-architecture combinations break (Qwen3-Next on M2 Ultra is the cautionary
   tale). Mitigation: only use models with a confirmed MLX-community release AND
   at least one independent benchmark / smoke-test report. No experimental kernels.
6. **MPS / MLX runtime co-existence** — Gemma runs on PyTorch-MPS; judge/proposer
   run on MLX. Both use Apple's Metal stack. First-time loading MLX after PyTorch
   may trigger a one-time shader recompile (~10s). Annoying but only once per
   process restart.
7. **Disk fills up during model downloads** — only 100 GB free. With the
   8-bit dedicated-model plan: Qwen3.6-35B-A3B-8bit (~38 GB) +
   Qwen3.6-27B-8bit (~28 GB) = ~66 GB just for the primary judge + proposer.
   Plus Opus-distilled-27B-4bit alternate (~14 GB) brings total to ~80 GB.
   Post-download free space: ~20 GB — uncomfortably tight given Gemma's
   already-cached ~24 GB and HuggingFace caches. **Mandatory before Day 1:**
   `huggingface-cli scan-cache` and prune; consider moving older weights to
   external storage. If still tight, defer the Opus-distilled alternate
   download until proposer validation actually fails.

## Success criteria

The plan is done when:

- Worker runs end-to-end with **zero subscription-token usage** outside this
  Claude Code chat session.
- A 24-hour Capraro Grounding sprint produces hit/identification/coherence/FPR
  numbers within ±10% of the Causality baseline.
- The system can be paused and resumed at any phase without data loss or
  recalibration.
- All operational scripts (`start_worker.sh`, `start_capraro_sprint.sh`) work
  without ANTHROPIC_API_KEY or claude-agent-sdk.

## What this plan explicitly does NOT do

- Doesn't introduce concurrent model execution. Serial swap is the contract.
- Doesn't change Gemma3-12B (the evaluator stays bf16 PyTorch-MPS, exactly as
  ADR-001 etc. specify).
- Doesn't touch the fitness function (ADR-018 ident-prioritized stays).
- Doesn't switch contrast-pair derivation method (still mean-diff on example
  sentences).
- Doesn't migrate to a different Mac. M2 Ultra 64GB is the platform; if we ever
  upgrade to M4/M5 Ultra with more RAM, Qwen3-Next-80B becomes viable and the
  whole plan can be revisited at that point.

## Pointers

- Architecture and model rationale: this doc.
- Hardware verification: see Bash output in this conversation, 2026-04-27.
- Existing judge implementation pattern: `src/judges/claude_judge.py` (use as
  the template for `local_mlx_judge.py`).
- Existing proposer implementation pattern: `src/strategies/novel_contrast.py`
  (the `claude-agent-sdk` calls are the only thing that change).
- Calibration set: 5 known Phase 1 true-positive detections (referenced in
  CLAUDE.md gotcha #5 — "Strict judge matters").
- Memory safety invariant: CLAUDE.md gotcha #5 — "Never run two sweeps
  concurrently on the same DB."
