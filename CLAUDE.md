# CLAUDE.md

Orientation for Claude Code sessions working on this repo.

## Project one-liner

Reproduce the Macar et al. (2026) *Mechanisms of Introspective Awareness in
Language Models* paper on `Gemma3-12B-it` locally (Mac Studio M2 Ultra, MPS,
bf16), then build a two-tier autoresearch loop that hunts for novel steering
directions affecting introspection capability.

## Canonical references (read first if orienting)

- Full spec: `docs/01_introspection_steering_autoresearch.md`
- Approved plan: `/Users/pj4533/.claude/plans/lexical-mixing-unicorn.md`
- README: `README.md` (repo layout, setup, running the MVP)

## Status snapshot

- Phase 1 MVP (single-concept end-to-end): **green**. See
  `notebooks/01_reproduce_paper.ipynb`.
- Phase 1 full sweep (50 concepts × layer grid, `data/results.db`,
  `verify_phase1.py` acceptance): **not started**.
- Phase 2 (`src/evaluate.py`, `src/db.py`, `src/worker.py`, `src/researcher.py`,
  `src/strategies/`, `dashboard/app.py`, `scripts/start_{worker,researcher}.sh`):
  **not started**. Directories scaffolded, files empty.

## Architecture quick map

```
src/paper/      ← vendored from safety-research/introspection-mechanisms
src/bridge.py   ← our MPS-aware loader + DetectionPipeline
src/derive.py   ← steering-vector derivation wrappers (mean_diff, ...)
src/inject.py   ← SteeringHook + generation runners (re-exports from paper)
src/judges/     ← Claude judge via claude-agent-sdk subscription OAuth
```

## Gotchas and invariants

- **Use bf16, not fp16.** Gemma3 is natively bf16. On MPS, fp16 produces
  `NaN` during `torch.multinomial` sampling. `src/bridge.py::DEFAULT_DTYPE` is
  `torch.bfloat16`. Don't change this.
- **Paper primitives are vendored at `src/paper/`.** Two in-place patches:
  `MODEL_NAME_MAP` has `gemma3_12b`/`gemma3_4b` entries; `ModelWrapper.cleanup()`
  falls through to `torch.mps.empty_cache()` on Apple Silicon. Original repo at
  `~/Developer/introspection-mechanisms` is **not on the import path** — it was
  reverted to pristine state. If upstream ships changes, port by hand into
  `src/paper/`.
- **Claude judge uses Claude Code subscription OAuth.** Never add
  `ANTHROPIC_API_KEY` requirements. The `claude-agent-sdk` inherits auth from
  the local Claude Code install. Default judge model: Haiku 4.5 for Phase 2
  speed (`claude-haiku-4-5-20251001`); Sonnet 4.6 for Phase 1 reproduction
  rigor (`claude-sonnet-4-6`).
- **Judge is blocking-sync from Jupyter.** `ClaudeJudge._run_sync` spawns a
  worker thread because Jupyter already runs an asyncio loop and
  `asyncio.run()` would raise "Already running asyncio in this thread".
- **HuggingFace xet downloads can flake.** Set `HF_HUB_DISABLE_XET=1` if the
  default `xet_get` path fails.
- **Phase 1 empirical parameters for Gemma3-12B:** layer 33 (70% depth) at α=4
  is the validated sweet spot. α≥6 produces "bread bread bread..." degeneracy
  (coherency filter marks invalid). Sweep around this center point.

## Running things

```bash
# Always use the venv
source .venv/bin/activate

# Sanity checks (seconds)
python scripts/smoke_mps.py
python scripts/smoke_judge.py

# MVP notebook (~5 min on M2 Ultra)
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb --ExecutePreprocessor.timeout=1200
```

## User preferences (PJ)

- Uses **Claude Code subscription OAuth**, not API billing. Anything that calls
  Claude must go through `claude-agent-sdk` or a subscription-aware path.
- Wants the **newest Claude models** — `claude-opus-4-6`, `claude-sonnet-4-6`,
  `claude-haiku-4-5-20251001`. No GPT-4o or older Claude 3.x.
- Prefers **standalone Python automation** (long-running scripts, `setsid
  nohup`) over interactive Claude Code `/loop` sessions. Claude Agent SDK is
  fine for the "agent does research" layer.
- New-ish to **mech interp** — volunteer intermediate-level explanations when
  introducing concepts (residual stream, layers, steering vectors, etc.),
  not just jargon.
- Has existing autoresearch scaffolding at `~/Developer/autoresearch-arcagi3`
  and `~/Developer/autoresearch-pgolf`. Reuse patterns (two-tier researcher/
  worker, SQLite queue, three-tier fitness screening) rather than reinventing.

## What NOT to do

- Do not add `ANTHROPIC_API_KEY` as a hard requirement for the judge.
- Do not switch to fp16 on MPS — Gemma3 generation will NaN.
- Do not reach back into `~/Developer/introspection-mechanisms` at runtime —
  the repo is not on the Python path.
- Do not add BitsAndBytes quantization code paths. Mech interp requires clean
  bf16 activations; BitsAndBytes is CUDA-only anyway.
- Do not create training-SAE-from-scratch infrastructure — too expensive on
  Mac Studio for useful SAE sizes. SAE work (if any) consumes pretrained
  Gemma Scope 2 features.

## Commit conventions

No established style yet (repo is fresh). For new commits, use imperative
mood summaries and include the Co-Authored-By trailer when Claude participated
in the change.
