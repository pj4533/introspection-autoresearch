# Local Judge Calibration Kit

**Status:** off-pipeline. None of these scripts touch the live worker / researcher /
public site. Read-only on `data/results.db`. Writes only under `data/calibration/`.

This is the Day-1 gating experiment for the local-pipeline plan
(`docs/local_pipeline_plan.md`). The goal: prove that some MLX-quantized local
judge can replicate Sonnet 4.6's strict semantic grading on the existing corpus
to within tolerance, before we refactor the worker.

The kit is designed so we can sweep multiple candidate judge models with
minimal repeated work. Each model gets its own SQLite cache and verdict file,
and `compare_verdicts.py` produces a single side-by-side report.

## Files

```
scripts/calibrate_judge/
├── README.md                   ← this file
├── build_calibration_set.py    ← samples a stratified set from data/results.db
├── run_local_judge.py          ← runs one MLX model against the calibration set
└── compare_verdicts.py         ← agreement metrics + markdown report

src/judges/local_mlx_judge.py    ← MLX-backed Judge implementation (parallel
                                   to claude_judge.py; not imported by the
                                   live pipeline)
```

Outputs live under:
```
data/calibration/
├── calibration_set.jsonl              ← stratified sample (one-time)
├── verdicts_<model_tag>.jsonl         ← per-model judgments
├── disagreements_<model_tag>.jsonl    ← rows where local != Sonnet
├── judge_cache_<model_tag>.sqlite     ← per-model judgment cache
└── report.md                          ← side-by-side acceptance report
```

## One-time setup

```bash
source .venv/bin/activate
pip install -U mlx mlx-lm huggingface-hub
export HF_HUB_DISABLE_XET=1   # known gotcha; keeps downloads from flaking
```

## Usage

### 1. Build the calibration set (once)

```bash
python scripts/calibrate_judge/build_calibration_set.py
```

This samples the production DB. Defaults are tuned to over-sample the EDGES of
the verdict space, where calibration drift hurts most:

- All 5 Phase-1 known true positives (literal regression fixtures).
- All Class-1 hits ever recorded (det+ident, very rare).
- 60 Class-2 hits (det only — the wall we're trying not to break).
- 60 Class-0 nulls (tests false-positive resistance).
- 30 incoherent cases (tests gibberish recognition).

Override with `--n-class-2`, `--n-class-0`, `--n-incoherent`, `--seed`.

Output: `data/calibration/calibration_set.jsonl`.

### 2. Download a candidate judge model

For the primary candidate from `docs/local_pipeline_plan.md`:

```bash
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-8bit \
  --local-dir ~/models/Qwen3.6-35B-A3B-8bit
```

Other candidates worth trying:
- `mlx-community/Qwen3.6-35B-A3B-4bit` (smaller, faster, may be too lossy)
- `mlx-community/GLM-4.7-Flash-8bit` (alternate; pin temp=0)
- `lmstudio-community/Phi-4-reasoning-plus-MLX-4bit` (speed-first fallback)

### 3. Run one judge against the calibration set

```bash
python scripts/calibrate_judge/run_local_judge.py \
    --model ~/models/Qwen3.6-35B-A3B-8bit
```

This produces `data/calibration/verdicts_Qwen3.6-35B-A3B-8bit.jsonl` (one row
per calibration item, with the local verdict alongside Sonnet's reference
verdict). Resume-aware via `--resume`. Smoke-test via `--limit 20`.

Repeat with each candidate model.

### 4. Generate the side-by-side report

```bash
python scripts/calibrate_judge/compare_verdicts.py \
    --results data/calibration/verdicts_Qwen3.6-35B-A3B-8bit.jsonl \
    --results data/calibration/verdicts_GLM-4.7-Flash-8bit.jsonl \
    --results data/calibration/verdicts_Phi-4-reasoning-plus-MLX-4bit.jsonl
```

Produces `data/calibration/report.md` with:
- Per-model overall agreement %
- Per-stratum breakdown (phase1_known_tp, phase2_class_2, phase2_class_0, etc.)
- Class confusion matrix (where the local judge drifts vs Sonnet)
- Per-flag (detected/identified/coherent) agreement
- PASS/FAIL on the gating criteria

### Acceptance criteria (from local_pipeline_plan.md)

A judge is acceptable if:
- **Class 2 (det only) agreement ≥ 95%** — the wall is the crucial signal.
- **Class 0 (null) agreement ≥ 90%** — false positives are corrosive to long
  hill-climbs.
- **Phase 1 known TPs: exact match all three flags** — the regression
  fixtures must reproduce literally.

Manual review of `data/calibration/disagreements_<model>.jsonl` is the
qualitative pass: if the disagreements look like the local judge being
*stricter* than Sonnet (which would help reduce false positives), accept.
If they look like the local judge missing real signal, reject.

## Tips

- **Don't run during a live worker sweep.** The local judge will load
  ~40 GB of MLX weights, and Gemma's PyTorch-MPS allocations on the same
  machine will OOM. Calibration is a stop-the-world experiment.
- **Cache is per-model.** Re-running `run_local_judge.py` with the same
  model is free after the first pass (uses `data/calibration/judge_cache_<tag>.sqlite`).
- **Bigger calibration sets are cheap.** A 200-item set takes 10-20 minutes
  on Qwen3.6-35B-A3B-8bit. Bump `--n-class-2 200 --n-class-0 200` if you
  want tighter confidence intervals on the agreement %.
- **The reference (Sonnet) verdicts are not ground truth.** They're our
  current operating standard. Disagreements where the local judge is
  *closer to a careful human reading* are wins, not losses. Use the
  disagreements file as a chance to tighten the prompt, not just a metric
  to chase.

## When this concludes

If a model passes: write up the choice in `docs/decisions.md` (ADR-019),
proceed to Day 2 of `docs/local_pipeline_plan.md`.

If no model passes: revisit the cost-cutting tactics in the plan
("Cheaper directions" section) and don't refactor the worker.
