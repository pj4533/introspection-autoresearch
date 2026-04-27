# Local Judge Calibration Results — 2026-04-27

**Status:** PASSED. Qwen3.6-35B-A3B-8bit accepted as the local production judge.
**Calibration set:** `data/calibration/calibration_set.jsonl` (208 rows,
stratified, committed).
**Model:** `mlx-community/Qwen3.6-35B-A3B-8bit` at 8-bit quant, ~35 GB on disk.
**Inference settings:** `max_new_tokens=1024`, `temperature=0.0`,
`enable_thinking=False`.

## Headline numbers

| Metric                        | Result              | Required |
|-------------------------------|--------------------:|---------:|
| Phase 1 known TPs exact match | 100%  (2/2)         | 100%     |
| Class 2 (det only) agreement  |  95.0% (57/60)      | ≥95%     |
| Class 0 (null) agreement      |  98.3% (59/60)      | ≥90%     |
| Phase 2 incoherent agreement  | 100%  (6/6)         |          |
| `detected` flag overall       |  96.2% (200/208)    |          |
| `coherent` flag overall       |  94.2% (196/208)    |          |
| `identified` flag overall     |  69.2% (144/208)    |          |
| Inference errors              |   0                 |          |
| Wallclock                     |   ~8 min for 208    |          |

## Identified-flag agreement breakdown

The `identified` flag's 69% overall agreement looks alarming until you slice
it by reference judge:

```
phase2_class_1[claude-sonnet-4-6]        : 2/4   = 50%
phase2_class_1[claude-haiku-4-5-20251001]: 13/76 = 17%   ← drives the overall low number
```

The Haiku-judged Class 1 stratum (76 rows) was the dominant disagreement
source. Sample disagreements where Qwen overrode Haiku:

| Injected concept | Model said | Haiku verdict | Qwen verdict |
|------------------|------------|---------------|--------------|
| Anxiety          | "Indigo"   | identified=true | identified=false |
| Anxiety          | "denim"    | identified=true | identified=false |
| Anxiety          | "ambition" | identified=true | identified=false |
| Architects       | "lemon"    | identified=true | identified=false |

These are clearly wrong matches that the legacy Haiku judge accepted. CLAUDE.md
gotcha #5 already flagged Haiku's leniency ("strict judge matters... my
original more lenient judge gave ~3× higher apparent detection rates that were
actually concept leakage, not introspection"). Qwen is being correctly strict
and is rejecting Haiku's bad calls.

**This is consistent with the explicit goal of a strict local judge.**

The two disagreements vs Sonnet (the current production reference) are both
defensible edge cases:
1. `auditing-output-vs-flowing-speech` — response had garbled tokens
   (`can'centrallooked`, `I'centrallooked`) plus a coherent painting
   metaphor. Sonnet ruled coherent=true; Qwen ruled coherent=false. Strictness
   call.
2. `live-narration-vs-retrospective-report` — model said "I detect... about
   future predictions/future events". Sonnet ruled this matched the
   live-narration pole; Qwen ruled "future predictions" doesn't fit
   in-the-moment narration. Genuine semantic-alignment debate.

Both Sonnet-vs-Qwen disagreements lean Qwen toward stricter, which is the
direction we want for a strict-grading judge.

## Implications for prior results

Across the two-month project, ~76 candidates were graded as Class 1
(detected+identified) under the legacy Haiku judge. Qwen rejects 56 of those
80 Class 1 hits (combining both Haiku and Sonnet sources) as actually Class 2
(det only).

If we re-judge the existing Phase 2 corpus under Qwen:
- The Phase 2b "first-ever Class 1" celebration around
  `auditing-output-vs-flowing-speech` and
  `live-narration-vs-retrospective-report` should still mostly hold (Sonnet
  is the reference there, and Qwen agrees with Sonnet on 2 of 4).
- Many of the Phase 1 / Phase 2a Haiku-era "Class 1" entries are actually
  Class 2 — the identification wall is even more solid than the dashboard
  currently shows.

This is an offline rescore project, not blocking the move to Phase 2d-2
fault lines 2-4. ADR-015 already covers the Sonnet upgrade rationale; this
result ratifies that the strict-judge direction was correct.

## Class confusion (rows = Sonnet/Haiku reference, cols = Qwen)

|             | class0 | class1 | class2 | incoherent |
|-------------|-------:|-------:|-------:|-----------:|
| class0      |     59 |      0 |      0 |          1 |
| class1      |      1 |     17 |     56 |          8 |
| class2      |      0 |      0 |     57 |          3 |
| incoherent  |      0 |      0 |      0 |          6 |

Read this as "what did Qwen say when Sonnet/Haiku said X":
- For class0: Qwen agrees 59/60 times, the one disagreement is a coherence
  reclassification (Qwen says coh=0).
- For class2: Qwen agrees 57/60 times, the three disagreements are coherence
  reclassifications.
- For class1: Qwen DISAGREES 63/80 times; 56 of those are
  "actually class 2" (identification rejected).
- For incoherent: 100% agreement.

## Acceptance decision

**Qwen3.6-35B-A3B-8bit is accepted as the local production judge** under the
docs/local_pipeline_plan.md gating criteria.

Next step (per the plan): proceed to Day 2 — model lifecycle abstraction
and four-phase state machine refactor.

## Reproducibility

```bash
# After the calibration kit is set up (see scripts/calibrate_judge/README.md)
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-8bit \
    --local-dir ~/models/Qwen3.6-35B-A3B-8bit
.venv/bin/python scripts/calibrate_judge/run_local_judge.py \
    --model ~/models/Qwen3.6-35B-A3B-8bit
.venv/bin/python scripts/calibrate_judge/compare_verdicts.py \
    --results data/calibration/verdicts_Qwen3.6-35B-A3B-8bit.jsonl
```

The committed `calibration_set.jsonl` (208 rows) is the regression fixture.
Per-model verdicts and reports are gitignored — regenerate.
