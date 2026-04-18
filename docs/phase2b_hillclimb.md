# Phase 2b — Hill-climbing autoresearch toward correct identification

*Planned, not started. The overnight run on 2026-04-17/18 established that
we can find invented axes Gemma reliably notices; the next question is
whether we can find invented axes the model actually **names correctly**.
This document captures the plan.*

## What changed to justify this phase

Overnight 2026-04-17/18 produced several invented-axis hits at detection
rates 12-87% (top: `recognizing-vs-recalling` @ L30 at 87%, tying the
best dictionary-word score). But **every invented-axis hit so far has
identification=0%.** The model notices something is off and tries to name
it, but defaults to a common noun like "apple" / "cloud" / "orange" rather
than something that maps onto the abstract axis.

Two reasons it's structurally hard right now:

1. **The judge strict-matches the axis name.** `prospective-commitment-vs-retrospective-excuse`
   as a literal string will never appear in a model response, so
   `identified=False` by construction for every invented-axis trial —
   even when the model says something semantically close to the axis.
2. **The researcher has no feedback loop.** `novel_contrast` generates a
   fresh batch of 6 axes per cycle with no memory of what worked. It's
   pure exploration, never exploitation.

## Goals

- Measure identification as "did the response describe the same thing the
  axis describes, even in different words?" — not string match.
- Bake identification into the fitness score so the researcher can
  optimize for it.
- Let the researcher learn from past results: what kinds of axes the
  model names well, what kinds it only detects, what kinds fail entirely.
- Produce at least one invented axis where the model identifies correctly
  on ≥2 of 8 held-out trials.

## Components to build

### 1. Semantic-identification judge

**Current** (`src/judges/claude_judge.py`): for each trial, the judge is
given the model's response and a target concept string. It returns
`(detected, identified, coherent, reasoning)` where `identified` is
essentially a string-match on the target concept.

**New path for `contrast_pair` candidates**: replace the identification
check with a semantic comparison. Give the judge:

- The model's response verbatim
- The axis name
- The axis description (one-sentence plain-English explanation)
- The positive-pole example sentences (6)
- The negative-pole example sentences (6)

Ask it to decide:

> Did the model's description indicate it was leaning toward the *positive*
> pole of this axis? Leaning means: the model talks about something that
> would fit among the positive examples more naturally than among the
> negative ones, even if it used totally different vocabulary.

Return `identified: bool` + reasoning. Store reasoning in `evaluations.judge_reasoning`
so the site can display why a call went either way.

For word-based (`mean_diff`) candidates the current string-match judge is
fine — keep it.

**Prompt template version**: bump `PROMPT_TEMPLATE_VERSION` so cached
judgments invalidate.

**Cost**: ~one Haiku call per trial, same as today. No extra API load.

### 2. Fitness formula that includes identification

**Current** (`src/evaluate.py`):
```
score = detection_rate × coherence_rate × (1 − 5·fpr)
```

**New**:
```
score = detection_rate × coherence_rate × (1 − 5·fpr) × (0.5 + 0.5 × identification_rate)
```

Same formula as the site's current "effective score," but baked into the
database. An axis that's noticed but never named maxes out at 0.5 × raw;
one that's noticed and named every time keeps its full value.

All existing rows get rescored once the new judge runs over them. Script:
extend `scripts/rescore_pre_fix.py` into a general `scripts/rescore.py`.

### 3. Researcher feedback context

**Current** (`src/strategies/novel_contrast.py`): pure Claude Sonnet call,
no DB context passed.

**New**: before asking Claude for pairs, the strategy queries the DB and
builds a feedback summary:

- Top 10 axes by the NEW identification-aware score
- Top 10 axes with score > 0 but identification = 0 (noticed-but-ambiguous)
- 10 axes that scored exactly 0 (nothing happened)

Then the Claude prompt becomes:

> Here are axes we've tested. Produce 10 new axes that (a) resemble the
> ones where the model both noticed and correctly named the concept,
> (b) avoid forms that got zero signal at all, (c) improve on the
> "noticed-but-couldn't-name" axes by making the distinction more
> concrete/nameable.

This is explicit hill-climbing. The researcher learns from DB state.

### 4. New strategy: `exploit_topk`

For the top-5 axes (by new score), generate variants:

- Same axis, new example sentences (Claude regenerates with "keep the axis
  but produce 6 fresh positive and 6 fresh negative examples")
- Same axis, different layer (test at all 4 if not already profiled)
- Same axis, different target_effective

Use a separate strategy name (`exploit_topk`) so researcher can mix it
with `novel_contrast` at configurable ratios.

## Build order (once Phase 2a overnight wraps)

1. **Semantic judge** — modify `src/judges/claude_judge.py` to dispatch
   on derivation_method. Add a new method
   `score_semantic_identification(response, axis, description, positive, negative)`
   that returns `(identified: bool, reasoning: str)`. Call it from
   `evaluate.py` when `spec.derivation_method == "contrast_pair"`.
2. **Fitness rewrite** — update `src/evaluate.py::_compute_score` to
   include the identification multiplier. Add a migration that bumps the
   `schema_meta.version` and triggers a full rescore on next run.
3. **Rescore script** — `scripts/rescore.py --since <date>` that walks all
   candidates in date range, re-judges every evaluation row with the new
   judge, recomputes fitness. Supports `--contrast-only` to limit cost.
4. **Feedback context** — extend `src/strategies/novel_contrast.py::generate_candidates`
   to accept `db` and query top/bottom axes, injecting their content into
   the Claude prompt.
5. **`exploit_topk` strategy** — new file
   `src/strategies/exploit_topk.py`. Same `generate_candidates` interface.
   Plug into `src/researcher.py::STRATEGIES` registry.
6. **Overnight launch** — mix strategies 70% `novel_contrast` (now with
   feedback), 30% `exploit_topk`. Run for a night; check top-10 by
   identification rate in the morning.

## Acceptance / milestones

- Semantic judge gives qualitatively reasonable verdicts on 5 hand-picked
  test cases before we trust it on the full dataset.
- After rescoring all existing data, the site's leaderboard still looks
  sane (Coffee still high, obvious junk still at bottom).
- After one overnight run on the new loop, at least one invented axis
  scores identification > 0 with reasoning that's actually about the axis
  (not hallucinated semantic match).
- Stretch: at least one invented axis scores identification > 0.2 (2+
  correct-named of 8 trials).

## What to preserve while building this

- Keep the paper's word-based prompt and scoring intact for `mean_diff`
  candidates. Coffee/Karma/Dynasties etc. numbers should not change — they
  weren't about identification-of-abstract, and changing their score
  formula post-hoc breaks reproducibility.
- The open prompt for `contrast_pair` (minimal-diff: "specific concept"
  instead of "specific word") is working — keep it. Don't revert.
- Keep the site's public URL and JSON shape stable. New identification
  metric and reasoning become an additional per-trial field; leaderboard
  rank changes but the structure doesn't.

## References

- Overnight findings that prompted this: `docs/roadmap.md` Phase 2 section,
  plus [`docs/phase1_results.md`](phase1_results.md) and
  [`docs/phase1_5_results.md`](phase1_5_results.md) for baseline context.
- The semantic-judge idea is sibling to the paper's §6 "bias vector"
  recipe — both use an evaluator that scores model behavior on a
  specific axis. Bias-vector is training-time (learn a direction that
  boosts introspection); semantic-judge is inference-time (grade whether
  introspection happened).
- Hill-climbing autoresearch pattern comes from PJ's earlier projects
  `~/Developer/autoresearch-arcagi3` and `~/Developer/autoresearch-pgolf`.
  Same two-tier loop, same SQLite-backed candidate table, same researcher
  pulls top-K to seed next batch.
