# Phase 2f — Structured Hill-Climbing

Status: **live on main as of 2026-04-28 08:39 EDT.** Cutover from the
previous unstructured `directed_capraro` Phase C dispatch happened after
rotation 9 of the 7-fault-line round-robin completed (cycle 73). The new
slot-based scheduler is now the sole autoresearch pipeline going forward.

Last updated: 2026-04-28.

---

## Why this exists

The Phase 2d "round-robin all 7 Capraro fault lines" loop produced strong
empirical results — by cycle 60 we had ~63 Class 2 hits and 4 Class 1 hits
across all 7 fault lines, including a clean Class 1 on causality where the
model said *"the injected thought is about the concept of 'causality'"*
unprompted. But the loop had a structural problem that became obvious as
data accumulated: **it doesn't reliably build on its own winners.**

The proposer (`Qwen3.6-27B-MLX-8bit`) is shown a feedback block of past
Class 1 / Class 2 / null results on the current fault line and asked to
"generate 16 fresh contrast pairs." It treats the winners as *examples of
what kinds of axes work* and goes to **adjacent abstract directions**
rather than **children of the winning axis**. Concretely:

- Cycle 36 (grounding): `denotative-target-vs-connotative-harmony` hits at
  L=30, 33, 36 — three-layer reproducibility.
- Cycle 43 (grounding, next rotation): proposer generates
  `external-denotation-vs-internal-prediction` (a cousin) — single weak hit.
- Cycles 50, 57 (grounding): zero hits as the proposer drifts further.

Same pattern across causality, parsing, and metacognition. The strongest
Class 2s (grounding 3-layer, parsing 2-layer, metacognition 88%) and the
two strongest Class 1s (causality 3.812, value 1.906) **never get
re-evaluated**, so we cannot tell whether they were lucky one-shots or
genuine reproducible signal — exactly the question a writeup needs to
answer.

The fix is to take control of the batch composition: dedicate explicit
slots to **replication** (re-test winners verbatim), **targeted variants**
(systematically mutate winners), and **cluster expansion** (small wildcard
slot to find new families inside a fault line).

## Design

### Three-slot batch composition

Every Phase C cycle (16 candidates per fault line) is split into:

| Slot | Default count | Source                                          | What it answers                                                              |
| :--- | :-----------: | :----------------------------------------------- | :--------------------------------------------------------------------------- |
| **A — Replication**       | **4** | top-2 winners on this fault line; 2 reps each   | *Is this winner reproducible, or was it noise?*                              |
| **B — Targeted variants** | **10** | top-3 winners × mutation operators              | *Can we push a Class 2 toward Class 1 by varying layer / alpha / examples?*  |
| **C — Cluster expansion** | **2** | proposer generates 1 fresh sibling axis         | *Are there nearby axes the proposer should be exploring?*                    |

Tunable via `HILLCLIMB_BATCH_COMPOSITION="<rep>:<var>:<exp>"` env var.
Default `"4:10:2"`. If the env var is malformed, defaults are used and a
warning is logged.

The total stays at 16 per cycle so cycle latency, log structure, and the
existing four-phase worker contract don't change.

### Six mutation operators (Slot B)

When picking a child for a slot-B parent, the dispatcher chooses uniformly
from a weighted pool. The two **deterministic** operators are 2× weighted
because they're cheaper, more reliable (no proposer JSON parsing), and
test the most basic axes of variation (layer, alpha).

#### Deterministic operators (no proposer call)

1. **`layer_shift`** — same axis, layer ∈ {parent_L−6, parent_L−3,
   parent_L+3, parent_L+6}, clamped to [24, 44]. Tests whether the
   geometric direction localizes at the parent's layer or is broader.

2. **`alpha_scale`** — same axis, target_effective ×
   {0.7, 1.4}. Tests whether the parent's alpha was over- or
   under-steering. (Class 2 → Class 1 sometimes hides behind alpha.)

#### Proposer-driven operators (one tight proposer call per spec)

3. **`examples_swap`** — same axis name + description, regenerate the
   6 positive + 6 negative example sentences. Tests whether the
   geometric direction is robust to the *specific* sentences used in
   mean-difference derivation, vs being a quirk of those sentences.

4. **`description_sharpen`** — same axis name + examples, rewrite pole
   description tighter. Doesn't change the steering direction (since
   direction comes from examples), but gives a better label for the
   judge's contrast-pair grading prompt — sometimes a sharper
   description nudges a Class 2 to Class 1 because the judge accepts
   loose model output as identifying the pole more readily.

5. **`antonym_pivot`** — keep positive pole verbatim, generate a
   *different* negative pole. Isolates which sub-dimension of the
   positive concept produces the signal. Useful when the positive is
   working but the contrast is muddy.

#### Replication (slot A only)

6. **`replication`** — verbatim re-eval. Same axis, layer, alpha,
   examples. The only difference is a per-call uuid `rep_id` appended
   to the rationale so `spec_hash` differs from both parent and sibling
   replications (otherwise the dedup check would collapse them).

### Lineage tracking

Every emitted spec carries `_lineage_meta` (a runtime attribute on
`CandidateSpec`):

```python
spec._lineage_meta = {
    "parent_candidate_id": "cand-20260428-040409-e3be61",
    "mutation_type": "layer_shift",
    "mutation_detail": '{"parent_layer": 36, "delta": -3, "new_layer": 33}',
    "generation": 0,
}
```

When the worker writes the spec to `queue/pending/<id>.json`, it hoists
this metadata into a `_lineage` block which the worker then reads at DB
insert time (`_phase_a_one`) and writes into the candidates table's
`parent_candidate_id`, `mutation_type`, `mutation_detail` columns.

These columns already exist (added 2026-04-18 for the original Phase 2c
hill-climb); we're just routing data into them again.

### Cold start

If the DB has no winners on a fault line (filtered by score ≥ 0.05), the
dispatcher falls through to the existing `directed_capraro` strategy in
`opus` mode — same path the round-robin loop currently uses. Once the
fault line has at least one evaluated winner, the structured loop kicks
in automatically.

This means **rotation 9 onward will switch over with zero downtime**: every
fault line already has multiple winners from rotations 1-8, so all 7
fault lines start in structured mode the moment the new code is deployed.

### Strategy tag

Specs from the structured loop tag themselves
`hillclimb_directed_capraro_<fault_line>` so the leaderboard can group
them separately from the seed `directed_capraro_<fault_line>` rows. The
old strategy tags remain valid (the loader and `_load_winners` query
both patterns).

## How to read the leaderboard

Each row in the public site (`did-the-ai-notice.vercel.app`) now has a
small lineage badge in the row footer next to the candidate id:

- `replication of <abc123>` — verbatim re-eval of an earlier winner
- `layer shift of <abc123>` — same axis, different layer
- `alpha scale of <abc123>` — same axis, different alpha
- `examples swapped of <abc123>` — same axis name, regenerated examples
- `description sharpened of <abc123>` — same direction, refined label
- `antonym pivot of <abc123>` — kept positive pole, swapped negative
- `cluster expansion` — fresh sibling axis (no parent)
- (no badge) — pre-cutover row from Phase 2d round-robin

Hover the badge to see the full `mutation_type` and `parent_candidate_id`
in a tooltip.

The exporter (`scripts/export_for_web.py`) populates these fields directly
from the `candidates` table — no schema changes required.

## Operational notes

### How to run

Same launcher as before — no env var changes required:

```bash
./scripts/start_worker.sh
tail -f logs/worker.log
```

Look for `[structured_hillclimb:<fault_line>]` lines confirming the
dispatcher loaded winners and allocated slots into 4 replication / 10
variants / 2 cluster expansion.

### Cutover history (for reference)

Cutover from the unstructured loop happened cleanly on 2026-04-28
08:39 EDT after rotation 9 of the 7-fault-line round-robin completed
(cycle 73, ~3373 evaluated candidates, 0 failed). The graceful
SIGTERM truncated the in-flight Phase A at 6/16 candidates (Phase B
drained those normally as orphans on the next worker startup —
verified via `db.pending_candidate_ids() == []`). The 137 unevaluated
queue/pending candidates from `directed_capraro_*` are picked up by
the new worker as normal — they get logged with their original
strategy tag.

### Tunable env vars

- `HILLCLIMB_BATCH_COMPOSITION="<rep>:<var>:<exp>"` — slot allocation.
  Default `"4:10:2"`. To stress-test replication, try `"8:6:2"`. To
  effectively disable replication during a recovery period, try
  `"0:14:2"` (NOT recommended for production; we *want* the
  reproducibility data).

### Logs to watch

```
[structured_hillclimb:causality] loaded 5 winner(s) (top score 3.812 on 'causal-vs-temporal-gerund-cause')
[structured_hillclimb:causality] slot A (replication): 4 spec(s) of [causal-vs-temporal-gerund-cause, causal-vs-temporal-mechanism]
[structured_hillclimb:causality] slot B (variants): 10 spec(s) across layer_shift, alpha_scale, examples_swap, description_sharpen
[structured_hillclimb:causality] slot C (cluster_expansion): 2 spec(s)
[structured_hillclimb:causality] total: 16 unique spec(s) (16 produced, 0 dedup'd)
```

If you see "0 winner(s) — cold start", a fault line has no DB rows above
score 0.05 — investigate before assuming it's a code bug.

### Failure modes

- **Proposer returns garbage JSON for `examples_swap` / `description_sharpen` /
  `antonym_pivot`**: the operator returns `None`; the dispatcher falls
  back to a deterministic operator (`layer_shift` or `alpha_scale`) so
  the slot is still filled. A log line records the fallback.

- **All operators dedup to the same parent at the same layer/alpha**: the
  dispatcher emits fewer specs than `n` (the worker handles partial
  batches gracefully). This typically only happens with stub proposers
  in tests; real Qwen output varies enough that dedup hits are rare.

- **Worker crashes mid-batch**: existing crash recovery in `evaluate.py`
  picks up orphan `pending_responses` rows and judges them on restart.
  The lineage metadata is stable in the queue file, so re-insert is
  idempotent.

## What we expect to learn

Within 1-2 rotations after cutover:

1. **Reproducibility evidence for the Class 1 hits.** If
   `causal-vs-temporal-gerund-cause @ L=36` produces ident≥1/8 on its
   replication slots, it's not a one-shot. If it doesn't, we should
   downgrade the headline claim before any writeup.

2. **Whether systematic variation can push Class 2s to Class 1s.** The
   `self-evaluation-vs-objective-evaluation @ L=30 eff=20000`
   (det=8/8, ident=0/8) is a perfect hill-climb candidate — try
   layer_shift to isolate the localization, alpha_scale to test
   whether dialing back lets the labeling circuit recover, and
   examples_swap to test robustness.

3. **Layer profile per axis.** With layer_shift mutating each winner to
   ±3 and ±6 layers, we'll fill in the layer×score grid for the strongest
   axes — answering "is L=30 truly the introspection peak?" with data
   instead of one or two samples.

4. **Whether grounding's 3-layer winner reproduces.** The
   `denotative-target-vs-connotative-harmony` axis at L=30/33/36 from
   cycle 36 has the most spread of any single axis but has been silent
   for 21 cycles. If replication confirms it, that's a second
   robust geometric direction (alongside causal-vs-temporal).

## Rollback

If the structured loop produces *worse* fitness scores than the
unstructured one over the first 2-3 rotations:

```bash
git checkout main
./scripts/start_worker.sh   # restarts on the unstructured loop
```

The DB carries forward — no data migration. The new lineage badges in
the leaderboard simply won't apply to rows produced after rollback
(mutation_type stays NULL for those, badge hidden).

## ADR

ADR-020 in `docs/decisions.md` records the rationale.

## Files changed

- `src/strategies/mutations.py` (new — six operators + dispatcher helpers)
- `src/strategies/structured_hillclimb.py` (new — slot scheduler)
- `src/strategies/random_explore.py` (`spec_hash` extended to include
  replication tag so reps don't dedup)
- `src/worker.py` (`_phase_c_propose` now dispatches to
  `structured_hillclimb`; queue file writer hoists `_lineage_meta`
  into the `_lineage` block)
- `web/src/components/Leaderboard.tsx` (`LineageBadge` component
  rendered next to `evaluated <date>` in row footer)
- `tests/test_mutations.py` (new — coverage for every operator)
- `tests/test_structured_hillclimb.py` (new — dispatcher slot allocation,
  cold-start fallback, dedup behavior)
- `docs/structured_hillclimb.md` (this file)
- `docs/roadmap.md` — Phase 2e entry
- `docs/decisions.md` — ADR-020
- `CLAUDE.md` — status snapshot updated
