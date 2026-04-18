# Phase 2c — Real autoresearch (hill-climbing with lineage)

*Started 2026-04-18. What Phase 2b called "autoresearch" was actually random
search with feedback-prompting. This phase implements the real thing —
Karpathy-style hill climbing with lineage, commit-on-improvement, and
revert-on-failure.*

## What was wrong with Phase 2b

Every candidate was independent. `novel_contrast` asked Claude Sonnet for
fresh axes each cycle. `exploit_topk` pulled frozen top-scorers and
generated variants in three dimensions (regen examples / alt effective /
alt layer). No state was maintained across cycles. A "good" candidate
couldn't evolve — the next batch was a new random draw, influenced only by
prompt-level feedback ("do more like these").

The score formula rewarded identification (×(0.5+0.5·ident) multiplier) but
the researcher had no mechanism to **build toward** identification.
Lineages of improvement were impossible because no candidate had a parent.

## What Phase 2c does

Proper hill climbing per lineage:

1. **Lineages** are tracked in the DB. Each lineage has an ID, a current
   leader (the best scoring candidate in its chain), a generation counter,
   and a history of committed mutations.
2. **Mutations** are small, targeted changes applied to a leader. Each
   mutation candidate has `parent_candidate_id` pointing to the leader it
   was derived from.
3. **Commit rule**: when a mutation is evaluated, if its effective score
   beats its parent's, it becomes the new leader of that lineage. Parent
   keeps its data but `is_leader=0`. If mutation score ≤ parent, mutation
   is "rejected" — stays in DB as a dead branch, lineage keeps its old
   leader.
4. **Seeds**: the top-N scoring contrast_pair candidates from Phase 2a/2b
   become initial leaders of N lineages (gen 0).
5. **The researcher**'s job becomes: for each active leader, propose K
   mutations. Evaluate all. Keep the winning mutation (or none) per
   lineage. Repeat next cycle.

## DB schema additions

`candidates` table gains:

- `lineage_id TEXT` — groups related candidates. NULL for pre-2c legacy
  candidates (they'll all be treated as rootless by the hillclimb loop).
- `parent_candidate_id TEXT` — NULL for seeds; the candidate.id this one
  was mutated from otherwise.
- `generation INTEGER NOT NULL DEFAULT 0` — depth in the lineage. Seed = 0.
- `is_leader INTEGER NOT NULL DEFAULT 0` — exactly one per lineage is 1.
- `mutation_type TEXT` — `seed | swap_positive | swap_negative | alt_effective | alt_layer | edit_description`. NULL for seeds and pre-2c legacy.
- `mutation_detail TEXT` — JSON describing the specific change, e.g.
  `{"index": 3, "old": "I noticed my words", "new": "I was watching myself talk"}`. Makes rejected mutations informative on the site.

`schema_meta.version` bumps to 3 (was 2).

Index added on `(lineage_id, is_leader)` for fast leader queries.

## Mutation types

Five mutation types. `hillclimb.py` samples them with weights tuned so the
cheap/fast ones run more often than the expensive Claude calls.

| Type | What changes | Cost | Weight |
|---|---|---|---|
| `alt_effective` | target_effective × uniform(0.8, 1.25) | free, deterministic | 3 |
| `alt_layer` | layer → nearest neighbor in {30, 33, 36, 40} | free | 1 |
| `swap_positive` | rewrite 1 of 6 positive examples (Claude small call) | 1 Claude call | 3 |
| `swap_negative` | rewrite 1 of 6 negative examples (Claude small call) | 1 Claude call | 3 |
| `edit_description` | rephrase the description | 1 Claude call | 1 |

The Claude calls for `swap_*` are tiny — "here's the axis + 6 examples +
which index to replace. Produce one NEW sentence under 15 words that
captures the same pole." Single sentence round-trip, ~2s per call.

## Commit logic (worker-side)

Pseudocode for `src/worker.py` after `evaluate_candidate` returns:

```python
parent_id = spec.parent_candidate_id
if parent_id:
    parent_score = db.get_score(parent_id)
    if new_score > parent_score:
        db.execute("UPDATE candidates SET is_leader=0 WHERE id=?", parent_id)
        db.execute("UPDATE candidates SET is_leader=1 WHERE id=?", spec.id)
        log(f"LINEAGE {lineage_id} gen{spec.generation}: "
            f"{parent_score:.3f} -> {new_score:.3f} (committed)")
    else:
        log(f"LINEAGE {lineage_id} gen{spec.generation}: "
            f"{parent_score:.3f} vs {new_score:.3f} (rejected)")
```

## Initial seeding

A one-shot script `scripts/seed_lineages.py`:

1. Pull top-N scoring contrast_pair candidates (by effective score).
2. For each, assign a fresh `lineage_id` (UUID-ish).
3. Update `generation=0`, `is_leader=1`, `parent_candidate_id=NULL`,
   `mutation_type='seed'`.
4. Commit.

Default N=10. Pre-Phase-2c candidates without score>0 stay unlineaged.

## Site changes

The whole POINT of autoresearch is visible cumulative progress. The site
must show it.

### New: lineage generation tag on each card

Leader cards show "gen N" badge. Rejected mutation cards (non-leaders)
show a muted "rejected gen N" tag and link back to their lineage's
current leader.

### New section: Lineage trees

One tree per lineage. Root = seed (gen 0), children = mutations that
succeeded or failed. Rendered top-down:

```
seed: recognizing-vs-recalling (0.766)
 ├─ gen1 swap_positive[3]: 0.766 → 0.812  ✓ COMMITTED
 │   ├─ gen2 alt_effective: 0.812 → 0.790  ✗ rejected
 │   ├─ gen2 swap_negative[2]: 0.812 → 0.875  ✓ COMMITTED
 │   │   └─ ...
 │   └─ gen2 swap_positive[1]: 0.812 → 0.550  ✗ rejected
 └─ gen1 alt_layer: 0.766 → 0.000  ✗ rejected
```

Each node clickable to see trial responses + mutation_detail.

### New section: Progress timeline

X-axis: wall-clock time. Y-axis: effective score. One line per lineage.
Only committed leaders are plotted (rejections aren't on the curve).
Shows which lineages are still climbing vs plateaued.

### Leaderboard now filters to current leaders by default

A new filter toggle: "all candidates" (current behavior — flat
leaderboard) vs "lineage leaders only" (one row per lineage showing the
current best). Default: lineage leaders.

## Acceptance

- At least one lineage shows ≥3 committed improvements (gen 0 → 3).
- At least one mutation improves `identification_rate` (the actual goal).
- Site renders lineage trees and progress timelines correctly with live
  data.

## Non-goals for this phase

- Crossover (combining two lineages into a child) — planned Phase 2d.
- Auto-retire plateaued lineages (after N rejections in a row, stop
  mutating). Can add later; for now we just let cycles revisit.
- Parallel mutation evaluation. Worker processes serially; that's fine.
- Visualizing rejected mutations' responses in detail (they're in DB;
  tree view shows the mutation and score; user can click to drill in
  later if we want).
