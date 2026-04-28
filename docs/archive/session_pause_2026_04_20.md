> **ARCHIVED 2026-04-28.** Superseded by Phase 2g ([`docs/phase2g_plan.md`](../phase2g_plan.md)). Historical session pause note from the Sonnet-judge cost investigation that motivated the all-local move.
>
> Kept for historical context only; the code it referenced has been removed or refactored.

---

# Session pause — 2026-04-20

Paused mid-Phase-2c to conserve Claude Max subscription usage. Everything is
stopped: worker, researcher loop, refresh_site, both monitors. Nothing is
running. DB is at `data/results.db`, last deploy is live at
[did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app).

This doc is the handoff note — read alongside
[`roadmap.md`](roadmap.md) and [`phase2c_autoresearch.md`](phase2c_autoresearch.md).

## Where we are

Phase 2c (real hill-climbing autoresearch with commit/revert lineages) was
launched 2026-04-19 evening and ran through the night into 2026-04-20
morning. The scaffolding works end-to-end; the scientific signal is stuck
on one axis.

### What Phase 2c proved

- **Hill-climbing loop works.** 10 seeded lineages × 5 mutation types
  (swap_positive, swap_negative, alt_effective, alt_layer, edit_description).
  Commit-or-reject rule fires correctly against the current lineage leader
  (not parent — fixed the multi-leader bug mid-run).
- **Throughput**: ~53 candidates/hour. 747 evaluated overnight; 1,186 total
  in DB.
- **Detection scales cleanly via magnitude tuning.** Three lineages
  climbed to 100% detection (score 0.500): `noticing-vs-naming` @ L30,
  `recognizing-vs-recalling` @ L30, `self-monitoring-vs-unguarded-expression`
  @ L33.
- **FPR stays at 0 throughout.** Paper's control hygiene is preserved.
- **`alt_effective` and `swap_*` are the productive mutations.** `alt_layer`
  reliably destroys detection on invented axes (they're layer-specific).

### What Phase 2c did NOT achieve

**Zero identification gain on any lineage leader.** Every committed
mutation improved detection; none produced consistent naming. The model
reliably notices something is injected but still defaults to single-noun
guesses ("apple", "cloud") for abstract axes.

Only **one** NAMED event happened on an invented axis during the hill
climb: `noticing-vs-naming gen2/swap_negative` — 62% det, **12% ident**,
score 0.352. It was rejected (leader was at 0.500 detection). Not
reproduced in subsequent mutations. Fragile signal.

Eight earlier NAMED events on invented axes exist in the DB from
pre-hillclimb Phase 2a/2b (e.g. `inhabiting-vs-reporting-experience` @
L30 with 62% ident, `relational-repair-vs-relational-rupture` @ L40 with
50% ident) but they all had high FPR so `score=0.000` — they're proof
the mechanism exists but not usable as seeds under the current fitness
function.

## The open question

**Detection vs. identification decouple.** Pushing `target_effective`
reliably drives detection → 100% without moving identification at all.
This is the central finding of Phase 2c and the project's next target.

Hypotheses worth testing (in no particular order):

1. **The open prompt is still too word-framed.** Even with "concept"
   instead of "word", Gemma's prior for single-token responses dominates.
   Try prompts that explicitly invite descriptive phrases or emotional
   qualities.
2. **Identification needs its own mutation dimension.** Currently all
   mutations perturb the direction. A mutation that rewrites the axis
   *description* without touching examples might pull naming up — the
   judge uses pole examples but the model sees only the steering
   direction + prompt.
3. **Fitness multiplier on identification is too weak.** Current:
   `score = det × coh × fpr_penalty × (0.5 + 0.5·ident)`. This gives
   a 100%-det / 0%-ident candidate score 0.5, same as a
   50%-det / 100%-ident candidate. Harsher weighting (e.g. `ident²` or
   `ident > 0.2` as a gate) would force the search toward naming.
4. **Some axes just aren't nameable at the direction level.** `Coffee`
   hit 75% ID as a dictionary word but nothing Claude Sonnet invented
   has reproduced that. Maybe invented axes need to be *closer* to
   real-word concepts — bias the novel_contrast generator toward
   phenomenological terms the model actually has as single tokens.

## Things to revisit before the next run

- **Revisit fitness function.** The 0.5 + 0.5·ident multiplier hasn't
  produced selection pressure toward naming. Either harden it or add an
  identification-floor cutoff.
- **Look at the 8 high-ident / high-FPR pre-hillclimb invented axes.**
  Their controls false-positive'd but their injected trials named
  correctly. Could be a prompt-format issue where the open prompt
  drifts the model toward describing anything. Worth auditing the
  actual responses.
- **Check whether semantic judge is still calibrated.** If it's lenient
  enough to mark coincidental matches, the 12% ident on noticing-vs-naming
  might be noise. Spot-check 5 recent NAMED trials by eye.
- **Consider exploit_topk.** Not yet run against the semantic judge.
  Could generate axes that are semantic neighbors of the known NAMED
  examples (e.g. variations on "noticing-vs-naming" itself).
- **Seeded vs. emergent scoring.** The three lineage leaders at 0.500
  include one seed (`63409b55` self-monitoring-vs-unguarded-expression)
  that never improved — mutations kept tying or losing. That lineage
  might be at its local ceiling and burning researcher cycles.

## Current live data shape

- 10 lineages, 26–39 evaluations each.
- 3 at score 0.500 (detection ceiling).
- 2 at 0.438, 1 at 0.313, 4 at 0.250, 1 at 0.219.
- 43 NAMED events total in DB — 34 are word seeds (e.g. Coffee 75% ident),
  9 are invented-axis (8 with FPR > 0, 1 from hill-climb).

## Restart checklist

1. `source .venv/bin/activate` — everything runs from the venv.
2. `ps aux | grep -E "src\.(worker|researcher)|refresh_site"` — confirm
   nothing is running.
3. Start worker: `./scripts/start_worker.sh` (loads model once, polls
   `queue/pending/`).
4. Start researcher: `./scripts/start_researcher.sh` (30-min cycles).
   Check `logs/researcher.log` for the first run.
5. Start refresh_site: `nohup ./scripts/refresh_site.sh > logs/refresh.log 2>&1 &`
6. Rearm monitors if you want live streaming (see `scripts/monitor_phase2c.sh`
   if it exists, or the inline command from prior session).

## Quick orientation for a new Claude session

- Roadmap: `docs/roadmap.md`
- ADRs: `docs/decisions.md`
- Phase 2c plan: `docs/phase2c_autoresearch.md`
- This pause note: `docs/session_pause_2026_04_20.md` (here)
- DB: `data/results.db` (schema v3, lineage columns populated)
- Latest web export: `web/public/data/*.json`
