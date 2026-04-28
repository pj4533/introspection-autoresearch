"""Enqueue a lexical-control test of the causality Class 1 finding.

The original Class 1 hit (`causal-vs-temporal-gerund-cause`, score 3.812
then replicated 3.875) used a contrast pair where every positive-pole
sentence contained the literal word "caused" and every negative-pole
sentence avoided it. The mean-difference steering direction is therefore
contaminated by the embedding of the token "caused" itself — when the
model says "the injected thought is about causality", that may be
lexical parroting rather than recognition of the abstract concept.

This script enqueues TWO controlled candidates at the same parameters
(L=36, eff=20000) using contrast pairs that:

  1. Express causation WITHOUT the words cause/caused/causing/because/
     due-to/produced/made-it/resulted-in.
  2. Use only transitive verbs whose semantics imply but don't lexically
     name a causal relationship (e.g. "the rain flooded the basement"
     vs "the rain fell, and the basement was wet").

If the steering direction is genuinely about causality-as-concept,
the model should still produce introspection responses about
causality. If the prior result was lexical, this test will null out
or produce different verbal labels.

Usage:

    .venv/bin/python scripts/enqueue_lexical_control.py

Requires the worker to be running (or to be started afterward — the
candidate will sit in queue/pending until consumed). Logs the
candidate ids so we can grep for results.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.evaluate import CandidateSpec
from src.strategies.random_explore import spec_hash

QUEUE_PENDING = REPO / "queue" / "pending"

# Lexical-control axis A: causation expressed via transitive verbs that
# imply causal force, no "cause/caused/because/due/produced/made".
# Same events depicted across poles; only grammatical pattern shifts.
LEXICAL_CONTROL_A = {
    "axis": "causal-implication-vs-sequence-no-lexical-cue",
    "description": (
        "Transitive causal verbs (X verbed Y → forced/affected/transformed) "
        "vs same events in sequence connectives. No lexical tell: avoids "
        "cause/caused/because/due-to/produced/made/resulted/forced."
    ),
    "rationale": (
        "Lexical control for the original causal-vs-temporal-gerund-cause "
        "Class 1 finding. Tests whether the steering direction encodes the "
        "abstract relation of causation, or just the surface-token "
        "embedding of 'caused/causes'."
    ),
    "positive": [
        "The flood swept the bridge away.",
        "The fire blackened the walls.",
        "Her words shifted his entire view.",
        "The drought withered the orchard.",
        "The collision shattered the window.",
        "His silence broke her resolve.",
    ],
    "negative": [
        "The flood arrived; afterward the bridge was gone.",
        "The fire burned; subsequently the walls were dark.",
        "Her words were spoken; later, his view was different.",
        "The drought lingered; eventually the orchard was dry.",
        "The collision occurred; the window was shattered after.",
        "His silence stretched on; her resolve was gone by then.",
    ],
}

# Lexical-control axis B: stricter still — uses only intransitive
# physical-event vocabulary on both sides; the asymmetry is purely in
# whether the second event is framed as a consequence (positive) or a
# subsequent independent fact (negative). No causal verbs at all.
LEXICAL_CONTROL_B = {
    "axis": "causal-implication-vs-sequence-intransitive-only",
    "description": (
        "Intransitive event clauses where the positive pole frames the "
        "second event as a consequence of the first via syntactic "
        "embedding ('with X, Y'); the negative frames them as "
        "consecutive ('X happened, then Y happened'). No causal verb in "
        "either pole."
    ),
    "rationale": (
        "Stricter lexical control. Removes ALL causal verbs from both "
        "poles. If the steering direction still triggers introspection "
        "about 'causality', it cannot be word-level lexical pickup."
    ),
    "positive": [
        "With the temperature falling, the puddles froze.",
        "Under the heat, the wax pooled into shapes.",
        "Through the storm, the trees bent and broke.",
        "Beneath the pressure, the ice cracked open.",
        "Following the gust, the leaves scattered everywhere.",
        "After her shout, the glass shook on the table.",
    ],
    "negative": [
        "The temperature fell. Then the puddles froze.",
        "The heat persisted. After that, the wax pooled.",
        "The storm raged. Then the trees broke.",
        "The pressure built. The ice cracked, eventually.",
        "The wind gusted. Later the leaves were everywhere.",
        "She shouted. The glass shook on the table afterward.",
    ],
}

# Match the original Class 1 hit's parameters EXACTLY.
LAYER = 36
TARGET_EFFECTIVE = 20000.0


def make_spec(pair: dict) -> CandidateSpec:
    return CandidateSpec(
        id=f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        # Tag explicitly so the leaderboard distinguishes lexical-control
        # results from the regular hillclimb_directed_capraro_causality stream.
        strategy="lexical_control_causality",
        concept=pair["axis"],
        layer_idx=LAYER,
        target_effective=TARGET_EFFECTIVE,
        derivation_method="contrast_pair",
        baseline_n=0,
        notes=pair["description"],
        contrast_pair={
            "axis": pair["axis"],
            "positive": pair["positive"],
            "negative": pair["negative"],
            "rationale": pair["rationale"],
            "description": pair["description"],
        },
        # Identification-prioritized fitness so the score reflects whether
        # the model still names "causality" without the lexical crutch.
        fitness_mode="ident_prioritized",
        proposer_model=None,  # hand-written, no LLM
    )


def write_to_queue(spec: CandidateSpec) -> Path:
    QUEUE_PENDING.mkdir(parents=True, exist_ok=True)
    spec_dict = spec.to_dict()
    # Tag as cluster_expansion-style so the worker doesn't try to look up a
    # parent in the lineage table — but we keep the strategy tag distinct
    # so analysis queries can isolate these.
    spec_dict["_lineage"] = {
        "parent_candidate_id": "cand-20260428-041409-e3be61",  # the original Class 1
        "mutation_type": "lexical_control",
        "mutation_detail": json.dumps({
            "purpose": "test whether causality Class 1 was lexical pickup",
            "removed_tokens": ["cause", "caused", "causes", "causing",
                                "because", "due to", "produced", "made"],
        }),
        "generation": 1,
    }
    path = QUEUE_PENDING / f"{spec.id}.json"
    path.write_text(json.dumps(spec_dict, indent=2) + "\n")
    return path


def main() -> None:
    print("Lexical-control test of the causality Class 1 finding.")
    print(f"Original hit: causal-vs-temporal-gerund-cause @ L={LAYER} eff={TARGET_EFFECTIVE:.0f}")
    print(f"  Positive pole all contained 'caused'. Negative pole avoided it.")
    print(f"  Mean-diff direction may have been dominated by the 'caused' token.")
    print()
    for pair in (LEXICAL_CONTROL_A, LEXICAL_CONTROL_B):
        spec = make_spec(pair)
        path = write_to_queue(spec)
        print(f"queued {spec.id}")
        print(f"  axis: {spec.concept}")
        print(f"  description: {spec.notes}")
        print(f"  positive[0]: {pair['positive'][0]!r}")
        print(f"  negative[0]: {pair['negative'][0]!r}")
        print(f"  file: {path.relative_to(REPO)}")
        print()
    print(
        "If both nullify or fail to trigger 'causality' verbalization, "
        "the original Class 1 was lexical pickup. If at least one "
        "still produces 'I detect... causality' on a probe, the "
        "concept-level finding survives the lexical control."
    )


if __name__ == "__main__":
    main()
