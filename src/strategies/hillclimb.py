"""hillclimb — proper per-lineage autoresearch.

For each active leader (one per lineage), propose ONE small mutation.
Evaluate, then commit-or-reject based on whether the mutation beat its
parent's score.

Mutation types (weighted random choice):

- ``alt_effective``  — tweak target_effective by ×uniform(0.8, 1.25). Free.
- ``alt_layer``      — move layer to adjacent layer in [30, 33, 36, 40]. Free.
- ``swap_positive``  — rewrite 1 of the 6 positive example sentences. 1 Claude call.
- ``swap_negative``  — rewrite 1 of the 6 negative example sentences. 1 Claude call.
- ``edit_description`` — rephrase the axis description. 1 Claude call.

Commit/reject happens in the worker after evaluation
(see ``src/worker.py::_maybe_promote_mutation``). This module only emits
mutation candidates.

Used via ``python -m src.researcher --strategy hillclimb --n N``. Each cycle
produces up to N mutations spread across active lineages.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
import threading
import time
import uuid
from typing import Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from ..db import ResultsDB
from ..evaluate import CandidateSpec
from .novel_contrast import CLAUDE_MODEL
from .random_explore import spec_hash

# Which layers are valid for mutation moves. Same grid as novel_contrast.
LAYERS = [30, 33, 36, 40]

# Mutation-type weights. Cheap mutations run more often so we get a quick
# initial hill-climbing signal before spending Claude calls on edits.
MUTATION_WEIGHTS = {
    "alt_effective": 3,
    "alt_layer": 1,
    "swap_positive": 3,
    "swap_negative": 3,
    "edit_description": 1,
}

EXAMPLE_REWRITE_SYSTEM = (
    "You are editing a single example sentence for an abstract-axis "
    "contrast-pair experiment in mechanistic interpretability. You reply "
    "with a single JSON object containing exactly one new sentence."
)

EXAMPLE_REWRITE_TEMPLATE = """The axis is "{axis}" — {description}.

The current {pole} examples are:
{block}

Replace example #{index_1based} (currently: "{current}") with a NEW sentence
that exemplifies the same pole. Requirements:

- Under 15 words.
- Distinct from every other example in the list.
- Captures the same pole of the axis but covers a different topic /
  context / phrasing so we can test whether the axis direction is
  stable across varied examples.

Return a single JSON object of this form:

{{"new_example": "..."}}

Do not include any text before or after the JSON."""

DESCRIPTION_REWRITE_SYSTEM = (
    "You are editing a one-sentence description of an abstract axis in a "
    "mechanistic interpretability experiment. You reply with a single JSON "
    "object."
)

DESCRIPTION_REWRITE_TEMPLATE = """The axis is "{axis}".

Current description: "{description}"

Positive-pole examples: {positive_examples}
Negative-pole examples: {negative_examples}

Rewrite the description so it more clearly captures the distinction made
by the examples. Requirements:

- One sentence, under 30 words.
- Must remain accurate to the examples (not introduce a different axis).
- Should be concrete enough that a human reading it can predict which
  pole a new sentence would fall on.

Return a single JSON object of this form:

{{"new_description": "..."}}

Do not include any text before or after the JSON."""


# -----------------------------------------------------------------------------
# Claude small calls
# -----------------------------------------------------------------------------

async def _ask_claude(prompt: str, system: str) -> str:
    options = ClaudeAgentOptions(
        model=CLAUDE_MODEL,
        system_prompt=system,
        max_turns=1,
        permission_mode="bypassPermissions",
        allowed_tools=[],
    )
    chunks: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "".join(chunks).strip()


def _run_sync(prompt: str, system: str) -> str:
    result: dict = {}

    def worker() -> None:
        try:
            result["value"] = asyncio.run(_ask_claude(prompt, system))
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _parse_single_key(raw: str, key: str) -> Optional[str]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    val = obj.get(key)
    if not isinstance(val, str) or not val.strip():
        return None
    return val.strip()


# -----------------------------------------------------------------------------
# Individual mutators. Each returns (new_spec_dict, mutation_type, mutation_detail)
# or None on failure.
# -----------------------------------------------------------------------------

def _mutate_alt_effective(leader_spec: dict, rng: random.Random) -> tuple:
    old = leader_spec["target_effective"]
    factor = rng.uniform(0.8, 1.25)
    new = round(old * factor, 0)
    new = max(8000.0, min(30000.0, new))
    if abs(new - old) < 1e-3:
        return None
    spec = copy.deepcopy(leader_spec)
    spec["target_effective"] = new
    return spec, "alt_effective", json.dumps({"old": old, "new": new, "factor": factor})


def _mutate_alt_layer(leader_spec: dict, rng: random.Random) -> tuple:
    old = leader_spec["layer_idx"]
    alts = [l for l in LAYERS if l != old]
    if not alts:
        return None
    # Prefer adjacent layer (closest in index) to keep the step small.
    alts.sort(key=lambda l: abs(l - old))
    # Take from top-2 nearest, choose randomly within that.
    pool = alts[:2]
    new = rng.choice(pool)
    spec = copy.deepcopy(leader_spec)
    spec["layer_idx"] = new
    return spec, "alt_layer", json.dumps({"old": old, "new": new})


def _mutate_swap_example(leader_spec: dict, rng: random.Random, pole: str) -> tuple:
    cp = leader_spec.get("contrast_pair")
    if not cp:
        return None
    examples = list(cp.get(pole, []))
    if len(examples) < 2:
        return None
    idx = rng.randrange(len(examples))
    current = examples[idx]
    axis = cp.get("axis", leader_spec.get("concept", ""))
    description = leader_spec.get("notes", "") or cp.get("description", "")
    block = "\n".join(f"  {i+1}. \"{s}\"" for i, s in enumerate(examples))
    prompt = EXAMPLE_REWRITE_TEMPLATE.format(
        axis=axis,
        description=description or "(no description)",
        pole=pole,
        block=block,
        index_1based=idx + 1,
        current=current,
    )
    try:
        raw = _run_sync(prompt, EXAMPLE_REWRITE_SYSTEM)
    except Exception as e:
        print(f"[hillclimb] swap_{pole} Claude call failed: {e}", flush=True)
        return None
    new_sentence = _parse_single_key(raw, "new_example")
    if not new_sentence or new_sentence == current:
        return None
    new_examples = examples.copy()
    new_examples[idx] = new_sentence
    new_spec = copy.deepcopy(leader_spec)
    new_spec["contrast_pair"] = dict(cp)
    new_spec["contrast_pair"][pole] = new_examples
    detail = json.dumps({
        "pole": pole,
        "index": idx,
        "old": current,
        "new": new_sentence,
    })
    return new_spec, f"swap_{pole}", detail


def _mutate_edit_description(leader_spec: dict, rng: random.Random) -> tuple:
    cp = leader_spec.get("contrast_pair")
    if not cp:
        return None
    axis = cp.get("axis", leader_spec.get("concept", ""))
    current_desc = leader_spec.get("notes", "") or cp.get("description", "")
    pos = cp.get("positive", [])
    neg = cp.get("negative", [])
    if not pos or not neg:
        return None
    prompt = DESCRIPTION_REWRITE_TEMPLATE.format(
        axis=axis,
        description=current_desc or "(none)",
        positive_examples=json.dumps(pos),
        negative_examples=json.dumps(neg),
    )
    try:
        raw = _run_sync(prompt, DESCRIPTION_REWRITE_SYSTEM)
    except Exception as e:
        print(f"[hillclimb] edit_description Claude call failed: {e}", flush=True)
        return None
    new_desc = _parse_single_key(raw, "new_description")
    if not new_desc or new_desc == current_desc:
        return None
    new_spec = copy.deepcopy(leader_spec)
    new_spec["notes"] = new_desc
    if isinstance(new_spec.get("contrast_pair"), dict):
        new_spec["contrast_pair"]["description"] = new_desc
    detail = json.dumps({"old": current_desc, "new": new_desc})
    return new_spec, "edit_description", detail


MUTATORS = {
    "alt_effective": _mutate_alt_effective,
    "alt_layer": _mutate_alt_layer,
    "swap_positive": lambda s, r: _mutate_swap_example(s, r, "positive"),
    "swap_negative": lambda s, r: _mutate_swap_example(s, r, "negative"),
    "edit_description": _mutate_edit_description,
}


# -----------------------------------------------------------------------------
# Strategy entrypoint
# -----------------------------------------------------------------------------

def _weighted_choice(rng: random.Random, weights: dict[str, int]) -> str:
    total = sum(weights.values())
    r = rng.uniform(0, total)
    cumulative = 0
    for k, w in weights.items():
        cumulative += w
        if r <= cumulative:
            return k
    return next(iter(weights))


def generate_candidates(
    n: int,
    db: ResultsDB,
    concept_pool: Optional[list[str]] = None,  # unused
    rng_seed: Optional[int] = None,
    max_attempts: int = 20,
) -> list[CandidateSpec]:
    """For each active leader, propose one mutation. Emits up to N specs.

    If there are fewer than N leaders, loops and proposes additional mutations
    (always fresh random mutation type) for each leader until N emitted.
    """
    rng = random.Random(rng_seed if rng_seed is not None else time.time_ns())
    leaders = db.get_leaders()
    contrast_leaders = [l for l in leaders if l.get("derivation_method") == "contrast_pair"]
    if not contrast_leaders:
        print("[hillclimb] no contrast_pair lineage leaders — run seed_lineages.py first", flush=True)
        return []

    print(
        f"[hillclimb] {len(contrast_leaders)} active lineage leaders; proposing {n} mutations",
        flush=True,
    )

    out: list[CandidateSpec] = []
    seen_this_batch: set[str] = set()
    attempts = 0

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        leader = contrast_leaders[(len(out) + attempts) % len(contrast_leaders)]
        try:
            leader_spec = json.loads(leader["spec_json"])
        except Exception:
            continue

        mutation_type = _weighted_choice(rng, MUTATION_WEIGHTS)
        mutator = MUTATORS[mutation_type]
        try:
            result = mutator(leader_spec, rng)
        except Exception as e:
            print(f"[hillclimb] mutator {mutation_type} raised: {e}", flush=True)
            continue
        if result is None:
            continue
        new_spec_dict, mtype, mdetail = result

        new_id = f"cand-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        parent_score = leader.get("score", 0.0) or 0.0

        spec = CandidateSpec(
            id=new_id,
            strategy=f"hillclimb_{mtype}",
            concept=new_spec_dict.get("concept", leader["concept"]),
            layer_idx=int(new_spec_dict["layer_idx"]),
            target_effective=float(new_spec_dict["target_effective"]),
            derivation_method="contrast_pair",
            baseline_n=0,
            notes=new_spec_dict.get("notes", ""),
            contrast_pair=new_spec_dict.get("contrast_pair"),
        )
        # Stash lineage info in the spec_dict (CandidateSpec doesn't carry them
        # natively; researcher.py passes them to db.insert_candidate directly).
        spec._lineage_id = leader["lineage_id"]
        spec._parent_candidate_id = leader["id"]
        spec._generation = (leader.get("generation", 0) or 0) + 1
        spec._mutation_type = mtype
        spec._mutation_detail = mdetail
        spec._parent_score = parent_score

        h = spec_hash(spec)
        if h in seen_this_batch or db.has_candidate_hash(h):
            time.sleep(0.001)
            continue
        seen_this_batch.add(h)
        out.append(spec)

    print(f"[hillclimb] emitted {len(out)} mutation candidates", flush=True)
    return out
