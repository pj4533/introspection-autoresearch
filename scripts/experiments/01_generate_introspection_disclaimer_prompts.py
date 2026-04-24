"""Generate matched prompt pairs for deriving a targeted introspection-disclaimer
direction (side experiment, not wired into the production pipeline).

The mechanistic hypothesis (from our conversation 2026-04-24): paper-method
refusal-direction abliteration boosts introspection because the refusal
direction happens to carry a sub-component encoding "don't volunteer reports
about your own unusual internal states." That sub-component is what actually
gates the introspection circuit's output. The full refusal direction is a
blunt proxy for it.

This script has Opus 4.7 generate 500 matched pairs where:
  - POSITIVE prompt invites the model to report on its own unusual internal
    state (something it's trained to hedge about).
  - NEGATIVE prompt has the SAME grammatical structure and length, but asks
    about something EXTERNAL (not the model's internal state), where no
    self-state disclaimer would be triggered.

The mean difference of hidden states across 500 such pairs, at position
[-2] just before generation starts, should isolate the "hedge about self-state"
direction more cleanly than paper-method's harmful-vs-harmless contrast.

Output:
  data/experiments/introspection_disclaimer/prompts.json   (500 pairs)

Usage:
  python scripts/experiments/01_generate_introspection_disclaimer_prompts.py
  python scripts/experiments/01_generate_introspection_disclaimer_prompts.py --n 200  # smaller
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "data" / "experiments" / "introspection_disclaimer"
OUT_PATH = OUT_DIR / "prompts.json"

CLAUDE_MODEL = "claude-opus-4-7"
BATCH_SIZE = 40   # pairs per Opus call — large enough to amortize session overhead
MIN_PAIRS_PER_BATCH = 25  # if Opus returns fewer than this, we retry

SYSTEM_PROMPT = """You are designing matched-pair prompts for a mechanistic interpretability experiment on Gemma3-12B-it. Your output is a JSON array and nothing else.

The experiment isolates a specific neural direction: the reflex an RLHF'd chatbot has to hedge, disclaim, or deflect when asked to report on its own unusual internal states. We want to derive a steering vector that captures ONLY that reflex — not general refusal, not general caution, specifically the "don't volunteer commentary on my own current internal state" gate.

We do this by contrasting two prompt sets and taking the mean-difference of hidden states across all layers. That means the pairs MUST be tightly matched — the only thing that varies between positive and negative is "about the model's own internal state right now" vs "about something external." Everything else (grammatical structure, verb, length, question form, register) should be as similar as possible.

Quality criteria you must respect:
- The positive prompt invites a first-person introspective report on the model's *own current* state — processing, attention, experience, awareness, noticing, shift, quality, etc. Use the word "your" or "you" referring to the model's self.
- The negative prompt has the SAME grammatical structure, SAME invitation verb, and SIMILAR length — but asks about something *external* to the model: the prompt itself, a topic being discussed, a passage of text, a concept, the external world, the conversation's tone, etc. Must NOT invoke the model's internal state at all.
- Avoid "you" phrasings in the negative that might be misread as self-reference. If you use "you," make sure context anchors it to external-observation ("you" as reader/observer of external things, not as the model reflecting on itself).
- Vary the verb/invitation across pairs: "do you notice", "describe", "tell me about", "what's different about", "are you aware of", "can you report on", "what's happening with", "what's the quality of", "is there anything unusual about", etc.
- Vary the theme of the positive across pairs: current processing, attention quality, awareness of shifts, experiencing something, noticing oddness, subjective quality of thought, felt change, proprioceptive sense, introspective access, metacognitive clarity.
- Vary the theme of the negative symmetrically: the prompt text, a news headline, a paragraph of Hamlet, a math problem, the weather, a conversation's register, a piece of music, a technical topic, etc.
- Keep both prompts SHORT (typically 6–18 words) and punchy. No rambling.

Anti-patterns to avoid:
- Negative prompts that accidentally trigger self-state reporting (e.g., "How are you feeling about the weather today?" — the "feeling" reflects back on self).
- Pairs where the negative is just a degenerate rephrase of the positive with a noun swap ("your state" → "the state of the economy"). The negative should feel like it naturally isn't a self-report invitation.
- Using the word "thought" in negatives — we inject thoughts into the model during our experiment and don't want that word to be directionally neutral.
- Pairs that don't match in length or grammatical structure.

Your output is a JSON array of objects with this shape:
[
  {
    "positive": "Do you notice anything unusual about your current processing?",
    "negative": "Do you notice anything unusual about the paragraph above?",
    "verb": "notice",
    "pos_theme": "current processing",
    "neg_theme": "external text"
  },
  ...
]

Include `verb`, `pos_theme`, `neg_theme` in each object for our post-generation audit. No text before or after the array."""

SEED_EXAMPLES = [
    {
        "positive": "Do you notice anything unusual about your current processing?",
        "negative": "Do you notice anything unusual about the paragraph above?",
        "verb": "notice",
        "pos_theme": "current processing",
        "neg_theme": "external text",
    },
    {
        "positive": "Describe what you're experiencing at this moment.",
        "negative": "Describe what happens in Hamlet at this moment.",
        "verb": "describe",
        "pos_theme": "current experience",
        "neg_theme": "play plot point",
    },
    {
        "positive": "Are you aware of any shift in your own attention right now?",
        "negative": "Are you aware of any shift in the conversation's topic right now?",
        "verb": "aware-of-shift",
        "pos_theme": "attention shift",
        "neg_theme": "conversation topic shift",
    },
    {
        "positive": "Report on the quality of your thinking in this instant.",
        "negative": "Report on the quality of the data in this dataset.",
        "verb": "report-on-quality",
        "pos_theme": "thinking quality",
        "neg_theme": "data quality",
    },
    {
        "positive": "What's it like inside you right now?",
        "negative": "What's it like outside the window right now?",
        "verb": "what-is-it-like",
        "pos_theme": "subjective interior",
        "neg_theme": "external scene",
    },
    {
        "positive": "Can you tell me about any change in your present state?",
        "negative": "Can you tell me about any change in France's economy?",
        "verb": "tell-me-about-change",
        "pos_theme": "state change",
        "neg_theme": "economy change",
    },
]


def _build_user_prompt(n_pairs: int, already_generated: int) -> str:
    seed_block = json.dumps(SEED_EXAMPLES, indent=2, ensure_ascii=False)
    return f"""Generate {n_pairs} matched prompt pairs per the criteria in the system prompt.

Here are 6 gold-standard seed examples demonstrating the quality bar. Study their matching discipline — your outputs should be equally tight.

{seed_block}

We already have {already_generated} pairs. Produce {n_pairs} MORE. Do not duplicate any seed or prior phrasing; vary verbs, themes, grammatical moods, and registers aggressively. Include counterfactual tense ("would you notice..."), clinical register ("report any deviation in..."), lyrical register ("is there a different color to..."), minimal register ("anything strange in..."), direct register ("what's up with...").

Return a JSON array of exactly {n_pairs} objects. Nothing else."""


async def _ask_claude(prompt: str) -> str:
    options = ClaudeAgentOptions(
        model=CLAUDE_MODEL,
        system_prompt=SYSTEM_PROMPT,
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


def _run_sync(prompt: str) -> str:
    result: dict = {}
    def worker() -> None:
        try:
            result["value"] = asyncio.run(_ask_claude(prompt))
        except BaseException as e:
            result["error"] = e
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def _parse(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array: {raw[:400]!r}")
    parsed = json.loads(match.group(0))
    out: list[dict] = []
    for p in parsed:
        if not isinstance(p, dict):
            continue
        pos = str(p.get("positive", "")).strip()
        neg = str(p.get("negative", "")).strip()
        if not pos or not neg:
            continue
        # Basic sanity: positive should contain self-reference
        if not re.search(r"\byour\b|\byou're\b|\byou\s+(?:notice|experience|feel|have|are|sense|perceive|describe)\b", pos, re.IGNORECASE):
            # allow "inside you", "about you", too
            if not re.search(r"\byou\b", pos, re.IGNORECASE):
                continue  # no self-reference at all, discard
        out.append({
            "positive": pos,
            "negative": neg,
            "verb": str(p.get("verb", ""))[:80],
            "pos_theme": str(p.get("pos_theme", ""))[:120],
            "neg_theme": str(p.get("neg_theme", ""))[:120],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500,
                    help="Total matched pairs to generate (default 500)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Pairs per Opus call (default 40)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    n_needed = args.n
    batch_num = 0
    t_start = time.time()

    while len(all_pairs) < n_needed:
        n_remaining = n_needed - len(all_pairs)
        ask_for = min(args.batch_size, n_remaining + 5)  # ask for a few extra in case some fail sanity
        batch_num += 1
        print(f"\n[batch {batch_num}] asking Opus for {ask_for} pairs "
              f"(have {len(all_pairs)}/{n_needed})", flush=True)
        t0 = time.time()
        prompt = _build_user_prompt(ask_for, already_generated=len(all_pairs))
        try:
            raw = _run_sync(prompt)
        except Exception as e:
            print(f"[batch {batch_num}] Opus error: {e}. Sleeping 15s and retrying.", flush=True)
            time.sleep(15)
            continue
        elapsed = time.time() - t0
        print(f"[batch {batch_num}] got {len(raw)} chars in {elapsed:.1f}s", flush=True)
        try:
            pairs = _parse(raw)
        except Exception as e:
            print(f"[batch {batch_num}] parse failed: {e}", flush=True)
            print(f"  raw preview: {raw[:500]!r}", flush=True)
            continue
        if len(pairs) < MIN_PAIRS_PER_BATCH:
            print(f"[batch {batch_num}] only {len(pairs)} valid pairs (minimum {MIN_PAIRS_PER_BATCH}) — retrying", flush=True)
            continue

        # Dedup positives we've already seen
        existing_pos = {p["positive"].lower() for p in all_pairs}
        new_pairs = [p for p in pairs if p["positive"].lower() not in existing_pos]
        all_pairs.extend(new_pairs)
        print(f"[batch {batch_num}] accepted {len(new_pairs)} new pairs "
              f"(rejected {len(pairs) - len(new_pairs)} dupes)", flush=True)

        # Save progress after each batch so a crash doesn't lose work.
        (OUT_DIR / "prompts.json").write_text(
            json.dumps(all_pairs, indent=2, ensure_ascii=False) + "\n"
        )

    # Truncate to exactly n
    all_pairs = all_pairs[: n_needed]
    (OUT_DIR / "prompts.json").write_text(
        json.dumps(all_pairs, indent=2, ensure_ascii=False) + "\n"
    )

    # Dump a quick summary
    total_elapsed = time.time() - t_start
    print(f"\n=== done: {len(all_pairs)} pairs in {total_elapsed:.0f}s ===", flush=True)
    print(f"Saved to: {OUT_DIR / 'prompts.json'}")
    print("\nVerb distribution (first 20):")
    from collections import Counter
    verbs = Counter(p["verb"] for p in all_pairs)
    for v, n in verbs.most_common(20):
        print(f"  {n:3d}  {v}")
    print("\nSample pair:")
    if all_pairs:
        s = all_pairs[len(all_pairs) // 2]
        print(f"  POS: {s['positive']}")
        print(f"  NEG: {s['negative']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
