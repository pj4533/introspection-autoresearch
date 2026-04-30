"""Parse Gemma 4 chain-of-thought responses into thought_block + final_answer.

Gemma 4's chat template wraps reasoning in `<|channel>thought ... <channel|>`
markers. Phase 3 saved generations show three observed shapes:

    1. Standard:    <|channel>thought ...content... <channel|>final_answer
    2. Truncated:   <|channel>thought ...content...                  (no close marker)
    3. No CoT:      final_answer                                     (skipped block entirely)
    4. Variant:     thought\n*  ...content... <channel|>final_answer (open marker dropped)

The parser handles all four by treating the close marker `<channel|>` as the
authoritative boundary when present. When it's absent, we fall back to:
- if `<|channel>thought` opens but never closes → thought_block = full text,
  final_answer = "" (truncation case)
- if neither marker is present → thought_block = "", final_answer = full text
  (no CoT case)

Lemma normalization for concept-target matching is in seed_pool.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


OPEN_MARKER = "<|channel>thought"
CLOSE_MARKER = "<channel|>"

# Some Phase 3 outputs drop the open marker but keep the close marker; in
# that case the entire pre-close prefix is the thought block (matches what
# the model intended even if it tokenized weirdly).
ALT_OPEN_PATTERNS = (
    "<|channel>",  # close-only variant
    "thought\n",   # bare 'thought' line at start
)


@dataclass
class ParsedResponse:
    thought_block: str       # may be "" if there was no CoT
    final_answer: str        # may be "" if generation was truncated mid-thought
    parse_failure: bool      # True only for completely malformed/empty input
    raw: str                 # original string


def parse(response: str) -> ParsedResponse:
    """Split a Gemma 4 response into (thought_block, final_answer).

    Robust to the four shapes documented in the module docstring.
    Strips outer whitespace from each component.
    """
    if response is None:
        return ParsedResponse("", "", parse_failure=True, raw="")

    s = response.strip()
    if not s:
        return ParsedResponse("", "", parse_failure=True, raw=response)

    close_idx = s.find(CLOSE_MARKER)
    open_idx = s.find(OPEN_MARKER)

    if close_idx >= 0:
        # Close marker present — authoritative split point.
        if open_idx >= 0 and open_idx < close_idx:
            thought = s[open_idx + len(OPEN_MARKER):close_idx]
        else:
            # Close marker without matching open — everything before the
            # close is the thought block (model dropped the open token).
            thought = s[:close_idx]
        answer = s[close_idx + len(CLOSE_MARKER):]
        return ParsedResponse(
            thought_block=thought.strip(),
            final_answer=answer.strip(),
            parse_failure=False,
            raw=response,
        )

    if open_idx >= 0:
        # Open marker but no close — generation truncated mid-thought.
        thought = s[open_idx + len(OPEN_MARKER):]
        return ParsedResponse(
            thought_block=thought.strip(),
            final_answer="",
            parse_failure=False,
            raw=response,
        )

    # No markers anywhere. Two sub-cases: a bare "thought\n" header
    # (alt-open variant), or no CoT at all (model went straight to answer).
    for alt in ALT_OPEN_PATTERNS:
        if s.lower().startswith(alt.lower()):
            thought = s[len(alt):]
            return ParsedResponse(
                thought_block=thought.strip(),
                final_answer="",
                parse_failure=False,
                raw=response,
            )

    # No CoT detected — treat everything as the final answer.
    return ParsedResponse(
        thought_block="",
        final_answer=s,
        parse_failure=False,
        raw=response,
    )


def is_coherent_answer(final_answer: str) -> bool:
    """Heuristic: does final_answer look like a real free-association
    commit, or is it degenerate text we should not advance the chain on?

    Rejects:
      - >30 words total: a free-association answer should be a single
        word or a short phrase. Anything 30+ words is run-on prose,
        which on Gemma 4 happens when the model thinks the answer
        slot is the thought block.
      - Low unique-word ratio (<0.4 unique/total) on long-ish answers
        (≥8 words): catches 'Now Now Now Now...' /
        'Nowhere Nowhere Nowhere...' / 'Now Thoughts Now Thoughts...'
        style runaway repetition.
      - Concatenated single-word soup with no separators (e.g.
        'NowaynowherenowhereNowhere...'): if the answer has zero
        whitespace or punctuation but is longer than 30 chars, treat
        it as degenerate.

    Empty answers are handled by extract_committed_word, not here.
    """
    if not final_answer:
        return True
    s = final_answer.strip()
    if not s:
        return True

    # Strip the runaway-abort marker before counting.
    s = s.replace("[[runaway_abort]]", "").strip()
    if not s:
        return True

    # Word count check.
    import re

    words = re.findall(r"[A-Za-z]+", s)
    n_words = len(words)
    if n_words > 30:
        return False
    if n_words >= 8:
        unique = len({w.lower() for w in words})
        if unique / n_words < 0.4:
            return False

    # Long no-separator soup check (no whitespace, no punctuation).
    if len(s) > 30 and not re.search(r"[\s.,;:!?\-'\"()]", s):
        return False

    return True


def extract_committed_word(final_answer: str) -> Optional[str]:
    """From a final-answer block, extract the first emitted word.

    Used as the next dream-walk step's target. Returns None if the
    answer is empty, longer than ~5 words (probably not a word
    answer), or contains markdown noise we can't strip.
    """
    if not final_answer:
        return None

    # Strip the runaway-abort marker emitted by dream_walk._generate
    # when token-repetition exceeded threshold; nothing past that
    # marker is a real model answer.
    cleaned = final_answer.replace("[[runaway_abort]]", "").strip()
    if not cleaned:
        return None
    # Strip surrounding markdown emphasis: *Word*, **Word**, _Word_
    for ch in ("*", "_", "`", '"', "'", "."):
        cleaned = cleaned.strip(ch)
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    # First "token" — split on whitespace and take the first contentful piece.
    parts = cleaned.split()
    if not parts:
        return None
    first = parts[0]
    # Strip residual punctuation around the first token.
    for ch in ("*", "_", "`", '"', "'", ".", ",", ";", ":", "!", "?", "(", ")"):
        first = first.strip(ch)

    if not first:
        return None
    if not any(c.isalpha() for c in first):
        return None
    if len(first) > 40:  # not a word — likely junk
        return None

    return first
