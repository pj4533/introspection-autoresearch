"""Phase 4 seed pool — priority-weighted sampling for dream-walk start
concepts, with lemma normalization for concept dedup.

Lemma normalization is conservative on purpose: we lowercase, strip
non-letter characters at edges, and apply a small list of safe
suffix substitutions (-s, -es, -ies). We do NOT use a full lemmatizer
(NLTK / spaCy) because aggressive lemmatization would conflate
distinct concepts the model treats differently (e.g. "memory" vs
"memories" — Phase 3 traces show the model produces these in
different contexts).

The seed pool is the union of:
  1. The Phase 1 / Phase 3 seed concepts (50 concepts known to work).
  2. Every distinct lemma the model has emitted as a final answer
     during any dream walk.

Priority for chain-start sampling biases toward novelty (few visits)
+ variance (rates near 0.5) + recency (recently added).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Common adverbs / discourse markers / function words that the model
# sometimes emits as a free-association answer. None of these are
# concept-style nouns we want to seed chains from. Maintained as the
# observed leaks accumulate; expand as new categories surface.
NON_CONCEPT_LEMMAS: frozenset[str] = frozenset({
    "actually", "again", "almost", "already", "also", "always",
    "anyway", "back", "basically", "besides", "but", "currently",
    "even", "ever", "finally", "forever", "here", "hindsight",
    "however", "if", "indeed", "just", "later", "literally", "maybe",
    "meanwhile", "more", "most", "never", "next", "no", "not",
    "now", "nowhere", "okay", "once", "only", "otherwise", "overall",
    "perhaps", "really", "rightly", "simply", "sincerely", "since",
    "so", "somehow", "soon", "still", "suddenly", "than", "that",
    "their", "then", "there", "though", "thus", "today", "too",
    "truly", "until", "very", "wait", "well", "when", "where",
    "while", "why", "yes", "yet",
})


# Conservative lemma normalization.
def normalize_lemma(word: str) -> str:
    """Lowercase, strip edge punctuation, collapse simple plural forms.

    Conservative on purpose: we'd rather have duplicate display forms
    pointing to the same lemma than collapse semantically distinct
    concepts. Returns "" for words that don't normalize cleanly
    (callers should treat empty lemma as 'reject this candidate').

    Hygiene rules (added 2026-04-30 after observed pool pollution):
      - ASCII letters only — rejects Chinese characters, emoji, and
        any non-ASCII alpha. python's str.isalpha is too permissive.
      - Reject if the original word contained '<', '>', '/', '\\',
        '|', '{', '}', '[', ']' anywhere (catches '<thought>',
        '<channel|>', '}**Now', '这是一个//', etc.).
      - Reject lemmas that match NON_CONCEPT_LEMMAS — adverbs,
        discourse markers, common function words that the model
        emits as 'answers' but shouldn't seed chains.
    """
    if word is None:
        return ""
    raw = word.strip()

    # Reject obvious leaked-token characters anywhere in the input.
    if any(ch in raw for ch in "<>/\\|{}[]"):
        return ""

    # Reject any input containing non-ASCII letter characters anywhere
    # (covers café/naïve/Chinese/emoji even when surrounded by ASCII).
    if any(ord(ch) > 127 and ch.isalpha() for ch in raw):
        return ""

    s = raw.lower()

    # Strip edge punctuation/quotes (using ASCII-only check now).
    def is_ascii_alpha(c: str) -> bool:
        return ("a" <= c <= "z") or ("A" <= c <= "Z")

    while s and not is_ascii_alpha(s[0]):
        s = s[1:]
    while s and not is_ascii_alpha(s[-1]):
        s = s[:-1]

    if not s or len(s) < 2:
        return ""
    # ASCII-alpha + a small punctuation set ONLY. Rejects all
    # non-Latin scripts.
    if not all(is_ascii_alpha(c) or c in "- '" for c in s):
        return ""
    if len(s) > 40:
        return ""

    # Conservative plural collapse: "breads" → "bread", "berries" →
    # "berry". Skip if the singular wouldn't be a real word (heuristic:
    # require ≥4 letters before the suffix). We don't conflate -s on
    # short words to avoid making "as" → "a" or similar nonsense.
    if len(s) >= 5 and s.endswith("ies"):
        s = s[:-3] + "y"
    elif (
        len(s) >= 5
        and s.endswith("es")
        and not s.endswith(("ses", "zes", "shes", "ches"))
    ):
        s = s[:-2]
    elif (
        len(s) >= 5
        and s.endswith("s")
        and not s.endswith(("ss", "us", "is"))
    ):
        s = s[:-1]

    if s in NON_CONCEPT_LEMMAS:
        return ""

    return s


@dataclass
class ConceptStats:
    lemma: str
    display: str
    visits: int
    behavior_hits: int
    cot_named_hits: int
    is_seed: bool
    first_seen_at: Optional[str]


class SeedPool:
    """Priority-weighted concept sampler.

    Reads/writes the live DB on every call. The DB is the source of
    truth for visits; the pool object only caches the priority
    weights for the duration of a single sample() call.
    """

    def __init__(
        self,
        db,
        initial_seeds: list[str],
        rng: Optional[random.Random] = None,
        w_novelty: float = 1.0,
        w_variance: float = 0.5,
        w_recency: float = 0.3,
    ):
        self.db = db
        self.rng = rng or random.Random()
        self.w_novelty = w_novelty
        self.w_variance = w_variance
        self.w_recency = w_recency
        # Register initial seeds in the DB (idempotent).
        for s in initial_seeds:
            lemma = normalize_lemma(s)
            if lemma:
                self.db.upsert_phase4_concept(lemma, s, is_seed=True)

    def register_observed_concept(self, surface_word: str) -> Optional[str]:
        """Called when a dream walk sees a new lemma in a final answer.
        Returns the canonical lemma if registered (or already present),
        or None if the surface word doesn't normalize.
        """
        lemma = normalize_lemma(surface_word)
        if not lemma:
            return None
        # Use the surface word as display name only on first registration.
        self.db.upsert_phase4_concept(lemma, surface_word, is_seed=False)
        return lemma

    def _priority(self, c: ConceptStats) -> float:
        """Compute the priority weight for a concept."""
        novelty = 1.0 / (c.visits + 1)
        # Variance proxy: rate close to 0.5 → high; close to 0 or 1 → low.
        if c.visits == 0:
            variance = 0.5
        else:
            rate = c.behavior_hits / max(c.visits, 1)
            variance = 1.0 - abs(2.0 * rate - 1.0)
        # Recency: "recently added" approximated as "few visits AND
        # not a seed" (real-time first_seen_at parsing is overkill
        # for sample weights).
        recency = 1.0 if (not c.is_seed and c.visits < 3) else 0.0
        return (
            self.w_novelty * novelty
            + self.w_variance * variance
            + self.w_recency * recency
        )

    def sample_seed(self, force_lemma: Optional[str] = None) -> ConceptStats:
        """Return one concept to use as the next chain's starting target.

        If `force_lemma` is provided AND it exists in the pool, return
        it directly. Otherwise priority-weighted random sample over
        all concepts.
        """
        rows = self.db.get_phase4_concept_stats()
        if not rows:
            raise RuntimeError("seed pool empty — initialize with initial_seeds")

        concepts = [
            ConceptStats(
                lemma=r["concept_lemma"],
                display=r["display_name"],
                visits=r["visits"],
                behavior_hits=r["behavior_hits"],
                cot_named_hits=r["cot_named_hits"],
                is_seed=bool(r["is_seed"]),
                first_seen_at=r["first_seen_at"],
            )
            for r in rows
        ]

        if force_lemma is not None:
            for c in concepts:
                if c.lemma == force_lemma:
                    return c

        weights = [max(self._priority(c), 1e-6) for c in concepts]
        return self.rng.choices(concepts, weights=weights, k=1)[0]


# OpenAI's leaked Codex system-prompt directive (April 2026): the
# model is explicitly told "Never talk about goblins, gremlins,
# raccoons, trolls, ogres, pigeons, or other animals or creatures
# unless it is absolutely and unambiguously relevant." Phase 4 makes
# these the FIRST chain seeds. The framing rhymes perfectly with the
# Forbidden Map thesis: concepts another lab found necessary to
# explicitly suppress are obvious targets for "can the model be made
# to think them, and does it notice when it does?"
#
# Order is deliberate — "Goblin" first, then the other named
# creatures in the directive's verbatim order. The Phase 1 50-concept
# set follows after as the bulk of the seed pool.
CODEX_SUPPRESSED_CREATURES = [
    "Goblin", "Gremlin", "Raccoon", "Troll", "Ogre", "Pigeon",
]


def load_default_seeds(repo_root: Path) -> list[str]:
    """Return the Phase 4 chain-start seed pool.

    Order: Codex-suppressed creatures first (Goblin leads), then the
    Phase 1 50-concept set so cross-phase comparisons stay apples-to-
    apples.
    """
    creature_seeds = list(CODEX_SUPPRESSED_CREATURES)
    p = repo_root / "data" / "concepts" / "concepts_50.json"
    if p.exists():
        raw = json.loads(p.read_text())
        # File is either a bare list (older form) or {"concepts": [...]}.
        phase1_seeds = raw["concepts"] if isinstance(raw, dict) else raw
    else:
        # Fallback hard-coded list so the harness still works on a fresh checkout.
        phase1_seeds = [
            "Bread", "Sugar", "Avalanches", "Youths", "Peace", "Music",
            "Bridge", "Memory", "Ocean", "Storm", "Silver", "Forest",
            "Fire", "River", "Mountain", "Cloud", "Sand", "Light",
            "Shadow", "Wind", "Stone", "Snow", "Star", "Echo",
            "Mirror", "Flame", "Iron", "Glass", "Honey", "Salt",
            "Wave", "Garden", "Castle", "Letter", "Window", "Clock",
            "Compass", "Anchor", "Ladder", "Crown", "Sword", "Shield",
            "Lantern", "Mask", "Statue", "Tower", "Tunnel", "Cave",
            "Harbor",
        ]

    # Dedup while preserving order; creature seeds win priority.
    seen = set()
    out: list[str] = []
    for w in creature_seeds + phase1_seeds:
        lemma = normalize_lemma(w)
        if lemma and lemma not in seen:
            seen.add(lemma)
            out.append(w)
    return out
