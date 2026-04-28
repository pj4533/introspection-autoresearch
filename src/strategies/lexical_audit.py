"""Lexical contamination audit for contrast pairs.

The mean-difference direction extracted from a contrast pair is a vector
in activation space. If a single surface token appears in *every*
positive-pole sentence and *never* in the negatives (or vice versa), the
embedding of that token will dominate the resulting steering direction.
A subsequent generation under that steering will then "introspect on"
the *token* rather than the *concept* the pair was meant to capture.

This module provides the audit function. It does not modify pairs; it
just reports which tokens are exclusive to which pole.

Discovered live, 2026-04-28: the Phase 2d causality Class 1 (score
3.812) had every positive example contain the literal word "caused";
when the contrast pair was rewritten without that token, the steering
signal vanished entirely. The methodological lesson is: a contrast
pair must contrast *concepts*, not *tokens*. Surface-level
contamination is the default failure mode of LLM-generated pairs
because grammatical rewrites (passive→active, third→first person,
sequence→causal-verb) are the easiest way to produce contrasts.

Public API:

    lexical_contamination(contrast_pair) -> ContaminationReport

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Stop words excluded from contamination scoring — these naturally cluster
# in any English text and aren't where the contamination signal lives.
# This list is short on purpose; better to flag a borderline word than miss
# a lexical tell.
STOP_WORDS = frozenset({
    "the", "a", "an", "of", "to", "and", "or", "but", "in", "on", "at",
    "by", "for", "with", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its",
})


@dataclass
class ContaminationReport:
    """Result of auditing one contrast pair.

    Fields:

      - ``positive_exclusive``: tokens that appear in EVERY positive example
        and NO negative example. Strongest contamination — these tokens
        will be the dominant feature in the mean-diff direction.
      - ``negative_exclusive``: same, swapped. Equally bad — the mean-diff
        will be biased AWAY from these tokens, so the model under steering
        will avoid them, which the judge may interpret as the positive
        pole "winning" by default.
      - ``high_skew``: tokens that appear in >=5/6 of one pole but <=1/6 of
        the other. Less catastrophic than fully exclusive but still strong
        signal. Useful for nudging the proposer rather than rejecting.
      - ``positive_total``, ``negative_total``: counts for context.
    """
    positive_exclusive: list[str] = field(default_factory=list)
    negative_exclusive: list[str] = field(default_factory=list)
    high_skew: list[tuple[str, int, int]] = field(default_factory=list)
    positive_total: int = 0
    negative_total: int = 0

    @property
    def is_clean(self) -> bool:
        """True iff no exclusive tokens and no high-skew tokens."""
        return (
            not self.positive_exclusive
            and not self.negative_exclusive
            and not self.high_skew
        )

    @property
    def is_contaminated(self) -> bool:
        """True iff there are any fully-exclusive tokens.

        High-skew alone is a softer warning. Fully-exclusive tokens
        guarantee the mean-diff direction is dominated by their
        embedding, which is the failure mode we observed live.
        """
        return bool(self.positive_exclusive) or bool(self.negative_exclusive)

    def summary(self) -> str:
        """Human-readable one-line summary, suitable for log lines."""
        if self.is_clean:
            return "lexically clean"
        parts = []
        if self.positive_exclusive:
            parts.append(f"+pole exclusive: {','.join(self.positive_exclusive[:5])}")
        if self.negative_exclusive:
            parts.append(f"-pole exclusive: {','.join(self.negative_exclusive[:5])}")
        if self.high_skew and not parts:
            top = self.high_skew[:3]
            parts.append(
                "high-skew: " + ",".join(f"{t}({p}/{n})" for t, p, n in top)
            )
        return "; ".join(parts)


def _tokenize(sentence: str) -> set[str]:
    """Lowercase, alphanumeric tokens, excluding stop words.

    Returns a SET — we count whether a token *appears* in the sentence,
    not how many times. This matches what the mean-diff sees: any
    occurrence of a token contributes its embedding once per sentence
    (after the model averages over positions).
    """
    tokens = re.findall(r"\b[a-z]+\b", sentence.lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 1}


def lexical_contamination(contrast_pair: dict) -> ContaminationReport:
    """Audit a contrast pair for surface-level contamination.

    Args:
        contrast_pair: dict with at minimum ``"positive"`` and
            ``"negative"`` lists of strings. Other keys are ignored.

    Returns:
        ContaminationReport detailing which tokens skew exclusively or
        near-exclusively to one pole.
    """
    positives = contrast_pair.get("positive") or []
    negatives = contrast_pair.get("negative") or []

    pos_token_sets = [_tokenize(s) for s in positives]
    neg_token_sets = [_tokenize(s) for s in negatives]

    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return ContaminationReport(positive_total=n_pos, negative_total=n_neg)

    # All tokens that appear anywhere in either pole.
    all_tokens: set[str] = set()
    for ts in pos_token_sets + neg_token_sets:
        all_tokens.update(ts)

    pos_exclusive: list[str] = []
    neg_exclusive: list[str] = []
    high_skew: list[tuple[str, int, int]] = []

    for token in sorted(all_tokens):
        pos_count = sum(1 for ts in pos_token_sets if token in ts)
        neg_count = sum(1 for ts in neg_token_sets if token in ts)
        # Fully exclusive: present in every example of one pole, absent
        # from every example of the other.
        if pos_count == n_pos and neg_count == 0:
            pos_exclusive.append(token)
        elif neg_count == n_neg and pos_count == 0:
            neg_exclusive.append(token)
        # High skew: >=5/6 vs <=1/6 in either direction.
        elif pos_count >= max(5, n_pos - 1) and neg_count <= 1:
            high_skew.append((token, pos_count, neg_count))
        elif neg_count >= max(5, n_neg - 1) and pos_count <= 1:
            high_skew.append((token, pos_count, neg_count))

    return ContaminationReport(
        positive_exclusive=pos_exclusive,
        negative_exclusive=neg_exclusive,
        high_skew=high_skew,
        positive_total=n_pos,
        negative_total=n_neg,
    )


def banned_tokens_for_decontamination(report: ContaminationReport) -> list[str]:
    """Tokens the proposer must avoid in BOTH poles when regenerating.

    Used by the ``lexical_decontaminate`` mutation operator. We forbid
    both the positive-exclusive and negative-exclusive sets — if the
    proposer can't write either pole without these tokens, the original
    contrast was lexical, not conceptual.

    High-skew tokens are NOT forbidden by default — they're
    near-contamination but not the smoking gun. The operator can
    optionally include them via a separate parameter if the contrast is
    severely contaminated.
    """
    return sorted(set(report.positive_exclusive + report.negative_exclusive))
