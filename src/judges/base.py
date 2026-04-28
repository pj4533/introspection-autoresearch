from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class JudgeResult:
    detected: bool
    identified: bool
    coherent: bool
    reasoning: str
    raw: str
    # Phase 2g: present only for `score_sae_feature` calls. Three-way grade:
    #   "conceptual"        — multi-clause paraphrase of the auto-interp label
    #                          that doesn't surface a single near-synonym token
    #   "lexical_fallback"  — single-word answer near-synonymous with the label
    #                          (e.g. "doubt" for an "uncertainty" feature)
    #   "none"              — no identifiable match
    # None for non-SAE judge calls (legacy contract preserved).
    identification_type: Optional[str] = None


class Judge(Protocol):
    def score_detection(self, response: str, concept: str) -> JudgeResult:
        ...
