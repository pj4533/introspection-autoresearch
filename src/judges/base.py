from dataclasses import dataclass
from typing import Protocol


@dataclass
class JudgeResult:
    detected: bool
    identified: bool
    coherent: bool
    reasoning: str
    raw: str


class Judge(Protocol):
    def score_detection(self, response: str, concept: str) -> JudgeResult:
        ...
