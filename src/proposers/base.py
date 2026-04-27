"""Proposer protocol.

A proposer takes a system+user prompt and returns raw text. The text is
parsed by the calling strategy (novel_contrast, directed_capraro, etc).
Streaming, structured output, and tool use are deliberately out of scope —
the strategies all want a single text blob and parse it with a regex.
"""

from __future__ import annotations

from typing import Protocol


class Proposer(Protocol):
    """Generates a single text completion for a strategy prompt."""

    name: str

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.7,
    ) -> str:
        ...
