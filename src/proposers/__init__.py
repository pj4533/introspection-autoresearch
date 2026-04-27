"""Proposer abstraction.

Strategies generate text by calling a `Proposer.generate(system, user, ...)`.
Two implementations:
  - `ClaudeProposer`: claude-agent-sdk wrapping (Opus 4.7 by default)
  - `LocalMLXProposer`: mlx_lm wrapping (Qwen3.6-27B 8-bit by default in
    docs/local_pipeline_plan.md)

Strategies that don't pass a proposer keep the legacy claude-agent-sdk
behavior so the old researcher.py still works during the refactor.
"""

from .base import Proposer
from .claude_proposer import ClaudeProposer
from .local_mlx_proposer import LocalMLXProposer
from .mock_proposer import MockProposer

__all__ = ["Proposer", "ClaudeProposer", "LocalMLXProposer", "MockProposer"]
