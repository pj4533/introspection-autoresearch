"""Proposer abstraction.

Strategies generate text by calling `Proposer.generate(system, user, ...)`.
Production: `LocalMLXProposer` (mlx_lm-backed). Tests: `MockProposer`.
"""

from .base import Proposer
from .local_mlx_proposer import LocalMLXProposer
from .mock_proposer import MockProposer

__all__ = ["Proposer", "LocalMLXProposer", "MockProposer"]
