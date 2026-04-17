"""Researcher strategies for generating candidate steering-direction specs."""

from .novel_contrast import generate_candidates as novel_contrast
from .random_explore import generate_candidates as random_explore

__all__ = ["random_explore", "novel_contrast"]
