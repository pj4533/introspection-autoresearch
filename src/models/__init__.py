"""Model lifecycle abstractions for the four-phase worker.

The registry pattern lets the worker swap between Gemma3-12B (PyTorch-MPS)
and MLX models (judge / proposer) without two large models ever being in
memory simultaneously. See docs/local_pipeline_plan.md.
"""

from .registry import (
    ModelHandle,
    GemmaHandle,
    MLXHandle,
    MockHandle,
    free_memory_gb,
    enforce_free_memory,
)

__all__ = [
    "ModelHandle",
    "GemmaHandle",
    "MLXHandle",
    "MockHandle",
    "free_memory_gb",
    "enforce_free_memory",
]
