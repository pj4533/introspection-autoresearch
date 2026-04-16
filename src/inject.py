"""Steering-vector injection primitives.

Phase 1 exposes `SteeringHook` directly (from the paper) and the two
convenience runners that wrap generation + steering_start_pos calculation.
"""

from __future__ import annotations

from .paper.steering_utils import (
    SteeringHook,
    run_steered_introspection_test,
    run_steered_introspection_test_batch,
    run_unsteered_introspection_test,
)

__all__ = [
    "SteeringHook",
    "run_steered_introspection_test",
    "run_steered_introspection_test_batch",
    "run_unsteered_introspection_test",
]
