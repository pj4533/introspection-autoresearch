"""Vendored primitives from safety-research/introspection-mechanisms (Macar et al. 2026).

Only the files we actually use are vendored: model_utils, steering_utils,
vector_utils. Copied on 2026-04-16 with two in-place patches for Apple Silicon
MPS support:

1. `MODEL_NAME_MAP` extended with `gemma3_12b` and `gemma3_4b`.
2. `ModelWrapper.cleanup()` falls through from `torch.cuda.empty_cache()` to
   `torch.mps.empty_cache()` on Apple Silicon.

The original repo at ~/Developer/introspection-mechanisms is no longer required
at runtime. If upstream ships improvements we want, diff + port by hand.
"""

from .model_utils import (
    MODEL_NAME_MAP,
    ModelWrapper,
    get_layer_at_fraction,
    load_model,
)
from .steering_utils import (
    IntrospectionPrompt,
    SteeringHook,
    run_steered_introspection_test,
    run_unsteered_introspection_test,
)
from .vector_utils import (
    extract_concept_vector,
    extract_concept_vector_with_baseline,
    get_baseline_words,
)

__all__ = [
    "MODEL_NAME_MAP",
    "ModelWrapper",
    "get_layer_at_fraction",
    "load_model",
    "IntrospectionPrompt",
    "SteeringHook",
    "run_steered_introspection_test",
    "run_unsteered_introspection_test",
    "extract_concept_vector",
    "extract_concept_vector_with_baseline",
    "get_baseline_words",
]
