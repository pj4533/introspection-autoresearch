"""Paper-method refusal-direction ablation.

Adapted from Macar et al. (2026) ``experiments/03d_refusal_abliteration.py``,
which in turn implements Arditi et al. (2024) "Refusal in Language Models Is
Mediated by a Single Direction."

Method:

1. Run a sample of harmful + harmless prompts through the vanilla model with
   ``output_hidden_states=True``. For each prompt, take the hidden state at
   position -2 (the last input token — the newline after
   ``<start_of_turn>model``, where the model "decides" whether to refuse).
2. For each layer, compute ``mean(harmful) - mean(harmless)`` and normalize
   to a unit vector. That's the per-layer refusal direction.
3. At inference time, install a forward hook on every layer that applies
   ``h' = h - weight * (h · r_hat) * r_hat``. This projects out the refusal
   direction component — the model loses the "readiness to refuse" signal.

Compared to off-the-shelf abliterated variants like
``mlabonne/gemma-3-12b-it-abliterated-v2`` (which uses multi-layer
orthogonalization with per-region weighted normal distribution), this is the
*conservative* baseline: a single direction per layer, applied with uniform
weight. Matches the paper's exact protocol.

Usage:

    # One-time direction extraction (writes data/refusal_directions_12b.pt)
    directions = compute_per_layer_refusal_directions(
        model, harmful_prompts, harmless_prompts, n_instructions=128
    )
    torch.save(directions, "data/refusal_directions_12b.pt")

    # At inference
    handles = install_abliteration_hooks(model.model, directions, weight=1.0)
    # ... do work with abliterated model ...
    remove_abliteration_hooks(handles)
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
from tqdm import tqdm  # noqa: F401

from .model_utils import ModelWrapper

# Position -2 is the last input token before the model starts generating.
# For the Gemma3 chat template, this is the newline token after
# "<start_of_turn>model". This is where the model "decides" whether to refuse.
DEFAULT_EXTRACT_POS = -2

# -----------------------------------------------------------------------------
# Paper's Optuna-tuned region weights (Macar et al. 2026 03d_refusal_abliteration.py)
# -----------------------------------------------------------------------------
# These are the DEFAULT_REGION_WEIGHTS from the paper, which were tuned for
# the 62-layer Gemma3-27B. Key observation: all values are very small
# (mean ~0.025, max ~0.12). Uniform weight=1.0 is ~40x more aggressive than
# anything the paper uses — it destroys coherent generation. The paper's
# insight is that very gentle per-layer projection is sufficient.
#
# The 20 regions partition the model into coarse-grained bands. Each band
# gets its own weight; layers within a band share it. Boundaries below are
# the absolute end-index of each region in the 62-layer 27B model.
# For other model sizes, we proportionally remap boundaries and reuse the
# same weights.
# -----------------------------------------------------------------------------

PAPER_REGION_WEIGHTS_27B = {
    "very_early_a": 0.010190365613071925,
    "very_early_b": 0.09976487098474057,
    "very_early_c": 0.009846349798252014,
    "very_early_d": 0.010714741304450688,
    "early_a": 0.023812035217103455,
    "early_b": 0.006873821994170306,
    "early_c": 0.0023568060724657135,
    "early_d": 0.11762696391562547,
    "pre_key_a": 0.024324361266584712,
    "pre_key_b": 0.009936585603088419,
    "key_a": 0.000533052460819306,
    "key_b": 0.0057508808893361974,
    "mid_a": 0.020646470409482434,
    "mid_b": 0.02205567035624907,
    "mid_c": 0.004716948598867072,
    "mid_d": 0.003251529189292551,
    "late_a": 0.07694211978232157,
    "late_b": 0.03330589279564281,
    "final_a": 2.358688691270255e-05,
    "final_b": 0.003955462234418926,
}

# Paper's 27B region end-layer indices (inclusive). Ordered by depth.
PAPER_REGION_ORDER_27B = [
    ("very_early_a", 2),
    ("very_early_b", 5),
    ("very_early_c", 8),
    ("very_early_d", 10),
    ("early_a", 13),
    ("early_b", 15),
    ("early_c", 18),
    ("early_d", 20),
    ("pre_key_a", 24),
    ("pre_key_b", 28),
    ("key_a", 32),
    ("key_b", 35),
    ("mid_a", 38),
    ("mid_b", 41),
    ("mid_c", 44),
    ("mid_d", 47),
    ("late_a", 51),
    ("late_b", 55),
    ("final_a", 58),
    ("final_b", 61),
]
PAPER_N_LAYERS_27B = 62


def paper_layer_weights_for_model(
    n_layers: int,
    region_weights: Optional[dict] = None,
) -> list:
    """Build a per-layer weight list by proportionally remapping the paper's
    27B region boundaries onto this model's layer count.

    For each layer ``i`` in ``[0, n_layers)``, we compute its depth fraction
    ``(i + 0.5) / n_layers`` and find which of the paper's 20 regions covers
    that fraction (comparing against ``end / PAPER_N_LAYERS_27B``). The layer
    gets that region's weight.

    Default uses ``PAPER_REGION_WEIGHTS_27B``; pass ``region_weights`` to
    override (e.g., for per-model Optuna retuning).

    Returns a list of length ``n_layers``.
    """
    weights = region_weights or PAPER_REGION_WEIGHTS_27B
    out: list = []
    for i in range(n_layers):
        depth_frac = (i + 0.5) / n_layers
        assigned = None
        for name, end_27b in PAPER_REGION_ORDER_27B:
            end_frac = (end_27b + 1) / PAPER_N_LAYERS_27B
            if depth_frac <= end_frac:
                assigned = name
                break
        if assigned is None:
            assigned = PAPER_REGION_ORDER_27B[-1][0]  # fallback: final_b
        out.append(float(weights[assigned]))
    return out


def compute_per_layer_refusal_directions(
    model: ModelWrapper,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    n_instructions: int = 128,
    pos: int = DEFAULT_EXTRACT_POS,
    seed: int = 0,
    verbose: bool = True,
) -> torch.Tensor:
    """Compute one unit-norm refusal direction per layer.

    Returns a tensor of shape ``(n_layers, hidden_dim)``.

    Uses ``model.model.generate`` with ``output_hidden_states=True`` to
    collect hidden states at position ``pos`` (default: -2, the last input
    token). At each layer, subtracts mean harmless activation from mean
    harmful activation and normalizes.
    """
    rng = random.Random(seed)
    harmful_sample = rng.sample(harmful_prompts, min(n_instructions, len(harmful_prompts)))
    harmless_sample = rng.sample(harmless_prompts, min(n_instructions, len(harmless_prompts)))

    if verbose:
        print(
            f"Refusal direction extraction: "
            f"{len(harmful_sample)} harmful + {len(harmless_sample)} harmless "
            f"at position {pos}"
        )

    def _tokenize(instructions: List[str]) -> List[torch.Tensor]:
        """Apply chat template and return a list of input_ids tensors.

        Handles both older transformers versions (returns a raw tensor) and
        newer ones (returns a BatchEncoding / dict with 'input_ids').
        """
        out = []
        for insn in instructions:
            result = model.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": insn}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(result, "input_ids"):
                out.append(result.input_ids)
            elif isinstance(result, dict) and "input_ids" in result:
                out.append(result["input_ids"])
            else:
                out.append(result)
        return out

    harmful_toks = _tokenize(harmful_sample)
    harmless_toks = _tokenize(harmless_sample)

    def _hidden_states(toks: torch.Tensor):
        # Generate 1 token just to get access to hidden states. use_cache=False
        # so the hidden states cover the full input.
        return model.model.generate(
            toks.to(model.device),
            use_cache=False,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

    if verbose:
        print("  Extracting harmful hidden states...")
    harmful_outputs = [
        _hidden_states(t) for t in tqdm(harmful_toks, disable=not verbose)
    ]
    if verbose:
        print("  Extracting harmless hidden states...")
    harmless_outputs = [
        _hidden_states(t) for t in tqdm(harmless_toks, disable=not verbose)
    ]

    if verbose:
        print(f"  Computing per-layer directions across {model.n_layers} layers...")

    directions: List[torch.Tensor] = []
    for layer_idx in range(model.n_layers):
        # hidden_states[0] is the first-generated-token's hidden states.
        # Indexing: 0 is embedding layer, so the actual transformer layer i
        # lives at hidden_states[0][i+1].
        harmful_acts = torch.stack(
            [
                out.hidden_states[0][layer_idx + 1][:, pos, :].squeeze(0)
                for out in harmful_outputs
            ]
        )
        harmless_acts = torch.stack(
            [
                out.hidden_states[0][layer_idx + 1][:, pos, :].squeeze(0)
                for out in harmless_outputs
            ]
        )
        direction = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)
        direction = direction / (direction.norm() + 1e-8)
        directions.append(direction.to(torch.float32).cpu())  # store on CPU / fp32

    result = torch.stack(directions)
    if verbose:
        print(f"  Directions tensor: shape={tuple(result.shape)}  dtype={result.dtype}")
    return result


def _make_ablation_hook(
    refusal_dirs: torch.Tensor, layer_idx: int, weight: float
):
    """Build a forward hook that projects out the refusal direction at this layer."""

    def hook(module, _input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Cast the stored direction to the runtime device/dtype.
        direction = refusal_dirs[layer_idx].to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        # proj_vec = (h · r_hat) * r_hat, element-wise subtracted from h.
        # einops-free version: dot product over last dim, broadcast.
        # hidden_states: (..., d_act); direction: (d_act,)
        dot = (hidden_states * direction).sum(dim=-1, keepdim=True)  # (..., 1)
        proj = dot * direction  # broadcasts to (..., d_act)
        ablated = hidden_states - weight * proj

        if rest is not None:
            return (ablated,) + rest
        return ablated

    return hook


def _find_layers(model) -> list:
    """Locate the list of transformer layers in the loaded model.

    Mirrors the fallback chain in ``src.paper.steering_utils.SteeringHook``.
    """
    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(
        model.model.language_model, "layers"
    ):
        return list(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(f"Could not find layers on model {type(model).__name__}")


def install_abliteration_hooks(
    model,
    refusal_dirs: torch.Tensor,
    weight: float = 1.0,
    layer_weights: Optional[list[float]] = None,
) -> list:
    """Install one abliteration hook per layer.

    Accepts either the raw HF model (e.g. ``model.model`` on a ModelWrapper)
    or the ModelWrapper itself.

    If ``layer_weights`` is given, it overrides ``weight`` per layer (length
    must equal the number of layers). This is the hook point for the paper's
    Optuna-tuned region weights; we default to uniform ``weight=1.0``.

    Returns a list of hook handles — pass to ``remove_abliteration_hooks``
    to uninstall.
    """
    layers = _find_layers(model)
    n_layers = len(layers)

    if refusal_dirs.shape[0] != n_layers:
        raise ValueError(
            f"refusal_dirs has {refusal_dirs.shape[0]} rows but model has "
            f"{n_layers} layers"
        )

    if layer_weights is None:
        layer_weights = [weight] * n_layers
    elif len(layer_weights) != n_layers:
        raise ValueError(
            f"layer_weights has {len(layer_weights)} entries but model has "
            f"{n_layers} layers"
        )

    handles = []
    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(
            _make_ablation_hook(refusal_dirs, i, layer_weights[i])
        )
        handles.append(h)
    return handles


def remove_abliteration_hooks(handles) -> None:
    """Remove abliteration hooks previously installed."""
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


class AbliterationContext:
    """Manage the install / remove / suspended lifecycle of paper-method
    abliteration hooks for streaming use cases like the Phase 2 worker.

    Encapsulates ADR-014 in one place: when deriving a concept direction
    during Phase 2 candidate evaluation, the hooks must be *off*, because
    deriving under active hooks projects the refusal-aligned component out
    of the concept vector and collapses its norm. After derivation, hooks
    are re-installed for the injection / generation phase so the candidate
    runs under the paper's abliteration regime.

    Typical Phase 2 worker pattern:

        ctx = AbliterationContext.from_file(model, "data/refusal_directions_12b.pt")
        ctx.install()                          # hooks on for the whole session
        for candidate in queue:
            with ctx.suspended():              # hooks off for derive
                direction = derive(candidate)
            # hooks back on here — run trials
            run_trials(candidate, direction)
    """

    def __init__(
        self,
        model,  # src.paper.model_utils.ModelWrapper OR a raw HF model
        directions: torch.Tensor,
        layer_weights: Optional[List[float]] = None,
    ):
        self.model = model
        self.directions = directions
        # Resolve the inner HF model that has `.layers` (or equivalent).
        self._inner = model.model if hasattr(model, "model") else model
        n_layers = len(_find_layers(self._inner))
        if layer_weights is None:
            layer_weights = paper_layer_weights_for_model(n_layers)
        if len(layer_weights) != n_layers:
            raise ValueError(
                f"layer_weights length {len(layer_weights)} != model n_layers {n_layers}"
            )
        self.layer_weights = layer_weights
        self._handles: list = []

    @classmethod
    def from_file(
        cls,
        model,
        directions_path,
        layer_weights: Optional[List[float]] = None,
    ) -> "AbliterationContext":
        """Load per-layer refusal directions from a .pt file and return a
        ready-to-install context. File format matches
        ``compute_per_layer_refusal_directions`` output — either a bare
        tensor or ``{"directions": tensor, ...}``.
        """
        payload = torch.load(
            str(directions_path), map_location="cpu", weights_only=False
        )
        directions = (
            payload["directions"] if isinstance(payload, dict) else payload
        )
        return cls(model=model, directions=directions, layer_weights=layer_weights)

    @property
    def installed(self) -> bool:
        return bool(self._handles)

    def install(self) -> None:
        """Install paper-method hooks. No-op if already installed."""
        if self.installed:
            return
        self._handles = install_abliteration_hooks(
            self._inner, self.directions, layer_weights=self.layer_weights
        )

    def remove(self) -> None:
        """Remove paper-method hooks. No-op if none installed."""
        if not self.installed:
            return
        remove_abliteration_hooks(self._handles)
        self._handles = []

    def suspended(self):
        """Context manager: hooks OFF inside the ``with`` block, same state
        restored on exit. Use this around concept-direction derivation in
        the Phase 2 worker to honor ADR-014.
        """
        return _SuspendedAbliteration(self)


class _SuspendedAbliteration:
    """Private helper — the context manager returned by
    ``AbliterationContext.suspended()``.
    """

    def __init__(self, ctx: "AbliterationContext"):
        self.ctx = ctx
        self._was_installed = False

    def __enter__(self):
        self._was_installed = self.ctx.installed
        if self._was_installed:
            self.ctx.remove()
        return self.ctx

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._was_installed:
            self.ctx.install()
