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
from tqdm import tqdm

from .model_utils import ModelWrapper

# Position -2 is the last input token before the model starts generating.
# For the Gemma3 chat template, this is the newline token after
# "<start_of_turn>model". This is where the model "decides" whether to refuse.
DEFAULT_EXTRACT_POS = -2


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
        return [
            model.tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": insn}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            for insn in instructions
        ]

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
