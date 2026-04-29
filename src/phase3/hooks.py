"""DecoderLayer wrapper for residual-stream capture + steering injection.

MLX (unlike PyTorch) has no `register_forward_hook` mechanism. We get
the same effect by replacing a target `DecoderLayer` with a thin
wrapper that delegates to the original layer and additionally:

  - captures the post-layer residual stream into a list (for concept-
    vector derivation), and/or
  - adds a steering vector to the post-layer residual stream (for
    paper-style steered generation).

The wrapper preserves the original layer's parameter tree by holding
the original as `self.original` — MLX's parameter walking is recursive
through `nn.Module` children, so weights remain reachable for the
quantized model exactly as before.

Use:

    capturer = ActivationCapturer()
    install_capture(model, layer_idx=42, capturer=capturer)
    model(prompt_ids)             # captures during forward
    activation = capturer.last    # (batch, seq, hidden_dim)
    uninstall(model, layer_idx=42)

For steering:

    install_steering(model, layer_idx=42, direction=v, strength=alpha)
    out = model.generate(...)
    uninstall(model, layer_idx=42)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ActivationCapturer:
    """Accumulates residual-stream activations from one or more hooked
    layers across a forward pass."""
    captures: list = field(default_factory=list)

    def __call__(self, h: mx.array) -> None:
        """Append a copy of `h` to the buffer."""
        # Use mx.eval so the capture is materialized; otherwise lazy
        # graph evaluation would hold references to the in-flight
        # forward computation.
        self.captures.append(h)

    @property
    def last(self) -> Optional[mx.array]:
        return self.captures[-1] if self.captures else None

    def reset(self) -> None:
        self.captures.clear()


class HookedDecoderLayer(nn.Module):
    """Wraps a Gemma4 DecoderLayer to support capture + injection.

    Signature matches gemma4_text.DecoderLayer.__call__:
        h, kvs, offset = layer(h, mask, c, per_layer_input=..., shared_kv=..., offset=...)

    Optional behavior controlled by attributes — set to None to disable.
    Both can be active simultaneously (e.g., capture under abliteration).
    """

    def __init__(self, original: nn.Module, layer_idx: int):
        super().__init__()
        self.original = original
        self.layer_idx = layer_idx
        self.capturer: Optional[ActivationCapturer] = None
        self.steering_direction: Optional[mx.array] = None
        self.steering_strength: float = 0.0
        # Optional refusal-ablation hook (h' = h - w * (h·r̂) r̂, paper §3.3).
        # When set, applied AFTER the original layer runs, BEFORE capture/
        # steering. r̂ should be unit-norm.
        self.refusal_dir: Optional[mx.array] = None
        self.refusal_weight: float = 0.0

    # Proxy plain-Python attributes (e.g. `.layer_type`) through to the
    # wrapped DecoderLayer. Gemma4TextModel._make_masks reads
    # `layer.layer_type` per layer to pick sliding-vs-full attention
    # masks. MLX nn.Module inherits from dict, and its __setattr__
    # routes nn.Module-typed values into the dict-self (because
    # nn.Module IS a dict). So `self.original` ends up in `self["original"]`,
    # not `self.__dict__["original"]`. We pull it from dict-self via
    # dict.get to avoid re-entering __getattr__.
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # MLX puts child Modules into self (the dict). Use dict.get
        # directly to avoid recursing back through this __getattr__.
        original = dict.get(self, "original", None)
        if original is None:
            raise AttributeError(name)
        return getattr(original, name)

    def __call__(self, h: mx.array, *args, **kwargs):
        h_out, kvs, offset = self.original(h, *args, **kwargs)

        # 1. Refusal-direction ablation (paper-method abliteration).
        if self.refusal_dir is not None and self.refusal_weight != 0.0:
            # Project h_out onto refusal_dir, subtract `weight * proj`.
            # h shape: (batch, seq, hidden_dim); refusal_dir: (hidden_dim,)
            r = self.refusal_dir
            proj = (h_out * r).sum(axis=-1, keepdims=True) * r
            h_out = h_out - self.refusal_weight * proj

        # 2. Capture (after abliteration, before steering — matches paper's
        #    "extract from already-abliterated residual when steering under
        #    abliteration mode").
        if self.capturer is not None:
            self.capturer(h_out)

        # 3. Steering injection.
        if self.steering_direction is not None and self.steering_strength != 0.0:
            h_out = h_out + self.steering_strength * self.steering_direction

        return h_out, kvs, offset


def _get_layers_list(model: nn.Module) -> list:
    """Return the .layers list from a Gemma4 multimodal Model or
    Gemma4TextModel directly."""
    # Multimodal Model: .language_model.model.layers (since Gemma4TextModel
    # wraps inside Model class — see gemma4_text.py line 578-584)
    # Direct: .model.layers (Gemma4TextModel)
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        "Could not locate transformer layer list on model "
        f"(type={type(model).__name__})"
    )


def install_hook(model: nn.Module, layer_idx: int) -> HookedDecoderLayer:
    """Replace `layers[layer_idx]` with a HookedDecoderLayer wrapping
    the original. Idempotent — re-installing returns the existing
    wrapper. Returns the wrapper so callers can configure capturer /
    steering / refusal."""
    layers = _get_layers_list(model)
    existing = layers[layer_idx]
    if isinstance(existing, HookedDecoderLayer):
        return existing
    wrapper = HookedDecoderLayer(existing, layer_idx=layer_idx)
    layers[layer_idx] = wrapper
    return wrapper


def uninstall_hook(model: nn.Module, layer_idx: int) -> None:
    """Restore the original layer (no-op if not currently wrapped)."""
    layers = _get_layers_list(model)
    cur = layers[layer_idx]
    if isinstance(cur, HookedDecoderLayer):
        layers[layer_idx] = cur.original


def uninstall_all(model: nn.Module) -> None:
    """Restore every wrapped layer in this model. Useful between
    experiments to ensure a clean baseline."""
    layers = _get_layers_list(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, HookedDecoderLayer):
            layers[i] = layer.original


def install_capture(
    model: nn.Module, layer_idx: int, capturer: ActivationCapturer
) -> HookedDecoderLayer:
    """Install a hook + bind a capturer."""
    h = install_hook(model, layer_idx)
    h.capturer = capturer
    return h


def install_steering(
    model: nn.Module,
    layer_idx: int,
    direction: mx.array,
    strength: float,
) -> HookedDecoderLayer:
    """Install a hook + bind a steering direction. `direction` shape
    (hidden_dim,); `strength` is alpha. Cleared by uninstall_hook."""
    h = install_hook(model, layer_idx)
    h.steering_direction = direction
    h.steering_strength = float(strength)
    return h


def install_refusal_ablation(
    model: nn.Module,
    refusal_dirs_per_layer: list,
    weights_per_layer: list,
) -> list[HookedDecoderLayer]:
    """Install paper-method §3.3 refusal-direction ablation across all
    layers.

    `refusal_dirs_per_layer`: list of (hidden_dim,) unit vectors, one
        per layer. Length must equal len(layers).
    `weights_per_layer`: list of floats, the per-layer projection
        scalar (typically the paper's Optuna-remapped values).

    Returns the list of wrappers so callers can later reach in and
    e.g. attach a capturer or steering on top of abliteration.
    """
    layers = _get_layers_list(model)
    if len(refusal_dirs_per_layer) != len(layers):
        raise ValueError(
            f"refusal_dirs length {len(refusal_dirs_per_layer)} "
            f"!= n_layers {len(layers)}"
        )
    if len(weights_per_layer) != len(layers):
        raise ValueError(
            f"weights length {len(weights_per_layer)} "
            f"!= n_layers {len(layers)}"
        )
    wrappers = []
    for i in range(len(layers)):
        h = install_hook(model, i)
        h.refusal_dir = refusal_dirs_per_layer[i]
        h.refusal_weight = float(weights_per_layer[i])
        wrappers.append(h)
    return wrappers
