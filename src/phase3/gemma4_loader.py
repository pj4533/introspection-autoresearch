"""Gemma 4 31B-IT MLX 8-bit loader for Phase 3.

Wraps `mlx_lm.load` with project-specific defaults and exposes the
underlying text-only model for hook installation.

Gemma 4 31B-it is multimodal at the architecture level
(`Gemma4ForConditionalGeneration`), but MLX-LM's `gemma4.Model` class
already strips vision/audio weights in its `sanitize()` method and
delegates everything to `language_model` (see
.venv/lib/python3.11/site-packages/mlx_lm/models/gemma4.py:32-90).
So we get a text-only inference path without any extra work.

Layer count: 60 (per `text_config.num_hidden_layers` in the model's
`config.json`). Phase 1's introspection peak on Gemma 3 12B was at
68.75% depth (L=33 of 48); the paper predicted ~70% on 27B; for
Gemma 4 31B, that maps to roughly L=42 (60 × 0.7 = 42).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

DEFAULT_MODEL_PATH = "~/models/gemma-4-31B-it-8bit"
N_LAYERS = 60          # text_config.num_hidden_layers
HIDDEN_DIM = 5376      # text_config.hidden_size
PREDICTED_PEAK_LAYER = 42   # ~70% depth, paper's prediction (Phase 1 / 27B)
# Empirically discovered Phase 3 peak from layer sweep 2026-04-29:
# Bread injection at L=25 produces clean "Yes. It was bread." detection,
# while L=15/35/42/50/55 either over-steer or fail to trigger. ~42% depth.
# Possibly tied to Gemma 4's attention pattern — full-attention layers
# at indices 5, 11, 17, 23, 29, 35, 41, 47, 53, 59; L=25 sits right
# after the L=23 full-attention layer.
EMPIRICAL_PEAK_LAYER = 25
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Gemma4Loaded:
    """Bundle of model + tokenizer with utility accessors."""
    model: nn.Module       # Gemma4Model (multimodal wrapper, text-only at runtime)
    tokenizer: object      # mlx_lm tokenizer (TokenizerWrapper)
    model_path: str

    @property
    def language_model(self) -> nn.Module:
        """Direct access to the Gemma4TextModel inside the multimodal wrapper."""
        return self.model.language_model

    @property
    def n_layers(self) -> int:
        return len(self.model.language_model.model.layers)

    @property
    def hidden_dim(self) -> int:
        return self.model.language_model.model.config.hidden_size


def load_gemma4(
    model_path: str = DEFAULT_MODEL_PATH,
    verbose: bool = True,
) -> Gemma4Loaded:
    """Load Gemma 4 31B-IT MLX 8-bit + tokenizer.

    Loads via mlx_lm.load. The 8-bit quantized weights live as
    safetensors shards under `model_path`; tokenizer is the standard
    Gemma 4 SentencePiece v2.
    """
    # Defer import so importing this module doesn't require mlx-lm.
    from mlx_lm import load as _mlx_load

    resolved = str(Path(model_path).expanduser())
    if verbose:
        print(f"[gemma4_loader] loading {resolved} ...", flush=True)
    model, tokenizer = _mlx_load(resolved)
    if verbose:
        n = len(model.language_model.model.layers)
        h = model.language_model.model.config.hidden_size
        print(f"[gemma4_loader] loaded. n_layers={n}  hidden_dim={h}", flush=True)
    return Gemma4Loaded(model=model, tokenizer=tokenizer, model_path=resolved)


def tokenize_chat_prompt(
    handle: Gemma4Loaded,
    user_message: str,
    system_message: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> mx.array:
    """Apply the Gemma 4 chat template + tokenize. Returns a 1-D
    int32 mx.array of token ids."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})
    rendered = handle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    ids = handle.tokenizer.encode(rendered, add_special_tokens=False)
    return mx.array(ids, dtype=mx.int32)


def forward_capture(
    handle: Gemma4Loaded,
    prompt_ids: mx.array,
    layer_idx: int,
) -> mx.array:
    """Run the prompt through the model and return the residual stream
    at `layer_idx`, shape (seq_len, hidden_dim) on the model's device.

    Lightweight — installs a temporary capture hook, runs one forward
    pass without sampling, returns the captured tensor, uninstalls.
    """
    from .hooks import ActivationCapturer, install_capture, uninstall_hook

    cap = ActivationCapturer()
    install_capture(handle.model, layer_idx, cap)
    try:
        # mlx_lm Model.__call__ takes (inputs, cache=None, ...). For a
        # single forward without generation we just call it with the
        # prompt ids; we don't use the logits, only the captured
        # residual stream.
        if prompt_ids.ndim == 1:
            inputs = prompt_ids[None, :]   # (1, seq_len)
        else:
            inputs = prompt_ids
        _ = handle.model(inputs)
        # Force evaluation so the capture is materialized.
        mx.eval(cap.last)
    finally:
        uninstall_hook(handle.model, layer_idx)
    h = cap.last                            # (1, seq_len, hidden_dim)
    return h.squeeze(0)
