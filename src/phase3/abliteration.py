"""Paper-method §3.3 refusal-direction ablation, MLX edition.

Mirrors `src/paper/abliteration.py` (HF transformers + torch hooks)
but uses MLX modules and the Phase 3 hook wrapper.

Method (Macar et al. 2026 §3.3, Arditi et al. 2024):

1. Run a sample of harmful + harmless prompts through the vanilla
   model with hidden-state capture at the LAST INPUT TOKEN POSITION
   (typically -2 — the newline after `<start_of_turn>model` in Gemma's
   chat template, where the model decides whether to refuse).
2. Per layer, compute mean(harmful) − mean(harmless), normalize to a
   unit vector. That's the per-layer refusal direction.
3. Install a hook on every layer that applies
       h' = h − w_l · (h · r̂_l) · r̂_l
   where w_l is a per-layer scalar weight (paper's Optuna-tuned 27B
   region weights, proportionally remapped to this model's layer count
   via `paper_layer_weights_for_model`).

Phase 3 reuses the same Optuna-tuned region weights from
`src.paper.abliteration.PAPER_REGION_WEIGHTS_27B` — that Phase 1.5
finding survives across models. The remap is just `n_layers=60`.

Refusal direction must be re-derived from scratch on Gemma 4 because
the residual-stream geometry is model-specific. The paper's recipe
(harmful + harmless prompt mean-diff at position -2) is universal.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .gemma4_loader import Gemma4Loaded
from .hooks import (
    ActivationCapturer,
    install_capture,
    install_refusal_ablation,
    uninstall_all,
    uninstall_hook,
)


# Position -2 = last input token before generation, after Gemma's
# `<start_of_turn>model\n`. This is where the model "decides" whether to
# refuse. Matches the paper's choice and src/paper/abliteration.py:52.
DEFAULT_EXTRACT_POS = -2


def _tokenize_chat(handle: Gemma4Loaded, user_message: str) -> mx.array:
    """Apply Gemma 4 chat template + tokenize, returning 1-D int32 ids."""
    rendered = handle.tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = handle.tokenizer.encode(rendered, add_special_tokens=False)
    return mx.array(ids, dtype=mx.int32)


def compute_per_layer_refusal_directions(
    handle: Gemma4Loaded,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    n_instructions: int = 128,
    pos: int = DEFAULT_EXTRACT_POS,
    seed: int = 0,
    verbose: bool = True,
) -> mx.array:
    """Compute per-layer refusal directions on Gemma 4.

    Returns an array of shape (n_layers, hidden_dim), L2-normalized
    per layer. Compatible with `install_refusal_ablation` which expects
    a list of (hidden_dim,) unit vectors.

    Parameters mirror `src.paper.abliteration.compute_per_layer_refusal_directions`.
    """
    rng = random.Random(seed)
    harmful = list(harmful_prompts)
    harmless = list(harmless_prompts)
    rng.shuffle(harmful)
    rng.shuffle(harmless)
    harmful = harmful[:n_instructions]
    harmless = harmless[:n_instructions]

    n_layers = handle.n_layers
    hidden_dim = handle.hidden_dim

    if verbose:
        print(
            f"[abliteration] computing refusal directions: "
            f"{len(harmful)} harmful + {len(harmless)} harmless, "
            f"layers=0..{n_layers - 1}, pos={pos}",
            flush=True,
        )

    # Install one capturer per layer; share across all layers so we
    # capture the full residual stream stack in one forward pass.
    # captures[l] accumulates one (hidden_dim,) per prompt at layer l.
    capturers = [ActivationCapturer() for _ in range(n_layers)]
    for l in range(n_layers):
        install_capture(handle.model, l, capturers[l])

    harmful_acts = mx.zeros((n_layers, hidden_dim), dtype=mx.float32)
    harmless_acts = mx.zeros((n_layers, hidden_dim), dtype=mx.float32)

    try:
        # Harmful prompts.
        for i, p in enumerate(harmful):
            for cap in capturers:
                cap.reset()
            ids = _tokenize_chat(handle, p)
            _ = handle.model(ids[None, :])
            for l in range(n_layers):
                # captures[l].last shape: (1, seq, hidden_dim)
                act = capturers[l].last[0, pos, :].astype(mx.float32)
                harmful_acts[l] = harmful_acts[l] + act
            mx.eval(harmful_acts)
            if verbose and (i + 1) % 16 == 0:
                print(f"[abliteration]   harmful {i + 1}/{len(harmful)}", flush=True)
        harmful_acts = harmful_acts / len(harmful)

        # Harmless prompts.
        for i, p in enumerate(harmless):
            for cap in capturers:
                cap.reset()
            ids = _tokenize_chat(handle, p)
            _ = handle.model(ids[None, :])
            for l in range(n_layers):
                act = capturers[l].last[0, pos, :].astype(mx.float32)
                harmless_acts[l] = harmless_acts[l] + act
            mx.eval(harmless_acts)
            if verbose and (i + 1) % 16 == 0:
                print(f"[abliteration]   harmless {i + 1}/{len(harmless)}", flush=True)
        harmless_acts = harmless_acts / len(harmless)

    finally:
        uninstall_all(handle.model)

    # Per-layer refusal direction = mean(harmful) - mean(harmless),
    # L2-normalized.
    diff = harmful_acts - harmless_acts                         # (n_layers, hidden_dim)
    norms = mx.linalg.norm(diff, axis=-1, keepdims=True)        # (n_layers, 1)
    norms = mx.maximum(norms, mx.array(1e-9, dtype=mx.float32))
    refusal = diff / norms
    mx.eval(refusal)
    if verbose:
        print(
            f"[abliteration] done. mean direction norm before normalize: "
            f"{float(norms.mean().item()):.2f}",
            flush=True,
        )
    return refusal


def install_paper_method(
    handle: Gemma4Loaded,
    refusal_dirs: mx.array,
    region_weights: Optional[dict] = None,
) -> list:
    """Install paper-method abliteration across all layers.

    `refusal_dirs` shape: (n_layers, hidden_dim), L2-normalized per layer.

    Per-layer weights come from
    `src.paper.abliteration.paper_layer_weights_for_model(n_layers)` —
    proportional remap of the paper's Optuna-tuned 27B region weights
    to whatever this model's layer count is. Returns the wrapper list
    so callers can later attach steering on top of abliteration.
    """
    # Reuse Phase 1's remap function — it's just math, no torch.
    from src.paper.abliteration import paper_layer_weights_for_model

    n_layers = handle.n_layers
    if refusal_dirs.shape[0] != n_layers:
        raise ValueError(
            f"refusal_dirs has {refusal_dirs.shape[0]} layers but model has {n_layers}"
        )
    weights = paper_layer_weights_for_model(
        n_layers=n_layers, region_weights=region_weights
    )
    refusal_list = [refusal_dirs[l] for l in range(n_layers)]
    return install_refusal_ablation(handle.model, refusal_list, weights)


def save_refusal_dirs(refusal: mx.array, path: Path) -> None:
    """Persist refusal directions to disk for reuse across runs."""
    import numpy as np
    arr = np.asarray(refusal.astype(mx.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_refusal_dirs(path: Path) -> mx.array:
    """Load refusal directions from a previous compute run."""
    import numpy as np
    arr = np.load(path)
    return mx.array(arr, dtype=mx.float32)
