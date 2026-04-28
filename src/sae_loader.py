"""SAE loading and decoder-direction extraction for Phase 2g.

Wraps the Gemma Scope 2 SAE checkpoint at
`google/gemma-scope-2-12b-it`, variant
`resid_post/layer_31_width_262k_l0_medium`. Returns the decoder
direction `W_dec[feature_idx]` as the steering vector for SAE-feature
injection candidates.

Caches loaded SAEs per (release, sae_id) so repeated lookups don't
re-load weights. The full 262k×3840 decoder matrix is ~2 GB in bf16.

The Gemma Scope 2 archive ships safetensors with three keys we care
about, all in fp32:
    W_enc:  (hidden_dim, n_features) = (3840, 262144)  -- not used here
    W_dec:  (n_features, hidden_dim) = (262144, 3840)  -- the steering rows
    threshold:  (n_features,)                          -- jump-relu activation gate
                                                          (not used for steering)

We hold W_dec only (and as bf16 to halve memory). When a feature is
requested, we slice one row, clone it, and move it to the model's
device + dtype.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


DEFAULT_RELEASE = "google/gemma-scope-2-12b-it"
DEFAULT_SAE_ID = "resid_post/layer_31_width_262k_l0_medium"


class _LoadedSAE:
    """In-memory SAE state. Just the W_dec rows and metadata we need."""

    def __init__(self, w_dec: torch.Tensor, n_features: int, hidden_dim: int) -> None:
        self.w_dec = w_dec               # (n_features, hidden_dim), bf16, CPU
        self.n_features = n_features
        self.hidden_dim = hidden_dim


def _resolve_sae_path(release: str, sae_id: str) -> Path:
    """Return the local path to the SAE folder (params.safetensors lives here)."""
    repo_dir = snapshot_download(
        repo_id=release,
        allow_patterns=[f"{sae_id}/config.json", f"{sae_id}/params.safetensors"],
    )
    return Path(repo_dir) / sae_id


@lru_cache(maxsize=4)
def _load_sae(release: str, sae_id: str) -> _LoadedSAE:
    sae_dir = _resolve_sae_path(release, sae_id)
    params_path = sae_dir / "params.safetensors"
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.safetensors not found at {params_path}. "
            "Run `hf download google/gemma-scope-2-12b-it "
            "--include 'resid_post/layer_31_width_262k_l0_medium/*'`."
        )
    with safe_open(str(params_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # Gemma Scope 2 uses lowercase `w_dec` / `w_enc`. Older Gemma Scope
        # used `W_dec`. Accept either.
        if "w_dec" in keys:
            tensor_key = "w_dec"
        elif "W_dec" in keys:
            tensor_key = "W_dec"
        else:
            raise KeyError(
                f"params.safetensors at {params_path} does not contain "
                f"'w_dec' or 'W_dec'. Found keys: {keys}"
            )
        w_dec = f.get_tensor(tensor_key).to(torch.bfloat16).contiguous()
    n_features, hidden_dim = w_dec.shape
    return _LoadedSAE(w_dec=w_dec, n_features=n_features, hidden_dim=hidden_dim)


def get_decoder_direction(
    release: str,
    sae_id: str,
    feature_idx: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return W_dec[feature_idx] on the requested device/dtype.

    Shape: (hidden_dim,) — for Gemma 3 12B-it this is (3840,).

    The returned tensor is a fresh clone; callers can scale/normalize it
    without mutating the cached SAE state.
    """
    sae = _load_sae(release, sae_id)
    if not (0 <= feature_idx < sae.n_features):
        raise IndexError(
            f"feature_idx {feature_idx} out of range for SAE "
            f"{release}/{sae_id} (n_features={sae.n_features})"
        )
    direction = sae.w_dec[feature_idx].detach().clone()
    return direction.to(device=device, dtype=dtype)


def get_neighbors(
    release: str,
    sae_id: str,
    feature_idx: int,
    n: int = 20,
    exclude_self: bool = True,
) -> list[tuple[int, float]]:
    """Return the n nearest features by W_dec cosine similarity.

    Returns: list of (other_feature_idx, cosine_similarity), descending.
    """
    sae = _load_sae(release, sae_id)
    w_dec_f32 = sae.w_dec.to(torch.float32)
    target = w_dec_f32[feature_idx]
    target_norm = target.norm()
    all_norms = w_dec_f32.norm(dim=1)
    sims = (w_dec_f32 @ target) / (all_norms * target_norm + 1e-9)
    if exclude_self:
        sims[feature_idx] = float("-inf")
    top = torch.topk(sims, k=n)
    return [(int(idx), float(score)) for idx, score in zip(top.indices, top.values)]


def n_features(release: str = DEFAULT_RELEASE, sae_id: str = DEFAULT_SAE_ID) -> int:
    """Return the SAE's feature count (262144 for the Phase 2g default)."""
    return _load_sae(release, sae_id).n_features


def hidden_dim(release: str = DEFAULT_RELEASE, sae_id: str = DEFAULT_SAE_ID) -> int:
    """Return the SAE's residual-stream dim (3840 for Gemma3-12B)."""
    return _load_sae(release, sae_id).hidden_dim
