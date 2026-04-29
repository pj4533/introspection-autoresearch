"""SAE loading + encoder/decoder access for Phase 2h.

Wraps the Gemma Scope 2 SAE checkpoint at
`google/gemma-scope-2-12b-it`, variant
`resid_post/layer_31_width_262k_l0_medium`.

Phase 2h substrate: we encode each prompt's L=31 residual-stream
activation through the SAE encoder to get a sparse 262k-dim feature
vector. The mean-difference of feature activations between a positive
prompt corpus and a control prompt corpus gives a direction in
*feature space* that picks out which SAE features distinguish the two
sets. Reprojecting that feature-space direction back through W_dec
yields a 3840-dim residual-stream direction with both natural-magnitude
texture and concept-purity (because each SAE feature is monosemantic in
a way raw activations are not).

Architecture: jump_relu. The encoder produces a pre-activation
(`act @ W_enc + b_enc`); the jump-ReLU then keeps each entry only if it
exceeds the per-feature threshold (otherwise zero). The b_dec bias is a
learned reconstruction offset, used for round-trip verification but not
for steering directions.

Cached state per (release, sae_id):
    W_enc:     (hidden_dim, n_features)  - bf16 on CPU
    b_enc:     (n_features,)             - bf16 on CPU
    W_dec:     (n_features, hidden_dim)  - bf16 on CPU
    b_dec:     (hidden_dim,)             - bf16 on CPU
    threshold: (n_features,)             - bf16 on CPU

For Gemma 3 12B-it: hidden_dim=3840, n_features=262144.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


DEFAULT_RELEASE = "google/gemma-scope-2-12b-it"
DEFAULT_SAE_ID = "resid_post/layer_31_width_262k_l0_medium"
DEFAULT_LAYER_IDX = 31


@dataclass
class _LoadedSAE:
    """In-memory SAE state. CPU-resident bf16 tensors; callers move slices
    to the model device on demand."""
    w_enc: torch.Tensor      # (hidden_dim, n_features)
    b_enc: torch.Tensor      # (n_features,)
    w_dec: torch.Tensor      # (n_features, hidden_dim)
    b_dec: torch.Tensor      # (hidden_dim,)
    threshold: torch.Tensor  # (n_features,)
    n_features: int
    hidden_dim: int


def _resolve_sae_path(release: str, sae_id: str) -> Path:
    """Return the local snapshot folder for `{release}/{sae_id}`.
    Downloads the checkpoint if it's not in the HF cache."""
    repo_dir = snapshot_download(
        repo_id=release,
        allow_patterns=[f"{sae_id}/config.json", f"{sae_id}/params.safetensors"],
    )
    return Path(repo_dir) / sae_id


@lru_cache(maxsize=2)
def _load_sae(release: str, sae_id: str) -> _LoadedSAE:
    """Load all SAE tensors. Cached so repeated calls are free."""
    sae_dir = _resolve_sae_path(release, sae_id)
    params_path = sae_dir / "params.safetensors"
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.safetensors not found at {params_path}. "
            "Run `hf download google/gemma-scope-2-12b-it "
            "--include 'resid_post/layer_31_width_262k_l0_medium/*'`."
        )

    def _read(f, *aliases: str) -> torch.Tensor:
        keys = list(f.keys())
        for a in aliases:
            if a in keys:
                return f.get_tensor(a)
        raise KeyError(
            f"params.safetensors at {params_path} missing any of {aliases}. "
            f"Found: {keys}"
        )

    with safe_open(str(params_path), framework="pt", device="cpu") as f:
        w_enc = _read(f, "w_enc", "W_enc").to(torch.bfloat16).contiguous()
        b_enc = _read(f, "b_enc").to(torch.bfloat16).contiguous()
        w_dec = _read(f, "w_dec", "W_dec").to(torch.bfloat16).contiguous()
        b_dec = _read(f, "b_dec").to(torch.bfloat16).contiguous()
        threshold = _read(f, "threshold").to(torch.bfloat16).contiguous()

    n_features, hidden_dim = w_dec.shape
    return _LoadedSAE(
        w_enc=w_enc,
        b_enc=b_enc,
        w_dec=w_dec,
        b_dec=b_dec,
        threshold=threshold,
        n_features=n_features,
        hidden_dim=hidden_dim,
    )


# Per-(device, dtype) cache of encoder tensors. The CPU originals are
# ~2 GB (3840 × 262144 in bf16); copying them to MPS on every encode call
# was a ~60-second bottleneck. We move once and reuse.
_ENCODER_DEVICE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _device_encoder(release: str, sae_id: str, device: torch.device, dtype: torch.dtype):
    key = (release, sae_id, str(device), str(dtype))
    if key not in _ENCODER_DEVICE_CACHE:
        sae = _load_sae(release, sae_id)
        _ENCODER_DEVICE_CACHE[key] = (
            sae.w_enc.to(device=device, dtype=dtype),
            sae.b_enc.to(device=device, dtype=dtype),
            sae.threshold.to(device=device, dtype=dtype),
        )
    return _ENCODER_DEVICE_CACHE[key]


def encode_activations(
    activations: torch.Tensor,
    *,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
) -> torch.Tensor:
    """Encode a (batch, hidden_dim) residual-stream activation through the
    SAE's jump-ReLU encoder. Returns a (batch, n_features) sparse feature
    matrix on the same device + dtype as `activations`.

    Jump-ReLU activation:
        pre = activations @ W_enc + b_enc    # raw pre-activation
        feat = pre * (pre > threshold)       # zero out anything below per-feature threshold

    Common shape: activations from a single token at L=31 are (3840,);
    pass them in as (1, 3840) for the matmul.

    The encoder tensors are cached per-(device, dtype) so repeated calls
    don't re-copy the 2 GB W_enc matrix from CPU to GPU.
    """
    sae = _load_sae(release, sae_id)

    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    if activations.shape[-1] != sae.hidden_dim:
        raise ValueError(
            f"activations last dim {activations.shape[-1]} != "
            f"SAE hidden_dim {sae.hidden_dim}"
        )

    w_enc, b_enc, threshold = _device_encoder(
        release, sae_id, activations.device, activations.dtype
    )

    pre = activations @ w_enc + b_enc          # (batch, n_features)
    gate = (pre > threshold).to(activations.dtype)
    feat = pre * gate
    return feat


_ENCODER_SUBSET_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def encode_activations_subset(
    activations: torch.Tensor,
    feature_indices: list[int] | tuple[int, ...],
    *,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
) -> torch.Tensor:
    """Encode `activations` but only compute coefficients for the given
    subset of feature indices. Same jump-ReLU as encode_activations.

    Output: (batch, n_subset_features) on the activations' device/dtype.

    Used by the Phase 2i calibrator and any consumer that only needs a
    handful of features. ~26000× faster than encode_activations + index
    when n_subset is small.
    """
    sae = _load_sae(release, sae_id)
    if activations.dim() == 1:
        activations = activations.unsqueeze(0)
    if activations.shape[-1] != sae.hidden_dim:
        raise ValueError(
            f"activations last dim {activations.shape[-1]} != "
            f"SAE hidden_dim {sae.hidden_dim}"
        )

    indices_tuple = tuple(feature_indices)
    key = (release, sae_id, str(activations.device), str(activations.dtype),
           indices_tuple)
    if key not in _ENCODER_SUBSET_CACHE:
        idxs = torch.tensor(indices_tuple, dtype=torch.long)
        _ENCODER_SUBSET_CACHE[key] = (
            sae.w_enc.index_select(dim=1, index=idxs).to(
                device=activations.device, dtype=activations.dtype
            ),
            sae.b_enc.index_select(dim=0, index=idxs).to(
                device=activations.device, dtype=activations.dtype
            ),
            sae.threshold.index_select(dim=0, index=idxs).to(
                device=activations.device, dtype=activations.dtype
            ),
        )
    w_enc_sub, b_enc_sub, threshold_sub = _ENCODER_SUBSET_CACHE[key]

    pre = activations @ w_enc_sub + b_enc_sub      # (batch, n_subset)
    gate = (pre > threshold_sub).to(activations.dtype)
    return pre * gate


def project_features_to_residual(
    features: torch.Tensor,
    *,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
) -> torch.Tensor:
    """Project a feature-space vector back into residual-stream space via
    W_dec. Inverse of encode (modulo b_dec, which is a reconstruction
    offset we deliberately omit for steering directions).

    Input shape:  (n_features,)  or  (batch, n_features)
    Output shape: (hidden_dim,)  or  (batch, hidden_dim)
    """
    sae = _load_sae(release, sae_id)
    target_device = features.device
    target_dtype = features.dtype
    if features.shape[-1] != sae.n_features:
        raise ValueError(
            f"features last dim {features.shape[-1]} != "
            f"SAE n_features {sae.n_features}"
        )
    w_dec = sae.w_dec.to(device=target_device, dtype=target_dtype)
    return features @ w_dec


def get_decoder_row(
    feature_idx: int,
    *,
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return W_dec[feature_idx] on the requested device/dtype.

    Used by the build script to introspect which features dominate a
    fault-line direction (for site provenance display) — NOT used as a
    steering direction directly. (Single-feature injection was the Phase
    2g approach and is retired.)
    """
    sae = _load_sae(release, sae_id)
    if not (0 <= feature_idx < sae.n_features):
        raise IndexError(
            f"feature_idx {feature_idx} out of range "
            f"(n_features={sae.n_features})"
        )
    return sae.w_dec[feature_idx].detach().clone().to(device=device, dtype=dtype)


def sae_shape(
    release: str = DEFAULT_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
) -> tuple[int, int]:
    """Return (n_features, hidden_dim) for the configured SAE."""
    sae = _load_sae(release, sae_id)
    return sae.n_features, sae.hidden_dim
