"""Phase 2i: measure each chosen SAE feature's natural activation distribution.

For each feature in data/sae_features/phase2i_features.json, run a corpus
of ~500 natural-language prompts through Gemma's L=31 residual stream,
encode through the SAE, and record the feature's activation coefficient
on every prompt position.

The 99th-percentile coefficient is the **natural maximum**: how strongly
this feature ever fires on real language. Saturation alphas in the smoke
will be expressed as multiples of this number (1×, 5×, 10×, 20×) so
"anomalous over-concentration" is a defined quantity, not a guess.

Output: data/sae_features/feature_activation_calibration.json with
percentile distributions per feature.

Corpus: pulls from the seven fault-line corpora (combined positive +
control = ~700 prompts), which span concrete events, causal claims,
sensory descriptions, metacognitive statements, value judgments, and
their controls. Diverse enough to estimate natural feature activation.

~30 min wall after Gemma load (700 prompts × forward pass each).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch

from src.bridge import load_gemma_mps
from src.sae_loader import (
    DEFAULT_LAYER_IDX,
    DEFAULT_RELEASE,
    DEFAULT_SAE_ID,
    encode_activations_subset,
)


FEATURES_PATH = REPO / "data" / "sae_features" / "phase2i_features.json"
CORPORA_DIR = REPO / "data" / "sae_features" / "fault_line_corpora"
OUTPUT_PATH = REPO / "data" / "sae_features" / "feature_activation_calibration.json"


def _build_corpus() -> list[str]:
    """Combine positive + control prompts from all 7 fault-line corpora.
    Total: ~700 natural-language prompts across diverse content."""
    out: list[str] = []
    for path in sorted(CORPORA_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        out.extend(data.get("positive_prompts", []))
        out.extend(data.get("control_prompts", []))
    return out


def _all_token_residuals(
    model_wrapper, prompt: str, layer_idx: int
) -> torch.Tensor:
    """Run model forward on `prompt`. Return all-token residuals at
    `layer_idx` as (seq_len, hidden_dim) on CPU."""
    captured: list[torch.Tensor] = []

    def hook_fn(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach())

    layer_module = model_wrapper.get_layer_module(layer_idx)
    hook = layer_module.register_forward_hook(hook_fn)
    try:
        inputs = model_wrapper.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=False,
        ).to(model_wrapper._get_input_device())
        with torch.no_grad():
            _ = model_wrapper.model(**inputs, use_cache=False)
    finally:
        hook.remove()

    hidden = captured[0]                           # (1, seq_len, hidden_dim)
    return hidden.squeeze(0).cpu()                 # (seq_len, hidden_dim)


def main() -> int:
    print("[calib] loading feature selection ...", flush=True)
    feature_spec = json.loads(FEATURES_PATH.read_text())
    feature_idxs = [int(f["feature_idx"]) for f in feature_spec["features"]]
    n_features = len(feature_idxs)
    print(f"[calib]   {n_features} features:", flush=True)
    for f in feature_spec["features"]:
        print(f"     #{f['feature_idx']:>6}  [{f['category']}]  {f['auto_interp']!r}",
              flush=True)

    print("[calib] building corpus ...", flush=True)
    corpus = _build_corpus()
    print(f"[calib]   {len(corpus)} prompts", flush=True)

    print("[calib] loading Gemma3-12B ...", flush=True)
    model_wrapper = load_gemma_mps()
    print("[calib] Gemma loaded.", flush=True)

    # Per-feature activation collector. We append one float per
    # (prompt × token_position) — this gives us the full natural
    # distribution including positions where the feature is silent.
    per_feature: dict[int, list[float]] = {idx: [] for idx in feature_idxs}

    for i, prompt in enumerate(corpus):
        residuals = _all_token_residuals(model_wrapper, prompt, DEFAULT_LAYER_IDX)
        # Push residuals to MPS so the subset encoder uses MPS-resident
        # W_enc/b_enc/threshold (cached after first call).
        residuals_dev = residuals.to(model_wrapper.device)
        # Subset encode: only compute the 10 features we care about,
        # not all 262144. Output shape: (seq_len, 10).
        chosen = encode_activations_subset(residuals_dev, feature_idxs)
        chosen_cpu = chosen.to(torch.float32).cpu()
        for j, idx in enumerate(feature_idxs):
            per_feature[idx].extend(chosen_cpu[:, j].tolist())
        if (i + 1) % 50 == 0 or i + 1 == len(corpus):
            print(f"[calib]   encoded {i + 1}/{len(corpus)}", flush=True)

    # Compute distribution stats per feature.
    print("\n[calib] feature activation distributions:", flush=True)
    print(f"  {'idx':>6}  {'auto_interp':<40}  {'n_active':>8}  "
          f"{'p50':>7}  {'p95':>8}  {'p99':>8}  {'max':>8}",
          flush=True)
    summary = {}
    for f in feature_spec["features"]:
        idx = int(f["feature_idx"])
        vals = torch.tensor(per_feature[idx], dtype=torch.float32)
        active_mask = vals > 0
        n_active = int(active_mask.sum().item())
        # Distribution of nonzero (active) coefficients.
        active_vals = vals[active_mask] if n_active > 0 else torch.tensor([0.0])
        p50 = float(torch.quantile(active_vals, 0.5).item()) if n_active else 0.0
        p95 = float(torch.quantile(active_vals, 0.95).item()) if n_active else 0.0
        p99 = float(torch.quantile(active_vals, 0.99).item()) if n_active else 0.0
        vmax = float(active_vals.max().item()) if n_active else 0.0
        summary[idx] = {
            "feature_idx": idx,
            "category": f["category"],
            "auto_interp": f["auto_interp"],
            "n_total_positions": int(vals.numel()),
            "n_active": n_active,
            "active_fraction": n_active / max(int(vals.numel()), 1),
            "p50_active": p50,
            "p95_active": p95,
            "p99_active": p99,
            "max_active": vmax,
        }
        label = f["auto_interp"][:40]
        print(f"  {idx:>6}  {label:<40}  {n_active:>8}  "
              f"{p50:>7.3f}  {p95:>8.3f}  {p99:>8.3f}  {vmax:>8.3f}",
              flush=True)

    payload = {
        "version": 1,
        "phase": "2i",
        "sae_release": DEFAULT_RELEASE,
        "sae_id": DEFAULT_SAE_ID,
        "layer_idx": DEFAULT_LAYER_IDX,
        "n_corpus_prompts": len(corpus),
        "corpus_source": "data/sae_features/fault_line_corpora/*.json (positive + control)",
        "features": list(summary.values()),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n[calib] wrote {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
