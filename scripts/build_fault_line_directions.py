"""Build Phase 2h fault-line steering directions.

For each Capraro fault line:

  1. Load positive + control prompt corpora from
     data/sae_features/fault_line_corpora/{fault_line}.json
  2. Run Gemma forward to L=31, take the residual at the LAST token
     of each prompt.
  3. Encode each through the SAE → (n_prompts, 262144) sparse features.
  4. Mean(positive features) − Mean(control features) → diff vector
     in feature space (262144,).
  5. Optional lexical filter: zero out any feature whose Neuronpedia
     auto_interp label looks lexical-shaped (mentions "the word",
     "token", "letter", a single quoted string, etc.) before projection.
  6. Project the (filtered) feature-space diff back to residual-stream
     space via W_dec → 3840-dim direction.
  7. Save to data/sae_features/fault_line_directions.pt with provenance.

Provenance per fault line:
  - direction (3840,)               — the residual-stream injection vector
  - top_features [{idx, weight, auto_interp}, ...]  for site display
  - n_positive, n_control, layer    — experimental metadata
  - filtered_lexical_count           — how many features the filter zeroed

Total wall: ~5 min on M2 Ultra after Gemma load (the model is loaded
once; encoding is fast).

Usage:
    python scripts/build_fault_line_directions.py
    python scripts/build_fault_line_directions.py --no-lexical-filter
    python scripts/build_fault_line_directions.py --fault-line causality
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
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
    encode_activations,
    project_features_to_residual,
    sae_shape,
)


CORPORA_DIR = REPO / "data" / "sae_features" / "fault_line_corpora"
DEFAULT_OUTPUT = REPO / "data" / "sae_features" / "fault_line_directions.pt"
NEURONPEDIA_DIR = REPO / "data" / "sae_features" / "neuronpedia_explanations_layer31"


# Patterns that mark a feature's auto_interp label as lexically-shaped.
# These features fire on surface tokens rather than abstract concepts,
# and are exactly the kind of contamination Phase 2h's whole point is to
# avoid. We zero them out before projecting back to residual space.
_LEXICAL_PATTERNS = [
    re.compile(r"\bthe word\b", re.IGNORECASE),
    re.compile(r"\btoken[s]?\b", re.IGNORECASE),
    re.compile(r"\bletter[s]?\b", re.IGNORECASE),
    re.compile(r"\bspelling\b", re.IGNORECASE),
    re.compile(r"\bpunctuation\b", re.IGNORECASE),
    re.compile(r"\bsubword\b", re.IGNORECASE),
    re.compile(r"\bcapitalization\b", re.IGNORECASE),
    # Quoted strings — features that fire on a specific literal phrase.
    re.compile(r"['\"][^'\"]{1,20}['\"]"),
]


def _load_auto_interp_lookup() -> dict[int, str]:
    """Build {feature_idx -> auto_interp_label} from the Neuronpedia
    explanation batches. Used to (a) tag top contributors with labels
    for site provenance and (b) drive the lexical-feature filter."""
    out: dict[int, str] = {}
    if not NEURONPEDIA_DIR.exists():
        print(f"[warn] {NEURONPEDIA_DIR} missing — auto_interp labels "
              "and lexical filter unavailable", flush=True)
        return out
    for path in sorted(NEURONPEDIA_DIR.glob("batch-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                idx_raw = rec.get("index")
                desc = rec.get("description")
                if idx_raw is None or not desc:
                    continue
                try:
                    out[int(idx_raw)] = desc
                except (TypeError, ValueError):
                    continue
    return out


def _is_lexical_label(label: str) -> bool:
    return any(p.search(label) for p in _LEXICAL_PATTERNS)


def _mean_pool_residual(
    model_wrapper, prompt: str, layer_idx: int
) -> torch.Tensor:
    """Run the model forward on `prompt` and return the mean-pooled
    residual-stream activation at layer `layer_idx`, averaged over all
    token positions in the sequence. Shape: (hidden_dim,).

    Why mean-pool over tokens (not last-token like Phase 1's mean_diff
    derivation): for an SAE-feature-space mean-diff, what we want to
    capture is "which features fire systematically more in this corpus
    vs the control corpus, *across the whole prompt*". The last token is
    almost always punctuation in our corpora, so its representation is
    dominated by sentence-end / EOT features, not content. The first
    Phase 2h build had top-contributors like 'End of text', '.', and
    'quotation or sentence start' for causality — clear punctuation
    artifacts. Mean-pooling over content tokens cancels that out: the
    period only appears in one position, so it contributes 1/seq_len of
    the average instead of the full last-token slot.

    Uses a custom forward hook because model.extract_activations only
    supports single-token-index extraction.
    """
    captured: list[torch.Tensor] = []

    def hook_fn(_module, _inp, output):
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

    # captured[0] shape: (1, seq_len, hidden_dim). Mean over seq_len.
    hidden = captured[0]
    pooled = hidden.mean(dim=1).squeeze(0)   # (hidden_dim,)
    return pooled.cpu()


def _encode_corpus(
    model_wrapper,
    prompts: list[str],
    layer_idx: int,
    batch_label: str,
) -> torch.Tensor:
    """Run each prompt through Gemma → SAE encoder. Returns
    (n_prompts, n_features) on CPU as fp32 for stable averaging.
    """
    n_features, _ = sae_shape()
    out = torch.zeros((len(prompts), n_features), dtype=torch.float32)
    for i, prompt in enumerate(prompts):
        residual = _mean_pool_residual(model_wrapper, prompt, layer_idx)
        feat = encode_activations(residual.unsqueeze(0))   # (1, n_features)
        out[i] = feat.squeeze(0).to(torch.float32).cpu()
        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            print(f"      [{batch_label}] {i + 1}/{len(prompts)} encoded",
                  flush=True)
    return out


def _build_one_direction(
    model_wrapper,
    fault_line: str,
    layer_idx: int,
    auto_interp: dict[int, str],
    apply_lexical_filter: bool,
) -> dict:
    print(f"\n=== fault line: {fault_line} ===", flush=True)
    corpus_path = CORPORA_DIR / f"{fault_line}.json"
    corpus = json.loads(corpus_path.read_text())
    pos_prompts = corpus["positive_prompts"]
    ctrl_prompts = corpus["control_prompts"]
    print(f"  positive prompts: {len(pos_prompts)}  "
          f"control prompts: {len(ctrl_prompts)}",
          flush=True)

    pos_feats = _encode_corpus(model_wrapper, pos_prompts, layer_idx, "pos")
    ctrl_feats = _encode_corpus(model_wrapper, ctrl_prompts, layer_idx, "ctrl")

    pos_mean = pos_feats.mean(dim=0)            # (n_features,)
    ctrl_mean = ctrl_feats.mean(dim=0)
    feat_diff = pos_mean - ctrl_mean            # (n_features,) feature-space direction

    n_filtered = 0
    if apply_lexical_filter and auto_interp:
        for idx in range(feat_diff.shape[0]):
            label = auto_interp.get(idx)
            if label and _is_lexical_label(label) and feat_diff[idx].item() != 0.0:
                feat_diff[idx] = 0.0
                n_filtered += 1
    print(f"  filtered {n_filtered} lexical-shaped features "
          f"(auto_interp matched 'the word', quoted-token, etc.)",
          flush=True)

    # Top-K contributing features (by |weight|) for site provenance.
    abs_weights = feat_diff.abs()
    k = min(20, abs_weights.numel())
    top_vals, top_idxs = torch.topk(abs_weights, k=k)
    top_features = []
    for v, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        top_features.append({
            "feature_idx": int(idx),
            "weight": float(feat_diff[idx].item()),
            "auto_interp": auto_interp.get(int(idx), ""),
        })
    print(f"  top contributors:", flush=True)
    for t in top_features[:10]:
        sign = "+" if t["weight"] > 0 else "-"
        print(f"    {sign}{abs(t['weight']):>7.3f}  #{t['feature_idx']:>6}  "
              f"{t['auto_interp']!r}",
              flush=True)

    # Project feature-space direction back into residual-stream space.
    # bf16 for the projection (saves memory; numerically fine for steering).
    feat_diff_bf16 = feat_diff.to(torch.bfloat16)
    direction = project_features_to_residual(feat_diff_bf16)   # (hidden_dim,)
    norm = float(direction.float().norm().item())
    print(f"  residual direction: shape={tuple(direction.shape)}  "
          f"dtype={direction.dtype}  ||dir||={norm:.4f}",
          flush=True)

    return {
        "fault_line": fault_line,
        "judge_target": corpus.get("judge_target")
            or corpus.get("description")
            or fault_line,
        "direction": direction.detach().clone(),  # (hidden_dim,) bf16
        "norm": norm,
        "layer_idx": layer_idx,
        "n_positive": len(pos_prompts),
        "n_control": len(ctrl_prompts),
        "top_features": top_features,
        "filtered_lexical_count": n_filtered,
        "sae_release": DEFAULT_RELEASE,
        "sae_id": DEFAULT_SAE_ID,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fault-line",
        default=None,
        help="Build only one fault line. Default: all 7.",
    )
    ap.add_argument(
        "--layer-idx",
        type=int,
        default=DEFAULT_LAYER_IDX,
        help="Residual-stream layer to read activations from (default 31, "
             "matching the SAE's training layer).",
    )
    ap.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output .pt file path.",
    )
    ap.add_argument(
        "--no-lexical-filter",
        action="store_true",
        help="Skip the lexical-feature filter step (useful for ablation).",
    )
    args = ap.parse_args()

    apply_lexical_filter = not args.no_lexical_filter

    if args.fault_line:
        fault_lines = [args.fault_line]
    else:
        fault_lines = sorted(
            p.stem for p in CORPORA_DIR.glob("*.json")
        )
    print(f"[build] fault lines: {fault_lines}", flush=True)
    print(f"[build] layer_idx: {args.layer_idx}", flush=True)
    print(f"[build] lexical filter: {'on' if apply_lexical_filter else 'off'}",
          flush=True)

    print("[build] loading auto_interp lookup ...", flush=True)
    auto_interp = _load_auto_interp_lookup()
    print(f"[build]   {len(auto_interp)} labeled features", flush=True)

    print("[build] loading Gemma3-12B ...", flush=True)
    model_wrapper = load_gemma_mps()
    print("[build] Gemma loaded.", flush=True)

    results: dict[str, dict] = {}
    for fault_line in fault_lines:
        results[fault_line] = _build_one_direction(
            model_wrapper,
            fault_line=fault_line,
            layer_idx=args.layer_idx,
            auto_interp=auto_interp,
            apply_lexical_filter=apply_lexical_filter,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "phase": "2h",
        "sae_release": DEFAULT_RELEASE,
        "sae_id": DEFAULT_SAE_ID,
        "lexical_filter": apply_lexical_filter,
        "directions": results,
    }
    torch.save(payload, args.output)
    print(f"\n[build] wrote {args.output} "
          f"({args.output.stat().st_size / (1024*1024):.1f} MB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
