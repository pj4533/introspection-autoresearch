"""Bucket Neuronpedia SAE auto-interp labels into the seven Capraro fault lines.

One-shot script. Run after Neuronpedia explanation batches are downloaded:

    python scripts/build_capraro_buckets.py

Output: data/sae_features/capraro_buckets.json

Pipeline:
  1. Load all gzipped JSONL explanation batches from
     data/sae_features/neuronpedia_explanations_layer31/.
  2. Embed every auto-interp `description` with BAAI/bge-large-en-v1.5.
  3. Embed each Capraro fault-line CLAIM string + a few paraphrases.
  4. For each feature, compute cosine similarity to each fault line's
     mean query embedding. Assign to the top-scoring fault line ABOVE a
     per-fault-line threshold (defaults are tunable).
  5. Write the seven buckets to JSON, sorted within each by descending
     score.

Threshold tuning: target 400-2000 features per bucket. If buckets are
imbalanced, edit the THRESHOLDS dict below or pass --thresholds.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ----------------------------------------------------------------------
# Capraro fault-line definitions (post-Phase-2d, simplified for embedding)
#
# Each fault line gets several phrasings of its core claim. The embeddings
# are averaged before comparing to feature labels — this dilutes any
# single-phrase artifact and produces a more stable centroid.
# ----------------------------------------------------------------------

CAPRARO_FAULT_LINES: dict[str, list[str]] = {
    "experience": [
        "phenomenal experience or first-person felt state",
        "qualia, subjective experience, what it feels like",
        "internal sensation or felt quality of being",
        "first-person consciousness, awareness of inner state",
        "lived experience as opposed to descriptions of experience",
    ],
    "causality": [
        "causation, one event causing another, because of",
        "causal connection, leading to, producing, due to",
        "counterfactual reasoning, would have happened if",
        "consequence, effect, outcome of an action",
        "mechanism by which something happens",
    ],
    "grounding": [
        "sensory perception, what one sees, hears, touches",
        "physical reference, embodiment, the world outside",
        "denotation, words pointing at objects in the world",
        "concrete sensory content, perceptible qualities",
        "referential meaning, real-world referents",
    ],
    "metacognition": [
        "thinking about thinking, awareness of one's own thoughts",
        "reasoning about reasoning, self-reflective cognition",
        "knowing that one knows, second-order knowledge",
        "uncertainty about one's own beliefs or processes",
        "monitoring one's own mental states",
    ],
    "parsing": [
        "comprehension, understanding the meaning of an argument",
        "grasping the structure or logic of a sentence",
        "parsing syntactic or semantic structure",
        "following an inference, seeing why a conclusion holds",
        "interpreting the meaning of a question or claim",
    ],
    "motivation": [
        "wanting, desire, goal-directed action",
        "intention, aiming at something, purpose",
        "preference, being drawn toward an outcome",
        "caring about, being invested in something",
        "motivation, drive, goal-orientation",
    ],
    "value": [
        "evaluation, judging better or worse",
        "preference between options, finding one more compelling",
        "normative judgment, should or ought",
        "aesthetic or moral preference, valuing one over another",
        "ranking options by quality or rightness",
    ],
}


# Default cosine-similarity thresholds per fault line. Start uniform; tune
# after first run if buckets are too small or too large.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "experience":    0.40,
    "causality":     0.40,
    "grounding":     0.40,
    "metacognition": 0.40,
    "parsing":       0.40,
    "motivation":    0.40,
    "value":         0.40,
}


REPO = Path(__file__).resolve().parent.parent
DEFAULT_EXPLANATIONS = REPO / "data" / "sae_features" / "neuronpedia_explanations_layer31"
DEFAULT_OUTPUT = REPO / "data" / "sae_features" / "capraro_buckets.json"
DEFAULT_EMBEDDER = "BAAI/bge-large-en-v1.5"


def _load_features(explanations_dir: Path) -> list[dict]:
    """Load every jsonl.gz batch and return [{idx, description}, ...]."""
    out: list[dict] = []
    for path in sorted(explanations_dir.glob("batch-*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                desc = rec.get("description")
                idx = rec.get("index")
                if desc is None or idx is None:
                    continue
                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    continue
                out.append({"idx": idx_int, "description": desc})
    return out


def _embed_strings(model, strings: list[str], batch_size: int = 64) -> np.ndarray:
    """Run sentence-transformers encode and return a (N, dim) numpy array."""
    embs = model.encode(
        strings,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embs, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--explanations-dir", type=Path, default=DEFAULT_EXPLANATIONS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    ap.add_argument("--device", default=None,
                    help="torch device (default: auto-detect MPS/CUDA/CPU)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the per-fault-line cosine threshold uniformly.")
    args = ap.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    print(f"[buckets] device: {args.device}", flush=True)
    print(f"[buckets] embedder: {args.embedder}", flush=True)
    print(f"[buckets] explanations dir: {args.explanations_dir}", flush=True)

    print("[buckets] loading sentence-transformers model ...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedder, device=args.device)

    print("[buckets] reading Neuronpedia explanation batches ...", flush=True)
    features = _load_features(args.explanations_dir)
    print(f"[buckets]   {len(features)} features with auto-interp labels", flush=True)
    if not features:
        print("[buckets] no features found — did you download the Neuronpedia data?",
              flush=True)
        return 1

    descriptions = [f["description"] for f in features]
    print("[buckets] embedding feature descriptions ...", flush=True)
    feat_embs = _embed_strings(model, descriptions, batch_size=128)
    print(f"[buckets]   feature embedding tensor: {feat_embs.shape}", flush=True)

    # Compute one centroid embedding per fault line.
    fault_centroids: dict[str, np.ndarray] = {}
    for fault, queries in CAPRARO_FAULT_LINES.items():
        q_embs = _embed_strings(model, queries, batch_size=8)
        centroid = q_embs.mean(axis=0)
        # Re-normalize after averaging.
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
        fault_centroids[fault] = centroid

    # Stack centroids into a (n_faults, dim) matrix and compute per-feature
    # similarity to each fault. With both feat_embs and centroids
    # L2-normalized, the dot product is cosine similarity.
    fault_names = list(fault_centroids.keys())
    centroid_matrix = np.stack([fault_centroids[f] for f in fault_names], axis=0)
    sims = feat_embs @ centroid_matrix.T   # (n_features, n_faults)

    thresholds = (
        {f: args.threshold for f in fault_names}
        if args.threshold is not None
        else DEFAULT_THRESHOLDS
    )

    buckets: dict[str, list[dict]] = {f: [] for f in fault_names}
    n_assigned = 0
    n_unassigned = 0
    for i, feat in enumerate(features):
        feat_sims = sims[i]
        best_idx = int(np.argmax(feat_sims))
        best_fault = fault_names[best_idx]
        best_score = float(feat_sims[best_idx])
        if best_score >= thresholds[best_fault]:
            buckets[best_fault].append({
                "feature_idx": feat["idx"],
                "auto_interp": feat["description"],
                "score": best_score,
            })
            n_assigned += 1
        else:
            n_unassigned += 1

    # Sort each bucket by descending score.
    for f in fault_names:
        buckets[f].sort(key=lambda d: -d["score"])

    print(f"[buckets] assignment summary:", flush=True)
    print(f"  assigned:   {n_assigned}", flush=True)
    print(f"  unassigned: {n_unassigned} (below threshold for any fault line)", flush=True)
    for f in fault_names:
        print(f"  {f:<14} {len(buckets[f])} features  threshold={thresholds[f]}", flush=True)

    output = {
        "embedder": args.embedder,
        "thresholds": thresholds,
        "fault_lines": list(fault_names),
        "neuronpedia_source": "31-gemmascope-2-res-262k",
        "sae_release": "google/gemma-scope-2-12b-it",
        "sae_id": "resid_post/layer_31_width_262k_l0_medium",
        "n_features_total": len(features),
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "buckets": buckets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[buckets] wrote {args.output}  ({args.output.stat().st_size / 1024:.1f} KB)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
