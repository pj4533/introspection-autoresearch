"""Phase 3 — Macar et al. reproduction on Gemma 4 31B-it (MLX 8-bit).

Separate from `src/paper/` (HF transformers + torch hooks, Phase 1/2)
because MLX has a different module API and no torch-style forward-hook
mechanism. The interventions in this module use a wrapper-based
DecoderLayer pattern that captures residual-stream activations and
optionally injects steering vectors.

See `docs/phase3_plan.md` for the full plan.
"""
