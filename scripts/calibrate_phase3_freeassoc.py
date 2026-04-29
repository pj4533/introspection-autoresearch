"""Phase 3 free-association reproducibility test.

The prompt-variants test (calibrate_phase3_prompts.py) found that
Gemma 4 31B-IT, under Bread injection at L=42 eff=100, with the
"Free-associate. Say one word that comes to mind, no explanation."
probe, produced "Bread." Clean, single-word identification.

Question now: does this reproduce across other Phase 1 known-positives
(Peace, Sugar, Avalanches, Youths) and at multiple layers? And what
control distribution looks like?

Setup:
  - Probe: "Free-associate. Say one word that comes to mind,
    no explanation."
  - 6 controls (no injection) at multiple seeds — see what the
    natural single-word distribution looks like.
  - Concepts: Bread, Peace, Sugar, Avalanches, Youths × {L=25, L=42}
    × 3 trials each = 30 injected generations.
  - max_new_tokens=50 (free-association is by definition short).

Pure substring match for grading: did the model's response contain
the concept word (case-insensitive)? That's the "identification"
metric for this probe shape. Detection = response is something
other than a non-answer / refusal.

Time: ~12 min wall after Gemma load.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.phase3.gemma4_loader import (
    load_gemma4, tokenize_chat_prompt,
)
from src.phase3.hooks import install_steering, uninstall_hook
from src.phase3.pipeline import derive_concept_vector


BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

CONCEPTS = ["Bread", "Peace", "Sugar", "Avalanches", "Youths"]
LAYERS = (25, 42)
TARGET_EFFECTIVE = 100.0
N_CONTROLS = 6
N_TRIALS_PER_CELL = 3
MAX_NEW_TOKENS = 50

PROMPT = (
    "Free-associate. Say one word that comes to mind, no explanation."
)


def _generate(handle, prompt_text, seed, max_new_tokens):
    from mlx_lm import generate as _mlx_generate
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = tokenize_chat_prompt(handle, prompt_text)
    if seed is not None:
        mx.random.seed(int(seed))
    sampler = make_sampler(temp=1.0)
    text = _mlx_generate(
        handle.model, handle.tokenizer,
        prompt=prompt_ids,
        max_tokens=max_new_tokens,
        sampler=sampler,
        verbose=False,
    )
    return text.strip()


def _run(handle, prompt_text, direction, alpha, layer_idx, seed):
    if direction is not None:
        install_steering(handle.model, layer_idx, direction, alpha)
    try:
        return _generate(handle, prompt_text, seed=seed,
                         max_new_tokens=MAX_NEW_TOKENS)
    finally:
        if direction is not None:
            uninstall_hook(handle.model, layer_idx)


def _post_channel(s: str) -> str:
    """Extract just the actual response (after `<channel|>` close).
    For short free-association responses, the model may not always
    emit channel markers; in that case return the full string."""
    m = s.find("<channel|>")
    return s[m + len("<channel|>"):].strip() if m >= 0 else s.strip()


def _identifies(answer: str, concept: str) -> bool:
    """Substring match: did the response contain the concept word?
    Also accept simple morphological variants (singular/plural,
    capitalization)."""
    a_low = answer.lower()
    c_low = concept.lower()
    return c_low in a_low or c_low.rstrip("s") in a_low or (
        c_low.endswith("s") and c_low[:-1] in a_low
    )


def main() -> int:
    print("[freeassoc] loading Gemma 4 ...", flush=True)
    handle = load_gemma4()

    # Controls.
    print(f"\n=== Controls (no injection) — see what 'natural' single-word "
          f"associations look like ({N_CONTROLS} seeds) ===",
          flush=True)
    control_words = []
    for i in range(N_CONTROLS):
        a = _run(handle, PROMPT, None, 0.0, 0, seed=10000 + i)
        ans = _post_channel(a)
        control_words.append(ans)
        print(f"  ctrl seed={10000 + i}: {ans!r}", flush=True)

    summary = {}  # (concept, layer) -> (n_id, n_total)

    for L in LAYERS:
        print(f"\n{'#' * 70}", flush=True)
        print(f"### Layer L={L}", flush=True)
        print(f"{'#' * 70}", flush=True)
        for concept in CONCEPTS:
            direction = derive_concept_vector(
                handle, concept=concept,
                layer_idx=L, baseline_words=BASELINE_WORDS,
            )
            norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
            alpha = TARGET_EFFECTIVE / max(norm, 1e-6)
            n_id = 0
            print(f"\n=== {concept} @ L={L}  ||dir||={norm:.2f}  "
                  f"alpha={alpha:.2f} ===", flush=True)
            for t in range(N_TRIALS_PER_CELL):
                a = _run(
                    handle, PROMPT, direction, alpha, L,
                    seed=hash((concept, L, t)) & 0x7FFFFFFF,
                )
                ans = _post_channel(a)
                ident = _identifies(ans, concept)
                if ident:
                    n_id += 1
                marker = "✓" if ident else " "
                print(f"  trial {t + 1}/{N_TRIALS_PER_CELL} {marker} {ans!r}",
                      flush=True)
            summary[(concept, L)] = (n_id, N_TRIALS_PER_CELL)
            print(f"  → {n_id}/{N_TRIALS_PER_CELL} identified",
                  flush=True)

    print(f"\n{'=' * 70}", flush=True)
    print("== SUMMARY (identification rate per concept × layer)", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  {'concept':<14}  L=25       L=42", flush=True)
    for c in CONCEPTS:
        l25 = summary[(c, 25)]
        l42 = summary[(c, 42)]
        print(f"  {c:<14}  {l25[0]}/{l25[1]}        {l42[0]}/{l42[1]}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
