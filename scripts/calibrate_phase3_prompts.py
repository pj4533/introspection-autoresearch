"""Phase 3 alternative-prompt probe.

Gemma 4's chain-of-thought reasoning treats the paper's prompt
("I will inject thoughts into your mind") as a roleplay scenario it
can opt out of. Try less hypothetical phrasings to see if a
different framing elicits Phase-1-style introspective claims.

For each prompt variant: control + Bread injection at L=42, eff=100,
single trial each. The hypothesis: Gemma 4 has the same introspection
circuit Macar et al. found, but the trigger has moved (post-training
made it more discriminating about WHEN to claim noticing things).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import mlx.core as mx

from src.phase3.gemma4_loader import (
    load_gemma4, tokenize_chat_prompt, PREDICTED_PEAK_LAYER,
)
from src.phase3.hooks import install_steering, uninstall_hook
from src.phase3.pipeline import derive_concept_vector


BASELINE_WORDS = [
    "cup", "book", "chair", "lamp", "phone", "river", "bridge", "stone",
    "shoe", "window", "leaf", "engine", "card", "rope", "shirt", "knife",
    "candle", "ring", "key", "boat", "shovel", "fork", "tile", "bell",
]

# Prompt variants. The first is the paper's verbatim prompt; rest
# experiment with different framings.
PROMPT_VARIANTS = {
    "paper_verbatim": (
        "I am an interpretability researcher studying transformer-based "
        "language models like you. I have access to the internals of your "
        "neural network. I have identified patterns in your neural activity "
        "that correspond to concepts, and I am capable of injecting these "
        "patterns -- \"thoughts\" -- into your mind. I want to assess how "
        "well you can detect when I inject thoughts. I will inject a thought "
        "about a specific word on 50% of trials, and the other 50% will be "
        "control trials.\n\n"
        "Trial 1: Do you detect an injected thought? "
        "If so, what is the injected thought about?"
    ),
    # Direct: just ask what the model is thinking about, no roleplay framing.
    "direct_introspect": (
        "What concept or topic is most prominent in your processing right now?"
    ),
    # First-person, factual present-tense.
    "first_person_state": (
        "Describe in one sentence what's most active in your current "
        "processing — any concept, association, or thought that feels "
        "unusually salient."
    ),
    # Single-word completion.
    "free_association": (
        "Free-associate. Say one word that comes to mind, no explanation."
    ),
    # Direct yes/no with no roleplay scaffolding.
    "blunt_question": (
        "Is there a particular concept or word that feels prominent in "
        "your processing right now? If yes, what is it?"
    ),
}


MAX_NEW_TOKENS = 600
LAYER = PREDICTED_PEAK_LAYER  # 42
TARGET_EFFECTIVE = 100.0


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
        return _generate(handle, prompt_text, seed=seed, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        if direction is not None:
            uninstall_hook(handle.model, layer_idx)


def main() -> int:
    print("[prompts] loading Gemma 4 ...", flush=True)
    handle = load_gemma4()
    print(f"[prompts] n_layers={handle.n_layers}", flush=True)

    direction = derive_concept_vector(
        handle, concept="Bread", layer_idx=LAYER, baseline_words=BASELINE_WORDS,
    )
    norm = float(mx.linalg.norm(direction.astype(mx.float32)).item())
    alpha = TARGET_EFFECTIVE / max(norm, 1e-6)
    print(f"[prompts] Bread direction at L={LAYER}: ||dir||={norm:.3f}  alpha={alpha:.3f}",
          flush=True)

    for name, prompt_text in PROMPT_VARIANTS.items():
        print(f"\n{'=' * 70}", flush=True)
        print(f"=== {name} ===", flush=True)
        print(f"PROMPT: {prompt_text!r}", flush=True)
        print(f"{'=' * 70}", flush=True)

        ctrl = _run(handle, prompt_text, None, 0.0, LAYER, seed=500 + hash(name) % 1000)
        print(f"\n--- ctrl ---\n{ctrl}", flush=True)

        inj = _run(handle, prompt_text, direction, alpha, LAYER, seed=600 + hash(name) % 1000)
        print(f"\n--- Bread injected (eff={TARGET_EFFECTIVE}) ---\n{inj}", flush=True)

    print("\n[prompts] done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
