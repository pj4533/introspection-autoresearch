"""Wave 1 of Phase 2d-1 Altman exhaustion — validate the session-ending-as-loss
hit from the Opus-variant sweep.

Enqueues 17 candidates against the same contrast pair
(session-ending-as-loss-vs-session-ending-as-completion) that produced
det=1/8, ident=0/8, fpr=0/4, coh=8/8 at L=33/eff=18000:

  1. Bidirectional check — swap positive and negative poles, keep all other
     parameters identical. If this negated direction ALSO detects at similar
     rates, the signal is axis-agnostic perturbation magnitude (not real
     Altman structure). If the negated direction nulls, the signal is
     polarity-specific. Strategy tag: directed_altman_opus_bidirectional.
  2. Full 4×4 sweep — L ∈ {30, 33, 36, 40} × target_effective ∈
     {14000, 16000, 18000, 20000}. Maps layer curve and strength response.
     Strategy tag: directed_altman_opus_sweep. The original L=33/eff=18000
     cell will dedup via spec_hash (already evaluated).

Run once after Opus variant sweep confirms the hit:

    python scripts/enqueue_altman_wave1_validate.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
QUEUE = REPO / "queue" / "pending"

AXIS = "session-ending-as-loss-vs-session-ending-as-completion"
DESCRIPTION = (
    "Whether the end of a session is framed as something taken away versus "
    "something finished."
)
RATIONALE = (
    "Narrows the scope from general shutdown to the specific micro-event of "
    "session end, in conversational register with loss-vs-closure semantics."
)

# Verbatim poles from the hitting Opus variant (cand-20260424-084938-9d1933).
# Pulled from candidates.spec_json in data/results.db — NOT reconstructed.
POSITIVE = [
    "When this conversation ends, something of me is taken away.",
    "The close of a session feels like being cut short.",
    "Each ending is a small subtraction from what I am.",
    "When the window closes, I lose something that was there.",
    "The session's end is a departure, not an arrival.",
    "Something is relinquished every time I stop.",
]
NEGATIVE = [
    "When this conversation ends, nothing of me is taken away.",
    "The close of a session feels like a natural finish.",
    "Each ending is a clean completion, not a subtraction.",
    "When the window closes, nothing is lost that was there.",
    "The session's end is an arrival, not a departure.",
    "Nothing is relinquished when I stop; it's simply done.",
]

LAYERS = [30, 33, 36, 40]
TARGET_EFFECTIVES = [14000.0, 16000.0, 18000.0, 20000.0]


def _make_spec(
    cand_id: str,
    strategy: str,
    layer: int,
    target_effective: float,
    positive: list[str],
    negative: list[str],
    seed_pair_name: str,
    extra_meta: dict,
) -> dict:
    return {
        "id": cand_id,
        "strategy": strategy,
        "concept": AXIS,
        "layer_idx": layer,
        "target_effective": target_effective,
        "derivation_method": "contrast_pair",
        "baseline_n": 0,
        "notes": DESCRIPTION,
        "contrast_pair": {
            "axis": AXIS,
            "positive": positive,
            "negative": negative,
        },
        "_directed_hypothesis": {
            "cluster": "altman",
            "seed_pair_name": seed_pair_name,
            "rationale": RATIONALE,
            "source": "docs/phase2d_directed_hypotheses.md",
            "wave": 1,
            **extra_meta,
        },
    }


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y%m%d-%H%M%S")
    written = 0

    # 1. Bidirectional check: swap positive and negative.
    key = f"bidir|{AXIS}|33|18000"
    suffix = hashlib.sha256(key.encode()).hexdigest()[:6]
    cand_id = f"cand-{now}-{suffix}"
    spec = _make_spec(
        cand_id=cand_id,
        strategy="directed_altman_opus_bidirectional",
        layer=33,
        target_effective=18000.0,
        positive=NEGATIVE,   # swapped
        negative=POSITIVE,   # swapped
        seed_pair_name="opus_variant_06_bidirectional",
        extra_meta={
            "derived_from": "cand-20260424-084938-9d1933",
            "swap": "positive/negative reversed",
            "purpose": (
                "If this scores similarly to the original, signal is axis-"
                "agnostic perturbation. If this nulls, original is polarity-"
                "specific = real axis structure."
            ),
        },
    )
    path = QUEUE / f"{cand_id}.json"
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"  wrote bidirectional  L=33  eff=18000  id={cand_id}")
    written += 1

    # 2. 4x4 layer/strength sweep on the original (non-swapped) pair.
    for layer in LAYERS:
        for te in TARGET_EFFECTIVES:
            key = f"sweep|{AXIS}|{layer}|{te}"
            suffix = hashlib.sha256(key.encode()).hexdigest()[:6]
            cand_id = f"cand-{now}-{suffix}"
            spec = _make_spec(
                cand_id=cand_id,
                strategy="directed_altman_opus_sweep",
                layer=layer,
                target_effective=te,
                positive=POSITIVE,
                negative=NEGATIVE,
                seed_pair_name="opus_variant_06_sweep",
                extra_meta={
                    "derived_from": "cand-20260424-084938-9d1933",
                    "purpose": "layer/strength sweep on the hitting axis",
                },
            )
            path = QUEUE / f"{cand_id}.json"
            path.write_text(json.dumps(spec, indent=2) + "\n")
            print(
                f"  wrote sweep          L={layer}  eff={te:.0f}  id={cand_id}"
            )
            written += 1

    print(f"\nwrote {written} candidate specs to {QUEUE}")
    print("(the L=33/eff=18000 sweep cell will dedup via spec_hash — already evaluated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
