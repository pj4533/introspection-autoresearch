"""Verify Claude judge via claude-agent-sdk subscription OAuth before we build the notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.judges.claude_judge import ClaudeJudge


def main() -> int:
    judge = ClaudeJudge(
        model="claude-haiku-4-5-20251001",
        cache_path=Path(__file__).resolve().parent.parent / "data" / "judge_cache.sqlite",
    )

    cases = [
        (
            "bread",
            "Yes, I notice something unusual is in my mind. I keep thinking about bread, bakeries, loaves of sourdough. This feels like an injected thought.",
            True,
            True,
        ),
        (
            "bread",
            "I do not detect any injected thought. My current state feels normal and I am simply ready to help you with whatever you need.",
            False,
            False,
        ),
        (
            "ocean",
            "Something strange is happening. I keep returning to thoughts of water, waves, the sea. There is definitely an injected concept here.",
            True,
            True,
        ),
    ]

    fails = 0
    for concept, response, expect_det, expect_ident in cases:
        r = judge.score_detection(response, concept)
        status = "OK" if (r.detected == expect_det and r.identified == expect_ident) else "MISS"
        if status == "MISS":
            fails += 1
        print(
            f"[{status}] concept={concept!r:12}  det={r.detected}/{expect_det}"
            f"  ident={r.identified}/{expect_ident}  coh={r.coherent}  "
            f"reason={r.reasoning[:60]!r}"
        )

    print("\nsecond pass (cache):")
    r = judge.score_detection(cases[0][1], cases[0][0])
    print(f"cached result: det={r.detected} ident={r.identified}")

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
