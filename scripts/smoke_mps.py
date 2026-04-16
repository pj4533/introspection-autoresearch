"""Verify MPS backend and vendored paper primitives work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def main() -> int:
    print(f"torch {torch.__version__}")
    print(f"mps available: {torch.backends.mps.is_available()}")
    print(f"mps built:     {torch.backends.mps.is_built()}")

    if not torch.backends.mps.is_available():
        print("FAIL: MPS not available.")
        return 1

    x = torch.randn(2048, 2048, dtype=torch.bfloat16, device="mps")
    y = x @ x.T
    torch.mps.synchronize()
    print(f"bf16 matmul ok: shape {tuple(y.shape)} dtype {y.dtype} device {y.device}")

    from src.paper import MODEL_NAME_MAP

    assert "gemma3_12b" in MODEL_NAME_MAP, "gemma3_12b patch missing from vendored copy"
    assert MODEL_NAME_MAP["gemma3_12b"] == "google/gemma-3-12b-it"
    print(f"vendored paper import ok: gemma3_12b -> {MODEL_NAME_MAP['gemma3_12b']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
