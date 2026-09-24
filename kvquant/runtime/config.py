"""Global constants for odmkv (per-token compressed KV)."""
from __future__ import annotations

# Bit levels we physically support per token (0 = eviction, 16 = bf16).
SUPPORTED_BITS: tuple[int, ...] = (0, 2, 3, 4, 8, 16)

# Default head_dim for Llama-3.x / Mistral / Qwen.
DEFAULT_HEAD_DIM: int = 128

# Numerical tolerances for kernel-vs-reference equivalence tests.
ATOL_BF16: float = 1e-2
ATOL_FP32: float = 1e-5
