"""odmkv — per-token compressed-KV cache and attention.

Storage: per-bit banks (kvquant.tq_adaptive_backend) — each token at its
own assigned bit level. Bits assigned per-token by Lagrangian over the
configured scorer (kvquant.allocator + kvquant.scorer).

Attention: q_len=1 reads the packed banks directly through a mixed-bit
Triton flash-decode kernel. Unsupported/masked and multi-token decode requests
fail explicitly; the production dispatcher has no materialize/SDPA fallback.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kvquant.runtime.backend_press import ODMNativeBackend
    from kvquant.runtime.storage.cache import ODMCache

__all__ = ["ODMCache", "ODMNativeBackend"]


def __getattr__(name: str):
    """Keep low-level kernel/layout imports independent of Transformers."""
    if name == "ODMCache":
        from kvquant.runtime.storage.cache import ODMCache
        return ODMCache
    if name == "ODMNativeBackend":
        from kvquant.runtime.backend_press import ODMNativeBackend
        return ODMNativeBackend
    raise AttributeError(name)
