"""Sequence-chunked prefill feed-forward, to cut the activation transient.

At long context the dominant *transient* allocation of a request is not the KV
cache at all -- it is the per-layer feed-forward intermediate.  For
Llama-3.1-8B (hidden 4096, intermediate 14336) a 32,768-token prefill holds
roughly ``3 x 32768 x 14336 x 2B = 2.8 GB`` live inside a single MLP call
(``up``, ``act(gate)``, and their product), which is far larger than the entire
compressed KV state.  That transient, not the cache, is what decides how long a
prompt fits on one card.

The feed-forward is row-independent, so evaluating it in sequence slices and
writing each slice into a preallocated output is the same computation with a
bounded working set.  Slicing changes the GEMM's ``M`` extent, so cuBLAS may
select a different kernel and the result is numerically equivalent rather than
bitwise identical -- well below quantization noise, but the reason this only
engages above ``min_tokens``.  Short prompts (the frozen H1/H2 matrix at
C=2048/4096) therefore keep their exact archived numerics.

The patch is scoped to a context manager and removed before CUDA-graph capture,
so the decode path is untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
import os

import torch


DEFAULT_CHUNK_TOKENS = 4096
DEFAULT_MIN_TOKENS = 8192


def _resolve(chunk_tokens: int | None, min_tokens: int | None) -> tuple[int, int]:
    if chunk_tokens is None:
        chunk_tokens = int(
            os.environ.get("R2_PREFILL_MLP_CHUNK", DEFAULT_CHUNK_TOKENS)
        )
    if min_tokens is None:
        min_tokens = int(
            os.environ.get("R2_PREFILL_MLP_MIN", DEFAULT_MIN_TOKENS)
        )
    return int(chunk_tokens), int(min_tokens)


def _iter_feed_forward(model):
    """Yield the per-layer feed-forward submodules of a decoder stack."""
    backbone = getattr(model, "model", model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and callable(getattr(mlp, "forward", None)):
            yield mlp


def _chunked_forward(original, chunk_tokens: int, min_tokens: int):
    def forward(hidden_states, *args, **kwargs):
        if (
            not isinstance(hidden_states, torch.Tensor)
            or hidden_states.dim() != 3
            or hidden_states.shape[1] < min_tokens
            or args
            or kwargs
        ):
            return original(hidden_states, *args, **kwargs)
        tokens = hidden_states.shape[1]
        out = torch.empty_like(hidden_states)
        for start in range(0, tokens, chunk_tokens):
            stop = min(start + chunk_tokens, tokens)
            out[:, start:stop] = original(hidden_states[:, start:stop])
        return out

    return forward


@contextmanager
def chunked_prefill_feed_forward(
    model, *, chunk_tokens: int | None = None, min_tokens: int | None = None,
):
    """Evaluate every decoder MLP in sequence slices for the enclosed block.

    ``chunk_tokens <= 0`` disables the transform, so a single environment
    variable (``R2_PREFILL_MLP_CHUNK=0``) restores the unsliced prefill.
    """
    chunk_tokens, min_tokens = _resolve(chunk_tokens, min_tokens)
    if chunk_tokens <= 0:
        yield
        return
    patched = []
    try:
        for mlp in _iter_feed_forward(model):
            patched.append((mlp, mlp.forward))
            mlp.forward = _chunked_forward(mlp.forward, chunk_tokens, min_tokens)
        yield
    finally:
        for mlp, original in patched:
            mlp.forward = original


__all__ = ["chunked_prefill_feed_forward", "DEFAULT_CHUNK_TOKENS", "DEFAULT_MIN_TOKENS"]
