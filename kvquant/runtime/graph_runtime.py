"""Process-lifetime CUDA resources shared by compressed graph decoders."""
from __future__ import annotations

import contextlib
import threading

import torch


# A fresh side stream per request leaves a cuBLAS workspace attached to every
# stream even after Python destroys the stream object (8.125 MiB/request on the
# supported CUDA stack).  Reuse one warmup stream per CUDA device so request memory
# reaches a stable plateau.
_WARMUP_STREAMS: dict[int, torch.cuda.Stream] = {}
_CAPTURE_LOCKS: dict[int, threading.Lock] = {}
_RESOURCE_LOCK = threading.Lock()


def canonical_cuda_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"CUDA graph runtime requires a CUDA device, got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def graph_warmup_stream(device: torch.device | str) -> torch.cuda.Stream:
    """Return the process-lifetime graph warmup stream for ``device``."""
    resolved = canonical_cuda_device(device)
    assert resolved.index is not None
    with _RESOURCE_LOCK:
        stream = _WARMUP_STREAMS.get(resolved.index)
        if stream is None:
            stream = torch.cuda.Stream(device=resolved)
            _WARMUP_STREAMS[resolved.index] = stream
    return stream


@contextlib.contextmanager
def graph_capture_guard(device: torch.device | str):
    """Serialize graph capture on one CUDA device across models/threads.

    CUDA permits ordinary work from other threads when capture uses
    ``thread_local`` error mode, but two captures must not race for the shared
    warmup stream or CUDA graph allocator state.  The lock is process-lifetime
    and device-scoped, so independent GPUs remain fully concurrent.
    """
    resolved = canonical_cuda_device(device)
    assert resolved.index is not None
    with _RESOURCE_LOCK:
        lock = _CAPTURE_LOCKS.get(resolved.index)
        if lock is None:
            lock = threading.Lock()
            _CAPTURE_LOCKS[resolved.index] = lock
    with lock:
        yield


__all__ = [
    "canonical_cuda_device",
    "graph_capture_guard",
    "graph_warmup_stream",
]
