"""Triton-accelerated helpers for TurboQuant.

Only *nearest-centroid lookup* is moved to Triton — the rotation
``x @ pi.T`` already uses cuBLAS and is fast. The searchsorted + clamp +
subtract + abs + where pipeline in PyTorch translates to 5–6 small kernel
launches per call; fusing them into a single Triton kernel cuts that to 1
kernel per call.

Design
------
* **GPU / multi-GPU safe**: every kernel launch is wrapped in
  ``with torch.cuda.device(tensor.device)``; all intermediate tensors live on
  the caller's device. Works under ``device_map="auto"`` with layers sharded
  across multiple GPUs.
* **Arch-portable**: BLOCK is a conservative 1024; ``num_warps`` is 4 which
  fits on any modern GPU (4090 / A100 / H100); all accumulators are fp32.
* **Bit-exact fallback**: when Triton / CUDA is unavailable, falls back to
  the original ``searchsorted + where`` PyTorch path, which is what the test
  oracle uses — so the Triton vs PyTorch equivalence is testable on any box.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _nearest_sorted_centroid(
        value,
        centroids,
        active,
        K: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        """Nearest index in one sorted scalar codebook.

        All native codebooks contain ``2**bits`` monotonically increasing
        Lloyd-Max centroids.  Sixteen or fewer centers retain the frozen linear
        scan: on RTX 4090 the scalar broadcast loads beat binary-search gathers
        for the 2/3/4-bit production tables.  The 256-entry 8-bit table uses
        lower-bound search, reducing it to eight decisions plus two neighbour
        loads.  The final strict ``right < left`` comparison preserves the
        linear oracle's tie rule: the smaller index wins.

        ``active`` may be a scalar or a vector mask.  Inactive lanes return a
        harmless index zero and never issue an out-of-bounds load.
        """
        if K <= 16:
            best_distance = tl.full(value.shape, float("inf"), dtype=tl.float32)
            best_index = tl.zeros(value.shape, dtype=tl.int32)
            for index in tl.static_range(0, K):
                center = tl.load(centroids + index).to(tl.float32)
                distance = tl.abs(value - center)
                better = active & (distance < best_distance)
                best_index = tl.where(better, index, best_index)
                best_distance = tl.where(better, distance, best_distance)
            return best_index
        else:
            # Search the closed interval [0, K - 1].  Native codebooks have a
            # power-of-two K, so exactly log2(K) iterations collapse every lane
            # to one index.  The half-open interval [0, K) would require an
            # extra iteration on the lower half and map some values one low.
            finite = active & (value == value) & (tl.abs(value) < float("inf"))
            low = tl.zeros(value.shape, dtype=tl.int32)
            high = tl.full(value.shape, K - 1, dtype=tl.int32)
            for _ in tl.static_range(0, SEARCH_STEPS):
                middle = (low + high) // 2
                center = tl.load(
                    centroids + middle,
                    mask=finite,
                    other=0.0,
                ).to(tl.float32)
                move_right = finite & (center < value)
                low = tl.where(move_right, middle + 1, low)
                high = tl.where(finite & ~move_right, middle, high)

            right = tl.minimum(low, K - 1)
            left = tl.minimum(tl.maximum(low - 1, 0), K - 1)
            left_center = tl.load(
                centroids + left,
                mask=finite,
                other=0.0,
            ).to(tl.float32)
            right_center = tl.load(
                centroids + right,
                mask=finite,
                other=0.0,
            ).to(tl.float32)
            left_distance = tl.abs(value - left_center)
            right_distance = tl.abs(value - right_center)
            return tl.where(
                finite,
                tl.where(right_distance < left_distance, right, left),
                0,
            )

    @triton.jit
    def _nearest_centroid_kernel(
        Y_ptr,                  # [N] fp32 values to quantise
        C_ptr,                  # [K] fp32 sorted centroids
        Out_ptr,                # [N] int32 output indices
        N,
        K: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """For each element of Y, find ``argmin_k |y - centroids[k]|``.

        Centroids are sorted, so this is an exact lower-bound search followed
        by a two-neighbour comparison.  It is byte-identical to the PyTorch
        ``searchsorted`` oracle, including lower-index tie breaking.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        if K <= 16:
            # Preserve the pre-optimization kernel body for the common small
            # tables; a nested helper changed code generation enough to regress
            # their measured latency despite doing the same comparisons.
            best_dist = tl.full([BLOCK], float("inf"), dtype=tl.float32)
            best_idx = tl.zeros([BLOCK], dtype=tl.int32)
            for k in tl.static_range(0, K):
                c = tl.load(C_ptr + k).to(tl.float32)
                dist = tl.abs(y - c)
                better = dist < best_dist
                best_idx = tl.where(better, k, best_idx)
                best_dist = tl.where(better, dist, best_dist)
        else:
            best_idx = _nearest_sorted_centroid(
                y, C_ptr, mask, K=K, SEARCH_STEPS=SEARCH_STEPS,
            )

        tl.store(Out_ptr + offs, best_idx, mask=mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mse_nearest_centroid(
    y: torch.Tensor,           # [..., D]  post-rotation values
    centroids: torch.Tensor,   # [K]       sorted ascending
) -> torch.Tensor:
    """For each scalar in *y*, return the index of the nearest entry in
    *centroids*. Result shape matches *y*, dtype int32.

    Equivalent to::

        ins = torch.searchsorted(centroids, y).clamp(1, K - 1)
        left  = (y - centroids[ins - 1]).abs()
        right = (y - centroids[ins]).abs()
        return torch.where(left <= right, ins - 1, ins)

    but fused into one kernel.
    """
    assert y.device == centroids.device, (
        f"y on {y.device}, centroids on {centroids.device}"
    )
    orig_shape = y.shape
    y_flat = y.contiguous().view(-1).to(torch.float32)
    c_flat = centroids.contiguous().to(torch.float32)
    N = y_flat.numel()
    K = c_flat.numel()
    if K < 2:
        raise ValueError(f"nearest-centroid lookup requires K>=2, got {K}")
    search_steps = (K - 1).bit_length()

    if not (HAS_TRITON and y_flat.is_cuda):
        # PyTorch fallback (also test oracle). Device-agnostic.
        return _nearest_centroid_reference(y_flat, c_flat).view(orig_shape)

    out = torch.empty(N, dtype=torch.int32, device=y_flat.device)
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    # Launch must happen on the tensor's CUDA context when the model is
    # sharded across multiple GPUs via device_map="auto".
    with torch.cuda.device(y_flat.device):
        _nearest_centroid_kernel[grid](
            y_flat, c_flat, out,
            N, K=K, SEARCH_STEPS=search_steps, BLOCK=BLOCK,
            num_warps=4,
        )
    return out.view(orig_shape)


def _nearest_centroid_reference(
    y: torch.Tensor, centroids: torch.Tensor,
) -> torch.Tensor:
    """Byte-identical PyTorch oracle — also used as CPU / no-triton fallback."""
    K = centroids.shape[0]
    y_f = y.to(torch.float32)
    c_f = centroids.to(torch.float32)
    ins = torch.searchsorted(c_f, y_f.contiguous())
    ins = ins.clamp(1, K - 1)
    left  = (y_f - c_f[ins - 1]).abs()
    right = (y_f - c_f[ins]).abs()
    selected = torch.where(left <= right, ins - 1, ins).to(torch.int32)
    # Match the frozen linear-scan kernel: every non-finite distance fails its
    # strict ``distance < best_distance`` test, leaving the initial code zero.
    return torch.where(torch.isfinite(y_f), selected, torch.zeros_like(selected))


__all__ = ["mse_nearest_centroid", "HAS_TRITON"]
