"""Native fixed-address decode-ring flush into shared-v2 bit arenas.

This module contains two implementations with the same storage contract:

``append_decode_2bit_workspace``
    A low-risk path which reads arbitrary ``[B,H,T,D]`` strides directly,
    normalizes into reusable fp32 workspaces, uses cuBLAS for the rotations,
    and encodes directly into each row's reserved 2-bit suffix.

``append_decode_2bit_fused``
    An experimental path which fuses normalization, rotation, nearest-
    centroid lookup, and byte packing.  K and V each require one Triton launch;
    a third launch publishes row lengths and debug tags.  It allocates no
    payload-sized temporary and never changes an arena address.

The fixed-2 paths deliberately derive the append destination from
``stable_row_capacity - reserve + decode_start`` instead of reading a row's
mutable ``seqlen``.  Therefore publishing the new length cannot race payload
writes, and an already captured compressed-attention graph remains valid.

``append_decode_mixed_workspace`` retains the same graph-stability contract
for allocator-selected 0/2/3/4/8/16 tags. It computes row-local ranks on CUDA.
Legacy reserved banks use ``offset + current_seqlen + rank``; production uses
one byte-tight segmented payload arena and publishes a compact, bidirectional
descriptor stream. Every store is bounded before logical history advances.
Zero-bit cells are published only to that history; no payload or attention
descriptor exists for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only import path.
    triton = None
    tl = None
    _HAS_TRITON = False


@dataclass
class NativeFlushWorkspace:
    """Reusable scratch for the cuBLAS-backed flush implementation."""

    normalized_k: torch.Tensor
    normalized_v: torch.Tensor
    rotated_k: torch.Tensor
    rotated_v: torch.Tensor
    ranks: torch.Tensor
    counts: torch.Tensor
    status: torch.Tensor
    rows: int
    max_tokens: int
    head_dim: int

    @property
    def reserved_bytes(self) -> int:
        """Persistent scratch bytes, reported separately from compressed KV."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.normalized_k,
                self.normalized_v,
                self.rotated_k,
                self.rotated_v,
                self.ranks,
                self.counts,
                self.status,
            )
        )

    @classmethod
    def allocate(
        cls,
        *,
        rows: int,
        max_tokens: int,
        head_dim: int,
        device: torch.device,
    ) -> "NativeFlushWorkspace":
        shape = (int(rows) * int(max_tokens), int(head_dim))
        buffers = [torch.empty(shape, device=device, dtype=torch.float32) for _ in range(4)]
        ranks = torch.empty(int(rows) * int(max_tokens), device=device, dtype=torch.int32)
        counts = torch.empty(5, int(rows), device=device, dtype=torch.int32)
        # invalid-tag, payload-overflow.
        status = torch.empty(2, device=device, dtype=torch.int32)
        return cls(
            *buffers, ranks, counts, status,
            int(rows), int(max_tokens), int(head_dim),
        )

    def validate(self, *, rows: int, tokens: int, head_dim: int, device: torch.device) -> None:
        if self.rows != rows or self.head_dim != head_dim or self.max_tokens < tokens:
            raise ValueError(
                "native flush workspace mismatch: "
                f"workspace=({self.rows},{self.max_tokens},{self.head_dim}) "
                f"request=({rows},{tokens},{head_dim})"
            )
        if self.normalized_k.device != device:
            raise ValueError("native flush workspace is on the wrong device")


if _HAS_TRITON:
    from kvquant.tq_triton import _nearest_sorted_centroid

    @triton.jit(
        do_not_specialize=["decode_start"],
        do_not_specialize_on_alignment=["decode_start"],
    )
    def _normalize_kv_ring_kernel(
        K,
        V,
        NK_OUT,
        NV_OUT,
        NORM_ARENA_K,
        NORM_ARENA_V,
        NORM_BASE,
        ROW_OFFSET,
        ROW_CAPACITY,
        decode_start,
        reserve: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        tokens: tl.constexpr,
        HAS_NORM_BASE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.program_id(1)
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, BLOCK_D)
        mask = d < head_dim
        k = tl.load(
            K
            + batch * key_stride_b
            + head * key_stride_h
            + token * key_stride_t
            + d * key_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V
            + batch * value_stride_b
            + head * value_stride_h
            + token * value_stride_t
            + d * value_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        norm_k = tl.sqrt(tl.sum(k * k, axis=0))
        norm_v = tl.sqrt(tl.sum(v * v, axis=0))
        work = (row * tokens + token) * head_dim + d
        tl.store(NK_OUT + work, k / (norm_k + 1.0e-10), mask=mask)
        tl.store(NV_OUT + work, v / (norm_v + 1.0e-10), mask=mask)

        initial = tl.load(ROW_CAPACITY + row).to(tl.int32) - reserve
        norm_base = tl.load(NORM_BASE).to(tl.int32) if HAS_NORM_BASE else 0
        destination = (
            norm_base + tl.load(ROW_OFFSET + row).to(tl.int32)
            + initial
            + decode_start
            + token
        )
        tl.store(NORM_ARENA_K + destination, norm_k)
        tl.store(NORM_ARENA_V + destination, norm_v)


    @triton.jit(
        do_not_specialize=["decode_start"],
        do_not_specialize_on_alignment=["decode_start"],
    )
    def _encode_rotated_kv_2bit_kernel(
        RK,
        RV,
        CENT_K,
        CENT_V,
        PACKED_K,
        PACKED_V,
        CODE_BASE,
        ROW_OFFSET,
        ROW_CAPACITY,
        decode_start,
        reserve: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        tokens: tl.constexpr,
        HAS_CODE_BASE: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.program_id(1)
        p = tl.arange(0, BLOCK_P)
        live = p < width
        work_base = (row * tokens + token) * head_dim
        packed_k = tl.zeros([BLOCK_P], dtype=tl.int32)
        packed_v = tl.zeros([BLOCK_P], dtype=tl.int32)
        for lane in tl.static_range(0, 4):
            d = p * 4 + lane
            lane_live = live & (d < head_dim)
            kval = tl.load(RK + work_base + d, mask=lane_live, other=0.0).to(tl.float32)
            vval = tl.load(RV + work_base + d, mask=lane_live, other=0.0).to(tl.float32)
            best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
            best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
            best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for centroid_idx in tl.static_range(0, 4):
                ck = tl.load(CENT_K + centroid_idx).to(tl.float32)
                cv = tl.load(CENT_V + centroid_idx).to(tl.float32)
                dk = tl.abs(kval - ck)
                dv = tl.abs(vval - cv)
                better_k = dk < best_k_dist
                better_v = dv < best_v_dist
                best_k_dist = tl.where(better_k, dk, best_k_dist)
                best_v_dist = tl.where(better_v, dv, best_v_dist)
                best_k = tl.where(better_k, centroid_idx, best_k)
                best_v = tl.where(better_v, centroid_idx, best_v)
            packed_k |= best_k << (2 * lane)
            packed_v |= best_v << (2 * lane)

        initial = tl.load(ROW_CAPACITY + row).to(tl.int32) - reserve
        destination = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + initial
            + decode_start
            + token
        )
        code_base = tl.load(CODE_BASE).to(tl.int32) if HAS_CODE_BASE else 0
        code = code_base + destination * width + p
        tl.store(PACKED_K + code, packed_k, mask=live)
        tl.store(PACKED_V + code, packed_v, mask=live)


    @triton.jit
    def _rotate_quant_lane(
        normalized,
        ROTATION,
        CENTROIDS,
        d,
        p,
        lane: tl.constexpr,
        head_dim: tl.constexpr,
        rotation_stride_0: tl.constexpr,
        rotation_stride_1: tl.constexpr,
        BLOCK_M: tl.constexpr,
        width: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        out_d = p * 4 + lane
        rotation_addr = (
            ROTATION
            + d[:, None] * rotation_stride_1
            + out_d[None, :] * rotation_stride_0
        )
        rotation_t = tl.load(rotation_addr).to(tl.float32)
        if IEEE:
            rotated = tl.dot(normalized, rotation_t, input_precision="ieee")
        else:
            rotated = tl.dot(normalized, rotation_t, input_precision="tf32")
        best_dist = tl.full((BLOCK_M, width), float("inf"), dtype=tl.float32)
        best_index = tl.zeros((BLOCK_M, width), dtype=tl.int32)
        for centroid_idx in tl.static_range(0, 4):
            centroid = tl.load(CENTROIDS + centroid_idx).to(tl.float32)
            distance = tl.abs(rotated - centroid)
            better = distance < best_dist
            best_dist = tl.where(better, distance, best_dist)
            best_index = tl.where(better, centroid_idx, best_index)
        return best_index


    @triton.jit(
        do_not_specialize=["decode_start"],
        do_not_specialize_on_alignment=["decode_start"],
    )
    def _rotate_encode_2bit_kernel(
        SOURCE,
        ROTATION,
        CENTROIDS,
        PACKED,
        NORMS,
        CODE_BASE,
        NORM_BASE,
        ROW_OFFSET,
        ROW_CAPACITY,
        decode_start,
        total_tokens,
        reserve: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        source_stride_b: tl.constexpr,
        source_stride_h: tl.constexpr,
        source_stride_t: tl.constexpr,
        source_stride_d: tl.constexpr,
        rotation_stride_0: tl.constexpr,
        rotation_stride_1: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HAS_CODE_BASE: tl.constexpr,
        HAS_NORM_BASE: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        live_m = m < total_tokens
        row = m // tokens
        token = m - row * tokens
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, head_dim)
        source_addr = (
            SOURCE
            + batch[:, None] * source_stride_b
            + head[:, None] * source_stride_h
            + token[:, None] * source_stride_t
            + d[None, :] * source_stride_d
        )
        x = tl.load(source_addr, mask=live_m[:, None], other=0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(x * x, axis=1))
        normalized = x / (norm[:, None] + 1.0e-10)
        # Existing append semantics are fp32 ``x @ pi.T``.  IEEE=True uses
        # full fp32 products; the profile also evaluates TF32 as an explicitly
        # labelled numerical implementation choice.
        # Four width-sized products avoid a layout-dependent reshape of a
        # [BLOCK_M,D] MMA result. Each product computes one of the four scalar
        # lanes packed into a byte; the aggregate FLOP count is unchanged.
        p = tl.arange(0, width)
        index0 = _rotate_quant_lane(
            normalized, ROTATION, CENTROIDS, d, p, 0,
            head_dim, rotation_stride_0, rotation_stride_1,
            BLOCK_M, width, IEEE,
        )
        index1 = _rotate_quant_lane(
            normalized, ROTATION, CENTROIDS, d, p, 1,
            head_dim, rotation_stride_0, rotation_stride_1,
            BLOCK_M, width, IEEE,
        )
        index2 = _rotate_quant_lane(
            normalized, ROTATION, CENTROIDS, d, p, 2,
            head_dim, rotation_stride_0, rotation_stride_1,
            BLOCK_M, width, IEEE,
        )
        index3 = _rotate_quant_lane(
            normalized, ROTATION, CENTROIDS, d, p, 3,
            head_dim, rotation_stride_0, rotation_stride_1,
            BLOCK_M, width, IEEE,
        )
        packed = index0 | (index1 << 2) | (index2 << 4) | (index3 << 6)

        initial = tl.load(ROW_CAPACITY + row, mask=live_m, other=reserve).to(tl.int32) - reserve
        destination = (
            tl.load(ROW_OFFSET + row, mask=live_m, other=0).to(tl.int32)
            + initial
            + decode_start
            + token
        )
        code_base = tl.load(CODE_BASE).to(tl.int32) if HAS_CODE_BASE else 0
        code = code_base + destination[:, None] * width + p[None, :]
        tl.store(PACKED + code, packed, mask=live_m[:, None])
        norm_base = tl.load(NORM_BASE).to(tl.int32) if HAS_NORM_BASE else 0
        norm_destination = norm_base + destination
        tl.store(NORMS + norm_destination, norm, mask=live_m)


    @triton.jit(
        do_not_specialize=["decode_start"],
        do_not_specialize_on_alignment=["decode_start"],
    )
    def _publish_decode_suffix_kernel(
        SEQLEN,
        ROW_CAPACITY,
        TAGS,
        decode_start,
        tokens: tl.constexpr,
        reserve: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.arange(0, BLOCK_T)
        initial = tl.load(ROW_CAPACITY + row).to(tl.int32) - reserve
        tl.store(SEQLEN + row, initial + decode_start + tokens)
        tl.store(
            TAGS + row * reserve + decode_start + token,
            2,
            mask=token < tokens,
        )


    @triton.jit
    def _rank_mixed_decode_kernel(
        TAGS,
        RANKS,
        COUNTS,
        INVALID,
        tokens: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        rows = tl.num_programs(0)
        token = tl.arange(0, BLOCK_T)
        valid = token < tokens
        tags = tl.load(TAGS + row * tokens + token, mask=valid, other=-1).to(tl.int32)
        allowed = (
            (tags == 0) | (tags == 2) | (tags == 3) | (tags == 4)
            | (tags == 8) | (tags == 16)
        )
        tl.atomic_max(INVALID, tl.max((valid & ~allowed).to(tl.int32), axis=0))
        chosen_rank = tl.zeros([BLOCK_T], dtype=tl.int32)

        selected = valid & (tags == 2)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(COUNTS + 0 * rows + row, tl.sum(selected.to(tl.int32), axis=0))

        selected = valid & (tags == 3)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(COUNTS + 1 * rows + row, tl.sum(selected.to(tl.int32), axis=0))

        selected = valid & (tags == 4)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(COUNTS + 2 * rows + row, tl.sum(selected.to(tl.int32), axis=0))

        selected = valid & (tags == 8)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(COUNTS + 3 * rows + row, tl.sum(selected.to(tl.int32), axis=0))

        selected = valid & (tags == 16)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(COUNTS + 4 * rows + row, tl.sum(selected.to(tl.int32), axis=0))
        tl.store(RANKS + row * tokens + token, chosen_rank, mask=valid)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _allocate_decode_segment_kernel(
        COUNTS,
        CODE_ROW_BASE,
        NORM_ROW_BASE,
        EXACT_ROW_BASE,
        SEGMENT_COUNTS,
        BUMP,
        OVERFLOW,
        segment,
        rows: tl.constexpr,
        code_capacity: tl.constexpr,
        norm_capacity: tl.constexpr,
        exact_capacity: tl.constexpr,
        slot_bytes: tl.constexpr,
        W2: tl.constexpr,
        W3: tl.constexpr,
        W4: tl.constexpr,
        W8: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Reserve one mixed segment from aggregate arenas on device."""
        index = tl.arange(0, BLOCK)
        valid = index < 5 * rows
        level = index // rows
        row = index - level * rows
        count = tl.load(COUNTS + level * rows + row, mask=valid, other=0).to(tl.int32)
        quant = valid & (level < 4)
        width = tl.where(
            level == 0, W2,
            tl.where(level == 1, W3, tl.where(level == 2, W4, W8)),
        )
        code_need = tl.where(quant, count * width, 0)
        norm_need = tl.where(quant, count, 0)
        exact_need = tl.where(valid & (level == 4), count, 0)
        code_prefix = tl.cumsum(code_need, axis=0) - code_need
        norm_prefix = tl.cumsum(norm_need, axis=0) - norm_need
        exact_prefix = tl.cumsum(exact_need, axis=0) - exact_need
        total_code = tl.sum(code_need, axis=0)
        total_norm = tl.sum(norm_need, axis=0)
        total_exact = tl.sum(exact_need, axis=0)
        code_start = tl.atomic_add(BUMP + 0, total_code)
        norm_start = tl.atomic_add(BUMP + 1, total_norm)
        # Codes and 16-bit slots share one byte pool: codes grow up from 0,
        # exact slots grow down from the top.  ``BUMP + 2`` counts exact slots
        # taken so far, so this segment occupies the next block below them.
        exact_taken = tl.atomic_add(BUMP + 2, total_exact)
        exact_start = exact_capacity - exact_taken - total_exact
        # Both fronts are monotone (code high-water rises, exact floor falls), so
        # checking them against each other at every segment catches any overlap.
        tl.atomic_max(
            OVERFLOW,
            (
                (exact_start < 0)
                | (norm_start + total_norm > norm_capacity)
                | (code_start + total_code > exact_start * slot_bytes)
            ).to(tl.int32),
        )
        descriptor = segment * 5 * rows + level * rows + row
        tl.store(SEGMENT_COUNTS + descriptor, count, mask=valid)
        quant_descriptor = segment * 4 * rows + level * rows + row
        tl.store(
            CODE_ROW_BASE + quant_descriptor,
            code_start + code_prefix,
            mask=quant,
        )
        tl.store(
            NORM_ROW_BASE + quant_descriptor,
            norm_start + norm_prefix,
            mask=quant,
        )
        tl.store(
            EXACT_ROW_BASE + segment * rows + row,
            exact_start + exact_prefix,
            mask=valid & (level == 4),
        )


    @triton.jit(
        do_not_specialize=["decode_start", "segment"],
        do_not_specialize_on_alignment=["decode_start", "segment"],
    )
    def _publish_decode_descriptors_kernel(
        TAGS,
        RANKS,
        COUNTS,
        DESCRIPTORS,
        DESCRIPTOR_COUNTS,
        DECODE_TAGS,
        OVERFLOW,
        decode_start,
        segment,
        rows: tl.constexpr,
        capacity: tl.constexpr,
        descriptor_stride: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Append compact, graph-stable indirection streams for attention.

        Payload allocation remains segmented and byte-tight.  A descriptor
        encodes a quant token as
        ``(segment * 4 + level) * descriptor_stride + rank`` and an exact
        token as ``segment * descriptor_stride + rank``.  The streams are
        logically separate so each attention tile executes only one QK/PV
        domain. Quant grows from the front and exact from the back of one
        row-capacity array.
        """
        row = tl.program_id(0)
        token = tl.arange(0, BLOCK_T)
        valid = token < tokens
        tag = tl.load(TAGS + row * tokens + token, mask=valid, other=0).to(tl.int32)
        rank = tl.load(RANKS + row * tokens + token, mask=valid, other=0).to(tl.int32)
        level = tl.where(
            tag == 2, 0,
            tl.where(tag == 3, 1, tl.where(tag == 4, 2, tl.where(tag == 8, 3, 4))),
        )
        c2 = tl.load(COUNTS + 0 * rows + row).to(tl.int32)
        c3 = tl.load(COUNTS + 1 * rows + row).to(tl.int32)
        c4 = tl.load(COUNTS + 2 * rows + row).to(tl.int32)
        c8 = tl.load(COUNTS + 3 * rows + row).to(tl.int32)
        ce = tl.load(COUNTS + 4 * rows + row).to(tl.int32)
        quant_total = c2 + c3 + c4 + c8
        quant_prefix = tl.where(
            level == 0, 0,
            tl.where(level == 1, c2, tl.where(level == 2, c2 + c3, c2 + c3 + c4)),
        )
        old_quant = tl.load(DESCRIPTOR_COUNTS + 0 * rows + row).to(tl.int32)
        old_exact = tl.load(DESCRIPTOR_COUNTS + 1 * rows + row).to(tl.int32)
        is_quant = valid & ((tag == 2) | (tag == 3) | (tag == 4) | (tag == 8))
        is_exact = valid & (tag == 16)
        quant_destination = old_quant + quant_prefix + rank
        exact_destination = capacity - 1 - (old_exact + rank)
        quant_descriptor = (
            (segment * 4 + level) * descriptor_stride + rank
        ).to(tl.int32)
        exact_descriptor = (segment * descriptor_stride + rank).to(tl.int32)
        tl.store(
            DESCRIPTORS + row * capacity + quant_destination,
            quant_descriptor,
            mask=is_quant & (quant_destination >= 0)
            & (quant_destination < capacity),
        )
        tl.store(
            DESCRIPTORS + row * capacity + exact_destination,
            exact_descriptor,
            mask=is_exact & (exact_destination >= 0)
            & (exact_destination < capacity),
        )
        tl.store(
            DECODE_TAGS + row * capacity + decode_start + token,
            tag,
            mask=valid,
        )
        tl.atomic_max(
            OVERFLOW,
            (old_quant + quant_total + old_exact + ce > capacity).to(tl.int32),
        )
        # One program owns the row; scalar publication is race-free.  Counts
        # become visible only after every descriptor store in the same stream.
        tl.store(DESCRIPTOR_COUNTS + 0 * rows + row, old_quant + quant_total)
        tl.store(DESCRIPTOR_COUNTS + 1 * rows + row, old_exact + ce)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _scatter_segment_norm_kernel(
        NORM_K_TMP,
        NORM_V_TMP,
        TAGS,
        RANKS,
        OUT_K,
        OUT_V,
        NORM_ROW_BASE,
        segment,
        rows: tl.constexpr,
        norm_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        tokens: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        tag = tl.load(TAGS + cell).to(tl.int32)
        level = tl.where(
            tag == 2, 0,
            tl.where(tag == 3, 1, tl.where(tag == 4, 2, tl.where(tag == 8, 3, -1))),
        )
        quant = level >= 0
        descriptor = segment * 4 * rows + level * rows + row
        base = tl.load(NORM_ROW_BASE + descriptor, mask=quant, other=0).to(tl.int32)
        destination = base + tl.load(RANKS + cell).to(tl.int32)
        live = quant & (destination >= 0) & (destination < norm_capacity)
        work = cell * head_dim
        tl.store(OUT_K + destination, tl.load(NORM_K_TMP + work), mask=live)
        tl.store(OUT_V + destination, tl.load(NORM_V_TMP + work), mask=live)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _scatter_segment_exact_kernel(
        K,
        V,
        TAGS,
        RANKS,
        OUT_K,
        OUT_V,
        EXACT_ROW_BASE,
        segment,
        rows: tl.constexpr,
        exact_capacity: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        token = cell - row * tokens
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, BLOCK_D)
        selected = tl.load(TAGS + cell).to(tl.int32) == 16
        base = tl.load(
            EXACT_ROW_BASE + segment * rows + row, mask=selected, other=0,
        ).to(tl.int32)
        destination = base + tl.load(RANKS + cell).to(tl.int32)
        live = selected & (destination >= 0) & (destination < exact_capacity) & (d < head_dim)
        k = tl.load(
            K + batch * key_stride_b + head * key_stride_h
            + token * key_stride_t + d * key_stride_d,
            mask=live, other=0.0,
        )
        v = tl.load(
            V + batch * value_stride_b + head * value_stride_h
            + token * value_stride_t + d * value_stride_d,
            mask=live, other=0.0,
        )
        tl.store(OUT_K + destination * head_dim + d, k, mask=live)
        tl.store(OUT_V + destination * head_dim + d, v, mask=live)


    @triton.jit
    def _validate_mixed_capacity_kernel(
        COUNTS,
        SEQLEN_2,
        CAPACITY_2,
        SEQLEN_3,
        CAPACITY_3,
        SEQLEN_4,
        CAPACITY_4,
        SEQLEN_8,
        CAPACITY_8,
        SEQLEN_16,
        CAPACITY_16,
        OVERFLOW,
    ):
        row = tl.program_id(0)
        rows = tl.num_programs(0)
        overflow = (
            (tl.load(SEQLEN_2 + row) + tl.load(COUNTS + 0 * rows + row)
             > tl.load(CAPACITY_2 + row))
            | (tl.load(SEQLEN_3 + row) + tl.load(COUNTS + 1 * rows + row)
               > tl.load(CAPACITY_3 + row))
            | (tl.load(SEQLEN_4 + row) + tl.load(COUNTS + 2 * rows + row)
               > tl.load(CAPACITY_4 + row))
            | (tl.load(SEQLEN_8 + row) + tl.load(COUNTS + 3 * rows + row)
               > tl.load(CAPACITY_8 + row))
            | (tl.load(SEQLEN_16 + row) + tl.load(COUNTS + 4 * rows + row)
               > tl.load(CAPACITY_16 + row))
        )
        tl.atomic_max(OVERFLOW, overflow.to(tl.int32))


    @triton.jit
    def _normalize_mixed_kv_kernel(
        K,
        V,
        NK_OUT,
        NV_OUT,
        NORM_K_TMP,
        NORM_V_TMP,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        token = tl.program_id(1)
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, BLOCK_D)
        live = d < head_dim
        k = tl.load(
            K + batch * key_stride_b + head * key_stride_h
            + token * key_stride_t + d * key_stride_d,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V + batch * value_stride_b + head * value_stride_h
            + token * value_stride_t + d * value_stride_d,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        norm_k = tl.sqrt(tl.sum(k * k, axis=0))
        norm_v = tl.sqrt(tl.sum(v * v, axis=0))
        work = (row * tokens + token) * head_dim + d
        tl.store(NK_OUT + work, k / (norm_k + 1.0e-10), mask=live)
        tl.store(NV_OUT + work, v / (norm_v + 1.0e-10), mask=live)
        # The first scalar of each future rotation output is temporary norm
        # scratch.  Norms are scattered before the two GEMMs overwrite it.
        scalar = (row * tokens + token) * head_dim
        tl.store(NORM_K_TMP + scalar, norm_k)
        tl.store(NORM_V_TMP + scalar, norm_v)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _prepare_segment_payload_kernel(
        K,
        V,
        TAGS,
        RANKS,
        NK_OUT,
        NV_OUT,
        NORM_ARENA_K,
        NORM_ARENA_V,
        NORM_ROW_BASE,
        EXACT_K,
        EXACT_V,
        EXACT_ROW_BASE,
        segment,
        rows: tl.constexpr,
        norm_capacity: tl.constexpr,
        exact_capacity: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Prepare only retained payloads; tag=0 performs no K/V work."""
        row = tl.program_id(0)
        token = tl.program_id(1)
        cell = row * tokens + token
        tag = tl.load(TAGS + cell).to(tl.int32)
        level = tl.where(
            tag == 2, 0,
            tl.where(tag == 3, 1, tl.where(tag == 4, 2, tl.where(tag == 8, 3, -1))),
        )
        quant = level >= 0
        exact = tag == 16
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, BLOCK_D)
        lane = d < head_dim
        source_k = (
            K + batch * key_stride_b + head * key_stride_h
            + token * key_stride_t + d * key_stride_d
        )
        source_v = (
            V + batch * value_stride_b + head * value_stride_h
            + token * value_stride_t + d * value_stride_d
        )
        rank = tl.load(RANKS + cell).to(tl.int32)
        if quant:
            k = tl.load(source_k, mask=lane, other=0.0).to(tl.float32)
            v = tl.load(source_v, mask=lane, other=0.0).to(tl.float32)
            norm_k = tl.sqrt(tl.sum(k * k, axis=0))
            norm_v = tl.sqrt(tl.sum(v * v, axis=0))
            work = cell * head_dim + d
            tl.store(
                NK_OUT + work, k / (norm_k + 1.0e-10),
                mask=lane,
            )
            tl.store(
                NV_OUT + work, v / (norm_v + 1.0e-10),
                mask=lane,
            )
            descriptor = segment * 4 * rows + level * rows + row
            base = tl.load(NORM_ROW_BASE + descriptor).to(tl.int32)
            destination = base + rank
            safe = (destination >= 0) & (destination < norm_capacity)
            tl.store(NORM_ARENA_K + destination, norm_k, mask=safe)
            tl.store(NORM_ARENA_V + destination, norm_v, mask=safe)
        elif exact:
            base = tl.load(EXACT_ROW_BASE + segment * rows + row).to(tl.int32)
            destination = base + rank
            safe = (
                (destination >= 0) & (destination < exact_capacity) & lane
            )
            k = tl.load(source_k, mask=safe, other=0.0)
            v = tl.load(source_v, mask=safe, other=0.0)
            tl.store(EXACT_K + destination * head_dim + d, k, mask=safe)
            tl.store(EXACT_V + destination * head_dim + d, v, mask=safe)


    @triton.jit
    def _scatter_mixed_norm_kernel(
        NORM_K_TMP,
        NORM_V_TMP,
        TAGS,
        RANKS,
        NORM_ARENA_K,
        NORM_ARENA_V,
        NORM_BASE,
        ROW_OFFSET,
        SEQLEN,
        ROW_CAPACITY,
        tag: tl.constexpr,
        head_dim: tl.constexpr,
        tokens: tl.constexpr,
        HAS_NORM_BASE: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        selected = tl.load(TAGS + cell).to(tl.int32) == tag
        destination = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(SEQLEN + row).to(tl.int32)
            + tl.load(RANKS + cell).to(tl.int32)
        )
        row_end = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(ROW_CAPACITY + row).to(tl.int32)
        )
        live = selected & (destination < row_end)
        norm_base = tl.load(NORM_BASE).to(tl.int32) if HAS_NORM_BASE else 0
        work = cell * head_dim
        tl.store(
            NORM_ARENA_K + norm_base + destination,
            tl.load(NORM_K_TMP + work),
            mask=live,
        )
        tl.store(
            NORM_ARENA_V + norm_base + destination,
            tl.load(NORM_V_TMP + work),
            mask=live,
        )


    @triton.jit
    def _scatter_mixed_exact_kernel(
        K,
        V,
        TAGS,
        RANKS,
        OUT_K,
        OUT_V,
        ROW_OFFSET,
        SEQLEN,
        ROW_CAPACITY,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        token = cell - row * tokens
        batch = row // num_heads
        head = row - batch * num_heads
        d = tl.arange(0, BLOCK_D)
        destination = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(SEQLEN + row).to(tl.int32)
            + tl.load(RANKS + cell).to(tl.int32)
        )
        row_end = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(ROW_CAPACITY + row).to(tl.int32)
        )
        selected = tl.load(TAGS + cell).to(tl.int32) == 16
        live = selected & (destination < row_end) & (d < head_dim)
        k = tl.load(
            K + batch * key_stride_b + head * key_stride_h
            + token * key_stride_t + d * key_stride_d,
            mask=live,
            other=0.0,
        )
        v = tl.load(
            V + batch * value_stride_b + head * value_stride_h
            + token * value_stride_t + d * value_stride_d,
            mask=live,
            other=0.0,
        )
        tl.store(OUT_K + destination * head_dim + d, k, mask=live)
        tl.store(OUT_V + destination * head_dim + d, v, mask=live)


    @triton.jit
    def _encode_rotated_mixed_kernel(
        RK,
        RV,
        TAGS,
        RANKS,
        CENT_K,
        CENT_V,
        PACKED_K,
        PACKED_V,
        CODE_BASE,
        ROW_OFFSET,
        SEQLEN,
        ROW_CAPACITY,
        tag: tl.constexpr,
        num_centroids: tl.constexpr,
        packing_bits: tl.constexpr,
        values_per_byte: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        tokens: tl.constexpr,
        HAS_CODE_BASE: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        selected = tl.load(TAGS + cell).to(tl.int32) == tag
        destination = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(SEQLEN + row).to(tl.int32)
            + tl.load(RANKS + cell).to(tl.int32)
        )
        row_end = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(ROW_CAPACITY + row).to(tl.int32)
        )
        # A scalar program-level branch is essential for the 8-bit bank: an
        # unselected cell must not execute even the exact binary lookup under
        # masks, nor issue payload reads or stores.
        if selected & (destination < row_end):
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            live = p < width
            packed_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            packed_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, values_per_byte):
                d = p * values_per_byte + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                if num_centroids <= 16:
                    best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                    best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                    best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                    best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                    for centroid_idx in tl.static_range(0, num_centroids):
                        ck = tl.load(CENT_K + centroid_idx).to(tl.float32)
                        cv = tl.load(CENT_V + centroid_idx).to(tl.float32)
                        dk = tl.abs(kval - ck)
                        dv = tl.abs(vval - cv)
                        better_k = dk < best_k_dist
                        better_v = dv < best_v_dist
                        best_k_dist = tl.where(better_k, dk, best_k_dist)
                        best_v_dist = tl.where(better_v, dv, best_v_dist)
                        best_k = tl.where(better_k, centroid_idx, best_k)
                        best_v = tl.where(better_v, centroid_idx, best_v)
                else:
                    best_k = _nearest_sorted_centroid(
                        kval,
                        CENT_K,
                        lane_live,
                        K=num_centroids,
                        SEARCH_STEPS=packing_bits,
                    )
                    best_v = _nearest_sorted_centroid(
                        vval,
                        CENT_V,
                        lane_live,
                        K=num_centroids,
                        SEARCH_STEPS=packing_bits,
                    )
                packed_k |= best_k << (packing_bits * lane)
                packed_v |= best_v << (packing_bits * lane)
            code_base = tl.load(CODE_BASE).to(tl.int32) if HAS_CODE_BASE else 0
            code = code_base + destination * width + p
            tl.store(PACKED_K + code, packed_k, mask=live)
            tl.store(PACKED_V + code, packed_v, mask=live)


    @triton.jit
    def _encode_rotated_mixed_3bit_kernel(
        RK,
        RV,
        TAGS,
        RANKS,
        CENT_K,
        CENT_V,
        PACKED_K,
        PACKED_V,
        CODE_BASE,
        ROW_OFFSET,
        SEQLEN,
        ROW_CAPACITY,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        tokens: tl.constexpr,
        HAS_CODE_BASE: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """Append true 3-bit codes directly into the fixed-address arena."""
        cell = tl.program_id(0)
        row = cell // tokens
        selected = tl.load(TAGS + cell).to(tl.int32) == 3
        destination = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(SEQLEN + row).to(tl.int32)
            + tl.load(RANKS + cell).to(tl.int32)
        )
        row_end = (
            tl.load(ROW_OFFSET + row).to(tl.int32)
            + tl.load(ROW_CAPACITY + row).to(tl.int32)
        )
        if selected & (destination < row_end):
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            live = p < width
            base_bit = p * 8
            first_d = base_bit // 3
            bit_offset = base_bit - first_d * 3
            word_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            word_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = first_d + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                for centroid_idx in tl.static_range(0, 8):
                    ck = tl.load(CENT_K + centroid_idx).to(tl.float32)
                    cv = tl.load(CENT_V + centroid_idx).to(tl.float32)
                    dk = tl.abs(kval - ck)
                    dv = tl.abs(vval - cv)
                    better_k = lane_live & (dk < best_k_dist)
                    better_v = lane_live & (dv < best_v_dist)
                    best_k_dist = tl.where(better_k, dk, best_k_dist)
                    best_v_dist = tl.where(better_v, dv, best_v_dist)
                    best_k = tl.where(better_k, centroid_idx, best_k)
                    best_v = tl.where(better_v, centroid_idx, best_v)
                word_k |= best_k << (3 * lane)
                word_v |= best_v << (3 * lane)
            packed_k = (word_k >> bit_offset) & 0xFF
            packed_v = (word_v >> bit_offset) & 0xFF
            code_base = tl.load(CODE_BASE).to(tl.int32) if HAS_CODE_BASE else 0
            code = code_base + destination * width + p
            tl.store(PACKED_K + code, packed_k, mask=live)
            tl.store(PACKED_V + code, packed_v, mask=live)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _encode_segment_mixed_kernel(
        RK,
        RV,
        TAGS,
        RANKS,
        CENT_K,
        CENT_V,
        PACKED_K,
        PACKED_V,
        CODE_ROW_BASE,
        tag: tl.constexpr,
        level_index: tl.constexpr,
        num_centroids: tl.constexpr,
        packing_bits: tl.constexpr,
        values_per_byte: tl.constexpr,
        segment,
        rows: tl.constexpr,
        code_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        selected = tl.load(TAGS + cell).to(tl.int32) == tag
        if selected:
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + level_index * rows + row,
            ).to(tl.int32)
            target = base + tl.load(RANKS + cell).to(tl.int32) * width + p
            live = (
                (p < width) & (target >= 0) & (target < code_capacity)
            )
            packed_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            packed_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, values_per_byte):
                d = p * values_per_byte + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d,
                    mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d,
                    mask=lane_live, other=0.0,
                ).to(tl.float32)
                if num_centroids <= 16:
                    best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                    best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                    best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                    best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                    for centroid_idx in tl.static_range(0, num_centroids):
                        ck = tl.load(CENT_K + centroid_idx).to(tl.float32)
                        cv = tl.load(CENT_V + centroid_idx).to(tl.float32)
                        dk = tl.abs(kval - ck)
                        dv = tl.abs(vval - cv)
                        better_k = lane_live & (dk < best_k_dist)
                        better_v = lane_live & (dv < best_v_dist)
                        best_k_dist = tl.where(better_k, dk, best_k_dist)
                        best_v_dist = tl.where(better_v, dv, best_v_dist)
                        best_k = tl.where(better_k, centroid_idx, best_k)
                        best_v = tl.where(better_v, centroid_idx, best_v)
                else:
                    best_k = _nearest_sorted_centroid(
                        kval,
                        CENT_K,
                        lane_live,
                        K=num_centroids,
                        SEARCH_STEPS=packing_bits,
                    )
                    best_v = _nearest_sorted_centroid(
                        vval,
                        CENT_V,
                        lane_live,
                        K=num_centroids,
                        SEARCH_STEPS=packing_bits,
                    )
                packed_k |= best_k << (packing_bits * lane)
                packed_v |= best_v << (packing_bits * lane)
            tl.store(PACKED_K + target, packed_k, mask=live)
            tl.store(PACKED_V + target, packed_v, mask=live)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _encode_segment_3bit_kernel(
        RK,
        RV,
        TAGS,
        RANKS,
        CENT_K,
        CENT_V,
        PACKED_K,
        PACKED_V,
        CODE_ROW_BASE,
        segment,
        rows: tl.constexpr,
        code_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        cell = tl.program_id(0)
        row = cell // tokens
        selected = tl.load(TAGS + cell).to(tl.int32) == 3
        if selected:
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + 1 * rows + row,
            ).to(tl.int32)
            target = base + tl.load(RANKS + cell).to(tl.int32) * width + p
            live = (
                (p < width) & (target >= 0) & (target < code_capacity)
            )
            base_bit = p * 8
            first_d = base_bit // 3
            bit_offset = base_bit - first_d * 3
            word_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            word_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = first_d + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d,
                    mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d,
                    mask=lane_live, other=0.0,
                ).to(tl.float32)
                best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                for centroid_idx in tl.static_range(0, 8):
                    ck = tl.load(CENT_K + centroid_idx).to(tl.float32)
                    cv = tl.load(CENT_V + centroid_idx).to(tl.float32)
                    dk = tl.abs(kval - ck)
                    dv = tl.abs(vval - cv)
                    better_k = lane_live & (dk < best_k_dist)
                    better_v = lane_live & (dv < best_v_dist)
                    best_k_dist = tl.where(better_k, dk, best_k_dist)
                    best_v_dist = tl.where(better_v, dv, best_v_dist)
                    best_k = tl.where(better_k, centroid_idx, best_k)
                    best_v = tl.where(better_v, centroid_idx, best_v)
                word_k |= best_k << (3 * lane)
                word_v |= best_v << (3 * lane)
            tl.store(PACKED_K + target, (word_k >> bit_offset) & 0xFF, mask=live)
            tl.store(PACKED_V + target, (word_v >> bit_offset) & 0xFF, mask=live)


    @triton.jit(
        do_not_specialize=["segment"],
        do_not_specialize_on_alignment=["segment"],
    )
    def _encode_segment_all_levels_kernel(
        RK,
        RV,
        TAGS,
        RANKS,
        CENT2_K,
        CENT2_V,
        CENT3_K,
        CENT3_V,
        CENT4_K,
        CENT4_V,
        CENT8_K,
        CENT8_V,
        PACKED_K,
        PACKED_V,
        CODE_ROW_BASE,
        segment,
        rows: tl.constexpr,
        code_capacity: tl.constexpr,
        head_dim: tl.constexpr,
        W2: tl.constexpr,
        W3: tl.constexpr,
        W4: tl.constexpr,
        W8: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """Encode every quant level with one tag-dispatched launch.

        The previous segmented path launched four grids over all logical
        cells (five cell-grids for D=128 because 8-bit needs two width tiles).
        This kernel launches two width tiles once and branches per program on
        the scalar token tag.  Tags 0 and 16 execute no centroid or payload
        work.  Each branch deliberately retains the frozen level-specific
        packing and lower-index tie rule, including the true three-byte/eight-
        code representation for 3-bit values.
        """
        cell = tl.program_id(0)
        width_block = tl.program_id(1)
        row = cell // tokens
        tag = tl.load(TAGS + cell).to(tl.int32)
        rank = tl.load(RANKS + cell).to(tl.int32)
        p = width_block * BLOCK_P + tl.arange(0, BLOCK_P)

        if (tag == 2) & (width_block == 0):
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + 0 * rows + row,
            ).to(tl.int32)
            target = base + rank * W2 + p
            live = (p < W2) & (target >= 0) & (target < code_capacity)
            packed_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            packed_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = p * 4 + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                for centroid_idx in tl.static_range(0, 4):
                    ck = tl.load(CENT2_K + centroid_idx).to(tl.float32)
                    cv = tl.load(CENT2_V + centroid_idx).to(tl.float32)
                    dk = tl.abs(kval - ck)
                    dv = tl.abs(vval - cv)
                    better_k = lane_live & (dk < best_k_dist)
                    better_v = lane_live & (dv < best_v_dist)
                    best_k_dist = tl.where(better_k, dk, best_k_dist)
                    best_v_dist = tl.where(better_v, dv, best_v_dist)
                    best_k = tl.where(better_k, centroid_idx, best_k)
                    best_v = tl.where(better_v, centroid_idx, best_v)
                packed_k |= best_k << (2 * lane)
                packed_v |= best_v << (2 * lane)
            tl.store(PACKED_K + target, packed_k, mask=live)
            tl.store(PACKED_V + target, packed_v, mask=live)

        elif (tag == 3) & (width_block == 0):
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + 1 * rows + row,
            ).to(tl.int32)
            target = base + rank * W3 + p
            live = (p < W3) & (target >= 0) & (target < code_capacity)
            base_bit = p * 8
            first_d = base_bit // 3
            bit_offset = base_bit - first_d * 3
            word_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            word_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = first_d + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                for centroid_idx in tl.static_range(0, 8):
                    ck = tl.load(CENT3_K + centroid_idx).to(tl.float32)
                    cv = tl.load(CENT3_V + centroid_idx).to(tl.float32)
                    dk = tl.abs(kval - ck)
                    dv = tl.abs(vval - cv)
                    better_k = lane_live & (dk < best_k_dist)
                    better_v = lane_live & (dv < best_v_dist)
                    best_k_dist = tl.where(better_k, dk, best_k_dist)
                    best_v_dist = tl.where(better_v, dv, best_v_dist)
                    best_k = tl.where(better_k, centroid_idx, best_k)
                    best_v = tl.where(better_v, centroid_idx, best_v)
                word_k |= best_k << (3 * lane)
                word_v |= best_v << (3 * lane)
            tl.store(
                PACKED_K + target, (word_k >> bit_offset) & 0xFF, mask=live,
            )
            tl.store(
                PACKED_V + target, (word_v >> bit_offset) & 0xFF, mask=live,
            )

        elif (tag == 4) & (width_block == 0):
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + 2 * rows + row,
            ).to(tl.int32)
            target = base + rank * W4 + p
            live = (p < W4) & (target >= 0) & (target < code_capacity)
            packed_k = tl.zeros([BLOCK_P], dtype=tl.int32)
            packed_v = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 2):
                d = p * 2 + lane
                lane_live = live & (d < head_dim)
                kval = tl.load(
                    RK + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                vval = tl.load(
                    RV + cell * head_dim + d, mask=lane_live, other=0.0,
                ).to(tl.float32)
                best_k_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_v_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_k = tl.zeros([BLOCK_P], dtype=tl.int32)
                best_v = tl.zeros([BLOCK_P], dtype=tl.int32)
                for centroid_idx in tl.static_range(0, 16):
                    ck = tl.load(CENT4_K + centroid_idx).to(tl.float32)
                    cv = tl.load(CENT4_V + centroid_idx).to(tl.float32)
                    dk = tl.abs(kval - ck)
                    dv = tl.abs(vval - cv)
                    better_k = lane_live & (dk < best_k_dist)
                    better_v = lane_live & (dv < best_v_dist)
                    best_k_dist = tl.where(better_k, dk, best_k_dist)
                    best_v_dist = tl.where(better_v, dv, best_v_dist)
                    best_k = tl.where(better_k, centroid_idx, best_k)
                    best_v = tl.where(better_v, centroid_idx, best_v)
                packed_k |= best_k << (4 * lane)
                packed_v |= best_v << (4 * lane)
            tl.store(PACKED_K + target, packed_k, mask=live)
            tl.store(PACKED_V + target, packed_v, mask=live)

        elif tag == 8:
            base = tl.load(
                CODE_ROW_BASE + segment * 4 * rows + 3 * rows + row,
            ).to(tl.int32)
            target = base + rank * W8 + p
            live = (p < W8) & (target >= 0) & (target < code_capacity)
            d = p
            kval = tl.load(
                RK + cell * head_dim + d, mask=live & (d < head_dim), other=0.0,
            ).to(tl.float32)
            vval = tl.load(
                RV + cell * head_dim + d, mask=live & (d < head_dim), other=0.0,
            ).to(tl.float32)
            best_k = _nearest_sorted_centroid(
                kval, CENT8_K, live, K=256, SEARCH_STEPS=8,
            )
            best_v = _nearest_sorted_centroid(
                vval, CENT8_V, live, K=256, SEARCH_STEPS=8,
            )
            tl.store(PACKED_K + target, best_k, mask=live)
            tl.store(PACKED_V + target, best_v, mask=live)


    @triton.jit(
        do_not_specialize=["decode_start"],
        do_not_specialize_on_alignment=["decode_start"],
    )
    def _publish_mixed_decode_kernel(
        TAGS_IN,
        COUNTS,
        SEQLEN_2,
        SEQLEN_3,
        SEQLEN_4,
        SEQLEN_8,
        SEQLEN_16,
        TAGS_OUT,
        decode_start,
        decode_capacity: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        rows = tl.num_programs(0)
        tl.store(SEQLEN_2 + row, tl.load(SEQLEN_2 + row) + tl.load(COUNTS + 0 * rows + row))
        tl.store(SEQLEN_3 + row, tl.load(SEQLEN_3 + row) + tl.load(COUNTS + 1 * rows + row))
        tl.store(SEQLEN_4 + row, tl.load(SEQLEN_4 + row) + tl.load(COUNTS + 2 * rows + row))
        tl.store(SEQLEN_8 + row, tl.load(SEQLEN_8 + row) + tl.load(COUNTS + 3 * rows + row))
        tl.store(SEQLEN_16 + row, tl.load(SEQLEN_16 + row) + tl.load(COUNTS + 4 * rows + row))
        token = tl.arange(0, BLOCK_T)
        live = token < tokens
        tags = tl.load(TAGS_IN + row * tokens + token, mask=live, other=0)
        tl.store(
            TAGS_OUT + row * decode_capacity + decode_start + token,
            tags,
            mask=live,
        )


def _validate_append(packed: Any, keys: torch.Tensor, values: torch.Tensor) -> tuple[dict[str, Any], int, int, int, int]:
    if not _HAS_TRITON or not keys.is_cuda:
        raise RuntimeError("native decode flush requires CUDA and Triton")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("decode K/V must be matching [B,H,T,D]")
    batch, heads, tokens, head_dim = keys.shape
    if (batch, heads, head_dim) != (
        packed.batch_size,
        packed.num_heads,
        packed.head_dim,
    ):
        raise ValueError("decode K/V shape does not match packed prefill")
    if keys.device != values.device or keys.device != packed.tags.device:
        raise ValueError("decode K/V and packed arena must share a CUDA device")
    reserve = int(packed.capacity.decode_2bit_reserve_per_row)
    if packed.decode_tags is None or reserve <= 0:
        raise RuntimeError("2-bit arena has no decode reserve")
    if packed.decode_len + tokens > reserve:
        raise RuntimeError("2-bit decode reserve exceeded")
    bank = packed.bank(2)
    required = ("stable_row_capacity",)
    if any(field not in bank for field in required):
        raise RuntimeError("native direct flush requires a fixed per-row reserve layout")
    return bank, batch, heads, tokens, head_dim


@torch.no_grad()
def append_decode_2bit_workspace(
    packed: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    workspace: NativeFlushWorkspace,
) -> None:
    """Append a strided ring prefix using reusable scratch plus cuBLAS."""
    bank, batch, heads, tokens, head_dim = _validate_append(packed, keys, values)
    rows = batch * heads
    workspace.validate(rows=rows, tokens=tokens, head_dim=head_dim, device=keys.device)
    if tokens == 0:
        return
    reserve = int(packed.capacity.decode_2bit_reserve_per_row)
    decode_start = int(packed.decode_len)
    count = rows * tokens
    normalized_k = workspace.normalized_k.narrow(0, 0, count)
    normalized_v = workspace.normalized_v.narrow(0, 0, count)
    rotated_k = workspace.rotated_k.narrow(0, 0, count)
    rotated_v = workspace.rotated_v.narrow(0, 0, count)
    has_shared_base = "code_base" in bank
    code_base = bank["code_base"] if has_shared_base else bank["offset"]
    norm_base = bank["norm_base"] if has_shared_base else bank["offset"]
    block_d = triton.next_power_of_2(head_dim)
    _normalize_kv_ring_kernel[(rows, tokens)](
        keys,
        values,
        normalized_k,
        normalized_v,
        bank["norms_k"],
        bank["norms_v"],
        norm_base,
        bank["offset"],
        bank["stable_row_capacity"],
        decode_start,
        reserve=reserve,
        num_heads=heads,
        head_dim=head_dim,
        key_stride_b=keys.stride(0),
        key_stride_h=keys.stride(1),
        key_stride_t=keys.stride(2),
        key_stride_d=keys.stride(3),
        value_stride_b=values.stride(0),
        value_stride_h=values.stride(1),
        value_stride_t=values.stride(2),
        value_stride_d=values.stride(3),
        tokens=tokens,
        HAS_NORM_BASE=has_shared_base,
        BLOCK_D=block_d,
        num_warps=4,
    )
    torch.mm(normalized_k, packed.pi_k.T, out=rotated_k)
    torch.mm(normalized_v, packed.pi_v.T, out=rotated_v)
    width = int(bank.get("packed_width", bank["packed_k"].shape[-1]))
    _encode_rotated_kv_2bit_kernel[(rows, tokens)](
        rotated_k,
        rotated_v,
        bank["cent_k"],
        bank["cent_v"],
        bank["packed_k"],
        bank["packed_v"],
        code_base,
        bank["offset"],
        bank["stable_row_capacity"],
        decode_start,
        reserve=reserve,
        head_dim=head_dim,
        width=width,
        tokens=tokens,
        HAS_CODE_BASE=has_shared_base,
        BLOCK_P=triton.next_power_of_2(width),
        num_warps=4,
    )
    _publish_decode_suffix_kernel[(rows,)](
        bank["seqlen"],
        bank["stable_row_capacity"],
        packed.decode_tags,
        decode_start,
        tokens=tokens,
        reserve=reserve,
        BLOCK_T=triton.next_power_of_2(tokens),
        num_warps=4,
    )
    packed.decode_len += int(tokens)


def _validate_mixed_append(
    packed: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not _HAS_TRITON or not keys.is_cuda:
        raise RuntimeError("native mixed-bit decode flush requires CUDA and Triton")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("decode K/V must be matching [B,H,T,D]")
    batch, heads, tokens, head_dim = keys.shape
    if (batch, heads, head_dim) != (
        packed.batch_size,
        packed.num_heads,
        packed.head_dim,
    ):
        raise ValueError("decode K/V shape does not match packed prefill")
    if bits.shape != keys.shape[:3]:
        raise ValueError(
            f"decode bit tags must have shape {tuple(keys.shape[:3])}, got {tuple(bits.shape)}"
        )
    if bits.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise TypeError(f"decode bit tags must be an integer tensor, got {bits.dtype}")
    if keys.device != values.device or keys.device != bits.device:
        raise ValueError("decode K/V, bit tags, and packed arena must share a CUDA device")
    arena = getattr(packed, "decode_arena", None)
    capacity = (
        int(arena.capacity_per_row)
        if arena is not None else int(packed.capacity.decode_capacity_per_row)
    )
    if packed.decode_tags is None or capacity <= 0:
        raise RuntimeError("native mixed-bit arenas have no decode reserve")
    if packed.decode_len + tokens > capacity:
        raise RuntimeError(
            f"native decode reserve exceeded: need={packed.decode_len + tokens}, "
            f"capacity={capacity}"
        )
    if arena is None:
        for bank in (*packed.quant_banks, packed.exact):
            if "stable_row_capacity" not in bank:
                raise RuntimeError(
                    "native mixed-bit flush requires fixed per-row reserve descriptors"
                )
    return batch, heads, tokens, head_dim


def _append_decode_segmented_payload(
    packed: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    tags: torch.Tensor,
    ranks: torch.Tensor,
    counts: torch.Tensor,
    workspace: NativeFlushWorkspace,
    *,
    batch: int,
    heads: int,
    tokens: int,
    head_dim: int,
) -> None:
    arena = packed.decode_arena
    rows = batch * heads
    count = rows * tokens
    segment = arena.validate_append(tokens, packed.decode_len)
    widths = tuple(
        int(bank.get("packed_width", bank["packed_k"].shape[-1]))
        for bank in packed.quant_banks
    )
    _allocate_decode_segment_kernel[(1,)](
        counts,
        arena.code_row_base,
        arena.norm_row_base,
        arena.exact_row_base,
        arena.counts,
        arena.bump,
        arena.overflow_flag,
        segment=segment,
        rows=rows,
        code_capacity=arena.code_capacity,
        norm_capacity=arena.norm_capacity,
        exact_capacity=arena.exact_capacity,
        slot_bytes=arena.slot_bytes,
        W2=widths[0],
        W3=widths[1],
        W4=widths[2],
        W8=widths[3],
        BLOCK=triton.next_power_of_2(5 * rows),
        num_warps=4,
    )
    # Allocation and every payload publication below are independently
    # bounds-masked.  Keep one sticky device failure flag and validate it once
    # after publication instead of launching three separate eq/assert pairs.

    import os as _os
    use_fused_prepare = _os.environ.get("R2_FLUSH_FUSED_PREPARE", "1") != "0"
    normalized_k = workspace.normalized_k.narrow(0, 0, count)
    normalized_v = workspace.normalized_v.narrow(0, 0, count)
    rotated_k = workspace.rotated_k.narrow(0, 0, count)
    rotated_v = workspace.rotated_v.narrow(0, 0, count)
    block_d = triton.next_power_of_2(head_dim)
    if use_fused_prepare:
        _prepare_segment_payload_kernel[(rows, tokens)](
            keys,
            values,
            tags,
            ranks,
            normalized_k,
            normalized_v,
            arena.norm_arena_k,
            arena.norm_arena_v,
            arena.norm_row_base,
            arena.exact_k,
            arena.exact_v,
            arena.exact_row_base,
            segment=segment,
            rows=rows,
            norm_capacity=arena.norm_capacity,
            exact_capacity=arena.exact_capacity,
            num_heads=heads,
            head_dim=head_dim,
            key_stride_b=keys.stride(0),
            key_stride_h=keys.stride(1),
            key_stride_t=keys.stride(2),
            key_stride_d=keys.stride(3),
            value_stride_b=values.stride(0),
            value_stride_h=values.stride(1),
            value_stride_t=values.stride(2),
            value_stride_d=values.stride(3),
            tokens=tokens,
            BLOCK_D=block_d,
            num_warps=4,
        )
    else:
        _normalize_mixed_kv_kernel[(rows, tokens)](
            keys,
            values,
            normalized_k,
            normalized_v,
            rotated_k,
            rotated_v,
            num_heads=heads,
            head_dim=head_dim,
            key_stride_b=keys.stride(0),
            key_stride_h=keys.stride(1),
            key_stride_t=keys.stride(2),
            key_stride_d=keys.stride(3),
            value_stride_b=values.stride(0),
            value_stride_h=values.stride(1),
            value_stride_t=values.stride(2),
            value_stride_d=values.stride(3),
            tokens=tokens,
            BLOCK_D=block_d,
            num_warps=4,
        )
        _scatter_segment_norm_kernel[(count,)](
            rotated_k,
            rotated_v,
            tags,
            ranks,
            arena.norm_arena_k,
            arena.norm_arena_v,
            arena.norm_row_base,
            segment=segment,
            rows=rows,
            norm_capacity=arena.norm_capacity,
            head_dim=head_dim,
            tokens=tokens,
            num_warps=4,
        )
        _scatter_segment_exact_kernel[(count,)](
            keys,
            values,
            tags,
            ranks,
            arena.exact_k,
            arena.exact_v,
            arena.exact_row_base,
            segment=segment,
            rows=rows,
            exact_capacity=arena.exact_capacity,
            num_heads=heads,
            head_dim=head_dim,
            key_stride_b=keys.stride(0),
            key_stride_h=keys.stride(1),
            key_stride_t=keys.stride(2),
            key_stride_d=keys.stride(3),
            value_stride_b=values.stride(0),
            value_stride_h=values.stride(1),
            value_stride_t=values.stride(2),
            value_stride_d=values.stride(3),
            tokens=tokens,
            BLOCK_D=block_d,
            num_warps=4,
        )

    torch.mm(normalized_k, packed.pi_k.T, out=rotated_k)
    torch.mm(normalized_v, packed.pi_v.T, out=rotated_v)
    unified_setting = _os.environ.get("R2_FLUSH_UNIFIED_ENCODE", "auto").lower()
    if unified_setting not in {"auto", "0", "1"}:
        raise ValueError(
            "R2_FLUSH_UNIFIED_ENCODE must be one of {'auto','0','1'}"
        )
    # Paired, alternating CUDA-event admission on the production target-1/2/4
    # histograms shows the one-launch kernel wins at every density.  ``auto``
    # therefore selects it without reading tags back to the host; ``0`` remains
    # only as a byte-exact regression oracle and diagnostic escape hatch.
    use_unified_encode = unified_setting in {"auto", "1"}
    if use_unified_encode:
        block_p = min(64, triton.next_power_of_2(max(widths)))
        _encode_segment_all_levels_kernel[
            (count, triton.cdiv(max(widths), block_p))
        ](
            rotated_k,
            rotated_v,
            tags,
            ranks,
            packed.quant_banks[0]["cent_k"],
            packed.quant_banks[0]["cent_v"],
            packed.quant_banks[1]["cent_k"],
            packed.quant_banks[1]["cent_v"],
            packed.quant_banks[2]["cent_k"],
            packed.quant_banks[2]["cent_v"],
            packed.quant_banks[3]["cent_k"],
            packed.quant_banks[3]["cent_v"],
            arena.code_arena_k,
            arena.code_arena_v,
            arena.code_row_base,
            segment=segment,
            rows=rows,
            code_capacity=arena.code_capacity,
            head_dim=head_dim,
            W2=widths[0],
            W3=widths[1],
            W4=widths[2],
            W8=widths[3],
            tokens=tokens,
            BLOCK_P=block_p,
            num_warps=4,
        )
    else:
        for level_index, (bank, level, width) in enumerate(
            zip(packed.quant_banks, (2, 3, 4, 8), widths)
        ):
            block_p = min(64, triton.next_power_of_2(width))
            grid = (count, triton.cdiv(width, block_p))
            if level == 3:
                _encode_segment_3bit_kernel[grid](
                    rotated_k,
                    rotated_v,
                    tags,
                    ranks,
                    bank["cent_k"],
                    bank["cent_v"],
                    arena.code_arena_k,
                    arena.code_arena_v,
                    arena.code_row_base,
                    segment=segment,
                    rows=rows,
                    code_capacity=arena.code_capacity,
                    head_dim=head_dim,
                    width=width,
                    tokens=tokens,
                    BLOCK_P=block_p,
                    num_warps=4,
                )
            else:
                packing_bits = 2 if level == 2 else 4 if level == 4 else 8
                _encode_segment_mixed_kernel[grid](
                    rotated_k,
                    rotated_v,
                    tags,
                    ranks,
                    bank["cent_k"],
                    bank["cent_v"],
                    arena.code_arena_k,
                    arena.code_arena_v,
                    arena.code_row_base,
                    tag=level,
                    level_index=level_index,
                    num_centroids=1 << level,
                    packing_bits=packing_bits,
                    values_per_byte=8 // packing_bits,
                    segment=segment,
                    rows=rows,
                    code_capacity=arena.code_capacity,
                    head_dim=head_dim,
                    width=width,
                    tokens=tokens,
                    BLOCK_P=block_p,
                    num_warps=4,
                )

    _publish_decode_descriptors_kernel[(rows,)](
        tags,
        ranks,
        counts,
        arena.descriptors,
        arena.descriptor_counts,
        packed.decode_tags,
        arena.overflow_flag,
        int(packed.decode_len),
        segment=segment,
        rows=rows,
        capacity=arena.capacity_per_row,
        descriptor_stride=arena.buffer_size,
        tokens=tokens,
        BLOCK_T=triton.next_power_of_2(tokens),
        num_warps=4,
    )
    torch._assert_async(
        arena.overflow_flag == 0,
        "native mixed-bit decode validation failed: invalid tag or arena "
        "capacity exceeded",
    )

    decode_start = int(packed.decode_len)
    arena.publish_host_segment(decode_start, tokens)
    packed.decode_len += int(tokens)


@torch.no_grad()
def append_decode_mixed_workspace(
    packed: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    bits: torch.Tensor,
    workspace: NativeFlushWorkspace,
) -> None:
    """Append allocator-selected ``0/2/3/4/8/16`` payloads in place.

    The allocation stays on CUDA.  A row-local rank maps each token directly
    to its bank's reserved CSR suffix; no host histogram, dynamic gather, bank
    rebuild, or graph recapture is involved. A 0-bit token has no rank in any
    payload bank, so it advances ``decode_len`` but causes no K/V store.
    """
    batch, heads, tokens, head_dim = _validate_mixed_append(
        packed, keys, values, bits,
    )
    rows = batch * heads
    workspace.validate(rows=rows, tokens=tokens, head_dim=head_dim, device=keys.device)
    if tokens == 0:
        return

    tags = bits.contiguous()
    count = rows * tokens
    ranks = workspace.ranks.narrow(0, 0, count)
    counts = workspace.counts
    arena = getattr(packed, "decode_arena", None)
    if arena is None:
        workspace.status.zero_()
        invalid = workspace.status.narrow(0, 0, 1)
        overflow = workspace.status.narrow(0, 1, 1)
    else:
        # The aggregate arena already owns a sticky, initially-zero device
        # failure flag.  Rank validation, allocation overflow, and descriptor
        # overflow all OR into it, so the segmented path needs neither a status
        # clear launch nor an intermediate synchronization assertion.
        invalid = arena.overflow_flag
        overflow = None
    block_t = triton.next_power_of_2(tokens)
    _rank_mixed_decode_kernel[(rows,)](
        tags,
        ranks,
        counts,
        invalid,
        tokens=tokens,
        BLOCK_T=block_t,
        num_warps=4,
    )
    if arena is not None:
        _append_decode_segmented_payload(
            packed,
            keys,
            values,
            tags,
            ranks,
            counts,
            workspace,
            batch=batch,
            heads=heads,
            tokens=tokens,
            head_dim=head_dim,
        )
        return
    torch._assert_async(
        invalid == 0,
        "native decode bit tags must be one of {0,2,3,4,8,16}",
    )
    banks = packed.quant_banks
    exact = packed.exact
    _validate_mixed_capacity_kernel[(rows,)](
        counts,
        banks[0]["seqlen"],
        banks[0]["stable_row_capacity"],
        banks[1]["seqlen"],
        banks[1]["stable_row_capacity"],
        banks[2]["seqlen"],
        banks[2]["stable_row_capacity"],
        banks[3]["seqlen"],
        banks[3]["stable_row_capacity"],
        exact["seqlen"],
        exact["stable_row_capacity"],
        overflow,
    )
    torch._assert_async(
        overflow == 0,
        "native mixed-bit per-row decode reserve exceeded",
    )

    normalized_k = workspace.normalized_k.narrow(0, 0, count)
    normalized_v = workspace.normalized_v.narrow(0, 0, count)
    rotated_k = workspace.rotated_k.narrow(0, 0, count)
    rotated_v = workspace.rotated_v.narrow(0, 0, count)
    block_d = triton.next_power_of_2(head_dim)
    _normalize_mixed_kv_kernel[(rows, tokens)](
        keys,
        values,
        normalized_k,
        normalized_v,
        rotated_k,
        rotated_v,
        num_heads=heads,
        head_dim=head_dim,
        key_stride_b=keys.stride(0),
        key_stride_h=keys.stride(1),
        key_stride_t=keys.stride(2),
        key_stride_d=keys.stride(3),
        value_stride_b=values.stride(0),
        value_stride_h=values.stride(1),
        value_stride_t=values.stride(2),
        value_stride_d=values.stride(3),
        tokens=tokens,
        BLOCK_D=block_d,
        num_warps=4,
    )

    for bank, level in zip(banks, (2, 3, 4, 8)):
        shared = "norm_base" in bank
        norm_base = bank["norm_base"] if shared else bank["offset"]
        _scatter_mixed_norm_kernel[(count,)](
            rotated_k,
            rotated_v,
            tags,
            ranks,
            bank["norms_k"],
            bank["norms_v"],
            norm_base,
            bank["offset"],
            bank["seqlen"],
            bank["stable_row_capacity"],
            tag=level,
            head_dim=head_dim,
            tokens=tokens,
            HAS_NORM_BASE=shared,
            num_warps=4,
        )
    _scatter_mixed_exact_kernel[(count,)](
        keys,
        values,
        tags,
        ranks,
        exact["keys"],
        exact["values"],
        exact["offset"],
        exact["seqlen"],
        exact["stable_row_capacity"],
        num_heads=heads,
        head_dim=head_dim,
        key_stride_b=keys.stride(0),
        key_stride_h=keys.stride(1),
        key_stride_t=keys.stride(2),
        key_stride_d=keys.stride(3),
        value_stride_b=values.stride(0),
        value_stride_h=values.stride(1),
        value_stride_t=values.stride(2),
        value_stride_d=values.stride(3),
        tokens=tokens,
        BLOCK_D=block_d,
        num_warps=4,
    )

    torch.mm(normalized_k, packed.pi_k.T, out=rotated_k)
    torch.mm(normalized_v, packed.pi_v.T, out=rotated_v)
    for bank, level in zip(banks, (2, 3, 4, 8)):
        width = int(bank.get("packed_width", bank["packed_k"].shape[-1]))
        block_p = min(64, triton.next_power_of_2(width))
        shared = "code_base" in bank
        code_base = bank["code_base"] if shared else bank["offset"]
        physical_bits = int(bank.get("physical_bits", 4 if level == 3 else level))
        if level == 3 and physical_bits == 3:
            _encode_rotated_mixed_3bit_kernel[
                (count, triton.cdiv(width, block_p))
            ](
                rotated_k,
                rotated_v,
                tags,
                ranks,
                bank["cent_k"],
                bank["cent_v"],
                bank["packed_k"],
                bank["packed_v"],
                code_base,
                bank["offset"],
                bank["seqlen"],
                bank["stable_row_capacity"],
                head_dim=head_dim,
                width=width,
                tokens=tokens,
                HAS_CODE_BASE=shared,
                BLOCK_P=block_p,
                num_warps=4,
            )
        else:
            packing_bits = 2 if level == 2 else 4 if level in (3, 4) else 8
            values_per_byte = 8 // packing_bits
            _encode_rotated_mixed_kernel[(count, triton.cdiv(width, block_p))](
                rotated_k,
                rotated_v,
                tags,
                ranks,
                bank["cent_k"],
                bank["cent_v"],
                bank["packed_k"],
                bank["packed_v"],
                code_base,
                bank["offset"],
                bank["seqlen"],
                bank["stable_row_capacity"],
                tag=level,
                num_centroids=1 << level,
                packing_bits=packing_bits,
                values_per_byte=values_per_byte,
                head_dim=head_dim,
                width=width,
                tokens=tokens,
                HAS_CODE_BASE=shared,
                BLOCK_P=block_p,
                num_warps=4,
            )

    decode_start = int(packed.decode_len)
    decode_capacity = int(packed.capacity.decode_capacity_per_row)
    _publish_mixed_decode_kernel[(rows,)](
        tags,
        counts,
        banks[0]["seqlen"],
        banks[1]["seqlen"],
        banks[2]["seqlen"],
        banks[3]["seqlen"],
        exact["seqlen"],
        packed.decode_tags,
        decode_start,
        decode_capacity=decode_capacity,
        tokens=tokens,
        BLOCK_T=block_t,
        num_warps=4,
    )
    packed.decode_len += int(tokens)


@torch.no_grad()
def append_decode_2bit_fused(
    packed: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    block_m: int = 16,
    ieee: bool = True,
) -> None:
    """Three-launch direct ring-to-reserve append without payload scratch."""
    bank, batch, heads, tokens, head_dim = _validate_append(packed, keys, values)
    if tokens == 0:
        return
    if head_dim % 4:
        raise ValueError("fused 2-bit flush requires head_dim divisible by four")
    rows = batch * heads
    reserve = int(packed.capacity.decode_2bit_reserve_per_row)
    decode_start = int(packed.decode_len)
    total_tokens = rows * tokens
    width = int(bank.get("packed_width", bank["packed_k"].shape[-1]))
    has_shared_base = "code_base" in bank
    code_base = bank["code_base"] if has_shared_base else bank["offset"]
    norm_base = bank["norm_base"] if has_shared_base else bank["offset"]
    grid = (triton.cdiv(total_tokens, int(block_m)),)
    common = dict(
        CODE_BASE=code_base,
        NORM_BASE=norm_base,
        ROW_OFFSET=bank["offset"],
        ROW_CAPACITY=bank["stable_row_capacity"],
        decode_start=decode_start,
        total_tokens=total_tokens,
        reserve=reserve,
        num_heads=heads,
        head_dim=head_dim,
        width=width,
        tokens=tokens,
        BLOCK_M=int(block_m),
        HAS_CODE_BASE=has_shared_base,
        HAS_NORM_BASE=has_shared_base,
        IEEE=bool(ieee),
        num_warps=4,
    )
    _rotate_encode_2bit_kernel[grid](
        keys,
        packed.pi_k,
        bank["cent_k"],
        bank["packed_k"],
        bank["norms_k"],
        source_stride_b=keys.stride(0),
        source_stride_h=keys.stride(1),
        source_stride_t=keys.stride(2),
        source_stride_d=keys.stride(3),
        rotation_stride_0=packed.pi_k.stride(0),
        rotation_stride_1=packed.pi_k.stride(1),
        **common,
    )
    _rotate_encode_2bit_kernel[grid](
        values,
        packed.pi_v,
        bank["cent_v"],
        bank["packed_v"],
        bank["norms_v"],
        source_stride_b=values.stride(0),
        source_stride_h=values.stride(1),
        source_stride_t=values.stride(2),
        source_stride_d=values.stride(3),
        rotation_stride_0=packed.pi_v.stride(0),
        rotation_stride_1=packed.pi_v.stride(1),
        **common,
    )
    _publish_decode_suffix_kernel[(rows,)](
        bank["seqlen"],
        bank["stable_row_capacity"],
        packed.decode_tags,
        decode_start,
        tokens=tokens,
        reserve=reserve,
        BLOCK_T=triton.next_power_of_2(tokens),
        num_warps=4,
    )
    packed.decode_len += int(tokens)


__all__ = [
    "NativeFlushWorkspace",
    "append_decode_2bit_fused",
    "append_decode_2bit_workspace",
    "append_decode_mixed_workspace",
]
