"""Asynchronous, fixed-arena prefill compaction for compressed attention.

Why a capacity plan is part of the API
--------------------------------------
The number of tokens assigned to every bit bank is a CUDA value.  A regular
``torch.Tensor`` allocation, however, needs a host integer.  Consequently an
implementation cannot simultaneously have all three of the following:

1. exact-size allocations chosen after seeing device-resident tags;
2. no device-to-host synchronization; and
3. ordinary PyTorch/CUDA tensor storage.

The v1 packer chooses (1) and gives up (2) indirectly through dynamic-shape
selection operators.  This module chooses (2): the caller supplies a
host-known :class:`NativePackingCapacity`, and every payload is compacted on
device into fixed-address arenas.  Capacity can be derived conservatively from
the allocation bit budget, or more tightly from a serving profile.  Both live
bytes and reserved bytes are exposed; unused reserve is real allocated memory
and is never presented as compression.

Hot-path properties
-------------------
* no ``torch.nonzero`` or boolean advanced indexing;
* no payload-sized allocation per bit level;
* one K code arena, one V code arena, one K norm arena, and one V norm arena;
* deterministic row-major ragged CSR order;
* 0-bit tokens write no payload;
* 3-bit codes use a dense little-endian bitstream (eight codes / three bytes);
* fixed per-level suffixes support pointer-stable mixed-bit decode appends;
* capacity overflow is recorded in a device flag and every store is bounded.

The only loop over bit levels launches directly into final arena views.  It
does not create level-sized indices, rotated values, masks, or packed
temporaries.  A pair of reusable, capacity-sized fp32 workspaces is used for
normalization and rotation; K and V reuse the same pair sequentially.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import threading
from typing import Any

import torch

from kvquant.tq_backend import (
    _load_or_compute_codebook,
    _unpack_indices,
    random_rotation,
)


try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only on CPU-only installs.
    triton = None
    tl = None
    _HAS_TRITON = False


QUANTIZED_LEVELS: tuple[int, ...] = (2, 3, 4, 8)
STORAGE_LEVELS: tuple[int, ...] = (2, 3, 4, 8, 16)
SUPPORTED_LEVELS: tuple[int, ...] = (0, 2, 3, 4, 8, 16)
SUPPORTED_TAG_DTYPES: tuple[torch.dtype, ...] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)
_RANK_CHUNK = 256
_ROTATION_CACHE_MAX_ENTRIES = 128
_CACHE_LOCK = threading.Lock()
_ROTATION_CACHE: OrderedDict[
    tuple[str, int | None, int, int],
    tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.cuda.Event | None,
    ],
] = OrderedDict()
_DEVICE_CODEBOOK_CACHE: dict[
    tuple[str, int | None, int, int], tuple[torch.Tensor, torch.cuda.Event | None]
] = {}


@dataclass(frozen=True)
class NativePackingPolicy:
    """Immutable allocator facts needed to prove shared-arena capacity."""

    target_avg_bits: float
    sink_tokens: int
    tail_tokens: int

    def __post_init__(self) -> None:
        if float(self.target_avg_bits) < 0:
            raise ValueError("target_avg_bits must be non-negative")
        if int(self.sink_tokens) < 0 or int(self.tail_tokens) < 0:
            raise ValueError("sink_tokens and tail_tokens must be non-negative")


def _packed_width(bits: int, head_dim: int) -> int:
    if bits == 3:
        return (3 * head_dim + 7) // 8
    effective_bits = 2 if bits == 2 else 4 if bits == 4 else 8
    values_per_byte = 8 // effective_bits
    return (head_dim + values_per_byte - 1) // values_per_byte


def _unpack_indices_true3(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Unpack the v2 dense 3-bit stream without materializing byte padding.

    Code ``d`` starts at bit ``3*d``.  A code can straddle two bytes, hence the
    16-bit gather before shifting.  This routine is a debug/materialization
    oracle only; production attention performs the same operation in Triton.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"expected uint8 packed codes, got {packed.dtype}")
    width = (3 * head_dim + 7) // 8
    if packed.shape[-1] != width:
        raise ValueError(
            f"true 3-bit payload width must be {width}, got {packed.shape[-1]}"
        )
    d = torch.arange(head_dim, device=packed.device, dtype=torch.long)
    bit = d * 3
    byte = bit // 8
    shift = bit % 8
    low = packed[..., byte].to(torch.int32)
    high_byte = torch.clamp(byte + 1, max=width - 1)
    high = packed[..., high_byte].to(torch.int32)
    high = torch.where((byte + 1) < width, high, torch.zeros_like(high))
    return ((low | (high << 8)) >> shift) & 0x7


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _shared_rotations(
    head_dim: int, device: torch.device, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Publish deterministic rotations safely across host/CUDA streams.

    Construction is serialized only on a cache miss.  Recording an event
    after the asynchronous QR/cast operations and waiting on every hit makes
    the immutable tensors safe to consume immediately from another stream.
    The bounded OrderedDict owns cache references only; an evicted rotation
    remains alive while a packed layer still references it.
    """
    if device.type != "cuda":
        raise RuntimeError("shared rotation cache requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (device.type, device.index, int(head_dim), int(seed))
    with _CACHE_LOCK:
        entry = _ROTATION_CACHE.get(key)
        if entry is None:
            with torch.cuda.device(device):
                pi_k = random_rotation(head_dim, device, torch.float32, seed=seed)
                pi_v = random_rotation(
                    head_dim, device, torch.float32, seed=seed + 2000,
                )
                tensors = (pi_k, pi_k.to(torch.bfloat16).contiguous(), pi_v)
                ready = torch.cuda.Event(blocking=False)
                ready.record(torch.cuda.current_stream(device))
            entry = (tensors, ready)
            _ROTATION_CACHE[key] = entry
            _ROTATION_CACHE.move_to_end(key)
            while len(_ROTATION_CACHE) > _ROTATION_CACHE_MAX_ENTRIES:
                _ROTATION_CACHE.popitem(last=False)
        else:
            _ROTATION_CACHE.move_to_end(key)
        tensors, ready = entry
        # Once publication completed globally, replace the event with a
        # permanent ready marker. Steady hits then need neither an event query
        # nor a stream node; an in-flight first miss still gets an explicit
        # wait on every consumer stream.
        if ready is not None and ready.query():
            ready = None
            _ROTATION_CACHE[key] = (tensors, None)
    if ready is not None:
        torch.cuda.current_stream(device).wait_event(ready)
    return tensors


def _shared_codebook(bits: int, head_dim: int, device: torch.device) -> torch.Tensor:
    """Publish one immutable device codebook safely across CUDA streams."""
    if device.type != "cuda":
        raise RuntimeError("shared codebook cache requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (device.type, device.index, int(bits), int(head_dim))
    with _CACHE_LOCK:
        entry = _DEVICE_CODEBOOK_CACHE.get(key)
        if entry is None:
            with torch.cuda.device(device):
                tensor = _load_or_compute_codebook(
                    bits, head_dim, device, torch.float32,
                )
                ready = torch.cuda.Event(blocking=False)
                ready.record(torch.cuda.current_stream(device))
            entry = (tensor, ready)
            _DEVICE_CODEBOOK_CACHE[key] = entry
        tensor, ready = entry
        if ready is not None and ready.query():
            ready = None
            _DEVICE_CODEBOOK_CACHE[key] = (tensor, None)
    if ready is not None:
        torch.cuda.current_stream(device).wait_event(ready)
    return tensor


@dataclass(frozen=True)
class NativePackingCapacity:
    """Host-known upper bounds for one layer's prefill payload.

    ``level_slots`` and ``exact_slots`` exclude the decode reserve.  The legacy
    ``decode_2bit_reserve_per_row`` field keeps old fixtures/API users working;
    production mixed-bit layouts use ``decode_reserve_per_row_by_level`` in
    storage-level order ``(2, 3, 4, 8, 16)``.
    """

    level_slots: tuple[int, int, int, int]
    exact_slots: int
    decode_2bit_reserve_per_row: int = 0
    decode_reserve_per_row_by_level: tuple[int, int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if len(self.level_slots) != len(QUANTIZED_LEVELS):
            raise ValueError("level_slots must contain bounds for 2, 3, 4, and 8 bits")
        reserves = self.resolved_decode_reserves()
        values = (*self.level_slots, self.exact_slots, *reserves)
        if any(int(value) < 0 for value in values):
            raise ValueError("packing capacities must be non-negative")

    def resolved_decode_reserves(self) -> tuple[int, int, int, int, int]:
        configured = self.decode_reserve_per_row_by_level
        if configured is None:
            return (int(self.decode_2bit_reserve_per_row), 0, 0, 0, 0)
        if len(configured) != len(STORAGE_LEVELS):
            raise ValueError(
                "decode_reserve_per_row_by_level must cover 2, 3, 4, 8, and 16 bits"
            )
        values = tuple(int(value) for value in configured)
        legacy = int(self.decode_2bit_reserve_per_row)
        if legacy and values[0] not in (0, legacy):
            raise ValueError(
                "conflicting 2-bit decode reserves in legacy and per-level fields"
            )
        if legacy and values[0] == 0:
            values = (legacy, *values[1:])
        return values

    @property
    def decode_capacity_per_row(self) -> int:
        return max(self.resolved_decode_reserves(), default=0)

    def resolved_level_slots(self, num_rows: int) -> tuple[int, int, int, int]:
        reserves = self.resolved_decode_reserves()
        return tuple(
            int(slots) + int(num_rows) * reserve
            for slots, reserve in zip(self.level_slots, reserves[:4])
        )

    def resolved_exact_slots(self, num_rows: int) -> int:
        return int(self.exact_slots) + int(num_rows) * self.resolved_decode_reserves()[4]

    @classmethod
    def from_policy_budget(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        target_avg_bits: float,
        sink_tokens: int = 0,
        tail_tokens: int = 0,
        per_request_rounding_bits: float = 16.0,
        slot_slack: int = 0,
        decode_2bit_reserve_per_row: int = 0,
        decode_reserve_per_row_by_level: tuple[int, int, int, int, int] | None = None,
    ) -> "NativePackingCapacity":
        """Conservative plan for the current layerwise allocator policy.

        The allocator constrains the nominal bit sum of non-sink cells.  For a
        given level ``b``, ``count_b <= total_bit_budget / b``.  Reserving that
        upper bound independently for every bank costs more than an exact
        histogram but needs neither a tag readback nor a changed allocation
        policy.  Protected sink/tail cells are added to the exact bank because
        tail protection is applied after the budget solve.

        ``per_request_rounding_bits`` covers the repair tolerance and fp32
        boundary rounding.  Sixteen bits per request is deliberately more
        conservative than the reference repair's half-bit aggregate stopping
        rule.
        """
        if min(batch_size, num_heads, seq_len) < 0:
            raise ValueError("batch_size, num_heads, and seq_len must be non-negative")
        if target_avg_bits < 0:
            raise ValueError("target_avg_bits must be non-negative")
        sink = min(int(sink_tokens), int(seq_len))
        tail = min(int(tail_tokens), int(seq_len))
        protected_per_row = min(int(seq_len), sink + tail)
        # Sinks are excluded from the lambda budget; tails are deliberately not
        # removed because they are allocated first and overwritten afterward.
        active_cells = int(batch_size) * int(num_heads) * (int(seq_len) - sink)
        nominal_budget = (
            float(target_avg_bits) * active_cells
            + float(per_request_rounding_bits) * int(batch_size)
        )
        slack = int(slot_slack)
        level_slots = tuple(
            min(
                int(batch_size) * int(num_heads) * int(seq_len),
                int(math.ceil(nominal_budget / bits)) + slack,
            )
            for bits in QUANTIZED_LEVELS
        )
        protected = int(batch_size) * int(num_heads) * protected_per_row
        exact = min(
            int(batch_size) * int(num_heads) * int(seq_len),
            protected + int(math.ceil(nominal_budget / 16.0)) + slack,
        )
        return cls(
            level_slots=level_slots,
            exact_slots=exact,
            decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
            decode_reserve_per_row_by_level=decode_reserve_per_row_by_level,
        )


@dataclass(frozen=True)
class NativeSharedPackingCapacity:
    """Aggregate capacity for device-based level bases.

    Unlike :class:`NativePackingCapacity`, mutually exclusive bit levels do
    not each reserve their worst case. ``quant_slots`` and
    ``code_bytes_per_tensor`` are shared by all quantized levels.  The launch
    bounds do not allocate storage; they only bound the grids of the four
    direct encoding kernels.
    """

    quant_slots: int
    code_bytes_per_tensor: int
    exact_slots: int
    level_launch_slots: tuple[int, int, int, int]
    decode_2bit_reserve_per_row: int = 0
    decode_reserve_per_row_by_level: tuple[int, int, int, int, int] | None = None

    def __post_init__(self) -> None:
        reserves = self.resolved_decode_reserves()
        values = (
            self.quant_slots,
            self.code_bytes_per_tensor,
            self.exact_slots,
            *self.level_launch_slots,
            *reserves,
        )
        if len(self.level_launch_slots) != len(QUANTIZED_LEVELS):
            raise ValueError("level_launch_slots must cover 2, 3, 4, and 8 bits")
        if any(int(value) < 0 for value in values):
            raise ValueError("shared packing capacities must be non-negative")

    def resolved_decode_reserves(self) -> tuple[int, int, int, int, int]:
        configured = self.decode_reserve_per_row_by_level
        if configured is None:
            return (int(self.decode_2bit_reserve_per_row), 0, 0, 0, 0)
        if len(configured) != len(STORAGE_LEVELS):
            raise ValueError(
                "decode_reserve_per_row_by_level must cover 2, 3, 4, 8, and 16 bits"
            )
        values = tuple(int(value) for value in configured)
        legacy = int(self.decode_2bit_reserve_per_row)
        if legacy and values[0] not in (0, legacy):
            raise ValueError(
                "conflicting 2-bit decode reserves in legacy and per-level fields"
            )
        if legacy and values[0] == 0:
            values = (legacy, *values[1:])
        return values

    @property
    def decode_capacity_per_row(self) -> int:
        return max(self.resolved_decode_reserves(), default=0)

    @classmethod
    def from_policy_budget(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        target_avg_bits: float,
        sink_tokens: int = 0,
        tail_tokens: int = 0,
        per_request_rounding_bits: float = 16.0,
        slot_slack: int = 0,
        code_slack_bytes: int = 256,
        decode_2bit_reserve_per_row: int = 0,
        decode_reserve_per_row_by_level: tuple[int, int, int, int, int] | None = None,
    ) -> "NativeSharedPackingCapacity":
        """Guaranteed aggregate arena under the nominal allocation budget.

        Code arena sizing uses the exact, rounded physical width of every
        level.  Dense 3-bit storage costs ``ceil(3*D/8)`` bytes per token, so
        for common ``D=128`` its byte-per-nominal-bit ratio is exactly 16.
        """
        if min(batch_size, num_heads, seq_len, head_dim) < 0:
            raise ValueError("shape values must be non-negative")
        sink = min(int(sink_tokens), int(seq_len))
        tail = min(int(tail_tokens), int(seq_len))
        protected_per_row = min(int(seq_len), sink + tail)
        rows = int(batch_size) * int(num_heads)
        active_cells = rows * (int(seq_len) - sink)
        nominal_budget = (
            float(target_avg_bits) * active_cells
            + float(per_request_rounding_bits) * int(batch_size)
        )
        probe = cls(
            quant_slots=0,
            code_bytes_per_tensor=0,
            exact_slots=0,
            level_launch_slots=(0, 0, 0, 0),
            decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
            decode_reserve_per_row_by_level=decode_reserve_per_row_by_level,
        )
        reserves = probe.resolved_decode_reserves()
        reserve_slots = tuple(rows * reserve for reserve in reserves)
        quant_slots = (
            int(math.ceil(nominal_budget / 2.0))
            + int(slot_slack)
            + sum(reserve_slots[:4])
        )
        worst_bytes_per_bit = max(
            _packed_width(bits, int(head_dim)) / bits for bits in QUANTIZED_LEVELS
        )
        code_bytes = (
            int(math.ceil(nominal_budget * worst_bytes_per_bit))
            + sum(
                slots * _packed_width(bits, int(head_dim))
                for bits, slots in zip(QUANTIZED_LEVELS, reserve_slots[:4])
            )
            + int(code_slack_bytes)
        )
        protected = rows * protected_per_row
        exact_slots = (
            protected
            + int(math.ceil(nominal_budget / 16.0))
            + int(slot_slack)
            + reserve_slots[4]
        )
        # Launch/workspace bounds cover live prefill cells only.  Decode
        # reserve is physical storage, not prefill work.
        launch = tuple(
            int(math.ceil(nominal_budget / bits)) + int(slot_slack)
            for bits in QUANTIZED_LEVELS
        )
        return cls(
            quant_slots=quant_slots,
            code_bytes_per_tensor=code_bytes,
            exact_slots=exact_slots,
            level_launch_slots=launch,
            decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
            decode_reserve_per_row_by_level=decode_reserve_per_row_by_level,
        )

    @classmethod
    def from_histogram(
        cls,
        *,
        counts: dict[int, int],
        head_dim: int,
        num_rows: int,
        decode_2bit_reserve_per_row: int = 0,
        decode_reserve_per_row_by_level: tuple[int, int, int, int, int] | None = None,
    ) -> "NativeSharedPackingCapacity":
        """Exact fixture/profile plan; caller owns how counts reached the host."""
        probe = cls(
            quant_slots=0,
            code_bytes_per_tensor=0,
            exact_slots=0,
            level_launch_slots=(0, 0, 0, 0),
            decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
            decode_reserve_per_row_by_level=decode_reserve_per_row_by_level,
        )
        reserves = probe.resolved_decode_reserves()
        reserve_slots = tuple(int(num_rows) * reserve for reserve in reserves)
        level_counts = tuple(int(counts.get(bits, 0)) for bits in QUANTIZED_LEVELS)
        quant_slots = sum(level_counts) + sum(reserve_slots[:4])
        code_bytes = sum(
            count * _packed_width(bits, int(head_dim))
            for bits, count in zip(QUANTIZED_LEVELS, level_counts)
        ) + sum(
            slots * _packed_width(bits, int(head_dim))
            for bits, slots in zip(QUANTIZED_LEVELS, reserve_slots[:4])
        )
        launch = level_counts
        return cls(
            quant_slots=quant_slots,
            code_bytes_per_tensor=code_bytes,
            exact_slots=int(counts.get(16, 0)) + reserve_slots[4],
            level_launch_slots=launch,
            decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
            decode_reserve_per_row_by_level=decode_reserve_per_row_by_level,
        )


if _HAS_TRITON:
    from kvquant.tq_triton import _nearest_sorted_centroid
    @triton.jit
    def _rank_and_chunk_count_kernel_v2(
        tags_ptr,
        rank_ptr,
        chunk_counts_ptr,
        invalid_tag_ptr,
        seq_len: tl.constexpr,
        chunks: tl.constexpr,
        rows: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        pos = chunk * BLOCK + tl.arange(0, BLOCK)
        valid = pos < seq_len
        # Validate before narrowing to uint8.  Otherwise values such as 256
        # would wrap to zero and be silently interpreted as eviction.
        source_tags = tl.load(tags_ptr + row * seq_len + pos, mask=valid, other=0)
        allowed = (
            (source_tags == 0) | (source_tags == 2) | (source_tags == 3)
            | (source_tags == 4) | (source_tags == 8) | (source_tags == 16)
        )
        invalid = tl.max((valid & ~allowed).to(tl.int32), axis=0)
        tl.atomic_max(invalid_tag_ptr, invalid)
        tags = source_tags.to(tl.int32)
        chosen_rank = tl.zeros([BLOCK], dtype=tl.int32)

        selected = valid & (tags == 2)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(chunk_counts_ptr + (0 * rows + row) * chunks + chunk, tl.sum(selected.to(tl.int32)))

        selected = valid & (tags == 3)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(chunk_counts_ptr + (1 * rows + row) * chunks + chunk, tl.sum(selected.to(tl.int32)))

        selected = valid & (tags == 4)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(chunk_counts_ptr + (2 * rows + row) * chunks + chunk, tl.sum(selected.to(tl.int32)))

        selected = valid & (tags == 8)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(chunk_counts_ptr + (3 * rows + row) * chunks + chunk, tl.sum(selected.to(tl.int32)))

        selected = valid & (tags == 16)
        prefix = tl.cumsum(selected.to(tl.int32), axis=0)
        chosen_rank = tl.where(selected, prefix - 1, chosen_rank)
        tl.store(chunk_counts_ptr + (4 * rows + row) * chunks + chunk, tl.sum(selected.to(tl.int32)))

        tl.store(rank_ptr + row * seq_len + pos, chosen_rank, mask=valid)


    @triton.jit
    def _compact_exact_kernel(
        keys_ptr,
        values_ptr,
        tags_ptr,
        rank_ptr,
        chunk_prefix_ptr,
        row_offsets_ptr,
        out_keys_ptr,
        out_values_ptr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        key_stride_b: tl.constexpr,
        key_stride_h: tl.constexpr,
        key_stride_t: tl.constexpr,
        key_stride_d: tl.constexpr,
        value_stride_b: tl.constexpr,
        value_stride_h: tl.constexpr,
        value_stride_t: tl.constexpr,
        value_stride_d: tl.constexpr,
        chunks: tl.constexpr,
        rank_chunk: tl.constexpr,
        exact_capacity,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        row = token // seq_len
        batch = row // num_heads
        head = row - batch * num_heads
        pos = token - row * seq_len
        chunk = pos // rank_chunk
        is_exact = tl.load(tags_ptr + token).to(tl.int32) == 16
        local = (
            tl.load(chunk_prefix_ptr + (4 * tl.num_programs(0) // seq_len + row) * chunks + chunk)
            + tl.load(rank_ptr + token)
        )
        destination = tl.load(row_offsets_ptr + 4 * (tl.num_programs(0) // seq_len) + row) + local
        valid = is_exact & (destination < exact_capacity) & (d < head_dim)
        source_k = (
            batch * key_stride_b
            + head * key_stride_h
            + pos * key_stride_t
            + d * key_stride_d
        )
        source_v = (
            batch * value_stride_b
            + head * value_stride_h
            + pos * value_stride_t
            + d * value_stride_d
        )
        target = destination * head_dim + d
        key = tl.load(keys_ptr + source_k, mask=valid, other=0.0)
        value = tl.load(values_ptr + source_v, mask=valid, other=0.0)
        tl.store(out_keys_ptr + target, key, mask=valid)
        tl.store(out_values_ptr + target, value, mask=valid)


    @triton.jit
    def _compact_normalize_kernel(
        source_ptr,
        tags_ptr,
        rank_ptr,
        chunk_prefix_ptr,
        storage_row_offsets_ptr,
        compute_row_offsets_ptr,
        storage_slot_bases_ptr,
        compute_slot_bases_ptr,
        storage_capacities_ptr,
        compute_capacities_ptr,
        normed_ptr,
        norms_ptr,
        storage_map_ptr,
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_heads: tl.constexpr,
        source_stride_b: tl.constexpr,
        source_stride_h: tl.constexpr,
        source_stride_t: tl.constexpr,
        source_stride_d: tl.constexpr,
        chunks: tl.constexpr,
        rows: tl.constexpr,
        rank_chunk: tl.constexpr,
        WRITE_MAP: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        tag = tl.load(tags_ptr + token).to(tl.int32)
        level = tl.where(
            tag == 2,
            0,
            tl.where(tag == 3, 1, tl.where(tag == 4, 2, tl.where(tag == 8, 3, -1))),
        )
        is_quant = level >= 0
        row = token // seq_len
        batch = row // num_heads
        head = row - batch * num_heads
        pos = token - row * seq_len
        chunk = pos // rank_chunk
        safe_level = tl.maximum(level, 0)
        local = (
            tl.load(chunk_prefix_ptr + (safe_level * rows + row) * chunks + chunk)
            + tl.load(rank_ptr + token)
        )
        storage_slot = (
            tl.load(storage_row_offsets_ptr + safe_level * rows + row) + local
        )
        compute_slot = (
            tl.load(compute_row_offsets_ptr + safe_level * rows + row) + local
        )
        storage_capacity = tl.load(storage_capacities_ptr + safe_level)
        compute_capacity = tl.load(compute_capacities_ptr + safe_level)
        storage_destination = (
            tl.load(storage_slot_bases_ptr + safe_level) + storage_slot
        )
        compute_destination = (
            tl.load(compute_slot_bases_ptr + safe_level) + compute_slot
        )
        compute_live = is_quant & (compute_slot < compute_capacity)
        storage_live = compute_live & (storage_slot < storage_capacity)
        values = tl.load(
            source_ptr
            + batch * source_stride_b
            + head * source_stride_h
            + pos * source_stride_t
            + d * source_stride_d,
            mask=compute_live & (d < head_dim),
            other=0.0,
        ).to(tl.float32)
        norm = tl.sqrt(tl.sum(values * values, axis=0))
        normalized = values / (norm + 1.0e-10)
        tl.store(
            normed_ptr + compute_destination * head_dim + d,
            normalized,
            mask=compute_live & (d < head_dim),
        )
        tl.store(norms_ptr + storage_destination, norm, mask=storage_live)
        # The encoder runs over compact compute slots but writes directly into
        # the final CSR storage slot (whose per-row reserve creates gaps).
        if WRITE_MAP:
            tl.store(
                storage_map_ptr + compute_destination,
                storage_slot,
                mask=compute_live,
            )


    @triton.jit
    def _encode_2bit_kernel(
        rotated_ptr,
        storage_map_ptr,
        centroid_ptr,
        packed_ptr,
        required_ptr,
        slot_bases_ptr,
        code_bases_ptr,
        capacity,
        code_capacity,
        workspace_capacity,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        slot = tl.program_id(0)
        slot_base = tl.load(slot_bases_ptr + 0)
        compute_slot = slot_base + slot
        if (
            (slot < tl.load(required_ptr + 0))
            & (slot < capacity)
            & (compute_slot < workspace_capacity)
        ):
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            code_base = tl.load(code_bases_ptr + 0)
            storage_slot = tl.load(storage_map_ptr + compute_slot)
            target = code_base + storage_slot * width + p
            live = (
                (p < width)
                & (storage_slot >= 0)
                & (target >= 0)
                & (target < code_capacity)
            )
            packed = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = p * 4 + lane
                value = tl.load(
                    rotated_ptr + compute_slot * head_dim + d,
                    mask=live & (d < head_dim),
                    other=0.0,
                ).to(tl.float32)
                best_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_index = tl.zeros([BLOCK_P], dtype=tl.int32)
                for k in tl.static_range(0, 4):
                    centroid = tl.load(centroid_ptr + k).to(tl.float32)
                    distance = tl.abs(value - centroid)
                    better = distance < best_dist
                    best_dist = tl.where(better, distance, best_dist)
                    best_index = tl.where(better, k, best_index)
                packed |= best_index << (2 * lane)
            tl.store(packed_ptr + target, packed, mask=live)


    @triton.jit
    def _encode_nibble_kernel(
        rotated_ptr,
        storage_map_ptr,
        centroid_ptr,
        packed_ptr,
        required_ptr,
        slot_bases_ptr,
        code_bases_ptr,
        required_index: tl.constexpr,
        num_centroids: tl.constexpr,
        capacity,
        code_capacity,
        workspace_capacity,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        slot = tl.program_id(0)
        slot_base = tl.load(slot_bases_ptr + required_index)
        compute_slot = slot_base + slot
        if (
            (slot < tl.load(required_ptr + required_index))
            & (slot < capacity)
            & (compute_slot < workspace_capacity)
        ):
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            code_base = tl.load(code_bases_ptr + required_index)
            storage_slot = tl.load(storage_map_ptr + compute_slot)
            target = code_base + storage_slot * width + p
            live = (
                (p < width)
                & (storage_slot >= 0)
                & (target >= 0)
                & (target < code_capacity)
            )
            packed = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 2):
                d = p * 2 + lane
                value = tl.load(
                    rotated_ptr + compute_slot * head_dim + d,
                    mask=live & (d < head_dim),
                    other=0.0,
                ).to(tl.float32)
                best_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_index = tl.zeros([BLOCK_P], dtype=tl.int32)
                for k in tl.static_range(0, num_centroids):
                    centroid = tl.load(centroid_ptr + k).to(tl.float32)
                    distance = tl.abs(value - centroid)
                    better = distance < best_dist
                    best_dist = tl.where(better, distance, best_dist)
                    best_index = tl.where(better, k, best_index)
                packed |= best_index << (4 * lane)
            tl.store(packed_ptr + target, packed, mask=live)


    @triton.jit
    def _encode_3bit_kernel(
        rotated_ptr,
        storage_map_ptr,
        centroid_ptr,
        packed_ptr,
        required_ptr,
        slot_bases_ptr,
        code_bases_ptr,
        capacity,
        code_capacity,
        workspace_capacity,
        head_dim: tl.constexpr,
        width: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """Encode eight 3-bit scalar codes into exactly three bytes.

        Each program lane owns one output byte.  Four neighbouring codes are
        sufficient to cover that byte even when its first bit starts inside a
        code; the assembled word is shifted to the byte boundary afterwards.
        """
        slot = tl.program_id(0)
        slot_base = tl.load(slot_bases_ptr + 1)
        compute_slot = slot_base + slot
        if (
            (slot < tl.load(required_ptr + 1))
            & (slot < capacity)
            & (compute_slot < workspace_capacity)
        ):
            p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            code_base = tl.load(code_bases_ptr + 1)
            storage_slot = tl.load(storage_map_ptr + compute_slot)
            target = code_base + storage_slot * width + p
            live = (
                (p < width)
                & (storage_slot >= 0)
                & (target >= 0)
                & (target < code_capacity)
            )
            base_bit = p * 8
            first_d = base_bit // 3
            bit_offset = base_bit - first_d * 3
            word = tl.zeros([BLOCK_P], dtype=tl.int32)
            for lane in tl.static_range(0, 4):
                d = first_d + lane
                lane_live = live & (d < head_dim)
                value = tl.load(
                    rotated_ptr + compute_slot * head_dim + d,
                    mask=lane_live,
                    other=0.0,
                ).to(tl.float32)
                best_dist = tl.full([BLOCK_P], float("inf"), dtype=tl.float32)
                best_index = tl.zeros([BLOCK_P], dtype=tl.int32)
                for k in tl.static_range(0, 8):
                    centroid = tl.load(centroid_ptr + k).to(tl.float32)
                    distance = tl.abs(value - centroid)
                    better = lane_live & (distance < best_dist)
                    best_dist = tl.where(better, distance, best_dist)
                    best_index = tl.where(better, k, best_index)
                word |= best_index << (3 * lane)
            packed = (word >> bit_offset) & 0xFF
            tl.store(packed_ptr + target, packed, mask=live)


    @triton.jit
    def _encode_8bit_kernel(
        rotated_ptr,
        storage_map_ptr,
        centroid_ptr,
        packed_ptr,
        required_ptr,
        slot_bases_ptr,
        code_bases_ptr,
        capacity,
        code_capacity,
        workspace_capacity,
        head_dim: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        slot = tl.program_id(0)
        slot_base = tl.load(slot_bases_ptr + 3)
        compute_slot = slot_base + slot
        if (
            (slot < tl.load(required_ptr + 3))
            & (slot < capacity)
            & (compute_slot < workspace_capacity)
        ):
            d = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
            code_base = tl.load(code_bases_ptr + 3)
            storage_slot = tl.load(storage_map_ptr + compute_slot)
            target = code_base + storage_slot * head_dim + d
            live = (
                (d < head_dim)
                & (storage_slot >= 0)
                & (target >= 0)
                & (target < code_capacity)
            )
            value = tl.load(
                rotated_ptr + compute_slot * head_dim + d,
                mask=live,
                other=0.0,
            ).to(tl.float32)
            best_index = _nearest_sorted_centroid(
                value,
                centroid_ptr,
                live,
                K=256,
                SEARCH_STEPS=8,
            )
            tl.store(packed_ptr + target, best_index, mask=live)


@dataclass
class NativePackedKVV2:
    """Decode-ready views backed by four shared fixed-address arenas."""

    quant_banks: tuple[dict[str, Any], ...]
    exact: dict[str, Any]
    pi_k: torch.Tensor
    pi_k_decode: torch.Tensor
    pi_v: torch.Tensor
    tags: torch.Tensor
    capacity: NativePackingCapacity | NativeSharedPackingCapacity
    required_slots: torch.Tensor
    required_code_bytes: torch.Tensor | None
    overflow_flag: torch.Tensor
    code_arena_k: torch.Tensor
    code_arena_v: torch.Tensor
    norm_arena_k: torch.Tensor
    norm_arena_v: torch.Tensor
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    storage_dtype: torch.dtype
    decode_arena: Any | None = None
    decode_tags: torch.Tensor | None = None
    decode_len: int = 0

    @property
    def num_rows(self) -> int:
        return self.batch_size * self.num_heads

    @property
    def total_seq_len(self) -> int:
        return self.seq_len + self.decode_len

    def bank(self, bits: int) -> dict[str, Any]:
        return self.quant_banks[QUANTIZED_LEVELS.index(bits)]

    def enable_segmented_decode(
        self,
        *,
        capacity_per_row: int,
        buffer_size: int,
        target_avg_bits: float,
        bit_levels: tuple[int, ...],
    ) -> None:
        """Allocate the aggregate mixed-bit suffix before graph capture."""
        if self.decode_arena is not None:
            raise RuntimeError("segmented decode arena is already configured")
        if any(self.capacity.resolved_decode_reserves()):
            raise RuntimeError(
                "segmented decode cannot coexist with per-bank suffix reserves"
            )
        if any(
            bank["cent_k"].data_ptr() != bank["cent_v"].data_ptr()
            for bank in self.quant_banks
        ):
            raise RuntimeError(
                "unified mixed-bit decode requires identical key/value MSE codebooks"
            )
        from kvquant.runtime.native_decode_arena import NativeDecodeArena

        self.decode_arena = NativeDecodeArena.allocate(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            capacity_per_row=capacity_per_row,
            buffer_size=buffer_size,
            target_avg_bits=target_avg_bits,
            bit_levels=bit_levels,
            device=self.tags.device,
            dtype=self.storage_dtype,
            codebooks=tuple(
                bank["cent_k"] for bank in self.quant_banks
            ),
        )
        self.decode_tags = torch.zeros(
            self.batch_size,
            self.num_heads,
            int(capacity_per_row),
            device=self.tags.device,
            dtype=torch.uint8,
        )

    def validate_capacity(self) -> None:
        """Explicit diagnostic synchronization; never call on the hot path."""
        required = self.required_slots.detach().cpu().tolist()
        reserves = self.capacity.resolved_decode_reserves()
        if isinstance(self.capacity, NativeSharedPackingCapacity):
            quant_need = sum(required[:4])
            exact_need = required[4]
            code_need = int(self.required_code_bytes.detach().cpu()[0])
            live_quant = list(required[:4])
            reserve_slots = [reserve * self.num_rows for reserve in reserves]
            live_quant = [
                count - reserve
                for count, reserve in zip(live_quant, reserve_slots[:4])
            ]
            failures = []
            if quant_need > self.capacity.quant_slots:
                failures.append(f"quant slots need={quant_need} cap={self.capacity.quant_slots}")
            if code_need > self.capacity.code_bytes_per_tensor:
                failures.append(
                    f"code bytes need={code_need} cap={self.capacity.code_bytes_per_tensor}"
                )
            if exact_need > self.capacity.exact_slots:
                failures.append(f"exact slots need={exact_need} cap={self.capacity.exact_slots}")
            for bits, need, have in zip(
                QUANTIZED_LEVELS, live_quant, self.capacity.level_launch_slots,
            ):
                if need > have:
                    failures.append(f"{bits}b launch need={need} cap={have}")
            workspace_capacity = self.capacity.quant_slots - sum(reserve_slots[:4])
            if sum(live_quant) > workspace_capacity:
                failures.append(
                    "compute workspace "
                    f"need={sum(live_quant)} cap={workspace_capacity}"
                )
            if failures:
                raise RuntimeError("native shared arena capacity exceeded: " + "; ".join(failures))
        else:
            capacities = [
                *(bank["packed_k"].shape[0] for bank in self.quant_banks),
                self.exact["keys"].shape[0],
            ]
            if any(need > have for need, have in zip(required, capacities)):
                detail = ", ".join(
                    f"{bits}b need={need} cap={have}"
                    for bits, need, have in zip(STORAGE_LEVELS, required, capacities)
                    if need > have
                )
                raise RuntimeError(f"native packing arena capacity exceeded: {detail}")

    @staticmethod
    def _payload_for_live(bank: dict[str, Any], live: torch.Tensor, field: str) -> torch.Tensor:
        """Return packed rows from either per-level views or a shared byte arena."""
        if "code_base" not in bank:
            return bank[field][live]
        width = int(bank["packed_width"])
        byte = torch.arange(width, device=live.device)
        indices = bank["code_base"].to(torch.long) + live[:, None] * width + byte[None, :]
        return bank[field][indices]

    @staticmethod
    def _norms_for_live(bank: dict[str, Any], live: torch.Tensor, field: str) -> torch.Tensor:
        if "norm_base" not in bank:
            return bank[field][live]
        return bank[field][bank["norm_base"].to(torch.long) + live]

    @staticmethod
    def _live_indices(bank: dict[str, Any], rows: int) -> torch.Tensor:
        width = int(bank["T_max"])
        columns = torch.arange(width, device=bank["offset"].device)
        live = columns.view(1, -1) < bank["seqlen"].view(rows, 1)
        indices = bank["offset"].to(torch.long).view(rows, 1) + columns.view(1, -1)
        return indices[live]

    @torch.no_grad()
    def materialize(self, *, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Debug oracle; dynamic gathers here are intentionally not production."""
        self.validate_capacity()
        if self.decode_arena is not None and int(
            self.decode_arena.overflow_flag.detach().cpu()[0]
        ):
            raise RuntimeError("segmented decode arena capacity exceeded")
        output_dtype = self.storage_dtype if dtype is None else dtype
        shape = (self.batch_size, self.num_heads, self.total_seq_len, self.head_dim)
        keys = torch.zeros(shape, device=self.tags.device, dtype=output_dtype)
        values = torch.zeros_like(keys)
        segmented = self.decode_arena is not None
        tags = self.tags
        if self.decode_len and not segmented:
            tags = torch.cat((tags, self.decode_tags[..., : self.decode_len]), dim=-1)
        flat_tags = tags.reshape(-1)
        if segmented:
            flat_tags = self.tags.reshape(self.num_rows, self.seq_len)
            flat_k = keys.view(
                self.num_rows, self.total_seq_len, self.head_dim,
            )[:, : self.seq_len]
            flat_v = values.view(
                self.num_rows, self.total_seq_len, self.head_dim,
            )[:, : self.seq_len]
        else:
            flat_k = keys.view(-1, self.head_dim)
            flat_v = values.view(-1, self.head_dim)

        for bits in QUANTIZED_LEVELS:
            bank = self.bank(bits)
            live = self._live_indices(bank, self.num_rows)
            if live.numel() == 0:
                continue
            packed_k = self._payload_for_live(bank, live, "packed_k")
            packed_v = self._payload_for_live(bank, live, "packed_v")
            norms_k = self._norms_for_live(bank, live, "norms_k")
            norms_v = self._norms_for_live(bank, live, "norms_v")
            physical_bits = int(bank.get("physical_bits", bits))
            if bits == 3 and physical_bits == 3:
                index_k = _unpack_indices_true3(packed_k, self.head_dim)
                index_v = _unpack_indices_true3(packed_v, self.head_dim)
            else:
                index_k = _unpack_indices(packed_k, bits, self.head_dim)
                index_v = _unpack_indices(packed_v, bits, self.head_dim)
            deq_k = (bank["cent_k"][index_k] @ self.pi_k) * norms_k.float().unsqueeze(-1)
            deq_v = (bank["cent_v"][index_v] @ self.pi_v) * norms_v.float().unsqueeze(-1)
            flat_k[flat_tags == bits] = deq_k.to(output_dtype)
            flat_v[flat_tags == bits] = deq_v.to(output_dtype)

        exact_live = self._live_indices(self.exact, self.num_rows)
        exact_mask = flat_tags == 16
        flat_k[exact_mask] = self.exact["keys"][exact_live].to(output_dtype)
        flat_v[exact_mask] = self.exact["values"][exact_live].to(output_dtype)

        if segmented and self.decode_len:
            arena = self.decode_arena
            decoded_k = keys[:, :, self.seq_len :].reshape(
                self.num_rows, self.decode_len, self.head_dim,
            )
            decoded_v = values[:, :, self.seq_len :].reshape_as(decoded_k)
            decode_tags = self.decode_tags[..., : self.decode_len].reshape(
                self.num_rows, self.decode_len,
            )
            for segment, (start, length) in enumerate(
                zip(arena.segment_starts, arena.segment_lengths)
            ):
                segment_tags = decode_tags[:, start : start + length]
                for level_index, bits in enumerate(QUANTIZED_LEVELS):
                    counts = arena.counts[segment, level_index]
                    max_count = int(counts.max().detach().cpu())
                    if max_count == 0:
                        continue
                    column = torch.arange(max_count, device=keys.device)
                    live = column.view(1, -1) < counts.view(-1, 1)
                    width = _packed_width(bits, self.head_dim)
                    byte = torch.arange(width, device=keys.device)
                    bases = arena.code_row_base[segment, level_index].long()
                    slots = bases.view(-1, 1) + column.view(1, -1) * width
                    addresses = slots[..., None] + byte.view(1, 1, -1)
                    packed_k = arena.code_arena_k[addresses[live]]
                    packed_v = arena.code_arena_v[addresses[live]]
                    norm_slots = (
                        arena.norm_row_base[segment, level_index].long().view(-1, 1)
                        + column.view(1, -1)
                    )
                    norms_k = arena.norm_arena_k[norm_slots[live]]
                    norms_v = arena.norm_arena_v[norm_slots[live]]
                    if bits == 3:
                        index_k = _unpack_indices_true3(packed_k, self.head_dim)
                        index_v = _unpack_indices_true3(packed_v, self.head_dim)
                    else:
                        index_k = _unpack_indices(packed_k, bits, self.head_dim)
                        index_v = _unpack_indices(packed_v, bits, self.head_dim)
                    bank = self.bank(bits)
                    deq_k = (bank["cent_k"][index_k] @ self.pi_k) * norms_k.float().unsqueeze(-1)
                    deq_v = (bank["cent_v"][index_v] @ self.pi_v) * norms_v.float().unsqueeze(-1)
                    mask = segment_tags == bits
                    target_k = decoded_k[:, start : start + length]
                    target_v = decoded_v[:, start : start + length]
                    target_k[mask] = deq_k.to(output_dtype)
                    target_v[mask] = deq_v.to(output_dtype)

                counts = arena.counts[segment, 4]
                max_count = int(counts.max().detach().cpu())
                if max_count:
                    column = torch.arange(max_count, device=keys.device)
                    live = column.view(1, -1) < counts.view(-1, 1)
                    slots = (
                        arena.exact_row_base[segment].long().view(-1, 1)
                        + column.view(1, -1)
                    )
                    target_k = decoded_k[:, start : start + length]
                    target_v = decoded_v[:, start : start + length]
                    mask = segment_tags == 16
                    target_k[mask] = arena.exact_k[slots[live]].to(output_dtype)
                    target_v[mask] = arena.exact_v[slots[live]].to(output_dtype)
        return keys, values

    @torch.no_grad()
    def append_decode_2bit(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        workspace: Any,
    ) -> None:
        """Stride-aware native append into the shared per-row 2-bit reserve.

        The workspace is owned by :class:`ODMCache` and reused serially by
        every layer in one flush.  Requiring it explicitly prevents an
        accidental per-layer payload allocation (four fp32 tensors per layer)
        from reappearing on the production path.
        """
        from kvquant.runtime.kernels.native_flush_v2 import append_decode_2bit_workspace

        append_decode_2bit_workspace(self, keys, values, workspace)

    @torch.no_grad()
    def append_decode(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        bits: torch.Tensor,
        *,
        workspace: Any,
    ) -> None:
        """Append a device-resident ``0/2/3/4/8/16`` decode allocation.

        Mixed production states use one pointer-stable aggregate arena plus
        fixed segment descriptors. Legacy fixtures may retain per-bank suffixes.
        """
        from kvquant.runtime.kernels.native_flush_v2 import append_decode_mixed_workspace

        append_decode_mixed_workspace(self, keys, values, bits, workspace)

    def reserved_storage_stats(self) -> dict[str, Any]:
        """Byte-exact persistent CUDA storages without synchronizing.

        Tensor views (all CSR slices and device level bases) are deduplicated
        by backing storage. Rotations and codebooks are separated: production
        uses a different rotation seed per layer, while codebooks are shared by
        every layer with the same device/head-dim/bit level.
        """
        seen: set[tuple[str, int | None, int]] = set()

        def unique_bytes(tensors) -> int:
            total = 0
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor):
                    continue
                storage = tensor.untyped_storage()
                size = int(storage.nbytes())
                if size == 0:
                    continue
                key = (tensor.device.type, tensor.device.index, storage.data_ptr())
                if key in seen:
                    continue
                seen.add(key)
                total += size
            return total

        payload = unique_bytes(
            (
                self.code_arena_k,
                self.code_arena_v,
                self.norm_arena_k,
                self.norm_arena_v,
                self.exact["keys"],
                self.exact["values"],
                *(() if self.decode_arena is None else (
                    self.decode_arena.code_arena_k,
                    self.decode_arena.code_arena_v,
                    self.decode_arena.norm_arena_k,
                    self.decode_arena.norm_arena_v,
                    self.decode_arena.exact_k,
                    self.decode_arena.exact_v,
                )),
            )
        )
        metadata_tensors: list[torch.Tensor | None] = [
            self.tags,
            self.overflow_flag,
            self.required_slots,
            self.required_code_bytes,
            self.decode_tags,
        ]
        for bank in self.quant_banks:
            metadata_tensors.extend(
                bank.get(field)
                for field in (
                    "offset",
                    "seqlen",
                    "stable_row_capacity",
                    "code_base",
                    "norm_base",
                )
            )
        metadata_tensors.extend(
            self.exact.get(field)
            for field in ("offset", "seqlen", "stable_row_capacity")
        )
        if self.decode_arena is not None:
            metadata_tensors.extend(
                (
                    self.decode_arena.code_row_base,
                    self.decode_arena.norm_row_base,
                    self.decode_arena.exact_row_base,
                    self.decode_arena.counts,
                    self.decode_arena.descriptors,
                    self.decode_arena.descriptor_counts,
                    self.decode_arena.bump,
                    self.decode_arena.overflow_flag,
                )
            )
        metadata = unique_bytes(metadata_tensors)
        rotation = unique_bytes((self.pi_k, self.pi_k_decode, self.pi_v))
        codebook = unique_bytes(
            (
                *(bank["cent_k"] for bank in self.quant_banks),
                *(bank["cent_v"] for bank in self.quant_banks),
                *(
                    ()
                    if self.decode_arena is None
                    else (self.decode_arena.unified_codebook,)
                ),
            )
        )
        shared = rotation + codebook
        dense = (
            self.batch_size
            * self.num_heads
            * self.total_seq_len
            * self.head_dim
            * 2
            * torch.empty((), dtype=self.storage_dtype).element_size()
        )
        return {
            "reserved_payload_bytes": payload,
            "metadata_bytes": metadata,
            "rotation_bytes": rotation,
            "shared_codebook_bytes": codebook,
            "shared_rotation_codebook_bytes": shared,
            "layer_owned_reserved_bytes": payload + metadata + rotation,
            "total_reserved_bytes": payload + metadata + shared,
            "dense_kv_bytes": dense,
            "reserved_ratio_vs_dense": (payload + metadata + shared) / max(dense, 1),
        }

    def live_storage_stats(self) -> dict[str, Any]:
        """Explicitly synchronized live-byte accounting for audit/reporting."""
        # Seqlens are the authoritative live counts after mixed-bit appends;
        # required_slots intentionally includes unused fixed-address reserve.
        quant_counts = [
            int(bank["seqlen"].sum(dtype=torch.int64).detach().cpu())
            for bank in self.quant_banks
        ]
        exact_count = int(
            self.exact["seqlen"].sum(dtype=torch.int64).detach().cpu()
        )
        decode_stats = (
            self.decode_arena.live_storage_stats()
            if self.decode_arena is not None else None
        )
        code_bytes = 2 * sum(
            count * _packed_width(bits, self.head_dim)
            for bits, count in zip(QUANTIZED_LEVELS, quant_counts)
        )
        norm_bytes = 2 * sum(quant_counts) * torch.empty((), dtype=torch.float32).element_size()
        exact_bytes = (
            exact_count
            * self.head_dim
            * 2
            * torch.empty((), dtype=self.storage_dtype).element_size()
        )
        dense = (
            self.batch_size
            * self.num_heads
            * self.total_seq_len
            * self.head_dim
            * 2
            * torch.empty((), dtype=self.storage_dtype).element_size()
        )
        payload = code_bytes + norm_bytes + exact_bytes
        if decode_stats is not None:
            code_bytes += decode_stats["live_code_bytes"]
            norm_bytes += decode_stats["live_norm_bytes"]
            exact_bytes += decode_stats["live_exact_bytes"]
            payload = code_bytes + norm_bytes + exact_bytes
            quant_decode = decode_stats["quantized_tokens"]
            exact_decode = decode_stats["exact_tokens"]
        else:
            quant_decode = 0
            exact_decode = 0
        return {
            "quantized_tokens": sum(quant_counts) + quant_decode,
            "exact_tokens": exact_count + exact_decode,
            "live_code_bytes": code_bytes,
            "live_norm_bytes": norm_bytes,
            "live_exact_bytes": exact_bytes,
            "live_payload_bytes": payload,
            "dense_kv_bytes": dense,
            "live_payload_ratio_vs_dense": payload / max(dense, 1),
        }


def _encode_workspace(
    rotated: torch.Tensor,
    storage_map: torch.Tensor,
    code_arena: torch.Tensor,
    banks: tuple[dict[str, Any], ...],
    required_slots: torch.Tensor,
    slot_bases: torch.Tensor,
    code_bases: torch.Tensor,
    launch_capacities: tuple[int, int, int, int],
    head_dim: int,
) -> None:
    for level_index, bits in enumerate(QUANTIZED_LEVELS):
        capacity = int(launch_capacities[level_index])
        if capacity == 0:
            continue
        width = _packed_width(bits, head_dim)
        block_p = min(64, triton.next_power_of_2(width))
        grid = (capacity, triton.cdiv(width, block_p))
        common = dict(
            capacity=capacity,
            code_capacity=code_arena.numel(),
            workspace_capacity=rotated.shape[0],
            head_dim=head_dim,
            width=width,
            BLOCK_P=block_p,
            num_warps=4,
        )
        if bits == 2:
            _encode_2bit_kernel[grid](
                rotated, storage_map, banks[level_index]["cent_k"], code_arena, required_slots,
                slot_bases, code_bases,
                **common,
            )
        elif bits == 3:
            _encode_3bit_kernel[grid](
                rotated,
                storage_map,
                banks[level_index]["cent_k"],
                code_arena,
                required_slots,
                slot_bases,
                code_bases,
                **common,
            )
        elif bits == 4:
            _encode_nibble_kernel[grid](
                rotated,
                storage_map,
                banks[level_index]["cent_k"],
                code_arena,
                required_slots,
                slot_bases,
                code_bases,
                required_index=level_index,
                num_centroids=1 << bits,
                **common,
            )
        else:
            _encode_8bit_kernel[grid](
                rotated,
                storage_map,
                banks[level_index]["cent_k"],
                code_arena,
                required_slots,
                slot_bases,
                code_bases,
                capacity=capacity,
                code_capacity=code_arena.numel(),
                workspace_capacity=rotated.shape[0],
                head_dim=head_dim,
                BLOCK_P=block_p,
                num_warps=4,
            )


@torch.no_grad()
def pack_native_prefill_v2(
    keys: torch.Tensor,
    values: torch.Tensor,
    bit_tags: torch.Tensor,
    *,
    capacity: NativePackingCapacity,
    seed: int = 42,
) -> NativePackedKVV2:
    """Compact dense prefill K/V asynchronously into fixed payload arenas."""
    if not _HAS_TRITON or not keys.is_cuda:
        raise RuntimeError("native packing v2 requires CUDA and Triton")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("expected matching K/V [B,H,T,D]")
    if bit_tags.shape != keys.shape[:3]:
        raise ValueError(f"expected bit tags {keys.shape[:3]}, got {bit_tags.shape}")
    if keys.device != values.device or keys.device != bit_tags.device:
        raise ValueError("K, V, and tags must share a CUDA device")
    if bit_tags.dtype not in SUPPORTED_TAG_DTYPES:
        raise TypeError(
            "bit tags must use uint8/int8/int16/int32/int64, "
            f"got {bit_tags.dtype}"
        )

    batch_size, num_heads, seq_len, head_dim = keys.shape
    if seq_len == 0:
        raise ValueError("native packing v2 currently requires a non-empty prefill")
    rows = batch_size * num_heads
    chunks = triton.cdiv(seq_len, _RANK_CHUNK)
    device = keys.device
    source_tags = bit_tags.contiguous()
    tags = source_tags.to(torch.uint8).contiguous()

    ranks = torch.empty(rows, seq_len, device=device, dtype=torch.int32)
    chunk_counts = torch.empty(
        len(STORAGE_LEVELS), rows, chunks, device=device, dtype=torch.int32,
    )
    invalid_tag = torch.zeros(1, device=device, dtype=torch.int32)
    _rank_and_chunk_count_kernel_v2[(rows, chunks)](
        source_tags,
        ranks,
        chunk_counts,
        invalid_tag,
        seq_len=seq_len,
        chunks=chunks,
        rows=rows,
        BLOCK=_RANK_CHUNK,
        num_warps=8,
    )
    torch._assert_async(
        invalid_tag == 0,
        "native packing bit tags must be one of {0,2,3,4,8,16}",
    )
    chunk_prefix = torch.cumsum(chunk_counts, dim=-1, dtype=torch.int32) - chunk_counts
    row_counts = chunk_counts.sum(dim=-1, dtype=torch.int32)
    compute_row_offsets = torch.cumsum(
        row_counts[:4], dim=1, dtype=torch.int32,
    ) - row_counts[:4]
    live_required_slots = row_counts[:4].sum(dim=1, dtype=torch.int32)
    decode_reserves = capacity.resolved_decode_reserves()
    row_strides = row_counts.clone()
    for level_index, reserve in enumerate(decode_reserves):
        if reserve:
            row_strides[level_index].add_(reserve)
    row_offsets = torch.cumsum(row_strides, dim=1, dtype=torch.int32) - row_strides
    required_slots = row_strides.sum(dim=1, dtype=torch.int32)

    level_capacities = capacity.resolved_level_slots(rows)
    exact_capacity = capacity.resolved_exact_slots(rows)
    all_capacities = (*level_capacities, exact_capacity)
    capacity_tensor = torch.tensor(all_capacities, device=device, dtype=torch.int32)
    overflow_flag = (required_slots > capacity_tensor).to(torch.uint8).amax().reshape(1)

    slot_bases_list: list[int] = []
    total_slots = 0
    for level_capacity in level_capacities:
        slot_bases_list.append(total_slots)
        total_slots += int(level_capacity)
    slot_bases = tuple(slot_bases_list)

    compute_slot_bases_list: list[int] = []
    total_compute_slots = 0
    for level_capacity in capacity.level_slots:
        compute_slot_bases_list.append(total_compute_slots)
        total_compute_slots += int(level_capacity)
    compute_slot_bases = tuple(compute_slot_bases_list)

    code_bases_list: list[int] = []
    total_code_bytes = 0
    for bits, level_capacity in zip(QUANTIZED_LEVELS, level_capacities):
        code_bases_list.append(total_code_bytes)
        total_code_bytes += int(level_capacity) * _packed_width(bits, head_dim)
    code_bases = tuple(code_bases_list)

    code_arena_k = torch.empty(total_code_bytes, device=device, dtype=torch.uint8)
    code_arena_v = torch.empty_like(code_arena_k)
    # Every live prefill norm is written below and reserved decode cells stay
    # outside seqlen until append writes them.  Zero-filling either region is
    # pure bandwidth and would make reserve size inflate prefill latency.
    norm_arena_k = torch.empty(total_slots, device=device, dtype=torch.float32)
    norm_arena_v = torch.empty_like(norm_arena_k)
    exact_k = torch.empty(exact_capacity, head_dim, device=device, dtype=keys.dtype)
    exact_v = torch.empty_like(exact_k)

    pi_k, pi_k_decode, pi_v = _shared_rotations(head_dim, device, seed)

    banks: list[dict[str, Any]] = []
    for level_index, bits in enumerate(QUANTIZED_LEVELS):
        level_capacity = level_capacities[level_index]
        width = _packed_width(bits, head_dim)
        code_base = code_bases[level_index]
        slot_base = slot_bases[level_index]
        centroids = _shared_codebook(bits, head_dim, device)
        banks.append(
            {
                "bits": bits,
                "physical_bits": bits,
                "packed_k": code_arena_k.narrow(0, code_base, level_capacity * width).view(
                    level_capacity, width
                ),
                "packed_v": code_arena_v.narrow(0, code_base, level_capacity * width).view(
                    level_capacity, width
                ),
                "norms_k": norm_arena_k.narrow(0, slot_base, level_capacity),
                "norms_v": norm_arena_v.narrow(0, slot_base, level_capacity),
                "cent_k": centroids,
                "cent_v": centroids,
                "offset": row_offsets[level_index],
                "seqlen": row_counts[level_index],
                "stable_row_capacity": row_strides[level_index],
                "T_max": seq_len + decode_reserves[level_index],
            }
        )
    quant_banks = tuple(banks)

    exact = {
        "keys": exact_k,
        "values": exact_v,
        "offset": row_offsets[4],
        "seqlen": row_counts[4],
        "stable_row_capacity": row_strides[4],
        "T_max": seq_len + decode_reserves[4],
    }

    # Exact cells bypass normalization/rotation and go straight to final CSR.
    block_d = triton.next_power_of_2(head_dim)
    _compact_exact_kernel[(rows * seq_len,)](
        keys,
        values,
        tags,
        ranks,
        chunk_prefix,
        row_offsets,
        exact_k,
        exact_v,
        seq_len=seq_len,
        head_dim=head_dim,
        num_heads=num_heads,
        key_stride_b=keys.stride(0),
        key_stride_h=keys.stride(1),
        key_stride_t=keys.stride(2),
        key_stride_d=keys.stride(3),
        value_stride_b=values.stride(0),
        value_stride_h=values.stride(1),
        value_stride_t=values.stride(2),
        value_stride_d=values.stride(3),
        chunks=chunks,
        rank_chunk=_RANK_CHUNK,
        exact_capacity=exact_capacity,
        BLOCK_D=block_d,
        num_warps=4,
    )

    # Reuse the same two payload-sized workspaces for K then V.  There is no
    # level-sized selected tensor and no packed-index temporary.
    normalized = torch.empty(
        total_compute_slots, head_dim, device=device, dtype=torch.float32,
    )
    rotated = torch.empty_like(normalized)
    storage_map = torch.empty(
        (total_compute_slots,), device=device, dtype=torch.int32,
    )
    slot_bases_tensor = torch.tensor(slot_bases, device=device, dtype=torch.int32)
    compute_slot_bases_tensor = torch.tensor(
        compute_slot_bases, device=device, dtype=torch.int32,
    )
    code_bases_tensor = torch.tensor(code_bases, device=device, dtype=torch.int32)
    level_capacities_tensor = capacity_tensor[:4]
    compute_capacities_tensor = torch.tensor(
        capacity.level_slots, device=device, dtype=torch.int32,
    )

    for source, norms, rotation, arena, write_map in (
        (keys, norm_arena_k, pi_k, code_arena_k, True),
        (values, norm_arena_v, pi_v, code_arena_v, False),
    ):
        _compact_normalize_kernel[(rows * seq_len,)](
            source,
            tags,
            ranks,
            chunk_prefix,
            row_offsets,
            compute_row_offsets,
            slot_bases_tensor,
            compute_slot_bases_tensor,
            level_capacities_tensor,
            compute_capacities_tensor,
            normalized,
            norms,
            storage_map,
            seq_len=seq_len,
            head_dim=head_dim,
            num_heads=num_heads,
            source_stride_b=source.stride(0),
            source_stride_h=source.stride(1),
            source_stride_t=source.stride(2),
            source_stride_d=source.stride(3),
            chunks=chunks,
            rows=rows,
            rank_chunk=_RANK_CHUNK,
            WRITE_MAP=write_map,
            BLOCK_D=block_d,
            num_warps=4,
        )
        torch.mm(normalized, rotation.T, out=rotated)
        _encode_workspace(
            rotated,
            storage_map,
            arena,
            quant_banks,
            live_required_slots,
            compute_slot_bases_tensor,
            code_bases_tensor,
            capacity.level_slots,
            head_dim,
        )

    return NativePackedKVV2(
        quant_banks=quant_banks,
        exact=exact,
        pi_k=pi_k,
        pi_k_decode=pi_k_decode,
        pi_v=pi_v,
        tags=tags,
        capacity=capacity,
        required_slots=required_slots,
        required_code_bytes=None,
        overflow_flag=overflow_flag,
        code_arena_k=code_arena_k,
        code_arena_v=code_arena_v,
        norm_arena_k=norm_arena_k,
        norm_arena_v=norm_arena_v,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        storage_dtype=keys.dtype,
        decode_tags=(
            torch.empty(
                batch_size,
                num_heads,
                capacity.decode_capacity_per_row,
                device=device,
                dtype=torch.uint8,
            )
            if capacity.decode_capacity_per_row
            else None
        ),
    )


@torch.no_grad()
def pack_native_prefill_shared_v2(
    keys: torch.Tensor,
    values: torch.Tensor,
    bit_tags: torch.Tensor,
    *,
    capacity: NativeSharedPackingCapacity,
    seed: int = 42,
) -> NativePackedKVV2:
    """Pack into aggregate quant arenas with device-resident level bases.

    This is the production candidate.  It removes the independent worst-case
    reservation of :func:`pack_native_prefill_v2`; the fused decode descriptor
    reads ``code_base`` and ``norm_base`` directly from device metadata.
    """
    if not _HAS_TRITON or not keys.is_cuda:
        raise RuntimeError("native shared packing v2 requires CUDA and Triton")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("expected matching K/V [B,H,T,D]")
    if bit_tags.shape != keys.shape[:3]:
        raise ValueError(f"expected bit tags {keys.shape[:3]}, got {bit_tags.shape}")
    if keys.device != values.device or keys.device != bit_tags.device:
        raise ValueError("K, V, and tags must share a CUDA device")
    if bit_tags.dtype not in SUPPORTED_TAG_DTYPES:
        raise TypeError(
            "bit tags must use uint8/int8/int16/int32/int64, "
            f"got {bit_tags.dtype}"
        )

    batch_size, num_heads, seq_len, head_dim = keys.shape
    if seq_len == 0:
        raise ValueError("native shared packing v2 currently requires a non-empty prefill")
    rows = batch_size * num_heads
    chunks = triton.cdiv(seq_len, _RANK_CHUNK)
    device = keys.device
    source_tags = bit_tags.contiguous()
    tags = source_tags.to(torch.uint8).contiguous()

    ranks = torch.empty(rows, seq_len, device=device, dtype=torch.int32)
    chunk_counts = torch.empty(
        len(STORAGE_LEVELS), rows, chunks, device=device, dtype=torch.int32,
    )
    invalid_tag = torch.zeros(1, device=device, dtype=torch.int32)
    _rank_and_chunk_count_kernel_v2[(rows, chunks)](
        source_tags,
        ranks,
        chunk_counts,
        invalid_tag,
        seq_len=seq_len,
        chunks=chunks,
        rows=rows,
        BLOCK=_RANK_CHUNK,
        num_warps=8,
    )
    torch._assert_async(
        invalid_tag == 0,
        "native packing bit tags must be one of {0,2,3,4,8,16}",
    )
    chunk_prefix = torch.cumsum(chunk_counts, dim=-1, dtype=torch.int32) - chunk_counts
    row_counts = chunk_counts.sum(dim=-1, dtype=torch.int32)
    compute_row_offsets = torch.cumsum(
        row_counts[:4], dim=1, dtype=torch.int32,
    ) - row_counts[:4]
    live_required_slots = row_counts[:4].sum(dim=1, dtype=torch.int32)
    decode_reserves = capacity.resolved_decode_reserves()
    row_strides = row_counts.clone()
    for level_index, reserve in enumerate(decode_reserves):
        if reserve:
            row_strides[level_index].add_(reserve)
    row_offsets = torch.cumsum(row_strides, dim=1, dtype=torch.int32) - row_strides
    required_slots = row_strides.sum(dim=1, dtype=torch.int32)

    quant_required = required_slots[:4]
    slot_bases = torch.cumsum(quant_required, dim=0, dtype=torch.int32) - quant_required
    compute_slot_bases = (
        torch.cumsum(live_required_slots, dim=0, dtype=torch.int32)
        - live_required_slots
    )
    widths = torch.tensor(
        [_packed_width(bits, head_dim) for bits in QUANTIZED_LEVELS],
        device=device,
        dtype=torch.int32,
    )
    level_code_bytes = quant_required * widths
    code_bases = torch.cumsum(level_code_bytes, dim=0, dtype=torch.int32) - level_code_bytes
    required_code_bytes = level_code_bytes.sum(dtype=torch.int32).reshape(1)
    required_quant_slots = quant_required.sum(dtype=torch.int32).reshape(1)
    launch_capacity_tensor = torch.tensor(
        capacity.level_launch_slots, device=device, dtype=torch.int32,
    )
    reserve_slots = tuple(rows * reserve for reserve in decode_reserves)
    workspace_capacity = max(int(capacity.quant_slots) - sum(reserve_slots[:4]), 0)
    required_compute_slots = live_required_slots.sum(dtype=torch.int32).reshape(1)
    overflow_flag = torch.stack(
        (
            required_quant_slots[0] > capacity.quant_slots,
            required_code_bytes[0] > capacity.code_bytes_per_tensor,
            required_slots[4] > capacity.exact_slots,
            (live_required_slots > launch_capacity_tensor).any(),
            required_compute_slots[0] > workspace_capacity,
        )
    ).to(torch.uint8).amax().reshape(1)

    # ``capacity`` is a worst case that prices *every* level as if it consumed
    # the whole nominal budget, so the code arena and the 16-bit arena are each
    # provisioned for the full budget even though one allocation cannot fill
    # both.  The exact requirement is already resolved on device above, and the
    # bases/offsets that address the arenas are those same cumsums — so reading
    # them back once per layer lets us allocate byte-exact storage with every
    # scatter still in bounds by construction.  ``R2_EXACT_ARENA=0`` restores
    # the archived worst-case sizing for A/B comparison.
    plan_slots = (
        int(capacity.quant_slots),
        int(capacity.code_bytes_per_tensor),
        int(capacity.exact_slots),
        max(int(capacity.quant_slots) - sum(reserve_slots[:4]), 0),
        *(int(value) for value in capacity.level_launch_slots),
    )
    if os.environ.get("R2_EXACT_ARENA", "1") != "0":
        plan_slots = tuple(
            int(value) for value in torch.stack(
                (
                    required_quant_slots[0],
                    required_code_bytes[0],
                    required_slots[4],
                    required_compute_slots[0],
                    *live_required_slots.unbind(0),
                )
            ).tolist()
        )
    quant_slots_alloc = max(plan_slots[0], 1)
    code_bytes_alloc = max(plan_slots[1], 1)
    exact_slots_alloc = max(plan_slots[2], 1)
    workspace_capacity = plan_slots[3]
    launch_capacities = tuple(plan_slots[4:8])

    code_arena_k = torch.empty(
        code_bytes_alloc, device=device, dtype=torch.uint8,
    )
    code_arena_v = torch.empty_like(code_arena_k)
    norm_arena_k = torch.empty(
        quant_slots_alloc, device=device, dtype=torch.float32,
    )
    norm_arena_v = torch.empty_like(norm_arena_k)
    exact_k = torch.empty(exact_slots_alloc, head_dim, device=device, dtype=keys.dtype)
    exact_v = torch.empty_like(exact_k)
    pi_k, pi_k_decode, pi_v = _shared_rotations(head_dim, device, seed)

    banks: list[dict[str, Any]] = []
    for level_index, bits in enumerate(QUANTIZED_LEVELS):
        centroids = _shared_codebook(bits, head_dim, device)
        banks.append(
            {
                "bits": bits,
                "physical_bits": bits,
                # Shared 1-D arenas; fused decode adds the device base before
                # its ordinary row offset. No dynamic Tensor view is needed.
                "packed_k": code_arena_k,
                "packed_v": code_arena_v,
                "norms_k": norm_arena_k,
                "norms_v": norm_arena_v,
                "packed_width": _packed_width(bits, head_dim),
                "code_base": code_bases[level_index : level_index + 1],
                "norm_base": slot_bases[level_index : level_index + 1],
                "cent_k": centroids,
                "cent_v": centroids,
                "offset": row_offsets[level_index],
                "seqlen": row_counts[level_index],
                "stable_row_capacity": row_strides[level_index],
                "T_max": seq_len + decode_reserves[level_index],
            }
        )
    quant_banks = tuple(banks)
    exact = {
        "keys": exact_k,
        "values": exact_v,
        "offset": row_offsets[4],
        "seqlen": row_counts[4],
        "stable_row_capacity": row_strides[4],
        "T_max": seq_len + decode_reserves[4],
    }

    block_d = triton.next_power_of_2(head_dim)
    _compact_exact_kernel[(rows * seq_len,)](
        keys,
        values,
        tags,
        ranks,
        chunk_prefix,
        row_offsets,
        exact_k,
        exact_v,
        seq_len=seq_len,
        head_dim=head_dim,
        num_heads=num_heads,
        key_stride_b=keys.stride(0),
        key_stride_h=keys.stride(1),
        key_stride_t=keys.stride(2),
        key_stride_d=keys.stride(3),
        value_stride_b=values.stride(0),
        value_stride_h=values.stride(1),
        value_stride_t=values.stride(2),
        value_stride_d=values.stride(3),
        chunks=chunks,
        rank_chunk=_RANK_CHUNK,
        exact_capacity=exact_slots_alloc,
        BLOCK_D=block_d,
        num_warps=4,
    )

    normalized = torch.empty(
        workspace_capacity, head_dim, device=device, dtype=torch.float32,
    )
    rotated = torch.empty_like(normalized)
    storage_map = torch.empty(
        (workspace_capacity,), device=device, dtype=torch.int32,
    )
    # A level may start beyond the aggregate arena only when overflow is
    # already flagged. Clamping remaining capacity keeps every scatter bounded.
    level_capacity_guard = (
        torch.full_like(slot_bases, quant_slots_alloc) - slot_bases
    ).clamp_min_(0)
    compute_capacity_guard = (
        torch.full_like(compute_slot_bases, workspace_capacity) - compute_slot_bases
    ).clamp_min_(0)
    for source, norms, rotation, arena, write_map in (
        (keys, norm_arena_k, pi_k, code_arena_k, True),
        (values, norm_arena_v, pi_v, code_arena_v, False),
    ):
        _compact_normalize_kernel[(rows * seq_len,)](
            source,
            tags,
            ranks,
            chunk_prefix,
            row_offsets,
            compute_row_offsets,
            slot_bases,
            compute_slot_bases,
            level_capacity_guard,
            compute_capacity_guard,
            normalized,
            norms,
            storage_map,
            seq_len=seq_len,
            head_dim=head_dim,
            num_heads=num_heads,
            source_stride_b=source.stride(0),
            source_stride_h=source.stride(1),
            source_stride_t=source.stride(2),
            source_stride_d=source.stride(3),
            chunks=chunks,
            rows=rows,
            rank_chunk=_RANK_CHUNK,
            WRITE_MAP=write_map,
            BLOCK_D=block_d,
            num_warps=4,
        )
        torch.mm(normalized, rotation.T, out=rotated)
        _encode_workspace(
            rotated,
            storage_map,
            arena,
            quant_banks,
            live_required_slots,
            compute_slot_bases,
            code_bases,
            launch_capacities,
            head_dim,
        )

    return NativePackedKVV2(
        quant_banks=quant_banks,
        exact=exact,
        pi_k=pi_k,
        pi_k_decode=pi_k_decode,
        pi_v=pi_v,
        tags=tags,
        capacity=capacity,
        required_slots=required_slots,
        required_code_bytes=required_code_bytes,
        overflow_flag=overflow_flag,
        code_arena_k=code_arena_k,
        code_arena_v=code_arena_v,
        norm_arena_k=norm_arena_k,
        norm_arena_v=norm_arena_v,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        storage_dtype=keys.dtype,
        decode_tags=(
            torch.empty(
                batch_size,
                num_heads,
                capacity.decode_capacity_per_row,
                device=device,
                dtype=torch.uint8,
            )
            if capacity.decode_capacity_per_row
            else None
        ),
    )


__all__ = [
    "NativePackingPolicy",
    "NativePackedKVV2",
    "NativePackingCapacity",
    "NativeSharedPackingCapacity",
    "QUANTIZED_LEVELS",
    "STORAGE_LEVELS",
    "pack_native_prefill_v2",
    "pack_native_prefill_shared_v2",
]
