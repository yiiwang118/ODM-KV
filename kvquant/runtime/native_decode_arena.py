"""Pointer-stable aggregate payload arena for mixed-bit decode flushes.

Prefill banks remain compact CSR regions. Periodic decode flushes are stored as
fixed-descriptor segments in these aggregate arenas: all quantized bit levels
share one K byte arena, one V byte arena, and one pair of norm arenas; exact
tokens share bounded BF16 arenas. Device bump pointers assign every segment's
absolute row bases. Thus mutually exclusive 2/3/4/8/16 outcomes consume one
budgeted pool instead of five worst-case per-row suffixes.

The arena owns immutable tensor addresses and a fixed descriptor shape. CUDA
graphs may capture attention before the first flush: descriptor counts start at
zero and later flushes only populate existing cells. A 0-bit token has no entry
in any payload descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading

import torch


QUANT_LEVELS: tuple[int, ...] = (2, 3, 4, 8)
PAYLOAD_LEVELS: tuple[int, ...] = (2, 3, 4, 8, 16)
_UNIFIED_CODEBOOK_LOCK = threading.Lock()
_UNIFIED_CODEBOOK_CACHE: dict[
    tuple[str, int | None, tuple[int, ...]],
    tuple[
        torch.Tensor,
        torch.cuda.Event | None,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
] = {}


def _shared_unified_codebook(codebooks: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Concatenate 2/3/4/8-bit scalar codebooks once per device/source set."""
    if len(codebooks) != 4 or any(table.ndim != 1 for table in codebooks):
        raise ValueError("unified decode codebook requires four flat scalar tables")
    if tuple(table.numel() for table in codebooks) != (4, 8, 16, 256):
        raise ValueError("unified decode codebook has an invalid level cardinality")
    device = codebooks[0].device
    dtype = codebooks[0].dtype
    if (
        device.type != "cuda"
        or any(table.device != device for table in codebooks)
        or any(table.dtype != dtype for table in codebooks)
    ):
        raise ValueError("unified decode codebooks must share one CUDA device and dtype")
    key = (device.type, device.index, tuple(table.data_ptr() for table in codebooks))
    with _UNIFIED_CODEBOOK_LOCK:
        entry = _UNIFIED_CODEBOOK_CACHE.get(key)
        if entry is None:
            unified = torch.cat(codebooks).contiguous()
            ready = torch.cuda.Event(blocking=False)
            ready.record(torch.cuda.current_stream(device))
            # Pin the source tables together with the concatenation.  A cache
            # key based only on data pointers is otherwise vulnerable to a
            # stale hit if an allocator later reuses all four addresses after
            # the original packed cache has been destroyed.
            entry = (unified, ready, codebooks)
            _UNIFIED_CODEBOOK_CACHE[key] = entry
        unified, ready, sources = entry
        if ready is not None and ready.query():
            ready = None
            _UNIFIED_CODEBOOK_CACHE[key] = (unified, None, sources)
    if ready is not None:
        torch.cuda.current_stream(device).wait_event(ready)
    return unified


def packed_width(bits: int, head_dim: int) -> int:
    if bits == 3:
        return (3 * int(head_dim) + 7) // 8
    return (int(head_dim) * int(bits) + 7) // 8


@dataclass
class NativeDecodeArena:
    """Aggregate mixed-bit suffix storage plus graph-visible descriptors."""

    code_arena_k: torch.Tensor
    code_arena_v: torch.Tensor
    norm_arena_k: torch.Tensor
    norm_arena_v: torch.Tensor
    exact_k: torch.Tensor
    exact_v: torch.Tensor
    code_row_base: torch.Tensor       # [S,4,R], absolute byte offset
    norm_row_base: torch.Tensor       # [S,4,R], absolute norm slot
    exact_row_base: torch.Tensor      # [S,R], absolute exact slot
    counts: torch.Tensor              # [S,5,R]
    descriptors: torch.Tensor         # [R,C], quant front / exact back
    descriptor_counts: torch.Tensor   # [2,R], quant/exact live descriptors
    unified_codebook: torch.Tensor    # [4+8+16+256], shared across layers
    bump: torch.Tensor                # int32 [code_bytes,norm_slots,exact_slots]
    overflow_flag: torch.Tensor       # int32 [1]
    rows: int
    head_dim: int
    capacity_per_row: int
    buffer_size: int
    max_segments: int
    target_avg_bits: float
    bit_levels: tuple[int, ...]
    segment_count: int = 0
    segment_starts: list[int] = field(default_factory=list)
    segment_lengths: list[int] = field(default_factory=list)

    @classmethod
    def allocate(
        cls,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        capacity_per_row: int,
        buffer_size: int,
        target_avg_bits: float,
        bit_levels: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        codebooks: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        rounding_bits_per_request: int = 16,
        code_slack_bytes: int = 256,
    ) -> "NativeDecodeArena":
        batch = int(batch_size)
        heads = int(num_heads)
        dim = int(head_dim)
        capacity = int(capacity_per_row)
        buffer = int(buffer_size)
        levels = tuple(sorted(set(int(level) for level in bit_levels)))
        if min(batch, heads, dim, capacity, buffer) < 0 or buffer == 0:
            raise ValueError("invalid segmented decode arena shape")
        if not levels or any(level not in (0, 2, 3, 4, 8, 16) for level in levels):
            raise ValueError(f"unsupported decode levels: {levels}")
        nonzero = tuple(level for level in levels if level != 0)
        if not nonzero:
            effective_target = 0.0
        else:
            lower = 0.0 if 0 in levels else float(nonzero[0])
            effective_target = max(
                lower, min(float(nonzero[-1]), float(target_avg_bits)),
            )
        rows = batch * heads
        max_segments = max(1, math.ceil(capacity / buffer))
        # Allocation is solved independently per request across H*T cells, and
        # once per periodic flush -- so the repair/rounding allowance is paid
        # per segment, not once for the whole generation.
        rounding_total = max_segments * batch * int(rounding_bits_per_request)
        nominal_bits = effective_target * rows * capacity + rounding_total
        # For K+V every storage level costs the same bytes per nominal bit
        # (``dim/8`` per tensor: a b-bit token needs ``b*dim/8`` and a 16-bit
        # token needs ``2*dim = 16*dim/8``).  So ONE pool of
        # ``nominal_bits * dim/8`` bytes provably covers any mix of quantized
        # codes and 16-bit slots, whereas sizing the two independently
        # provisions the whole budget twice -- and the second copy measurably
        # goes unused (decode 16-bit token count is 0 at target 1 and 2).
        # Codes grow up from byte 0; exact slots grow down from the top.
        slot_bytes = dim * torch.empty((), dtype=dtype).element_size()
        pool_bytes = (
            int(math.ceil(nominal_bits * dim / 8.0))
            + int(code_slack_bytes)
            + slot_bytes * batch          # per-request exact rounding allowance
        )
        # Slot-align so the bf16 view covers the pool exactly.
        pool_bytes = -(-pool_bytes // slot_bytes) * slot_bytes
        pool_slots = pool_bytes // slot_bytes
        # Norm slots are inexpensive and one per retained quantized token. The
        # full logical bound avoids coupling this metadata proof to bit budget.
        norm_capacity = rows * capacity
        # A level set whose floor is already 16 bits stores every retained cell
        # raw, so the joint pool has to hold one slot per logical cell.
        floor_level = 0.0 if 0 in levels else float(nonzero[0] if nonzero else 0)
        if floor_level >= 16.0:
            pool_slots = rows * capacity + batch
            pool_bytes = pool_slots * slot_bytes

        pool_k = torch.empty(pool_bytes, device=device, dtype=torch.uint8)
        pool_v = torch.empty_like(pool_k)
        # Two views over one allocation: byte-addressed codes and slot-addressed
        # 16-bit rows.  Collision between the two growth directions is proved on
        # device by the segment allocator, not by these views.
        code_k, code_v = pool_k, pool_v
        exact_k = pool_k.view(dtype).view(pool_slots, dim)
        exact_v = pool_v.view(dtype).view(pool_slots, dim)
        norm_k = torch.empty(norm_capacity, device=device, dtype=torch.float32)
        norm_v = torch.empty_like(norm_k)
        code_base = torch.zeros(
            max_segments, 4, rows, device=device, dtype=torch.int32,
        )
        norm_base = torch.zeros_like(code_base)
        exact_base = torch.zeros(
            max_segments, rows, device=device, dtype=torch.int32,
        )
        counts = torch.zeros(
            max_segments, 5, rows, device=device, dtype=torch.int32,
        )
        descriptors = torch.empty(
            rows, capacity, device=device, dtype=torch.int32,
        )
        descriptor_counts = torch.zeros(
            2, rows, device=device, dtype=torch.int32,
        )
        unified_codebook = _shared_unified_codebook(codebooks)
        bump = torch.zeros(3, device=device, dtype=torch.int32)
        overflow = torch.zeros(1, device=device, dtype=torch.int32)
        return cls(
            code_k, code_v, norm_k, norm_v, exact_k, exact_v,
            code_base, norm_base, exact_base, counts,
            descriptors, descriptor_counts,
            unified_codebook,
            bump, overflow,
            rows, dim, capacity, buffer, max_segments, effective_target, levels,
        )

    @property
    def code_capacity(self) -> int:
        return int(self.code_arena_k.numel())

    @property
    def norm_capacity(self) -> int:
        return int(self.norm_arena_k.numel())

    @property
    def exact_capacity(self) -> int:
        return int(self.exact_k.shape[0])

    @property
    def slot_bytes(self) -> int:
        """Bytes of one 16-bit K (or V) row, i.e. the exact-slot stride."""
        return int(self.head_dim * self.exact_k.element_size())

    @property
    def reserved_bytes(self) -> int:
        # ``code_arena_*`` and ``exact_*`` are two views over ONE pool, so count
        # distinct backing storages instead of summing views.
        tensors = (
            self.code_arena_k, self.code_arena_v,
            self.norm_arena_k, self.norm_arena_v,
            self.exact_k, self.exact_v,
            self.code_row_base, self.norm_row_base, self.exact_row_base,
            self.counts,
            self.descriptors, self.descriptor_counts,
            self.unified_codebook,
            self.bump, self.overflow_flag,
        )
        seen: set[int] = set()
        total = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.data_ptr() in seen:
                continue
            seen.add(storage.data_ptr())
            total += int(storage.nbytes())
        return total

    def validate_append(self, tokens: int, logical_start: int) -> int:
        tokens = int(tokens)
        if tokens > self.buffer_size:
            raise RuntimeError(
                f"segmented decode append exceeds descriptor stride: "
                f"tokens={tokens}, buffer_size={self.buffer_size}"
            )
        if logical_start + tokens > self.capacity_per_row:
            raise RuntimeError(
                f"segmented decode capacity exceeded: need={logical_start + tokens}, "
                f"capacity={self.capacity_per_row}"
            )
        # Segment metadata is sized for full ring flushes plus at most one
        # terminal remainder.  Accepting arbitrary short flushes would exhaust
        # ``max_segments`` before the advertised logical capacity and turn a
        # valid-looking append API into a delayed failure.
        if (
            tokens != self.buffer_size
            and logical_start + tokens != self.capacity_per_row
        ):
            raise RuntimeError(
                "segmented decode requires full-buffer flushes except for the "
                f"terminal remainder: start={logical_start}, tokens={tokens}, "
                f"buffer_size={self.buffer_size}, capacity={self.capacity_per_row}"
            )
        if self.segment_count >= self.max_segments:
            raise RuntimeError(
                f"segmented decode descriptor capacity exceeded: "
                f"need segment {self.segment_count + 1}, max={self.max_segments}"
            )
        return self.segment_count

    def publish_host_segment(self, logical_start: int, tokens: int) -> None:
        self.segment_starts.append(int(logical_start))
        self.segment_lengths.append(int(tokens))
        self.segment_count += 1

    def live_counts(self) -> dict[int, int]:
        """Explicitly synchronized audit counts, never used by production."""
        if self.segment_count == 0:
            return {level: 0 for level in PAYLOAD_LEVELS}
        totals = self.counts[: self.segment_count].sum(dim=(0, 2), dtype=torch.int64)
        host = totals.detach().cpu().tolist()
        return {level: int(host[index]) for index, level in enumerate(PAYLOAD_LEVELS)}

    def live_storage_stats(self) -> dict[str, int]:
        counts = self.live_counts()
        code_per_tensor = sum(
            counts[level] * packed_width(level, self.head_dim)
            for level in QUANT_LEVELS
        )
        norm_per_tensor = sum(counts[level] for level in QUANT_LEVELS) * 4
        exact_per_tensor = counts[16] * self.head_dim * self.exact_k.element_size()
        return {
            "segments": self.segment_count,
            "quant_descriptor_count": sum(
                counts[level] for level in QUANT_LEVELS
            ),
            "exact_descriptor_count": counts[16],
            "descriptor_reserved_bytes": (
                self.descriptors.numel() * self.descriptors.element_size()
                + self.descriptor_counts.numel() * self.descriptor_counts.element_size()
            ),
            "logical_capacity_per_row": self.capacity_per_row,
            "quantized_tokens": sum(counts[level] for level in QUANT_LEVELS),
            "exact_tokens": counts[16],
            "retained_tokens": sum(counts.values()),
            "evicted_tokens": self.rows * sum(self.segment_lengths) - sum(counts.values()),
            "live_code_bytes": 2 * code_per_tensor,
            "live_norm_bytes": 2 * norm_per_tensor,
            "live_exact_bytes": 2 * exact_per_tensor,
            "live_payload_bytes": 2 * (code_per_tensor + norm_per_tensor + exact_per_tensor),
        }
