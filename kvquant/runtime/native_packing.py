"""Direct GPU packing for the native compressed-attention path.

This module deliberately does *not* use ``TurboQuantAdaptiveKVCacheState``.
The old path first stored each token with an explicit ``(batch, head, seq)``
position and subsequently sorted/regrouped those banks for decode.  Prefill
already owns dense ``[B, H_kv, T, D]`` K/V tensors and dense bit tags, so that
intermediate representation is unnecessary: row-major selection is already
CSR order.

The supported production configuration is intentionally narrow:

* K and V use TurboQuant MSE codebooks;
* bit tags are ``{0, 2, 3, 4, 8, 16}`` (0 means physically evicted);
* outlier-channel splitting is disabled;
* one K rotation and one V rotation are shared by every bit level.

``pack_native_prefill`` selects all quantized tokens once, normalizes/rotates
K once and V once, then performs only the level-specific centroid lookup and
bit packing.  Its hot path has no host readback (``item``, ``tolist``, or
``mask.any``); the only Python loop is over the four compile-time bit levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from kvquant.tq_backend import (
    _load_or_compute_codebook,
    _pack_indices,
    _unpack_indices,
    random_rotation,
)
from kvquant.tq_triton import mse_nearest_centroid


QUANTIZED_LEVELS: tuple[int, ...] = (2, 3, 4, 8)
SUPPORTED_LEVELS: tuple[int, ...] = (0, 2, 3, 4, 8, 16)


def _packed_width(bits: int, head_dim: int) -> int:
    """Number of physical bytes used by one code vector.

    Three-bit codes intentionally use the existing nibble representation.  It
    is therefore a logical 3-bit codebook with a physical 4-bit index.
    """
    effective_bits = 2 if bits == 2 else 4 if bits in (3, 4) else 8
    values_per_byte = 8 // effective_bits
    return (head_dim + values_per_byte - 1) // values_per_byte


def _csr_metadata(rows: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int32 CSR ``offset`` and ``seqlen`` for already row-sorted data."""
    counts = torch.bincount(rows, minlength=num_rows).to(torch.int32)
    offset = torch.zeros(num_rows, dtype=torch.int32, device=rows.device)
    offset[1:] = counts[:-1].cumsum(0)
    return offset, counts


def _normalize_rotate(x: torch.Tensor, rotation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and rotate a compact token matrix exactly once."""
    x_float = x.float()
    norms = x_float.norm(dim=-1)
    rotated = (x_float / (norms.unsqueeze(-1) + 1e-10)) @ rotation.T
    return rotated, norms


@dataclass
class NativePackedKV:
    """Native prefill storage, with no saved per-token position triples.

    ``quant_banks`` can be passed directly to the ragged-bank portion of
    :func:`odmkv.kernels.fused_decode.fused_decode`.  ``exact`` is also CSR
    and is kept flat; the native decode integration should pass its ``offset``
    to the exact-source kernel instead of padding it to ``[rows, T_max, D]``.

    When ``decode_2bit_reserve_per_row`` is nonzero, the 2-bit bank is created
    at its final fixed capacity and decode rings append into the reserved
    suffix in place.  That is the storage contract needed by CUDA Graph replay.
    """

    quant_banks: tuple[dict[str, Any], ...]
    exact: dict[str, Any]
    pi_k: torch.Tensor
    pi_k_decode: torch.Tensor
    pi_v: torch.Tensor
    tags: torch.Tensor
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    storage_dtype: torch.dtype
    decode_2bit_reserve_per_row: int = 0
    decode_tags: torch.Tensor | None = None
    decode_len: int = 0

    @property
    def num_rows(self) -> int:
        return self.batch_size * self.num_heads

    def bank(self, bits: int) -> dict[str, Any]:
        """Return a fixed-level bank without data-dependent GPU inspection."""
        if bits not in QUANTIZED_LEVELS:
            raise KeyError(f"No quantized bank for {bits}-bit tags")
        return self.quant_banks[QUANTIZED_LEVELS.index(bits)]

    @property
    def total_seq_len(self) -> int:
        return self.seq_len + self.decode_len

    def _occupied_payload_indices(self, bank: dict[str, Any]) -> torch.Tensor:
        """Flat payload indices in row-major logical-token order."""
        capacity = bank.get("stable_row_capacity")
        if capacity is None:
            return torch.arange(
                bank["norms_k"].shape[0], device=self.tags.device,
            )
        # T_max is a host upper bound. The device seqlen mask selects only live
        # cells, so reserved holes are never dequantized or exposed.
        width = int(bank["T_max"])
        column = torch.arange(width, device=self.tags.device)
        live = column.view(1, -1) < bank["seqlen"].view(-1, 1)
        indices = bank["offset"].to(torch.long).view(-1, 1) + column.view(1, -1)
        return indices[live]

    @torch.no_grad()
    def append_decode_2bit(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Quantize a full decode ring directly into fixed 2-bit CSR storage.

        Every row receives the same ``T`` new positions in standard fixed-batch
        decode. No history tensor is concatenated and all payload pointers stay
        stable, so an already captured attention graph remains valid.
        """
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("decode K/V must be matching [B,H,T,D]")
        B, H, tokens, D = keys.shape
        if (B, H, D) != (self.batch_size, self.num_heads, self.head_dim):
            raise ValueError("decode K/V shape does not match packed prefill")
        if self.decode_tags is None or self.decode_2bit_reserve_per_row <= 0:
            raise RuntimeError("native 2-bit bank was created without decode reserve")
        if self.decode_len + tokens > self.decode_2bit_reserve_per_row:
            raise RuntimeError(
                f"native 2-bit decode reserve exceeded: need {self.decode_len + tokens}, "
                f"capacity={self.decode_2bit_reserve_per_row}"
            )
        bank = self.bank(2)
        if "stable_row_capacity" not in bank:
            raise RuntimeError("2-bit bank has no fixed CSR capacity")

        flat_k = keys.reshape(-1, D)
        flat_v = values.reshape(-1, D)
        rotated_k, norms_k = _normalize_rotate(flat_k, self.pi_k)
        rotated_v, norms_v = _normalize_rotate(flat_v, self.pi_v)
        idx_k = mse_nearest_centroid(rotated_k, bank["cent_k"])
        idx_v = mse_nearest_centroid(rotated_v, bank["cent_v"])
        packed_k = _pack_indices(idx_k, 2).contiguous()
        packed_v = _pack_indices(idx_v, 2).contiguous()

        rows = torch.arange(self.num_rows, device=keys.device).repeat_interleave(tokens)
        within = torch.arange(tokens, device=keys.device).repeat(self.num_rows)
        destination = (
            bank["offset"][rows].to(torch.long)
            + bank["seqlen"][rows].to(torch.long)
            + within
        )
        bank["packed_k"].index_copy_(0, destination, packed_k)
        bank["packed_v"].index_copy_(0, destination, packed_v)
        bank["norms_k"].index_copy_(0, destination, norms_k.float())
        bank["norms_v"].index_copy_(0, destination, norms_v.float())
        bank["seqlen"].add_(tokens)
        self.decode_tags[..., self.decode_len:self.decode_len + tokens].fill_(2)
        self.decode_len += int(tokens)

    def materialize(self, *, dtype: torch.dtype | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize into dense K/V; 0-bit positions remain exact zeros.

        This is a correctness oracle and fallback, not the decode hot path.
        Payload order within every bank is the same row-major order as the
        corresponding dense tag mask, so no position tensor is needed.
        """
        output_dtype = self.storage_dtype if dtype is None else dtype
        shape = (self.batch_size, self.num_heads, self.total_seq_len, self.head_dim)
        keys = torch.zeros(shape, device=self.tags.device, dtype=output_dtype)
        values = torch.zeros_like(keys)
        flat_k = keys.view(-1, self.head_dim)
        flat_v = values.view(-1, self.head_dim)
        tags = self.tags
        if self.decode_len:
            tags = torch.cat(
                (tags, self.decode_tags[..., :self.decode_len]), dim=-1,
            )
        flat_tags = tags.reshape(-1)

        for bits in QUANTIZED_LEVELS:
            bank = self.bank(bits)
            live = self._occupied_payload_indices(bank)
            idx_k = _unpack_indices(bank["packed_k"][live], bits, self.head_dim)
            idx_v = _unpack_indices(bank["packed_v"][live], bits, self.head_dim)
            deq_k = (bank["cent_k"][idx_k] @ self.pi_k) * bank["norms_k"].float().unsqueeze(-1)
            deq_v = (bank["cent_v"][idx_v] @ self.pi_v) * bank["norms_v"].float().unsqueeze(-1)
            flat_k[flat_tags == bits] = deq_k.to(output_dtype)
            flat_v[flat_tags == bits] = deq_v.to(output_dtype)

        exact_mask = flat_tags == 16
        flat_k[exact_mask] = self.exact["keys"].to(output_dtype)
        flat_v[exact_mask] = self.exact["values"].to(output_dtype)
        return keys, values

    def physical_storage_stats(self) -> dict[str, Any]:
        """Return byte-exact storage and physical-eviction accounting.

        Tensor sizes are derived from shape metadata only, so this method does
        not synchronize CUDA.  Rotation/codebook bytes are disclosed
        separately because they are one-per-layer shared metadata, whereas
        code/norm/exact bytes scale with tokens.
        """
        total_tokens = self.batch_size * self.num_heads * self.total_seq_len
        exact_tokens = self.exact["keys"].shape[0]
        quantized_tokens = 0
        payload_bytes = 0
        csr_bytes = 0
        codebook_bytes = 0
        per_level: dict[int, dict[str, int]] = {}

        for bits in QUANTIZED_LEVELS:
            bank = self.bank(bits)
            count = int(bank.get("prefill_tokens", bank["norms_k"].shape[0]))
            if bits == 2:
                count += self.decode_len * self.num_rows
            quantized_tokens += count
            code_bytes = _tensor_bytes(bank["packed_k"]) + _tensor_bytes(bank["packed_v"])
            norm_bytes = _tensor_bytes(bank["norms_k"]) + _tensor_bytes(bank["norms_v"])
            level_csr_bytes = _tensor_bytes(bank["offset"]) + _tensor_bytes(bank["seqlen"])
            # K/V use the same scalar MSE codebook tensor (two kernel arguments,
            # one physical allocation).
            level_codebook_bytes = _tensor_bytes(bank["cent_k"])
            payload_bytes += code_bytes + norm_bytes
            csr_bytes += level_csr_bytes
            codebook_bytes += level_codebook_bytes
            per_level[bits] = {
                "tokens": count,
                "logical_code_bits_per_scalar": bits,
                "physical_code_bits_per_scalar": 4 if bits == 3 else bits,
                "code_bytes_kv": code_bytes,
                "norm_bytes_kv": norm_bytes,
                "csr_bytes": level_csr_bytes,
            }

        evicted_tokens = total_tokens - quantized_tokens - exact_tokens
        exact_payload_bytes = _tensor_bytes(self.exact["keys"]) + _tensor_bytes(self.exact["values"])
        exact_csr_bytes = _tensor_bytes(self.exact["offset"]) + _tensor_bytes(self.exact["seqlen"])
        tags_bytes = _tensor_bytes(self.tags)
        if self.decode_tags is not None:
            tags_bytes += _tensor_bytes(self.decode_tags)
        rotation_bytes = (
            _tensor_bytes(self.pi_k)
            + _tensor_bytes(self.pi_k_decode)
            + _tensor_bytes(self.pi_v)
        )
        dense_bytes = total_tokens * self.head_dim * 2 * _dtype_bytes(self.storage_dtype)
        evicted_dense_bytes = evicted_tokens * self.head_dim * 2 * _dtype_bytes(self.storage_dtype)
        total_physical = (
            payload_bytes + exact_payload_bytes + tags_bytes + csr_bytes
            + exact_csr_bytes + rotation_bytes + codebook_bytes
        )
        return {
            "total_tokens": total_tokens,
            "quantized_tokens": quantized_tokens,
            "exact_tokens": exact_tokens,
            "evicted_tokens": evicted_tokens,
            "evicted_dense_payload_bytes": evicted_dense_bytes,
            "quantized_payload_bytes": payload_bytes,
            "exact_payload_bytes": exact_payload_bytes,
            "tags_bytes": tags_bytes,
            "csr_bytes": csr_bytes + exact_csr_bytes,
            "rotation_bytes": rotation_bytes,
            "codebook_bytes": codebook_bytes,
            "total_physical_bytes": total_physical,
            "dense_kv_bytes": dense_bytes,
            "physical_ratio_vs_dense": total_physical / max(dense_bytes, 1),
            "per_level": per_level,
        }


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _dtype_bytes(dtype: torch.dtype) -> int:
    # An empty scalar is enough to query a dtype and never allocates on CUDA.
    return torch.empty((), dtype=dtype).element_size()


@torch.no_grad()
def pack_native_prefill(
    keys: torch.Tensor,
    values: torch.Tensor,
    bit_tags: torch.Tensor,
    *,
    seed: int = 42,
    decode_2bit_reserve_per_row: int = 0,
) -> NativePackedKV:
    """Pack dense prefill K/V directly into decode-ready CSR storage.

    Args:
        keys, values: ``[B, H_kv, T, D]`` on the same device.
        bit_tags: integer ``[B, H_kv, T]`` tensor whose values are drawn from
            ``SUPPORTED_LEVELS``.  The allocation stage owns value validation;
            omitting a second GPU-to-host validation here keeps packing fully
            asynchronous.
        seed: K rotation seed.  V uses ``seed + 2000``, matching the existing
            MSE/MSE adaptive backend byte-for-byte.
        decode_2bit_reserve_per_row: fixed-capacity suffix allocated in every
            row of the 2-bit bank for direct graph-tail appends.
    """
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError(f"Expected matching K/V [B,H,T,D], got {keys.shape} and {values.shape}")
    if bit_tags.shape != keys.shape[:3]:
        raise ValueError(f"Expected bit tags {keys.shape[:3]}, got {bit_tags.shape}")
    if keys.device != values.device or keys.device != bit_tags.device:
        raise ValueError("K, V, and bit tags must be on the same device")
    if bit_tags.dtype.is_floating_point or bit_tags.dtype == torch.bool:
        raise TypeError(f"bit tags must be an integer tensor, got {bit_tags.dtype}")
    if decode_2bit_reserve_per_row < 0:
        raise ValueError("decode_2bit_reserve_per_row must be non-negative")

    batch_size, num_heads, seq_len, head_dim = keys.shape
    num_rows = batch_size * num_heads
    device = keys.device
    tags = bit_tags.to(dtype=torch.uint8).contiguous()
    flat_tags = tags.reshape(-1)
    flat_keys = keys.reshape(-1, head_dim)
    flat_values = values.reshape(-1, head_dim)

    # A single compacting selection is shared by K, V, tags, and row ids.
    quant_mask = (flat_tags > 0) & (flat_tags < 16)
    quant_flat_indices = torch.nonzero(quant_mask, as_tuple=False).squeeze(-1)
    quant_tags = flat_tags.index_select(0, quant_flat_indices)
    quant_rows = torch.div(quant_flat_indices, seq_len, rounding_mode="floor")
    quant_keys = flat_keys.index_select(0, quant_flat_indices)
    quant_values = flat_values.index_select(0, quant_flat_indices)

    compute_dtype = torch.float32
    pi_k = random_rotation(head_dim, device, compute_dtype, seed=seed)
    pi_v = random_rotation(head_dim, device, compute_dtype, seed=seed + 2000)
    rotated_k, norms_k = _normalize_rotate(quant_keys, pi_k)
    rotated_v, norms_v = _normalize_rotate(quant_values, pi_v)

    quant_banks: list[dict[str, Any]] = []
    for bits in QUANTIZED_LEVELS:
        level_mask = quant_tags == bits
        level_rows = quant_rows[level_mask]
        level_k = rotated_k[level_mask]
        level_v = rotated_v[level_mask]
        level_norms_k = norms_k[level_mask].contiguous()
        level_norms_v = norms_v[level_mask].contiguous()
        cent_k = _load_or_compute_codebook(bits, head_dim, device, compute_dtype)
        cent_v = cent_k
        if level_k.shape[0] == 0:
            # Triton does not accept a zero-sized launch grid.  Tensor shape is
            # host metadata, so this does not read a CUDA value or synchronize.
            indices_k = torch.empty_like(level_k, dtype=torch.int32)
            indices_v = torch.empty_like(level_v, dtype=torch.int32)
            packed_k = torch.empty(
                0, _packed_width(bits, head_dim), device=device, dtype=torch.uint8,
            )
            packed_v = torch.empty_like(packed_k)
        else:
            indices_k = mse_nearest_centroid(level_k, cent_k)
            indices_v = mse_nearest_centroid(level_v, cent_v)
            packed_k = _pack_indices(indices_k, bits).contiguous()
            packed_v = _pack_indices(indices_v, bits).contiguous()
        offset, seqlen = _csr_metadata(level_rows, num_rows)
        stable_row_capacity = None
        if bits == 2 and decode_2bit_reserve_per_row > 0:
            # Directly create the final fixed-address decode arena. Initial
            # compact payload is scattered once into each row's prefix; future
            # all-2-bit rings append into the reserved suffix in place.
            row_capacity = seqlen.to(torch.long) + decode_2bit_reserve_per_row
            stable_offset = torch.zeros_like(offset)
            stable_offset[1:] = row_capacity[:-1].cumsum(0).to(torch.int32)
            total_capacity = level_k.shape[0] + num_rows * decode_2bit_reserve_per_row
            width = _packed_width(bits, head_dim)
            stable_pk = torch.zeros(
                total_capacity, width, device=device, dtype=torch.uint8,
            )
            stable_pv = torch.zeros_like(stable_pk)
            stable_nk = torch.zeros(total_capacity, device=device, dtype=torch.float32)
            stable_nv = torch.zeros_like(stable_nk)
            if level_k.shape[0] != 0:
                within = torch.arange(level_k.shape[0], device=device) - offset[
                    level_rows
                ].to(torch.long)
                destination = stable_offset[level_rows].to(torch.long) + within
                stable_pk.index_copy_(0, destination, packed_k)
                stable_pv.index_copy_(0, destination, packed_v)
                stable_nk.index_copy_(0, destination, level_norms_k.float())
                stable_nv.index_copy_(0, destination, level_norms_v.float())
            packed_k, packed_v = stable_pk, stable_pv
            level_norms_k, level_norms_v = stable_nk, stable_nv
            offset = stable_offset
            stable_row_capacity = row_capacity.to(torch.int32)
        # ``T_max`` is a host integer consumed by the fused kernel's split-K
        # heuristic.  ``seq_len`` is a safe sync-free upper bound; empty levels
        # are statically visible from the compact tensor shape and use zero.
        t_max = (
            seq_len + decode_2bit_reserve_per_row
            if stable_row_capacity is not None
            else (seq_len if level_k.shape[0] != 0 else 0)
        )
        bank_dict = {
            "bits": bits,
            "physical_bits": 4 if bits == 3 else bits,
            "packed_k": packed_k,
            "norms_k": level_norms_k,
            "cent_k": cent_k,
            "packed_v": packed_v,
            "norms_v": level_norms_v,
            "cent_v": cent_v,
            "offset": offset,
            "seqlen": seqlen,
            "T_max": t_max,
            "prefill_tokens": level_k.shape[0],
        }
        if stable_row_capacity is not None:
            bank_dict["stable_row_capacity"] = stable_row_capacity
        quant_banks.append(bank_dict)

    exact_mask = flat_tags == 16
    exact_flat_indices = torch.nonzero(exact_mask, as_tuple=False).squeeze(-1)
    exact_rows = torch.div(exact_flat_indices, seq_len, rounding_mode="floor")
    exact_offset, exact_seqlen = _csr_metadata(exact_rows, num_rows)
    exact_keys = flat_keys.index_select(0, exact_flat_indices).contiguous()
    exact_values = flat_values.index_select(0, exact_flat_indices).contiguous()
    exact = {
        "keys": exact_keys,
        "values": exact_values,
        "offset": exact_offset,
        "seqlen": exact_seqlen,
        "T_max": seq_len if exact_keys.shape[0] != 0 else 0,
    }

    return NativePackedKV(
        quant_banks=tuple(quant_banks),
        exact=exact,
        pi_k=pi_k,
        # Decode rotates BF16 queries with a tensor-core matmul. Keep the
        # immutable cast next to the packed layer instead of replaying a
        # float32->bf16 conversion for every layer and generated token.
        pi_k_decode=pi_k.to(torch.bfloat16).contiguous(),
        pi_v=pi_v,
        tags=tags,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        storage_dtype=keys.dtype,
        decode_2bit_reserve_per_row=int(decode_2bit_reserve_per_row),
        decode_tags=(
            torch.empty(
                batch_size, num_heads, decode_2bit_reserve_per_row,
                device=device, dtype=torch.uint8,
            )
            if decode_2bit_reserve_per_row > 0 else None
        ),
    )


# Short integration-friendly alias.
pack_native_kv = pack_native_prefill
