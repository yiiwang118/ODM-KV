from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from kvquant.tq_backend import (
    MSEQuantized,
    ProdQuantized,
    TurboQuantProd,
    ValueQuantized,
    ValueQuantizedLike,
    _concat_prod_quantized,
    _concat_value_quantized,
    _mask_prod_quantized,
    _mask_value_quantized,
    dequantize_values,
    normalize_key_quantizer,
    normalize_value_quantizer,
    quantize_values,
    TurboQuantMSE,
)


def _empty_positions(device: torch.device) -> torch.Tensor:
    return torch.empty((0, 3), dtype=torch.long, device=device)


@dataclass
class ExactBank:
    device: torch.device
    dtype: torch.dtype
    positions: torch.Tensor = field(init=False)
    keys: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    values: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = _empty_positions(self.device)

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])

    def append(self, positions: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        if positions.numel() == 0:
            return
        self.positions = torch.cat([self.positions, positions], dim=0)
        self.keys = keys.clone() if self.keys is None else torch.cat([self.keys, keys], dim=0)
        self.values = values.clone() if self.values is None else torch.cat([self.values, values], dim=0)

    def truncate(self, seq_len: int) -> None:
        if self.size == 0:
            return
        keep_mask = self.positions[:, 2] < seq_len
        if not keep_mask.any():
            self.positions = _empty_positions(self.device)
            self.keys = None
            self.values = None
            return
        self.positions = self.positions[keep_mask].contiguous()
        self.keys = self.keys[keep_mask].contiguous()
        self.values = self.values[keep_mask].contiguous()

    def materialize_into(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self.size == 0 or self.keys is None or self.values is None:
            return
        pos = self.positions
        keys[pos[:, 0], pos[:, 1], pos[:, 2]] = self.keys.to(keys.dtype)
        values[pos[:, 0], pos[:, 1], pos[:, 2]] = self.values.to(values.dtype)


@dataclass
class QuantizedBank:
    bits: int
    head_dim: int
    value_group_size: int
    device: torch.device
    dtype: torch.dtype
    key_quantizer_type: str = "prod"
    value_quantizer: str = "minmax"
    seed: int = 42
    # OCS: outlier channel indices (per-head), None = disabled
    outlier_indices: Optional[torch.Tensor] = None   # [n_out] long tensor
    regular_indices: Optional[torch.Tensor] = None   # [d - n_out] long tensor
    outlier_min_bits: int = 4
    positions: torch.Tensor = field(init=False)
    key_quantized: Optional[ProdQuantized | MSEQuantized] = field(default=None, init=False, repr=False)
    value_quantized: Optional[ValueQuantizedLike] = field(default=None, init=False, repr=False)
    # OCS outlier key storage: kept at fp16 until materialize time to avoid
    # the dequant→requant round-trip per append (which accumulated precision
    # loss O(T) with the number of appends). Scalar MinMax is applied lazily.
    _outlier_key_raw: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.positions = _empty_positions(self.device)
        self.key_quantizer_type = normalize_key_quantizer(self.key_quantizer_type)
        self.value_quantizer = normalize_value_quantizer(self.value_quantizer)

        # MinMax at 1-bit has n_levels=1 (only min or max), unusable.
        # Force value to MSE which gives proper 2-centroid quantization.
        if self.bits == 1 and self.value_quantizer == "minmax":
            self.value_quantizer = "mse"

        # OCS: key quantizer operates on regular channels only
        key_dim = self.head_dim
        if self.outlier_indices is not None:
            key_dim = self.head_dim - self.outlier_indices.shape[0]
            self.outlier_bits = max(self.bits, self.outlier_min_bits)
        else:
            self.outlier_bits = self.bits

        if self.key_quantizer_type == "prod":
            self.key_quantizer = TurboQuantProd(
                key_dim, self.bits,
                device=self.device, dtype=self.dtype, seed=self.seed,
            )
        else:
            self.key_quantizer = TurboQuantMSE(
                key_dim, self.bits,
                device=self.device, dtype=self.dtype, seed=self.seed,
            )

        # Value quantizer (MSE instance, only created when value_quantizer=="mse")
        self.value_mse_quantizer = (
            TurboQuantMSE(
                self.head_dim, self.bits,
                device=self.device, dtype=self.dtype, seed=self.seed + 2000,
            )
            if self.value_quantizer == "mse"
            else None
        )

    @property
    def size(self) -> int:
        return int(self.positions.shape[0])

    def _concat_key_quantized(self, left, right):
        if self.key_quantizer_type == "prod":
            return _concat_prod_quantized(left, right)
        return _concat_value_quantized(left, right)  # works for MSEQuantized too

    def _mask_key_quantized(self, q, mask):
        if self.key_quantizer_type == "prod":
            return _mask_prod_quantized(q, mask)
        return _mask_value_quantized(q, mask)  # works for MSEQuantized too

    def append(self, positions: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        if positions.numel() == 0:
            return
        self.positions = torch.cat([self.positions, positions], dim=0)

        # ── Key quantization (with optional OCS) ────────────────────
        if self.outlier_indices is not None:
            k_out = keys[:, self.outlier_indices]      # [N, n_out]
            k_reg = keys[:, self.regular_indices]      # [N, d - n_out]
            # Regular channels → TurboQuant
            new_key_q = self.key_quantizer.quantize(k_reg)
            # Outlier channels → scalar MinMax at outlier_bits
            self._append_outlier_keys(k_out)
        else:
            new_key_q = self.key_quantizer.quantize(keys)

        new_value_q = quantize_values(
            values,
            self.bits,
            self.value_group_size,
            quantizer=self.value_quantizer,
            mse_quantizer=self.value_mse_quantizer,
            seed=self.seed + 2000,
        )
        self.key_quantized = self._concat_key_quantized(self.key_quantized, new_key_q)
        self.value_quantized = _concat_value_quantized(self.value_quantized, new_value_q)

    def _append_outlier_keys(self, k_out: torch.Tensor) -> None:
        """Append fp16 outlier key channels. Quantisation is deferred to
        ``materialize_into`` so scale/zero aren't recomputed per append
        (and so old data never gets re-quantised with a new scale, which
        would compound precision loss every call).
        """
        if self._outlier_key_raw is None:
            self._outlier_key_raw = k_out.detach().clone()
        else:
            self._outlier_key_raw = torch.cat(
                [self._outlier_key_raw, k_out.detach()], dim=0,
            )

    def truncate(self, seq_len: int) -> None:
        if self.size == 0:
            return
        keep_mask = self.positions[:, 2] < seq_len
        if not keep_mask.any():
            self.positions = _empty_positions(self.device)
            self.key_quantized = None
            self.value_quantized = None
            self._outlier_key_raw = None
            return
        self.positions = self.positions[keep_mask].contiguous()
        self.key_quantized = self._mask_key_quantized(self.key_quantized, keep_mask)
        self.value_quantized = _mask_value_quantized(self.value_quantized, keep_mask)
        if self._outlier_key_raw is not None:
            self._outlier_key_raw = self._outlier_key_raw[keep_mask].contiguous()

    def materialize_into(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if self.size == 0 or self.key_quantized is None or self.value_quantized is None:
            return
        pos = self.positions

        # ── Key dequantization ──────────────────────────────────────
        if self.outlier_indices is not None and self._outlier_key_raw is not None:
            # Regular channels from TurboQuant
            deq_reg = self.key_quantizer.dequantize(self.key_quantized).to(keys.dtype)
            # Outlier channels: scalar MinMax over the *current* full range.
            # Single forward quant-then-dequant with no stored uint8 artefact
            # → no compounding error across appends.
            raw = self._outlier_key_raw
            b = self.outlier_bits
            max_int = (1 << b) - 1
            mn = raw.amin(dim=0, keepdim=True)
            mx = raw.amax(dim=0, keepdim=True)
            scale = ((mx - mn) / max_int).clamp(min=1e-10)
            codes = ((raw - mn) / scale).clamp_(0, max_int).round_()
            deq_out = (codes * scale + mn).to(keys.dtype)
            # Merge into full-dim key
            N = deq_reg.shape[0]
            deq_keys = torch.empty(N, self.head_dim, device=keys.device, dtype=keys.dtype)
            deq_keys[:, self.outlier_indices] = deq_out
            deq_keys[:, self.regular_indices] = deq_reg
        else:
            deq_keys = self.key_quantizer.dequantize(self.key_quantized).to(keys.dtype)

        # ── Value dequantization (unchanged) ────────────────────────
        deq_values = dequantize_values(
            self.value_quantized,
            self.value_group_size,
            quantizer=self.value_quantizer,
            mse_quantizer=self.value_mse_quantizer,
        ).to(values.dtype)

        keys[pos[:, 0], pos[:, 1], pos[:, 2]] = deq_keys
        values[pos[:, 0], pos[:, 1], pos[:, 2]] = deq_values


@dataclass
class TurboQuantAdaptiveKVCacheState:
    head_dim: int
    bit_levels: tuple[int, ...]
    value_group_size: int
    device: torch.device
    dtype: torch.dtype
    key_quantizer: str = "prod"
    value_quantizer: str = "minmax"
    seed: int = 42
    # OCS: outlier channel indices (shared across all banks in this state)
    outlier_indices: Optional[torch.Tensor] = None   # [n_out] long
    regular_indices: Optional[torch.Tensor] = None   # [d - n_out] long
    outlier_min_bits: int = 4

    def __post_init__(self) -> None:
        levels = tuple(sorted(set(int(level) for level in self.bit_levels)))
        self.key_quantizer = normalize_key_quantizer(self.key_quantizer)
        min_bits = 2 if self.key_quantizer == "prod" else 1
        invalid = [level for level in levels if level not in (0, 16) and level < min_bits]
        if invalid:
            raise ValueError(
                f"Adaptive TurboQuant backend requires key bits >= {min_bits} "
                f"for key_quantizer={self.key_quantizer!r}; unsupported levels: {invalid}"
            )
        self.value_quantizer = normalize_value_quantizer(self.value_quantizer)
        group_size = min(self.value_group_size, self.head_dim)
        if self.head_dim % group_size != 0:
            group_size = math.gcd(self.head_dim, group_size)
        self.value_group_size = max(1, group_size)
        self.bit_levels = levels
        self.batch_size = 0
        self.num_heads = 0
        self.seq_len = 0
        self._quantized_banks = {
            bits: QuantizedBank(
                bits=bits,
                head_dim=self.head_dim,
                key_quantizer_type=self.key_quantizer,
                value_quantizer=self.value_quantizer,
                value_group_size=self.value_group_size,
                device=self.device,
                dtype=self.dtype,
                seed=self.seed,
                outlier_indices=self.outlier_indices,
                regular_indices=self.regular_indices,
                outlier_min_bits=self.outlier_min_bits,
            )
            for bits in self.bit_levels
            if bits not in (0, 16)
        }
        self._exact_bank = ExactBank(device=self.device, dtype=self.dtype)
        self._masked_positions = _empty_positions(self.device)

    def _reset(self) -> None:
        self.batch_size = 0
        self.num_heads = 0
        self.seq_len = 0
        self._exact_bank = ExactBank(device=self.device, dtype=self.dtype)
        self._masked_positions = _empty_positions(self.device)
        for bits in list(self._quantized_banks):
            self._quantized_banks[bits] = QuantizedBank(
                bits=bits,
                head_dim=self.head_dim,
                key_quantizer_type=self.key_quantizer,
                value_quantizer=self.value_quantizer,
                value_group_size=self.value_group_size,
                device=self.device,
                dtype=self.dtype,
                seed=self.seed,
                outlier_indices=self.outlier_indices,
                regular_indices=self.regular_indices,
                outlier_min_bits=self.outlier_min_bits,
            )

    def _validate_shapes(self, keys: torch.Tensor, values: torch.Tensor, bits: torch.Tensor) -> None:
        if keys.shape != values.shape:
            raise ValueError(f"Mismatched key/value shapes: {keys.shape} vs {values.shape}")
        if keys.dim() != 4:
            raise ValueError(f"Expected keys/values with shape [B, H, T, D], got {keys.shape}")
        if bits.shape != keys.shape[:3]:
            raise ValueError(f"Expected bits shape {keys.shape[:3]}, got {bits.shape}")
        if keys.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {keys.shape[-1]}")
        if self.batch_size == 0:
            self.batch_size = int(keys.shape[0])
            self.num_heads = int(keys.shape[1])
        elif self.batch_size != int(keys.shape[0]) or self.num_heads != int(keys.shape[1]):
            raise ValueError(
                f"Adaptive backend state shape changed from (B={self.batch_size}, H={self.num_heads}) "
                f"to (B={keys.shape[0]}, H={keys.shape[1]})"
            )

    def _positions_for_mask(self, mask: torch.Tensor, seq_offset: int) -> torch.Tensor:
        positions = mask.nonzero(as_tuple=False)
        if positions.numel() == 0:
            return _empty_positions(self.device)
        positions = positions.to(device=self.device, dtype=torch.long)
        positions[:, 2] += seq_offset
        return positions

    def prefill(self, keys: torch.Tensor, values: torch.Tensor, bits: torch.Tensor) -> None:
        self._reset()
        self.append(keys, values, bits)

    def append(self, keys: torch.Tensor, values: torch.Tensor, bits: torch.Tensor) -> None:
        self._validate_shapes(keys, values, bits)
        bits = bits.to(device=self.device, dtype=torch.int32)
        seq_offset = self.seq_len
        new_len = int(keys.shape[2])

        for level in self.bit_levels:
            mask = bits == level
            if not mask.any():
                continue
            local_positions = mask.nonzero(as_tuple=False)
            local_vectors = keys[local_positions[:, 0], local_positions[:, 1], local_positions[:, 2]]
            local_values = values[local_positions[:, 0], local_positions[:, 1], local_positions[:, 2]]
            positions = self._positions_for_mask(mask, seq_offset)
            if level == 0:
                self._masked_positions = torch.cat([self._masked_positions, positions], dim=0)
                continue

            if level == 16:
                self._exact_bank.append(positions, local_vectors, local_values)
                continue

            self._quantized_banks[level].append(positions, local_vectors, local_values)

        self.seq_len += new_len

    # ------------------------------------------------------------------
    # Decode-buffer fast path
    # ------------------------------------------------------------------

    def append_all_exact(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Route every new token to the ExactBank — no per-level loop, no mask ops.

        Semantically equivalent to ``append(keys, values, bits=full(16))`` but
        avoids the five ``mask.any()`` / ``mask.nonzero()`` GPU syncs per call
        that the generic loop incurs. Used by the decode-time buffer path
        where bits are pinned to 16 by construction, so there is no per-token
        branching to dispatch.
        """
        # Shape validation — match _validate_shapes without needing bits.
        if keys.shape != values.shape:
            raise ValueError(f"Mismatched key/value shapes: {keys.shape} vs {values.shape}")
        if keys.dim() != 4:
            raise ValueError(f"Expected keys/values with shape [B, H, T, D], got {keys.shape}")
        if keys.shape[-1] != self.head_dim:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {keys.shape[-1]}")
        if self.batch_size == 0:
            self.batch_size = int(keys.shape[0])
            self.num_heads = int(keys.shape[1])
        elif self.batch_size != int(keys.shape[0]) or self.num_heads != int(keys.shape[1]):
            raise ValueError(
                f"Adaptive backend state shape changed from (B={self.batch_size}, H={self.num_heads}) "
                f"to (B={keys.shape[0]}, H={keys.shape[1]})"
            )

        B, H, T, D = keys.shape
        if T == 0:
            return

        seq_offset = self.seq_len
        device = self.device

        # Build (b, h, s) positions directly via arange — identical row-major
        # ordering to mask.nonzero() on an all-True [B, H, T] mask.
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, H, T).reshape(-1)
        h_idx = torch.arange(H, device=device).view(1, H, 1).expand(B, H, T).reshape(-1)
        s_idx = (torch.arange(T, device=device) + seq_offset).view(1, 1, T).expand(B, H, T).reshape(-1)
        positions = torch.stack([b_idx, h_idx, s_idx], dim=-1).to(torch.long)

        # reshape(-1, D) gives the same row-major flat order as fancy-indexing
        # with (b_idx, h_idx, s_idx).
        keys_flat = keys.reshape(-1, D).contiguous()
        values_flat = values.reshape(-1, D).contiguous()

        self._exact_bank.append(positions, keys_flat, values_flat)
        self.seq_len += T

    def truncate(self, seq_len: int) -> None:
        seq_len = int(seq_len)
        if seq_len >= self.seq_len:
            return
        self.seq_len = seq_len
        self._exact_bank.truncate(seq_len)
        if self._masked_positions.numel() > 0:
            keep_mask = self._masked_positions[:, 2] < seq_len
            self._masked_positions = self._masked_positions[keep_mask].contiguous()
        for bank in self._quantized_banks.values():
            bank.truncate(seq_len)

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys = torch.zeros(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        values = torch.zeros_like(keys)
        self._exact_bank.materialize_into(keys, values)
        for bank in self._quantized_banks.values():
            bank.materialize_into(keys, values)
        return keys, values

    def materialize_quantized_into(
        self, keys: torch.Tensor, values: torch.Tensor,
    ) -> None:
        """Write only quantized bank tokens into existing tensors (in-place).

        Skips exact bank (fp16 tokens already in the HF cache) and the
        zero-allocation of a full tensor.  Used during prefill sync where
        the HF cache already contains the original fp16 data.
        """
        for bank in self._quantized_banks.values():
            bank.materialize_into(keys, values)

    def get_buffer_tokens(
        self, buffer_start_seq: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return buffer tokens (seq >= *buffer_start_seq*) from the exact bank.

        Returns ``(positions, keys, values)`` without modifying the bank.
        """
        bank = self._exact_bank
        if bank.size > 0 and bank.keys is not None:
            mask = bank.positions[:, 2] >= buffer_start_seq
            if mask.any():
                return bank.positions[mask], bank.keys[mask], bank.values[mask]
        empty = torch.empty(0, self.head_dim, device=self.device, dtype=self.dtype)
        return _empty_positions(self.device), empty, empty.clone()

    def flush_buffer(self, buffer_start_seq: int, new_bits: torch.Tensor) -> None:
        """Re-quantize buffer tokens from the exact bank.

        Removes entries with ``seq_pos >= buffer_start_seq`` from the exact
        bank and routes each to the bank matching its new bit level.
        """
        bank = self._exact_bank
        if bank.size == 0 or bank.keys is None:
            return
        mask = bank.positions[:, 2] >= buffer_start_seq
        if not mask.any():
            return

        buf_pos = bank.positions[mask]
        buf_keys = bank.keys[mask]
        buf_values = bank.values[mask]

        # Remove buffer entries from exact bank
        keep = ~mask
        if keep.any():
            bank.positions = bank.positions[keep].contiguous()
            bank.keys = bank.keys[keep].contiguous()
            bank.values = bank.values[keep].contiguous()
        else:
            bank.positions = _empty_positions(self.device)
            bank.keys = None
            bank.values = None

        # Route to target banks
        new_bits = new_bits.to(device=self.device, dtype=torch.int32)
        routed = torch.zeros(new_bits.shape[0], dtype=torch.bool, device=self.device)
        for level in self.bit_levels:
            level_mask = new_bits == level
            if not level_mask.any():
                continue
            routed |= level_mask
            pos = buf_pos[level_mask]
            k = buf_keys[level_mask]
            v = buf_values[level_mask]
            if level == 0:
                self._masked_positions = torch.cat(
                    [self._masked_positions, pos], dim=0,
                )
            elif level == 16:
                bank.append(pos, k, v)
            else:
                self._quantized_banks[level].append(pos, k, v)

        # Safety: return unrouted tokens to exact bank (prevents silent data loss)
        if not routed.all():
            orphan = ~routed
            bank.append(buf_pos[orphan], buf_keys[orphan], buf_values[orphan])

    def masked_key_indices(self) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self._masked_positions.numel() == 0:
            return None
        return (
            self._masked_positions[:, 0],
            self._masked_positions[:, 1],
            self._masked_positions[:, 2],
        )
