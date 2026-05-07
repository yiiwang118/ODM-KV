from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from codebook import compute_codebook


_CODEBOOK_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _make_generator(device: torch.device, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def random_rotation(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
) -> torch.Tensor:
    gen = _make_generator(device, seed)
    mat = torch.randn(dim, dim, device=device, dtype=torch.float32, generator=gen)
    q, r = torch.linalg.qr(mat)
    diag = torch.diag(r)
    signs = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    return (q * signs.unsqueeze(0)).to(dtype)


def _load_or_compute_codebook(bits: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (bits, dim)
    cached = _CODEBOOK_CACHE.get(key)
    if cached is None:
        centroids = compute_codebook(2 ** bits, dim)
        cached = torch.from_numpy(centroids.astype(np.float32))
        _CODEBOOK_CACHE[key] = cached
    return cached.to(device=device, dtype=dtype)


class MSEQuantized(NamedTuple):
    indices: torch.Tensor
    norms: torch.Tensor
    bits: int


class ProdQuantized(NamedTuple):
    mse_indices: torch.Tensor
    qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    norms: torch.Tensor
    mse_bits: int


class ValueQuantized(NamedTuple):
    data: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    bits: int


ValueQuantizedLike = ValueQuantized | MSEQuantized


def normalize_value_quantizer(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "group": "minmax",
        "group_minmax": "minmax",
        "minmax": "minmax",
        "mse": "mse",
        "turboquant": "mse",
        "turboquant_mse": "mse",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported value quantizer: {name!r}")
    return aliases[normalized]


def normalize_key_quantizer(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {
        "prod": "prod",
        "turboquant_prod": "prod",
        "turboquantprod": "prod",
        "mse": "mse",
        "turboquant_mse": "mse",
        "turboquantmse": "mse",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported key quantizer: {name!r}")
    return aliases[normalized]


def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    dim = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
        eff_bits = 1
    elif bits == 2:
        vals_per_byte = 4
        eff_bits = 2
    elif bits <= 4:
        vals_per_byte = 2
        eff_bits = 4
    else:
        return indices.to(torch.uint8)

    padded_dim = ((dim + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_dim > dim:
        indices = F.pad(indices.to(torch.uint8), (0, padded_dim - dim), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * eff_bits
    return (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)


def _unpack_indices(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    if bits == 1:
        vals_per_byte = 8
        eff_bits = 1
    elif bits == 2:
        vals_per_byte = 4
        eff_bits = 2
    elif bits <= 4:
        vals_per_byte = 2
        eff_bits = 4
    else:
        return packed.long()

    mask = (1 << eff_bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * eff_bits
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    return unpacked[..., :dim].long()


def _pack_qjl_signs(projected: torch.Tensor) -> torch.Tensor:
    signs = (projected > 0).to(torch.uint8)
    dim = signs.shape[-1]
    if dim % 8 != 0:
        signs = F.pad(signs, (0, 8 - dim % 8), value=0)
    reshaped = signs.reshape(*signs.shape[:-1], -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
    return (reshaped * powers).sum(dim=-1, dtype=torch.uint8)


def _unpack_qjl_signs(packed: torch.Tensor, dim: int) -> torch.Tensor:
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
    signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :dim]
    return 2.0 * signs - 1.0


class TurboQuantMSE:
    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 42,
    ):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype
        # Keep the quantizer's linear algebra in fp32 so bf16/fp16 model weights
        # can still use the backend without dtype-mismatch matmuls.
        self.compute_dtype = torch.float32
        self.pi = random_rotation(dim, device, self.compute_dtype, seed=seed)
        self.centroids = _load_or_compute_codebook(bits, dim, device, self.compute_dtype)

    def quantize(self, x: torch.Tensor) -> MSEQuantized:
        x_float = x.float()
        norms = x_float.norm(dim=-1, keepdim=False)
        x_unit = x_float / (norms.unsqueeze(-1) + 1e-10)
        y = x_unit @ self.pi.T

        # Fused nearest-centroid lookup (Triton on CUDA, PyTorch elsewhere).
        # Replaces searchsorted + clamp + 2× subtract + 2× abs + where (5-6
        # kernel launches) with a single kernel launch. Launch is wrapped in
        # the tensor's CUDA context so device_map="auto" remains safe.
        from kvquant.tq_triton import mse_nearest_centroid
        indices = mse_nearest_centroid(y, self.centroids)
        return MSEQuantized(indices=_pack_indices(indices, self.bits), norms=norms, bits=self.bits)

    def dequantize(self, q: MSEQuantized) -> torch.Tensor:
        indices = _unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = y_hat @ self.pi
        return x_hat * q.norms.float().unsqueeze(-1)


class TurboQuantProd:
    def __init__(
        self,
        dim: int,
        bits: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 42,
    ):
        if bits < 2:
            raise ValueError("TurboQuantProd requires at least 2 bits.")
        self.dim = dim
        self.bits = bits
        self.device = device
        self.dtype = dtype
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1, device=device, dtype=dtype, seed=seed)
        gen = _make_generator(device, seed + 1000)
        self.s = torch.randn(dim, dim, device=device, dtype=torch.float32, generator=gen)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def quantize(self, x: torch.Tensor) -> ProdQuantized:
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)
        residual = x.float() - x_hat
        residual_norms = residual.norm(dim=-1)
        projected = torch.matmul(residual.float(), self.s.T)
        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=_pack_qjl_signs(projected),
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q: ProdQuantized) -> torch.Tensor:
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)
        signs = _unpack_qjl_signs(q.qjl_signs, self.dim)
        x_qjl = torch.matmul(signs, self.s)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))
        return x_mse + x_qjl

    def attention_score(self, query: torch.Tensor, quantized_key: ProdQuantized) -> torch.Tensor:
        mse_q = MSEQuantized(
            indices=quantized_key.mse_indices,
            norms=quantized_key.norms,
            bits=quantized_key.mse_bits,
        )
        k_mse = self.mse_quantizer.dequantize(mse_q)
        scores_mse = torch.matmul(query.float(), k_mse.float().transpose(-2, -1))

        q_sketched = torch.matmul(query.float(), self.s.T)
        signs = _unpack_qjl_signs(quantized_key.qjl_signs, self.dim)
        scores_qjl = torch.matmul(q_sketched, signs.transpose(-2, -1))
        scores_qjl = scores_qjl * (self.qjl_scale * quantized_key.residual_norms.unsqueeze(-2))
        return scores_mse + scores_qjl.to(scores_mse.dtype)


def unpack_values(vq: ValueQuantized) -> torch.Tensor:
    bits = vq.bits
    packed = vq.data
    if bits == 2:
        v0 = packed & 0x03
        v1 = (packed >> 2) & 0x03
        v2 = (packed >> 4) & 0x03
        v3 = (packed >> 6) & 0x03
        return torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 4)
    if bits == 4:
        v0 = packed & 0x0F
        v1 = (packed >> 4) & 0x0F
        return torch.stack([v0, v1], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    return packed


def _quantize_values_minmax(v: torch.Tensor, bits: int, group_size: int) -> ValueQuantized:
    orig_shape = v.shape
    dim = orig_shape[-1]
    n_groups = dim // group_size
    if dim % group_size != 0 or n_groups == 0:
        raise ValueError(f"head_dim {dim} must be divisible by group_size {group_size}")

    grouped = v.reshape(*orig_shape[:-1], n_groups, group_size)
    v_min = grouped.min(dim=-1, keepdim=True).values
    v_max = grouped.max(dim=-1, keepdim=True).values

    n_levels = 2 ** bits - 1
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)
    zero = v_min

    v_q = ((grouped - zero) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*orig_shape[:-1], dim)

    if bits == 2:
        if dim % 4 != 0:
            raise ValueError(f"2-bit value packing requires dim divisible by 4, got {dim}")
        v4 = v_q_flat.reshape(*orig_shape[:-1], dim // 4, 4)
        packed = v4[..., 0] | (v4[..., 1] << 2) | (v4[..., 2] << 4) | (v4[..., 3] << 6)
        v_q_flat = packed
    elif bits == 4:
        if dim % 2 != 0:
            raise ValueError(f"4-bit value packing requires dim divisible by 2, got {dim}")
        v2 = v_q_flat.reshape(*orig_shape[:-1], dim // 2, 2)
        packed = v2[..., 0] | (v2[..., 1] << 4)
        v_q_flat = packed

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1),
        zeros=zero.squeeze(-1),
        bits=bits,
    )


def _dequantize_values_minmax(vq: ValueQuantized, group_size: int) -> torch.Tensor:
    data = unpack_values(vq).float()
    dim = data.shape[-1]
    batch_shape = data.shape[:-1]
    n_groups = dim // group_size
    data = data.reshape(*batch_shape, n_groups, group_size)
    scales = vq.scales.unsqueeze(-1)
    zeros = vq.zeros.unsqueeze(-1)
    return (data * scales + zeros).reshape(*batch_shape, dim)


def quantize_values(
    v: torch.Tensor,
    bits: int,
    group_size: int,
    *,
    quantizer: str = "minmax",
    mse_quantizer: Optional[TurboQuantMSE] = None,
    seed: int = 42,
) -> ValueQuantizedLike:
    value_quantizer = normalize_value_quantizer(quantizer)
    if value_quantizer == "mse":
        q = mse_quantizer or TurboQuantMSE(
            v.shape[-1],
            bits,
            device=v.device,
            dtype=v.dtype,
            seed=seed,
        )
        return q.quantize(v)
    return _quantize_values_minmax(v, bits, group_size)


def dequantize_values(
    vq: ValueQuantizedLike,
    group_size: int,
    *,
    quantizer: str = "minmax",
    mse_quantizer: Optional[TurboQuantMSE] = None,
) -> torch.Tensor:
    value_quantizer = normalize_value_quantizer(quantizer)
    if value_quantizer == "mse":
        if mse_quantizer is None:
            raise ValueError("mse_quantizer is required to dequantize MSE-quantized values.")
        if not isinstance(vq, MSEQuantized):
            raise TypeError(f"Expected MSEQuantized values, got {type(vq).__name__}.")
        return mse_quantizer.dequantize(vq)
    if not isinstance(vq, ValueQuantized):
        raise TypeError(f"Expected ValueQuantized values, got {type(vq).__name__}.")
    return _dequantize_values_minmax(vq, group_size)


def _slice_prod_quantized(q: Optional[ProdQuantized], length: int) -> Optional[ProdQuantized]:
    if q is None:
        return None
    return ProdQuantized(
        mse_indices=q.mse_indices[..., :length, :].contiguous(),
        qjl_signs=q.qjl_signs[..., :length, :].contiguous(),
        residual_norms=q.residual_norms[..., :length].contiguous(),
        norms=q.norms[..., :length].contiguous(),
        mse_bits=q.mse_bits,
    )


def _slice_value_quantized(q: Optional[ValueQuantizedLike], length: int) -> Optional[ValueQuantizedLike]:
    if q is None:
        return None
    if isinstance(q, MSEQuantized):
        return MSEQuantized(
            indices=q.indices[..., :length, :].contiguous(),
            norms=q.norms[..., :length].contiguous(),
            bits=q.bits,
        )
    return ValueQuantized(
        data=q.data[..., :length, :].contiguous(),
        scales=q.scales[..., :length, :].contiguous(),
        zeros=q.zeros[..., :length, :].contiguous(),
        bits=q.bits,
    )


def _mask_prod_quantized(q: Optional[ProdQuantized], mask: torch.Tensor) -> Optional[ProdQuantized]:
    """Boolean-mask select from a flat ProdQuantized (first dim is token axis)."""
    if q is None:
        return None
    return ProdQuantized(
        mse_indices=q.mse_indices[mask].contiguous(),
        qjl_signs=q.qjl_signs[mask].contiguous(),
        residual_norms=q.residual_norms[mask].contiguous(),
        norms=q.norms[mask].contiguous(),
        mse_bits=q.mse_bits,
    )


def _mask_value_quantized(q: Optional[ValueQuantizedLike], mask: torch.Tensor) -> Optional[ValueQuantizedLike]:
    """Boolean-mask select from a flat value quantization payload (first dim is token axis)."""
    if q is None:
        return None
    if isinstance(q, MSEQuantized):
        return MSEQuantized(
            indices=q.indices[mask].contiguous(),
            norms=q.norms[mask].contiguous(),
            bits=q.bits,
        )
    return ValueQuantized(
        data=q.data[mask].contiguous(),
        scales=q.scales[mask].contiguous(),
        zeros=q.zeros[mask].contiguous(),
        bits=q.bits,
    )


def _concat_prod_quantized(left: Optional[ProdQuantized], right: ProdQuantized) -> ProdQuantized:
    if left is None:
        return right
    return ProdQuantized(
        mse_indices=torch.cat([left.mse_indices, right.mse_indices], dim=-2),
        qjl_signs=torch.cat([left.qjl_signs, right.qjl_signs], dim=-2),
        residual_norms=torch.cat([left.residual_norms, right.residual_norms], dim=-1),
        norms=torch.cat([left.norms, right.norms], dim=-1),
        mse_bits=left.mse_bits,
    )


def _concat_value_quantized(
    left: Optional[ValueQuantizedLike],
    right: ValueQuantizedLike,
) -> ValueQuantizedLike:
    if left is None:
        return right
    if isinstance(left, MSEQuantized) and isinstance(right, MSEQuantized):
        return MSEQuantized(
            indices=torch.cat([left.indices, right.indices], dim=-2),
            norms=torch.cat([left.norms, right.norms], dim=-1),
            bits=left.bits,
        )
    if isinstance(left, ValueQuantized) and isinstance(right, ValueQuantized):
        return ValueQuantized(
            data=torch.cat([left.data, right.data], dim=-2),
            scales=torch.cat([left.scales, right.scales], dim=-2),
            zeros=torch.cat([left.zeros, right.zeros], dim=-2),
            bits=left.bits,
        )
    raise TypeError(f"Mismatched value quantized payloads: {type(left).__name__} vs {type(right).__name__}")


def _value_quantized_memory_bytes(q: Optional[ValueQuantizedLike]) -> int:
    if q is None:
        return 0
    if isinstance(q, MSEQuantized):
        return q.indices.nelement() * q.indices.element_size() + q.norms.nelement() * q.norms.element_size()
    return (
        q.data.nelement() * q.data.element_size()
        + q.scales.nelement() * q.scales.element_size()
        + q.zeros.nelement() * q.zeros.element_size()
    )


@dataclass
class TurboQuantKVCacheState:
    head_dim: int
    key_bits: int
    value_bits: int
    value_group_size: int
    buffer_size: int
    device: torch.device
    dtype: torch.dtype
    value_quantizer: str = "minmax"
    key_quantizer_type: str = "prod"
    seed: int = 42

    def __post_init__(self) -> None:
        self.value_quantizer = normalize_value_quantizer(self.value_quantizer)
        self.key_quantizer_type = normalize_key_quantizer(self.key_quantizer_type)
        min_key_bits = 2 if self.key_quantizer_type == "prod" else 1
        if self.key_bits < min_key_bits:
            raise ValueError(
                f"TurboQuant backend requires key_bits >= {min_key_bits} "
                f"for key_quantizer={self.key_quantizer_type!r}."
            )
        group_size = min(self.value_group_size, self.head_dim)
        if self.head_dim % group_size != 0:
            group_size = math.gcd(self.head_dim, group_size)
        self.value_group_size = max(1, group_size)
        if self.key_quantizer_type == "prod":
            self.key_quantizer = TurboQuantProd(
                self.head_dim, self.key_bits,
                device=self.device, dtype=self.dtype, seed=self.seed,
            )
        else:
            self.key_quantizer = TurboQuantMSE(
                self.head_dim, self.key_bits,
                device=self.device, dtype=self.dtype, seed=self.seed,
            )
        self.value_mse_quantizer = (
            TurboQuantMSE(
                self.head_dim,
                self.value_bits,
                device=self.device,
                dtype=self.dtype,
                seed=self.seed + 2000,
            )
            if self.value_quantizer == "mse"
            else None
        )
        self.seq_len = 0
        self.key_quantized: Optional[ProdQuantized] = None
        self.value_quantized: Optional[ValueQuantizedLike] = None
        self.key_buffer: Optional[torch.Tensor] = None
        self.value_buffer: Optional[torch.Tensor] = None
        self._last_flush_dequant: tuple[int, torch.Tensor, torch.Tensor] | None = None

    @property
    def quantized_len(self) -> int:
        if self.key_quantized is None:
            return 0
        return int(self.key_quantized.norms.shape[-1])

    @property
    def buffer_len(self) -> int:
        if self.key_buffer is None:
            return 0
        return int(self.key_buffer.shape[-2])

    def prefill(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        seq_len = int(keys.shape[-2])
        self.seq_len = seq_len
        self.key_quantized = None
        self.value_quantized = None
        self.key_buffer = None
        self.value_buffer = None

        if seq_len <= self.buffer_size:
            self.key_buffer = keys.clone()
            self.value_buffer = values.clone()
            return

        n_quant = seq_len - self.buffer_size
        keys_to_quant = keys[..., :n_quant, :]
        values_to_quant = values[..., :n_quant, :]
        self.key_quantized = self.key_quantizer.quantize(keys_to_quant)
        self.value_quantized = quantize_values(
            values_to_quant,
            self.value_bits,
            self.value_group_size,
            quantizer=self.value_quantizer,
            mse_quantizer=self.value_mse_quantizer,
            seed=self.seed + 2000,
        )
        self.key_buffer = keys[..., n_quant:, :].clone()
        self.value_buffer = values[..., n_quant:, :].clone()

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        append_len = int(keys.shape[-2])
        self.seq_len += append_len
        self._last_flush_dequant: tuple[int, torch.Tensor, torch.Tensor] | None = None
        if self.key_buffer is None:
            self.key_buffer = keys.clone()
            self.value_buffer = values.clone()
        else:
            self.key_buffer = torch.cat([self.key_buffer, keys], dim=-2)
            self.value_buffer = torch.cat([self.value_buffer, values], dim=-2)

        if self.buffer_len > self.buffer_size:
            self._flush_buffer(self.buffer_len - self.buffer_size)

    def _flush_buffer(self, n_flush: int) -> None:
        if n_flush <= 0 or self.key_buffer is None or self.value_buffer is None:
            return
        flush_start = self.quantized_len  # position in full sequence where flushed tokens sit
        keys_flush = self.key_buffer[..., :n_flush, :]
        values_flush = self.value_buffer[..., :n_flush, :]
        self.key_buffer = self.key_buffer[..., n_flush:, :].contiguous()
        self.value_buffer = self.value_buffer[..., n_flush:, :].contiguous()
        new_key_q = self.key_quantizer.quantize(keys_flush)
        new_value_q = quantize_values(
            values_flush,
            self.value_bits,
            self.value_group_size,
            quantizer=self.value_quantizer,
            mse_quantizer=self.value_mse_quantizer,
            seed=self.seed + 2000,
        )
        if self.key_quantizer_type == "prod":
            self.key_quantized = _concat_prod_quantized(self.key_quantized, new_key_q)
        else:
            self.key_quantized = _concat_value_quantized(self.key_quantized, new_key_q)
        self.value_quantized = _concat_value_quantized(self.value_quantized, new_value_q)
        # Cache dequantized flushed tokens for incremental HF cache update
        self._last_flush_dequant = (
            flush_start,
            self.key_quantizer.dequantize(new_key_q).to(self.dtype),
            dequantize_values(
                new_value_q, self.value_group_size,
                quantizer=self.value_quantizer,
                mse_quantizer=self.value_mse_quantizer,
            ).to(self.dtype),
        )

    def truncate(self, seq_len: int) -> None:
        seq_len = int(seq_len)
        if seq_len >= self.seq_len:
            return
        keep_quant = min(seq_len, self.quantized_len)
        keep_buffer = max(0, seq_len - keep_quant)
        if self.key_quantizer_type == "prod":
            self.key_quantized = _slice_prod_quantized(self.key_quantized, keep_quant)
        else:
            self.key_quantized = _slice_value_quantized(self.key_quantized, keep_quant)
        self.value_quantized = _slice_value_quantized(self.value_quantized, keep_quant)
        if self.key_buffer is not None:
            self.key_buffer = self.key_buffer[..., :keep_buffer, :].contiguous()
            self.value_buffer = self.value_buffer[..., :keep_buffer, :].contiguous()
        self.seq_len = seq_len

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        keys_parts: list[torch.Tensor] = []
        values_parts: list[torch.Tensor] = []

        if self.key_quantized is not None and self.value_quantized is not None:
            keys_parts.append(self.key_quantizer.dequantize(self.key_quantized).to(self.dtype))
            values_parts.append(
                dequantize_values(
                    self.value_quantized,
                    self.value_group_size,
                    quantizer=self.value_quantizer,
                    mse_quantizer=self.value_mse_quantizer,
                ).to(self.dtype)
            )
        if self.key_buffer is not None and self.value_buffer is not None:
            keys_parts.append(self.key_buffer)
            values_parts.append(self.value_buffer)

        if not keys_parts:
            empty = torch.zeros(0, device=self.device, dtype=self.dtype)
            return empty, empty
        if len(keys_parts) == 1:
            return keys_parts[0], values_parts[0]
        return torch.cat(keys_parts, dim=-2), torch.cat(values_parts, dim=-2)

    def memory_bytes(self) -> dict[str, int]:
        key_bytes = 0
        value_bytes = 0
        buffer_bytes = 0
        if self.key_quantized is not None:
            key_bytes += self.key_quantized.mse_indices.nelement()
            key_bytes += self.key_quantized.qjl_signs.nelement()
            key_bytes += self.key_quantized.residual_norms.nelement() * 2
            key_bytes += self.key_quantized.norms.nelement() * 2
        value_bytes += _value_quantized_memory_bytes(self.value_quantized)
        if self.key_buffer is not None:
            buffer_bytes += self.key_buffer.nelement() * self.key_buffer.element_size()
        if self.value_buffer is not None:
            buffer_bytes += self.value_buffer.nelement() * self.value_buffer.element_size()
        return {
            "quantized_keys": key_bytes,
            "quantized_values": value_bytes,
            "buffer": buffer_bytes,
            "total": key_bytes + value_bytes + buffer_bytes,
        }
