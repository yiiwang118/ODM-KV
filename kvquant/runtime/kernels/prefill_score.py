"""Native GPU primitives for the analytical prefill scorer.

The attention forward already owns post-RoPE Q.  Replaying q_proj is wasteful,
but naively inverting every Q element into an fp32 tensor is also expensive
(roughly 2 GiB at Llama B8/T2k).  The kernel below fuses inverse scaled-RoPE
with the suffix reduction and writes only ``[B,H,D]`` query means.
"""
from __future__ import annotations

import torch

from kvquant.attention_utils import _invert_rotary_pos_emb_q

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover - CPU development hosts
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _inverse_rope_suffix_mean_kernel(
        Q,
        COS,
        SIN,
        OUT,
        sq_b: tl.constexpr,
        sq_h: tl.constexpr,
        sq_t: tl.constexpr,
        sq_d: tl.constexpr,
        sc_b: tl.constexpr,
        sc_t: tl.constexpr,
        sc_d: tl.constexpr,
        so_b: tl.constexpr,
        so_h: tl.constexpr,
        so_d: tl.constexpr,
        T_CORE,
        H: tl.constexpr,
        D: tl.constexpr,
        N_SINK: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        b = bh // H
        h = bh - b * H
        d_valid = d < D
        half = D // 2
        pair_d = tl.where(d < half, d + half, d - half)
        pair_sign = tl.where(d < half, -1.0, 1.0)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Runtime loop bound avoids compiling a new, fully unrolled kernel for
        # every prompt length (and keeps 16k/32k-context compile size bounded).
        for t0 in tl.range(0, T_CORE, BLOCK_T):
            t_core = t0 + tl.arange(0, BLOCK_T)
            t = t_core + N_SINK
            valid = (t_core < T_CORE)[:, None] & d_valid[None, :]
            q_off = b * sq_b + h * sq_h + t[:, None] * sq_t
            q = tl.load(Q + q_off + d[None, :] * sq_d, mask=valid, other=0.0).to(tl.float32)
            q_pair = tl.load(
                Q + q_off + pair_d[None, :] * sq_d,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            rotate_q = q_pair * pair_sign[None, :]
            rope_off = b * sc_b + t[:, None] * sc_t + d[None, :] * sc_d
            cos = tl.load(COS + rope_off, mask=valid, other=0.0).to(tl.float32)
            sin = tl.load(SIN + rope_off, mask=valid, other=0.0).to(tl.float32)
            denom = tl.maximum(cos * cos + sin * sin, 1.0e-30)
            q_pre = (q * cos - rotate_q * sin) / denom
            acc += tl.sum(q_pre, axis=0)

        out_off = b * so_b + h * so_h + d * so_d
        tl.store(OUT + out_off, acc / T_CORE, mask=d_valid)


    @triton.jit
    def _gqa_logits_kernel(
        Q,
        K,
        OUT,
        sq_b: tl.constexpr,
        sq_h: tl.constexpr,
        sq_d: tl.constexpr,
        sk_b: tl.constexpr,
        sk_h: tl.constexpr,
        sk_t: tl.constexpr,
        sk_d: tl.constexpr,
        so_b: tl.constexpr,
        so_h: tl.constexpr,
        so_g: tl.constexpr,
        so_t: tl.constexpr,
        T,
        SCALE,
        H_KV: tl.constexpr,
        GROUPS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bhg = tl.program_id(0)
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        g = bhg % GROUPS
        bh = bhg // GROUPS
        h_kv = bh % H_KV
        b = bh // H_KV
        h_q = h_kv * GROUPS + g
        d = tl.arange(0, BLOCK_D)
        d_valid = d < D
        q = tl.load(
            Q + b * sq_b + h_q * sq_h + d * sq_d,
            mask=d_valid,
            other=0.0,
        ).to(tl.float32)
        valid = (t < T)[:, None] & d_valid[None, :]
        k = tl.load(
            K + b * sk_b + h_kv * sk_h + t[:, None] * sk_t + d[None, :] * sk_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(k * q[None, :], axis=1) * SCALE
        out_off = b * so_b + h_kv * so_h + g * so_g + t * so_t
        tl.store(OUT + out_off, logits, mask=t < T)


    @triton.jit
    def _weighted_value_sum_kernel(
        P,
        V,
        OUT,
        sp_b: tl.constexpr,
        sp_h: tl.constexpr,
        sp_t: tl.constexpr,
        sv_b: tl.constexpr,
        sv_h: tl.constexpr,
        sv_t: tl.constexpr,
        sv_d: tl.constexpr,
        so_b: tl.constexpr,
        so_h: tl.constexpr,
        so_d: tl.constexpr,
        T,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        b = bh // H
        h = bh - b * H
        d_valid = d < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for t0 in tl.range(0, T, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            t_valid = t < T
            p = tl.load(
                P + b * sp_b + h * sp_h + t * sp_t,
                mask=t_valid,
                other=0.0,
            ).to(tl.float32)
            valid = t_valid[:, None] & d_valid[None, :]
            value = tl.load(
                V + b * sv_b + h * sv_h + t[:, None] * sv_t + d[None, :] * sv_d,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(value * p[:, None], axis=0)
        tl.store(
            OUT + b * so_b + h * so_h + d * so_d,
            acc,
            mask=d_valid,
        )


    @triton.jit
    def _joint_mse_score_kernel(
        P,
        K,
        V,
        O,
        SCORE,
        sp_b: tl.constexpr,
        sp_h: tl.constexpr,
        sp_t: tl.constexpr,
        sk_b: tl.constexpr,
        sk_h: tl.constexpr,
        sk_t: tl.constexpr,
        sk_d: tl.constexpr,
        sv_b: tl.constexpr,
        sv_h: tl.constexpr,
        sv_t: tl.constexpr,
        sv_d: tl.constexpr,
        so_b: tl.constexpr,
        so_h: tl.constexpr,
        so_d: tl.constexpr,
        ss_b: tl.constexpr,
        ss_h: tl.constexpr,
        ss_t: tl.constexpr,
        T,
        EPSILON,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t = tl.program_id(1) * BLOCK_T + tl.arange(0, BLOCK_T)
        b = bh // H
        h = bh - b * H
        d = tl.arange(0, BLOCK_D)
        d_valid = d < D
        o = tl.load(
            O + b * so_b + h * so_h + d * so_d,
            mask=d_valid,
            other=0.0,
        ).to(tl.float32)
        valid = (t < T)[:, None] & d_valid[None, :]
        k = tl.load(
            K + b * sk_b + h * sk_h + t[:, None] * sk_t + d[None, :] * sk_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(
            V + b * sv_b + h * sv_h + t[:, None] * sv_t + d[None, :] * sv_d,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        vn2 = tl.sum(value * value, axis=1)
        vo = tl.sum(value * o[None, :], axis=1)
        on2 = tl.sum(o * o, axis=0)
        kn2 = tl.sum(k * k, axis=1)
        irrep = tl.maximum(vn2 - 2.0 * vo + on2, 0.0)
        p = tl.load(
            P + b * sp_b + h * sp_h + t * sp_t,
            mask=t < T,
            other=0.0,
        ).to(tl.float32)
        score = (p + EPSILON) * (p + EPSILON) * (vn2 + (kn2 / D) * irrep)
        tl.store(
            SCORE + b * ss_b + h * ss_h + t * ss_t,
            score,
            mask=t < T,
        )


def inverse_rope_suffix_mean(
    query: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    n_sink: int,
) -> torch.Tensor:
    """Return the pre-RoPE suffix mean without materialising pre-RoPE Q.

    Args:
        query: post-RoPE attention Q, ``[B,H,T,D]``.
        cos/sin: the exact position embedding tensors used by attention,
            ``[B,T,D]`` or ``[T,D]``.  Scaled RoPE is supported.
        n_sink: excluded prefix length.

    Returns:
        An fp32 ``[B,H,D]`` tensor.  The production CUDA path launches one
        Triton reduction; the CPU fallback is intentionally only an oracle for
        tests and diagnostics.
    """
    if query.dim() != 4:
        raise ValueError(f"query must be [B,H,T,D], got {tuple(query.shape)}")
    batch, heads, seq, dim = query.shape
    if not 0 <= n_sink < seq:
        raise ValueError(f"n_sink={n_sink} must be in [0,{seq})")
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.shape != sin.shape or cos.shape[-2:] != (seq, dim):
        raise ValueError(
            f"cos/sin must end in {(seq, dim)}, got {tuple(cos.shape)}/{tuple(sin.shape)}"
        )
    if cos.shape[0] not in (1, batch):
        raise ValueError(f"RoPE batch {cos.shape[0]} cannot broadcast to Q batch {batch}")
    cos = cos.to(device=query.device)
    sin = sin.to(device=query.device)
    if cos.shape[0] == 1 and batch != 1:
        cos = cos.expand(batch, -1, -1)
        sin = sin.expand(batch, -1, -1)

    if not (_HAS_TRITON and query.is_cuda):
        return _invert_rotary_pos_emb_q(query, cos, sin)[:, :, n_sink:].mean(dim=2)

    if dim % 2:
        raise ValueError(f"RoPE head dimension must be even, got {dim}")
    out = torch.empty(batch, heads, dim, device=query.device, dtype=torch.float32)
    block_d = min(32, triton.next_power_of_2(dim))
    block_t = 32
    t_core = seq - n_sink
    grid = (batch * heads, triton.cdiv(dim, block_d))
    # Triton otherwise launches on the process's current CUDA device.  Under
    # ``device_map`` that can differ from the decoder layer that owns Q/K/V,
    # producing a cross-device pointer error even though every tensor is
    # correctly colocated.
    with torch.cuda.device(query.device):
        _inverse_rope_suffix_mean_kernel[grid](
            query,
            cos,
            sin,
            out,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            cos.stride(0),
            cos.stride(1),
            cos.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            t_core,
            H=heads,
            D=dim,
            N_SINK=n_sink,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=4,
        )
    return out


def gqa_query_key_logits(
    mean_query: torch.Tensor,
    keys: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Compute fp32 ``Q_mean K^T`` without expanding GQA K or casting K.

    ``mean_query`` is ``[B,H_q,D]`` and ``keys`` is ``[B,H_kv,T,D]``.
    The result is ``[B,H_kv,G,T]``.  CUDA uses one Triton operator; the CPU
    branch is a numerical oracle.
    """
    if mean_query.dim() != 3 or keys.dim() != 4:
        raise ValueError("mean_query/keys must be [B,Hq,D] and [B,Hkv,T,D]")
    batch, h_q, dim = mean_query.shape
    if keys.shape[0] != batch or keys.shape[-1] != dim or h_q % keys.shape[1]:
        raise ValueError(
            f"incompatible mean-query/key shapes {tuple(mean_query.shape)}/{tuple(keys.shape)}"
        )
    h_kv, seq = keys.shape[1], keys.shape[2]
    groups = h_q // h_kv
    if not (_HAS_TRITON and mean_query.is_cuda and keys.is_cuda):
        q = mean_query.reshape(batch, h_kv, groups, dim).float()
        return torch.einsum("bhgd,bhtd->bhgt", q, keys.float()) * float(scaling)

    out = torch.empty(
        batch, h_kv, groups, seq,
        device=keys.device, dtype=torch.float32,
    )
    block_d = triton.next_power_of_2(dim)
    block_t = 32
    with torch.cuda.device(keys.device):
        _gqa_logits_kernel[(batch * h_kv * groups, triton.cdiv(seq, block_t))](
            mean_query,
            keys,
            out,
            mean_query.stride(0),
            mean_query.stride(1),
            mean_query.stride(2),
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            keys.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            seq,
            float(scaling),
            H_KV=h_kv,
            GROUPS=groups,
            D=dim,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=4,
        )
    return out


def weighted_value_sum(probability: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Return fp32 ``sum_t probability_t * value_t`` with no fp32 V copy."""
    if probability.dim() != 3 or values.dim() != 4:
        raise ValueError("probability/values must be [B,H,T] and [B,H,T,D]")
    if probability.shape != values.shape[:3]:
        raise ValueError(
            f"incompatible probability/value shapes {tuple(probability.shape)}/{tuple(values.shape)}"
        )
    batch, heads, seq, dim = values.shape
    if not (_HAS_TRITON and probability.is_cuda and values.is_cuda):
        return torch.einsum("bht,bhtd->bhd", probability.float(), values.float())

    out = torch.empty(batch, heads, dim, device=values.device, dtype=torch.float32)
    block_d = min(32, triton.next_power_of_2(dim))
    block_t = 32
    with torch.cuda.device(values.device):
        _weighted_value_sum_kernel[(batch * heads, triton.cdiv(dim, block_d))](
            probability,
            values,
            out,
            probability.stride(0),
            probability.stride(1),
            probability.stride(2),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            values.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            seq,
            H=heads,
            D=dim,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=4,
        )
    return out


def joint_mse_score(
    probability: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    expected_value: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Evaluate ExpectedJointMSE in one native kernel.

    K/V stay in their attention dtype and are converted in registers.  This
    replaces the old full-size fp32 K/V temporaries plus seven elementwise
    tensors with one fp32 ``[B,H,T]`` score output.
    """
    if keys.shape != values.shape or probability.shape != keys.shape[:3]:
        raise ValueError("probability and K/V shapes are incompatible")
    if expected_value.shape != (keys.shape[0], keys.shape[1], keys.shape[3]):
        raise ValueError("expected_value must be [B,H,D]")
    batch, heads, seq, dim = keys.shape
    if not (_HAS_TRITON and probability.is_cuda and keys.is_cuda and values.is_cuda):
        p = probability.float()
        k = keys.float()
        value = values.float()
        out = expected_value.float()
        vn2 = value.square().sum(-1)
        vo = torch.einsum("bhtd,bhd->bht", value, out)
        on2 = out.square().sum(-1, keepdim=True)
        irrep = (vn2 - 2.0 * vo + on2).clamp_min(0.0)
        kn2 = k.square().sum(-1)
        return (p + float(epsilon)).square() * (vn2 + (kn2 / dim) * irrep)

    score = torch.empty_like(probability, dtype=torch.float32)
    block_d = triton.next_power_of_2(dim)
    block_t = 16
    with torch.cuda.device(keys.device):
        _joint_mse_score_kernel[(batch * heads, triton.cdiv(seq, block_t))](
            probability,
            keys,
            values,
            expected_value,
            score,
            probability.stride(0),
            probability.stride(1),
            probability.stride(2),
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            keys.stride(3),
            values.stride(0),
            values.stride(1),
            values.stride(2),
            values.stride(3),
            expected_value.stride(0),
            expected_value.stride(1),
            expected_value.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            seq,
            float(epsilon),
            H=heads,
            D=dim,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=4,
        )
    return score
