"""Experimental GQA=4 specialization for native compressed decode.

This module is intentionally isolated from the production dispatch.  It keeps
the production 0/2/16 storage contract (ragged two-bit codes, ragged exact
prefix, and a raw decode tail) but uses four query rows per Triton program
instead of padding the query tile to ``BM=16``.  The purpose is to measure
whether the production kernel's padded accumulator footprint is material on
Ada before changing the supported path.

The public wrapper mirrors :func:`odmkv.kernels.fused_decode.fused_decode`
for the production layout.  It raises for ``G != 4`` or a 3/4/8-bit bank; it is
not a fallback implementation.
"""
from __future__ import annotations

import math
import os

import torch

from kvquant.runtime.kernels import fused_decode as _fd

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = _fd._HAS_TRITON
except Exception:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _two_bit_single_query(
        qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE,
        off, sl, s_id, scaling, m_i, l_i, acc,
        D: tl.constexpr, BYTES: tl.constexpr, NS: tl.constexpr,
        BT: tl.constexpr,
    ):
        """CUDA-core 2-bit attention for one real query (no MMA M padding)."""
        d = tl.arange(0, D)
        byte = d // 4
        shift = (d % 4) * 2
        pbase = tl.load(PBASE) + off * BYTES
        nbase = tl.load(NBASE) + off
        split_tokens = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * split_tokens
        t_hi = tl.minimum(t_lo + split_tokens, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            pk_t = tl.load(
                PK + pbase + t[None, :] * BYTES + byte[:, None],
                mask=valid[None, :], other=0,
            )
            index_k = (pk_t >> shift[:, None]) & 3
            norm_k = tl.load(NK + nbase + t, mask=valid, other=0.0)
            khat_t = tl.load(CK + index_k).to(tl.float32) * norm_k[None, :]
            score = tl.sum(qr[:, None].to(tl.float32) * khat_t, axis=0) * scaling
            score = tl.where(valid, score, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(score, axis=0))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            prob = tl.where(score == -float("inf"), 0.0, tl.exp(score - m_new))
            l_i = l_i * alpha + tl.sum(prob, axis=0)

            pv = tl.load(
                PV + pbase + t[:, None] * BYTES + byte[None, :],
                mask=valid[:, None], other=0,
            )
            index_v = (pv >> shift[None, :]) & 3
            norm_v = tl.load(NV + nbase + t, mask=valid, other=0.0)
            vhat = tl.load(CV + index_v).to(tl.float32) * norm_v[:, None]
            acc = acc * alpha + tl.sum(prob[:, None] * vhat, axis=0)
            m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _raw_single_query(
        q, K, V, off, sl, s_id, scaling, m_i, l_i, acc_cb, acc_raw,
        D: tl.constexpr, NS: tl.constexpr, BT: tl.constexpr,
    ):
        """Raw BF16 exact/tail source for one query and one split."""
        d = tl.arange(0, D)
        base = off * D
        split_tokens = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * split_tokens
        t_hi = tl.minimum(t_lo + split_tokens, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            kt = tl.load(
                K + base + t[None, :] * D + d[:, None],
                mask=valid[None, :], other=0.0,
            ).to(tl.float32)
            score = tl.sum(q[:, None].to(tl.float32) * kt, axis=0) * scaling
            score = tl.where(valid, score, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(score, axis=0))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            prob = tl.where(score == -float("inf"), 0.0, tl.exp(score - m_new))
            l_i = l_i * alpha + tl.sum(prob, axis=0)
            vt = tl.load(
                V + base + t[:, None] * D + d[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)
            acc_cb *= alpha
            acc_raw = acc_raw * alpha + tl.sum(prob[:, None] * vt, axis=0)
            m_i = m_new
        return m_i, l_i, acc_cb, acc_raw

    @triton.jit
    def _row_scalar(x, row: tl.constexpr, BM: tl.constexpr):
        rows = tl.arange(0, BM)
        return tl.sum(tl.where(rows == row, x, 0.0), axis=0)

    @triton.jit
    def _row_vector(x, row: tl.constexpr, BM: tl.constexpr):
        rows = tl.arange(0, BM)
        return tl.sum(tl.where(rows[:, None] == row, x, 0.0), axis=0)

    @triton.jit
    def _two_bit_hybrid(
        qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE,
        off, sl, s_id, scaling, qmask, m_i, l_i, a0, a1, a2, a3,
        D: tl.constexpr, BYTES: tl.constexpr, NS: tl.constexpr,
        BM: tl.constexpr, BT: tl.constexpr,
    ):
        """MMA QK with shared K, but only four real value accumulators."""
        d = tl.arange(0, D)
        byte = d // 4
        shift = (d % 4) * 2
        pbase = tl.load(PBASE) + off * BYTES
        nbase = tl.load(NBASE) + off
        split_tokens = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * split_tokens
        t_hi = tl.minimum(t_lo + split_tokens, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            pk_t = tl.load(
                PK + pbase + t[None, :] * BYTES + byte[:, None],
                mask=valid[None, :], other=0,
            )
            index_k = (pk_t >> shift[:, None]) & 3
            norm_k = tl.load(NK + nbase + t, mask=valid, other=0.0)
            khat_t = (tl.load(CK + index_k) * norm_k[None, :]).to(tl.bfloat16)
            score = tl.dot(qr, khat_t) * scaling
            score = tl.where(valid[None, :] & qmask[:, None], score, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(score, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            prob = tl.where(score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(prob, axis=1)

            pv = tl.load(
                PV + pbase + t[:, None] * BYTES + byte[None, :],
                mask=valid[:, None], other=0,
            )
            index_v = (pv >> shift[None, :]) & 3
            norm_v = tl.load(NV + nbase + t, mask=valid, other=0.0)
            vhat = tl.load(CV + index_v).to(tl.float32) * norm_v[:, None]
            alpha0 = _row_scalar(alpha, 0, BM)
            alpha1 = _row_scalar(alpha, 1, BM)
            alpha2 = _row_scalar(alpha, 2, BM)
            alpha3 = _row_scalar(alpha, 3, BM)
            p0 = _row_vector(prob, 0, BM)
            p1 = _row_vector(prob, 1, BM)
            p2 = _row_vector(prob, 2, BM)
            p3 = _row_vector(prob, 3, BM)
            a0 = a0 * alpha0 + tl.sum(p0[:, None] * vhat, axis=0)
            a1 = a1 * alpha1 + tl.sum(p1[:, None] * vhat, axis=0)
            a2 = a2 * alpha2 + tl.sum(p2[:, None] * vhat, axis=0)
            a3 = a3 * alpha3 + tl.sum(p3[:, None] * vhat, axis=0)
            m_i = m_new
        return m_i, l_i, a0, a1, a2, a3

    @triton.jit
    def _raw_hybrid(
        q, K, V, off, sl, s_id, scaling, qmask, m_i, l_i,
        cb0, cb1, cb2, cb3, raw0, raw1, raw2, raw3,
        D: tl.constexpr, NS: tl.constexpr, BM: tl.constexpr,
        BT: tl.constexpr,
    ):
        d = tl.arange(0, D)
        base = off * D
        split_tokens = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * split_tokens
        t_hi = tl.minimum(t_lo + split_tokens, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            kt = tl.load(
                K + base + t[None, :] * D + d[:, None],
                mask=valid[None, :], other=0.0,
            ).to(tl.bfloat16)
            score = tl.dot(q, kt) * scaling
            score = tl.where(valid[None, :] & qmask[:, None], score, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(score, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            prob = tl.where(score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(prob, axis=1)
            vt = tl.load(
                V + base + t[:, None] * D + d[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.float32)
            alpha0 = _row_scalar(alpha, 0, BM)
            alpha1 = _row_scalar(alpha, 1, BM)
            alpha2 = _row_scalar(alpha, 2, BM)
            alpha3 = _row_scalar(alpha, 3, BM)
            p0 = _row_vector(prob, 0, BM)
            p1 = _row_vector(prob, 1, BM)
            p2 = _row_vector(prob, 2, BM)
            p3 = _row_vector(prob, 3, BM)
            cb0 *= alpha0
            cb1 *= alpha1
            cb2 *= alpha2
            cb3 *= alpha3
            raw0 = raw0 * alpha0 + tl.sum(p0[:, None] * vt, axis=0)
            raw1 = raw1 * alpha1 + tl.sum(p1[:, None] * vt, axis=0)
            raw2 = raw2 * alpha2 + tl.sum(p2[:, None] * vt, axis=0)
            raw3 = raw3 * alpha3 + tl.sum(p3[:, None] * vt, axis=0)
            m_i = m_new
        return (m_i, l_i, cb0, cb1, cb2, cb3,
                raw0, raw1, raw2, raw3)

    @triton.jit
    def _split_g4_kernel(
        QR, Q,
        PK, NK, PV, NV, CK, CV, PBASE, NBASE, OFF, SL,
        EK, EV, OFFE, SLE,
        TK, TV, OFFT, SLT,
        M_OUT, L_OUT, CB_OUT, RAW_OUT, scaling,
        D: tl.constexpr, B2: tl.constexpr,
        HAS_2BIT: tl.constexpr, HAS_EXACT: tl.constexpr,
        HAS_TAIL: tl.constexpr, NS: tl.constexpr, BT: tl.constexpr,
    ):
        """One program per (KV row, real query, split), with no MMA padding."""
        r = tl.program_id(0)
        q_id = tl.program_id(1)
        s_id = tl.program_id(2)
        G: tl.constexpr = 4
        d = tl.arange(0, D)
        qr = tl.load(QR + (r * G + q_id) * D + d).to(tl.bfloat16)
        q = tl.load(Q + (r * G + q_id) * D + d).to(tl.bfloat16)
        m_i = -float("inf")
        l_i = 0.0
        acc_cb = tl.zeros([D], tl.float32)
        acc_raw = tl.zeros([D], tl.float32)

        if HAS_2BIT:
            m_i, l_i, acc_cb = _two_bit_single_query(
                qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE,
                tl.load(OFF + r), tl.load(SL + r), s_id, scaling,
                m_i, l_i, acc_cb, D, B2, NS, BT,
            )
        if HAS_EXACT:
            m_i, l_i, acc_cb, acc_raw = _raw_single_query(
                q, EK, EV, tl.load(OFFE + r), tl.load(SLE + r),
                s_id, scaling, m_i, l_i, acc_cb, acc_raw, D, NS, BT,
            )
        if HAS_TAIL:
            m_i, l_i, acc_cb, acc_raw = _raw_single_query(
                q, TK, TV, tl.load(OFFT + r), tl.load(SLT + r),
                s_id, scaling, m_i, l_i, acc_cb, acc_raw, D, NS, BT,
            )

        po = (r * NS + s_id) * G + q_id
        tl.store(M_OUT + po, m_i)
        tl.store(L_OUT + po, l_i)
        base = po * D + d
        tl.store(CB_OUT + base, acc_cb)
        tl.store(RAW_OUT + base, acc_raw)

    @triton.jit
    def _split_g4_hybrid_kernel(
        QR, Q,
        PK, NK, PV, NV, CK, CV, PBASE, NBASE, OFF, SL,
        EK, EV, OFFE, SLE,
        TK, TV, OFFT, SLT,
        M_OUT, L_OUT, CB_OUT, RAW_OUT, scaling,
        D: tl.constexpr, B2: tl.constexpr,
        HAS_2BIT: tl.constexpr, HAS_EXACT: tl.constexpr,
        HAS_TAIL: tl.constexpr, NS: tl.constexpr, BT: tl.constexpr,
    ):
        """Shared tensor-core scores plus four unpadded value accumulators."""
        r = tl.program_id(0)
        s_id = tl.program_id(1)
        BM: tl.constexpr = 16
        G: tl.constexpr = 4
        rows = tl.arange(0, BM)
        d = tl.arange(0, D)
        qmask = rows < G
        qr = tl.load(
            QR + (r * G + rows)[:, None] * D + d[None, :],
            mask=qmask[:, None], other=0.0,
        ).to(tl.bfloat16)
        q = tl.load(
            Q + (r * G + rows)[:, None] * D + d[None, :],
            mask=qmask[:, None], other=0.0,
        ).to(tl.bfloat16)
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        cb0 = tl.zeros([D], tl.float32)
        cb1 = tl.zeros([D], tl.float32)
        cb2 = tl.zeros([D], tl.float32)
        cb3 = tl.zeros([D], tl.float32)
        raw0 = tl.zeros([D], tl.float32)
        raw1 = tl.zeros([D], tl.float32)
        raw2 = tl.zeros([D], tl.float32)
        raw3 = tl.zeros([D], tl.float32)

        if HAS_2BIT:
            m_i, l_i, cb0, cb1, cb2, cb3 = _two_bit_hybrid(
                qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE,
                tl.load(OFF + r), tl.load(SL + r), s_id, scaling, qmask,
                m_i, l_i, cb0, cb1, cb2, cb3, D, B2, NS, BM, BT,
            )
        if HAS_EXACT:
            (m_i, l_i, cb0, cb1, cb2, cb3,
             raw0, raw1, raw2, raw3) = _raw_hybrid(
                q, EK, EV, tl.load(OFFE + r), tl.load(SLE + r), s_id,
                scaling, qmask, m_i, l_i, cb0, cb1, cb2, cb3,
                raw0, raw1, raw2, raw3, D, NS, BM, BT,
            )
        if HAS_TAIL:
            (m_i, l_i, cb0, cb1, cb2, cb3,
             raw0, raw1, raw2, raw3) = _raw_hybrid(
                q, TK, TV, tl.load(OFFT + r), tl.load(SLT + r), s_id,
                scaling, qmask, m_i, l_i, cb0, cb1, cb2, cb3,
                raw0, raw1, raw2, raw3, D, NS, BM, BT,
            )

        po = (r * NS + s_id) * G
        tl.store(M_OUT + po + rows, m_i, mask=qmask)
        tl.store(L_OUT + po + rows, l_i, mask=qmask)
        tl.store(CB_OUT + (po + 0) * D + d, cb0)
        tl.store(CB_OUT + (po + 1) * D + d, cb1)
        tl.store(CB_OUT + (po + 2) * D + d, cb2)
        tl.store(CB_OUT + (po + 3) * D + d, cb3)
        tl.store(RAW_OUT + (po + 0) * D + d, raw0)
        tl.store(RAW_OUT + (po + 1) * D + d, raw1)
        tl.store(RAW_OUT + (po + 2) * D + d, raw2)
        tl.store(RAW_OUT + (po + 3) * D + d, raw3)


def _raw_source(source_k, source_v, source_meta, nr: int, dim: int, device):
    if source_k is None:
        k, v, sl = _fd._empty_exact(nr, dim, device)
        off = torch.zeros(nr, device=device, dtype=torch.int32)
        return k, v, off, sl, 0, False
    if isinstance(source_meta, dict):
        off = source_meta["offset"].to(device, torch.int32).contiguous()
        sl = source_meta["seqlen"].to(device, torch.int32).contiguous()
        tmax = int(source_meta.get("T_max", 0))
    else:
        stride = source_k.shape[1]
        off = (torch.arange(nr, device=device, dtype=torch.int32) * stride).contiguous()
        sl = (source_meta if source_meta is not None else
              torch.full((nr,), stride, device=device, dtype=torch.int32))
        sl = sl.to(device, torch.int32).contiguous()
        tmax = int(stride)
    return (source_k.to(torch.bfloat16).contiguous(),
            source_v.to(torch.bfloat16).contiguous(), off, sl, tmax, True)


def fused_decode_g4(
    Q,
    quant_banks,
    exact_k,
    exact_v,
    exact_seqlen,
    pi_k,
    pi_v,
    scaling,
    *,
    tail_k=None,
    tail_v=None,
    tail_seqlen=None,
    tail_offset=None,
    BT: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    variant: str = "single",
):
    """Experimental production-layout decode for exactly four Q heads/KV head."""
    if not _HAS_TRITON:
        raise RuntimeError("fused_decode_g4 requires CUDA + Triton")
    nr, groups, dim = Q.shape
    if groups != 4:
        raise ValueError(f"fused_decode_g4 requires G=4, got {groups}")
    unsupported = sorted({int(bank["bits"]) for bank in quant_banks} - {2})
    if unsupported:
        raise ValueError(f"fused_decode_g4 only supports 0/2/16 layout; got {unsupported}")

    device = Q.device
    bf = torch.bfloat16
    q = Q.to(bf).contiguous()
    qr = torch.matmul(q, pi_k.to(bf).transpose(-1, -2)).contiguous()
    slot = _fd._slot(quant_banks, 2, nr, dim, device)
    has_2bit = _fd._slot_tmax(quant_banks, 2) > 0

    ek, ev, off_e, sl_e, te, has_exact = _raw_source(
        exact_k, exact_v, exact_seqlen, nr, dim, device,
    )
    tk, tv, off_t_default, sl_t, tt, has_tail = _raw_source(
        tail_k, tail_v, tail_seqlen, nr, dim, device,
    )
    if tail_offset is None:
        off_t = off_t_default
    else:
        off_t = tail_offset.to(device, torch.int32).contiguous()

    BT = int(os.environ.get("R2_G4_BT", BT))
    num_warps = int(os.environ.get("R2_G4_WARPS", num_warps))
    num_stages = int(os.environ.get("R2_G4_STAGES", num_stages))
    tmax = max(_fd._slot_tmax(quant_banks, 2), te, tt)
    ns = max(1, min(16, math.ceil(256 / nr)))
    ns = min(ns, max(1, tmax // 128))
    forced_ns = os.environ.get("R2_G4_NS")
    if forced_ns and int(forced_ns) > 0:
        ns = int(forced_ns)

    _, _, _, bytes2 = _fd._pp(2, dim)
    m = torch.empty(nr, ns, groups, device=device, dtype=torch.float32)
    l = torch.empty_like(m)
    cb = torch.empty(nr, ns, groups, dim, device=device, dtype=torch.float32)
    raw = torch.empty_like(cb)
    if variant == "single":
        kernel = _split_g4_kernel
        grid = (nr, groups, ns)
    elif variant == "hybrid":
        kernel = _split_g4_hybrid_kernel
        grid = (nr, ns)
    else:
        raise ValueError(f"unknown G4 prototype variant {variant!r}")
    kernel[grid](
        qr, q, *slot,
        ek, ev, off_e, sl_e,
        tk, tv, off_t, sl_t,
        m, l, cb, raw, scaling,
        D=dim, B2=bytes2,
        HAS_2BIT=has_2bit, HAS_EXACT=has_exact and te > 0,
        HAS_TAIL=has_tail and tt > 0,
        NS=ns, BT=BT, num_warps=num_warps, num_stages=num_stages,
    )

    if ns == 1:
        cb_o, raw_o, l_o = cb[:, 0], raw[:, 0], l[:, 0]
    else:
        cb_o = torch.empty(nr, groups, dim, device=device, dtype=torch.float32)
        raw_o = torch.empty_like(cb_o)
        l_o = torch.empty(nr, groups, device=device, dtype=torch.float32)
        _fd._combine_kernel[(nr,)](
            m, l, cb, raw, cb_o, raw_o, l_o,
            NS=ns, D=dim, BM=groups, G=groups, num_warps=4,
        )
    return (cb_o @ pi_v.float() + raw_o) / l_o.clamp(min=1e-20).unsqueeze(-1)


__all__ = ["fused_decode_g4", "_HAS_TRITON"]
