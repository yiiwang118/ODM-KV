"""Single fused decode kernel — all bit levels + exact + tail in ONE launch,
with split-K occupancy (matching FA2's flash_fwd_splitkv decode kernel).

The multi-op ``banked_decode_attention`` driver issues ~1 kernel per bit-bank +
a PyTorch cross-bank online-softmax combine + rotations — ~115 kernels and
hundreds of ops per decode step (94% of the fused-vs-fp16 gap is that launch
overhead; see profile_decode.py / cuda_graph_probe.py).

This kernel folds everything into ONE launch (grid = NR × n_split):
  * quant levels 2/3/4/8 via a per-level device function (absent = seqlen 0);
  * exact 16-bit prefix + the bf16 decode tail as two raw sources;
with a shared online softmax and TWO value accumulators —
  acc_cb  (quant, value-codebook space, rotated by pi_v at the end) and
  acc_raw (exact, raw space, no rotation).
Split-K partials (m,l,acc_cb,acc_raw) are combined on the host, which then does
only ``out = (acc_cb @ pi_v + acc_raw) / l``. The tail is a second raw source,
so no per-step exact rebuild is needed.

``fused_decode_one_level`` (V1) is kept for the single-level unit test.
"""
from __future__ import annotations

import os as _os_flags
import threading

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_TRITON = False

# Load packed payloads as int32 words instead of per-element bytes.  The byte
# path indexes with ``byte = d // VPB`` — a computed index Triton cannot prove
# affine, so it emits one 1-byte load per element (PTX ``ld.global.b8`` x64 per
# tile pair) and the nominal byte saving never reaches the memory system: a
# component ladder measured the 2-bit loads at 24x their traffic lower bound,
# costing 2.4x more time than loading the same tokens in bf16.  Word loads are
# affine ([W, BT] with W = BYTES/4), vectorize, and the codes are recovered in
# registers by a broadcast shift — for little-endian packing the bit offset of
# value j inside its word is EFF*j for every EFF in {2, 4, 8}, so the extracted
# codes are identical to the byte path's.  Word alignment holds because every
# level's per-token BYTES (32/48/64/128) is a multiple of 4, hence so is every
# cumsum level base, and the arenas start allocation-aligned.
#
# The dequantized tiles are bit-identical; final outputs may still differ at
# ULP scale because the changed register layout lets Triton pick a different
# tl.dot operand layout, reassociating the D-length accumulation — the same
# class of change as altering the split-K width.
#
# The switch is threaded into the kernels as a constexpr ARGUMENT, never read
# inside @jit as a captured global: Triton's compilation cache does not key on
# captured constexpr values, so a flag captured from the environment silently
# reuses whichever binary a previous process compiled — an A/B run measured
# the control twice that way.  Default on; R2_WIDE_LOAD=0 rolls back.
_WIDE_LOAD = _os_flags.environ.get("R2_WIDE_LOAD", "1") == "1"


def _pp(bits, D, physical_bits=None):
    if bits == 3 and physical_bits == 3:
        return 3, 1, 7, (3 * D + 7) // 8
    eff = 1 if bits == 1 else 2 if bits == 2 else 4 if bits <= 4 else 8
    vpb = 8 // eff
    return eff, vpb, (1 << eff) - 1, (D + vpb - 1) // vpb


if _HAS_TRITON:

    @triton.jit
    def _ql(qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE, off, sl, s_id, r, scaling,
            qmask, m_i, l_i, acc,
            D: tl.constexpr, BYTES: tl.constexpr, VPB: tl.constexpr, EFF: tl.constexpr,
            MASK: tl.constexpr, NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
            WIDE: tl.constexpr):
        """Fold one quant bit-level's split-s tokens into (m_i, l_i, acc_cb).

        ``off`` is the row's start index in a FLAT token array — for the padded
        layout it is ``r*T_max`` (rectangular view); for the ragged layout it is
        ``cumsum(counts)`` with zero padding. The kernel is identical either way;
        the ragged layout just removes the ~4× padding (both storage and the
        masked loads). Loop bound is the per-row count ``sl`` (data-dependent
        trip count, CUDA-graph-safe)."""
        d = tl.arange(0, D)
        byte = d // VPB
        shift = (d % VPB) * EFF
        # Shared-budget arenas carry device-resident level bases. Legacy and
        # per-level banks pass a cached zero scalar, retaining their old layout.
        pbase = tl.load(PBASE) + off * BYTES
        nbase = tl.load(NBASE) + off
        if WIDE and (EFF == 2 or EFF == 4 or EFF == 8):
            PKW = PK.to(tl.pointer_type(tl.int32))
            PVW = PV.to(tl.pointer_type(tl.int32))
            # ``pbase`` is runtime data (device-resident level base), so the
            # vectorizer cannot see its alignment on its own and would fall
            # back to scalar 4-byte loads.  Promise 8-byte alignment (v2.b32),
            # not 16: level code bases are cumsums whose smallest observed
            # granule is 8 bytes — the true-3-bit region carries a non-48-
            # multiple guard tail, so 16 is genuinely violated (a measured
            # base of 6856 crashed a v4 load with a misaligned address).
            # Every per-token stride (32/48/64/128) and every region size is
            # a multiple of 8, so 8 holds for all levels.
            wbase = tl.multiple_of(pbase // 4, 2)
        TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * TS
        # Split-local early exit: never iterate past this row's real token count.
        # A split whose range starts beyond ``sl`` (common at high eviction /
        # absent banks, sl=0) runs zero iterations instead of masked loads+dots.
        t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            if WIDE and (EFF == 2 or EFF == 4 or EFF == 8):
                W: tl.constexpr = BYTES // 4
                VW: tl.constexpr = 32 // EFF
                kw = tl.load(PKW + wbase + t[None, :] * W + tl.arange(0, W)[:, None],
                             mask=valid[None, :], other=0)
                ikT = tl.reshape(
                    (kw[:, None, :] >> (EFF * tl.arange(0, VW))[None, :, None]) & MASK,
                    (D, BT))
            else:
                pkT = tl.load(PK + pbase + t[None, :] * BYTES + byte[:, None],
                              mask=valid[None, :], other=0)
                ikT = (pkT >> shift[:, None]) & MASK
            nk = tl.load(NK + nbase + t, mask=valid, other=0.0).to(tl.float32)
            khatT = (tl.load(CK + ikT) * nk[None, :]).to(tl.bfloat16)
            s = tl.dot(qr, khatT) * scaling
            s = tl.where(valid[None, :] & qmask[:, None], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(s == -float("inf"), 0.0, tl.exp(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(p, axis=1)
            if WIDE and (EFF == 2 or EFF == 4 or EFF == 8):
                vw = tl.load(PVW + wbase + t[:, None] * W + tl.arange(0, W)[None, :],
                             mask=valid[:, None], other=0)
                iv = tl.reshape(
                    (vw[:, :, None] >> (EFF * tl.arange(0, VW))[None, None, :]) & MASK,
                    (BT, D))
            else:
                pv = tl.load(PV + pbase + t[:, None] * BYTES + byte[None, :],
                             mask=valid[:, None], other=0)
                iv = (pv >> shift[None, :]) & MASK
            nv = tl.load(NV + nbase + t, mask=valid, other=0.0).to(tl.float32)
            vhat = (tl.load(CV + iv) * nv[:, None]).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vhat)
            m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _ql3(qr, PK, NK, PV, NV, CK, CV, PBASE, NBASE, off, sl, s_id, r, scaling,
             qmask, m_i, l_i, acc,
             D: tl.constexpr, BYTES: tl.constexpr,
             NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr):
        """Fold one dense true-3-bit bank into the online softmax state."""
        d = tl.arange(0, D)
        bit = d * 3
        byte = bit // 8
        shift = bit - byte * 8
        pbase = tl.load(PBASE) + off * BYTES
        nbase = tl.load(NBASE) + off
        TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * TS
        t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            pk_lo = tl.load(
                PK + pbase + t[None, :] * BYTES + byte[:, None],
                mask=valid[None, :], other=0,
            ).to(tl.int32)
            pk_hi = tl.load(
                PK + pbase + t[None, :] * BYTES + byte[:, None] + 1,
                mask=valid[None, :] & ((byte + 1)[:, None] < BYTES), other=0,
            ).to(tl.int32)
            ikT = ((pk_lo | (pk_hi << 8)) >> shift[:, None]) & 7
            nk = tl.load(NK + nbase + t, mask=valid, other=0.0).to(tl.float32)
            khatT = (tl.load(CK + ikT) * nk[None, :]).to(tl.bfloat16)
            s = tl.dot(qr, khatT) * scaling
            s = tl.where(valid[None, :] & qmask[:, None], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(s == -float("inf"), 0.0, tl.exp(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(p, axis=1)
            pv_lo = tl.load(
                PV + pbase + t[:, None] * BYTES + byte[None, :],
                mask=valid[:, None], other=0,
            ).to(tl.int32)
            pv_hi = tl.load(
                PV + pbase + t[:, None] * BYTES + byte[None, :] + 1,
                mask=valid[:, None] & ((byte + 1)[None, :] < BYTES), other=0,
            ).to(tl.int32)
            iv = ((pv_lo | (pv_hi << 8)) >> shift[None, :]) & 7
            nv = tl.load(NV + nbase + t, mask=valid, other=0.0).to(tl.float32)
            vhat = (tl.load(CV + iv) * nv[:, None]).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vhat)
            m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _ql_segments(
        qr, PK, NK, PV, NV, CK, CV, CBASE, NBASE, COUNTS,
        s_id, r, scaling, qmask, m_i, l_i, acc,
        D: tl.constexpr, BYTES: tl.constexpr, VPB: tl.constexpr,
        EFF: tl.constexpr, MASK: tl.constexpr, LEVEL: tl.constexpr,
        NR: tl.constexpr, MAX_SEGMENTS: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
    ):
        """Fold every populated segment of one quantized decode level."""
        d = tl.arange(0, D)
        byte = d // VPB
        shift = (d % VPB) * EFF
        for segment in range(MAX_SEGMENTS):
            sl = tl.load(COUNTS + (segment * 5 + LEVEL) * NR + r)
            pbase = tl.load(CBASE + (segment * 4 + LEVEL) * NR + r)
            nbase = tl.load(NBASE + (segment * 4 + LEVEL) * NR + r)
            TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
            t_lo = s_id * TS
            t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
            for t0 in range(t_lo, t_hi, BT):
                t = t0 + tl.arange(0, BT)
                valid = t < sl
                pkT = tl.load(
                    PK + pbase + t[None, :] * BYTES + byte[:, None],
                    mask=valid[None, :], other=0,
                )
                ikT = (pkT >> shift[:, None]) & MASK
                nk = tl.load(NK + nbase + t, mask=valid, other=0.0).to(tl.float32)
                khatT = (tl.load(CK + ikT) * nk[None, :]).to(tl.bfloat16)
                score = tl.dot(qr, khatT) * scaling
                score = tl.where(
                    valid[None, :] & qmask[:, None], score, -float("inf"),
                )
                m_new = tl.maximum(m_i, tl.max(score, axis=1))
                alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
                probability = tl.where(
                    score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]),
                )
                l_i = l_i * alpha + tl.sum(probability, axis=1)
                pv = tl.load(
                    PV + pbase + t[:, None] * BYTES + byte[None, :],
                    mask=valid[:, None], other=0,
                )
                iv = (pv >> shift[None, :]) & MASK
                nv = tl.load(NV + nbase + t, mask=valid, other=0.0).to(tl.float32)
                vhat = (tl.load(CV + iv) * nv[:, None]).to(tl.bfloat16)
                acc = acc * alpha[:, None] + tl.dot(
                    probability.to(tl.bfloat16), vhat,
                )
                m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _ql3_segments(
        qr, PK, NK, PV, NV, CK, CV, CBASE, NBASE, COUNTS,
        s_id, r, scaling, qmask, m_i, l_i, acc,
        D: tl.constexpr, BYTES: tl.constexpr, LEVEL: tl.constexpr,
        NR: tl.constexpr, MAX_SEGMENTS: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
    ):
        d = tl.arange(0, D)
        bit = d * 3
        byte = bit // 8
        shift = bit - byte * 8
        for segment in range(MAX_SEGMENTS):
            sl = tl.load(COUNTS + (segment * 5 + LEVEL) * NR + r)
            pbase = tl.load(CBASE + (segment * 4 + LEVEL) * NR + r)
            nbase = tl.load(NBASE + (segment * 4 + LEVEL) * NR + r)
            TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
            t_lo = s_id * TS
            t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
            for t0 in range(t_lo, t_hi, BT):
                t = t0 + tl.arange(0, BT)
                valid = t < sl
                pk_lo = tl.load(
                    PK + pbase + t[None, :] * BYTES + byte[:, None],
                    mask=valid[None, :], other=0,
                ).to(tl.int32)
                pk_hi = tl.load(
                    PK + pbase + t[None, :] * BYTES + byte[:, None] + 1,
                    mask=valid[None, :] & ((byte + 1)[:, None] < BYTES), other=0,
                ).to(tl.int32)
                ikT = ((pk_lo | (pk_hi << 8)) >> shift[:, None]) & 7
                nk = tl.load(NK + nbase + t, mask=valid, other=0.0).to(tl.float32)
                khatT = (tl.load(CK + ikT) * nk[None, :]).to(tl.bfloat16)
                score = tl.dot(qr, khatT) * scaling
                score = tl.where(
                    valid[None, :] & qmask[:, None], score, -float("inf"),
                )
                m_new = tl.maximum(m_i, tl.max(score, axis=1))
                alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
                probability = tl.where(
                    score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]),
                )
                l_i = l_i * alpha + tl.sum(probability, axis=1)
                pv_lo = tl.load(
                    PV + pbase + t[:, None] * BYTES + byte[None, :],
                    mask=valid[:, None], other=0,
                ).to(tl.int32)
                pv_hi = tl.load(
                    PV + pbase + t[:, None] * BYTES + byte[None, :] + 1,
                    mask=valid[:, None] & ((byte + 1)[None, :] < BYTES), other=0,
                ).to(tl.int32)
                iv = ((pv_lo | (pv_hi << 8)) >> shift[None, :]) & 7
                nv = tl.load(NV + nbase + t, mask=valid, other=0.0).to(tl.float32)
                vhat = (tl.load(CV + iv) * nv[:, None]).to(tl.bfloat16)
                acc = acc * alpha[:, None] + tl.dot(
                    probability.to(tl.bfloat16), vhat,
                )
                m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _ql_descriptor_stream(
        qr, PK, NK, PV, NV, CBASE, NBASE, DESCRIPTORS, DESCRIPTOR_COUNTS,
        CODEBOOK, s_id, r, scaling, qmask, m_i, l_i, acc,
        D: tl.constexpr, NR: tl.constexpr, CAPACITY: tl.constexpr,
        DESCRIPTOR_STRIDE: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
    ):
        """Fold all 2/3/4/8-bit decode payloads as one dense token stream."""
        d = tl.arange(0, D)
        sl = tl.load(DESCRIPTOR_COUNTS + r).to(tl.int32)
        TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * TS
        t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            descriptor = tl.load(
                DESCRIPTORS + r * CAPACITY + t, mask=valid, other=0,
            ).to(tl.int32)
            segment_level = descriptor // DESCRIPTOR_STRIDE
            rank = descriptor - segment_level * DESCRIPTOR_STRIDE
            segment = segment_level // 4
            level = segment_level - segment * 4
            bits = tl.where(
                level == 0, 2,
                tl.where(level == 1, 3, tl.where(level == 2, 4, 8)),
            ).to(tl.int32)
            width = (D * bits + 7) // 8
            table_base = tl.where(
                level == 0, 0,
                tl.where(level == 1, 4, tl.where(level == 2, 12, 28)),
            ).to(tl.int32)
            code_mask = tl.where(
                level == 0, 3,
                tl.where(level == 1, 7, tl.where(level == 2, 15, 255)),
            ).to(tl.int32)
            base_index = (segment * 4 + level) * NR + r
            pbase = tl.load(CBASE + base_index, mask=valid, other=0).to(tl.int32)
            nbase = tl.load(NBASE + base_index, mask=valid, other=0).to(tl.int32)
            bit_position = d[:, None] * bits[None, :]
            byte = bit_position // 8
            shift = bit_position - byte * 8
            payload_base = pbase[None, :] + rank[None, :] * width[None, :]
            load_mask = valid[None, :] & (byte < width[None, :])
            pk_low = tl.load(
                PK + payload_base + byte,
                mask=load_mask, other=0,
            ).to(tl.int32)
            pk_high = tl.load(
                PK + payload_base + byte + 1,
                mask=load_mask & (bits[None, :] == 3)
                & ((byte + 1) < width[None, :]),
                other=0,
            ).to(tl.int32)
            index_k = (
                ((pk_low | (pk_high << 8)) >> shift)
                & code_mask[None, :]
            )
            norm_k = tl.load(
                NK + nbase + rank, mask=valid, other=0.0,
            ).to(tl.float32)
            key = (
                tl.load(CODEBOOK + table_base[None, :] + index_k)
                * norm_k[None, :]
            ).to(tl.bfloat16)
            score = tl.dot(qr, key) * scaling
            score = tl.where(
                valid[None, :] & qmask[:, None], score, -float("inf"),
            )
            m_new = tl.maximum(m_i, tl.max(score, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            probability = tl.where(
                score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]),
            )
            l_i = l_i * alpha + tl.sum(probability, axis=1)

            bit_position_v = bits[:, None] * d[None, :]
            byte_v = bit_position_v // 8
            shift_v = bit_position_v - byte_v * 8
            payload_base_v = pbase[:, None] + rank[:, None] * width[:, None]
            load_mask_v = valid[:, None] & (byte_v < width[:, None])
            pv_low = tl.load(
                PV + payload_base_v + byte_v,
                mask=load_mask_v, other=0,
            ).to(tl.int32)
            pv_high = tl.load(
                PV + payload_base_v + byte_v + 1,
                mask=load_mask_v & (bits[:, None] == 3)
                & ((byte_v + 1) < width[:, None]),
                other=0,
            ).to(tl.int32)
            index_v = (
                ((pv_low | (pv_high << 8)) >> shift_v)
                & code_mask[:, None]
            )
            norm_v = tl.load(
                NV + nbase + rank, mask=valid, other=0.0,
            ).to(tl.float32)
            value = (
                tl.load(CODEBOOK + table_base[:, None] + index_v)
                * norm_v[:, None]
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                probability.to(tl.bfloat16), value,
            )
            m_i = m_new
        return m_i, l_i, acc

    @triton.jit
    def _exact(q, EK, EV, off, sl, s_id, r, scaling, qmask, m_i, l_i, acc_cb, acc_raw,
               D: tl.constexpr, NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr):
        """Fold one raw exact source's split-s tokens into acc_raw. ``off`` = row
        start in the flat token array (padded: r*T_max; ragged: cumsum)."""
        d = tl.arange(0, D)
        ebase = off * D
        TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * TS
        t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)   # split-local early exit
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            ekT = tl.load(EK + ebase + t[None, :] * D + d[:, None],
                          mask=valid[None, :], other=0.0).to(tl.bfloat16)   # [D,BT]
            s = tl.dot(q, ekT) * scaling
            s = tl.where(valid[None, :] & qmask[:, None], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(s == -float("inf"), 0.0, tl.exp(s - m_new[:, None]))
            l_i = l_i * alpha + tl.sum(p, axis=1)
            ev = tl.load(EV + ebase + t[:, None] * D + d[None, :],
                         mask=valid[:, None], other=0.0).to(tl.bfloat16)     # [BT,D]
            acc_cb = acc_cb * alpha[:, None]
            acc_raw = acc_raw * alpha[:, None] + tl.dot(p.to(tl.bfloat16), ev)
            m_i = m_new
        return m_i, l_i, acc_cb, acc_raw

    @triton.jit
    def _exact_segments(
        q, EK, EV, EBASE, COUNTS, s_id, r, scaling,
        qmask, m_i, l_i, acc_cb, acc_raw,
        D: tl.constexpr, NR: tl.constexpr, MAX_SEGMENTS: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
    ):
        d = tl.arange(0, D)
        for segment in range(MAX_SEGMENTS):
            sl = tl.load(COUNTS + (segment * 5 + 4) * NR + r)
            off = tl.load(EBASE + segment * NR + r)
            ebase = off * D
            TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
            t_lo = s_id * TS
            t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
            for t0 in range(t_lo, t_hi, BT):
                t = t0 + tl.arange(0, BT)
                valid = t < sl
                ekT = tl.load(
                    EK + ebase + t[None, :] * D + d[:, None],
                    mask=valid[None, :], other=0.0,
                ).to(tl.bfloat16)
                score = tl.dot(q, ekT) * scaling
                score = tl.where(
                    valid[None, :] & qmask[:, None], score, -float("inf"),
                )
                m_new = tl.maximum(m_i, tl.max(score, axis=1))
                alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
                probability = tl.where(
                    score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]),
                )
                l_i = l_i * alpha + tl.sum(probability, axis=1)
                ev = tl.load(
                    EV + ebase + t[:, None] * D + d[None, :],
                    mask=valid[:, None], other=0.0,
                ).to(tl.bfloat16)
                acc_cb = acc_cb * alpha[:, None]
                acc_raw = acc_raw * alpha[:, None] + tl.dot(
                    probability.to(tl.bfloat16), ev,
                )
                m_i = m_new
        return m_i, l_i, acc_cb, acc_raw

    @triton.jit
    def _exact_descriptor_stream(
        q, EK, EV, EBASE, DESCRIPTORS, DESCRIPTOR_COUNTS,
        s_id, r, scaling, qmask, m_i, l_i, acc_cb, acc_raw,
        D: tl.constexpr, NR: tl.constexpr, CAPACITY: tl.constexpr,
        DESCRIPTOR_STRIDE: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
    ):
        """Fold every exact decode token through one compact descriptor stream."""
        d = tl.arange(0, D)
        sl = tl.load(DESCRIPTOR_COUNTS + NR + r).to(tl.int32)
        TS = ((sl + NS - 1) // NS + BT - 1) // BT * BT
        t_lo = s_id * TS
        t_hi = tl.minimum(t_lo + TS, ((sl + BT - 1) // BT) * BT)
        for t0 in range(t_lo, t_hi, BT):
            t = t0 + tl.arange(0, BT)
            valid = t < sl
            descriptor = tl.load(
                DESCRIPTORS + r * CAPACITY + (CAPACITY - 1 - t),
                mask=valid, other=0,
            ).to(tl.int32)
            segment = descriptor // DESCRIPTOR_STRIDE
            rank = descriptor - segment * DESCRIPTOR_STRIDE
            off = tl.load(EBASE + segment * NR + r, mask=valid, other=0).to(tl.int32)
            slot = off + rank
            exact_k = tl.load(
                EK + slot[None, :] * D + d[:, None],
                mask=valid[None, :], other=0.0,
            ).to(tl.bfloat16)
            score = tl.dot(q, exact_k) * scaling
            score = tl.where(
                valid[None, :] & qmask[:, None], score, -float("inf"),
            )
            m_new = tl.maximum(m_i, tl.max(score, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            probability = tl.where(
                score == -float("inf"), 0.0, tl.exp(score - m_new[:, None]),
            )
            l_i = l_i * alpha + tl.sum(probability, axis=1)
            exact_v = tl.load(
                EV + slot[:, None] * D + d[None, :],
                mask=valid[:, None], other=0.0,
            ).to(tl.bfloat16)
            acc_cb = acc_cb * alpha[:, None]
            acc_raw = acc_raw * alpha[:, None] + tl.dot(
                probability.to(tl.bfloat16), exact_v,
            )
            m_i = m_new
        return m_i, l_i, acc_cb, acc_raw

    @triton.jit
    def _fused_split_kernel(
        QR, Q,
        PK0, NK0, PV0, NV0, CK0, CV0, PB0, NB0, OFF0, SL0,
        PK1, NK1, PV1, NV1, CK1, CV1, PB1, NB1, OFF1, SL1,
        PK2, NK2, PV2, NV2, CK2, CV2, PB2, NB2, OFF2, SL2,
        PK3, NK3, PV3, NV3, CK3, CV3, PB3, NB3, OFF3, SL3,
        EK, EV, OFFE, SLE,      # exact 16-bit prefix
        TK, TV, OFFT, SLT,      # decode tail (raw)
        DPK, DNK, DPV, DNV, DCBASE, DNBASE, DCOUNTS,
        DEK, DEV, DEBASE,        # aggregate segmented decode suffix
        DQDESC, DEDESC, DDCOUNTS, DCODEBOOK,
        M_OUT, L_OUT, CB_OUT, RAW_OUT, scaling,
        G: tl.constexpr, D: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr,
        B4: tl.constexpr, B8: tl.constexpr, TRUE3: tl.constexpr,
        HAS0: tl.constexpr, HAS1: tl.constexpr, HAS2: tl.constexpr,
        HAS3: tl.constexpr, HAS_EXACT: tl.constexpr, HAS_TAIL: tl.constexpr,
        HAS_SEGMENTS: tl.constexpr, USE_DESCRIPTOR_STREAMS: tl.constexpr,
        NR: tl.constexpr, MAX_SEGMENTS: tl.constexpr,
        DESCRIPTOR_CAPACITY: tl.constexpr, DESCRIPTOR_STRIDE: tl.constexpr,
        NS: tl.constexpr, BM: tl.constexpr, BT: tl.constexpr,
        SM: tl.constexpr, WIDE: tl.constexpr,
    ):
        r = tl.program_id(0)
        s_id = tl.program_id(1)
        offs_m = tl.arange(0, BM)
        d = tl.arange(0, D)
        qmask = offs_m < G
        qr = tl.load(QR + (r * G + offs_m)[:, None] * D + d[None, :],
                     mask=qmask[:, None], other=0.0).to(tl.bfloat16)
        q = tl.load(Q + (r * G + offs_m)[:, None] * D + d[None, :],
                    mask=qmask[:, None], other=0.0).to(tl.bfloat16)
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc_cb = tl.zeros([BM, D], tl.float32)
        acc_raw = tl.zeros([BM, D], tl.float32)
        # quant levels (2-bit VPB4/EFF2/MASK3, 3&4-bit VPB2/EFF4/MASK15, 8-bit VPB1/EFF8/MASK255)
        if HAS0:
            m_i, l_i, acc_cb = _ql(qr, PK0, NK0, PV0, NV0, CK0, CV0, PB0, NB0,
                                   tl.load(OFF0 + r), tl.load(SL0 + r),
                                   s_id, r, scaling, qmask, m_i, l_i, acc_cb, D, B2, 4, 2, 3, NS, BM, BT,
                                   WIDE)
        if HAS_SEGMENTS and not USE_DESCRIPTOR_STREAMS:
            m_i, l_i, acc_cb = _ql_segments(
                qr, DPK, DNK, DPV, DNV, CK0, CV0,
                DCBASE, DNBASE, DCOUNTS,
                s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                D, B2, 4, 2, 3, 0, NR, MAX_SEGMENTS, NS, BM, BT,
            )
        if HAS1:
            if TRUE3:
                m_i, l_i, acc_cb = _ql3(
                    qr, PK1, NK1, PV1, NV1, CK1, CV1, PB1, NB1,
                    tl.load(OFF1 + r), tl.load(SL1 + r),
                    s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                    D, B3, NS, BM, BT,
                )
            else:
                m_i, l_i, acc_cb = _ql(qr, PK1, NK1, PV1, NV1, CK1, CV1, PB1, NB1,
                                       tl.load(OFF1 + r), tl.load(SL1 + r),
                                       s_id, r, scaling, qmask, m_i, l_i, acc_cb, D, B4, 2, 4, 15, NS, BM, BT,
                                   WIDE)
        if HAS_SEGMENTS and not USE_DESCRIPTOR_STREAMS:
            m_i, l_i, acc_cb = _ql3_segments(
                qr, DPK, DNK, DPV, DNV, CK1, CV1,
                DCBASE, DNBASE, DCOUNTS,
                s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                D, B3, 1, NR, MAX_SEGMENTS, NS, BM, BT,
            )
        if HAS2:
            m_i, l_i, acc_cb = _ql(qr, PK2, NK2, PV2, NV2, CK2, CV2, PB2, NB2,
                                   tl.load(OFF2 + r), tl.load(SL2 + r),
                                   s_id, r, scaling, qmask, m_i, l_i, acc_cb, D, B4, 2, 4, 15, NS, BM, BT,
                                   WIDE)
        if HAS_SEGMENTS and not USE_DESCRIPTOR_STREAMS:
            m_i, l_i, acc_cb = _ql_segments(
                qr, DPK, DNK, DPV, DNV, CK2, CV2,
                DCBASE, DNBASE, DCOUNTS,
                s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                D, B4, 2, 4, 15, 2, NR, MAX_SEGMENTS, NS, BM, BT,
            )
        if HAS3:
            m_i, l_i, acc_cb = _ql(qr, PK3, NK3, PV3, NV3, CK3, CV3, PB3, NB3,
                                   tl.load(OFF3 + r), tl.load(SL3 + r),
                                   s_id, r, scaling, qmask, m_i, l_i, acc_cb, D, B8, 1, 8, 255, NS, BM, BT,
                                   WIDE)
        if HAS_SEGMENTS and not USE_DESCRIPTOR_STREAMS:
            m_i, l_i, acc_cb = _ql_segments(
                qr, DPK, DNK, DPV, DNV, CK3, CV3,
                DCBASE, DNBASE, DCOUNTS,
                s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                D, B8, 1, 8, 255, 3, NR, MAX_SEGMENTS, NS, BM, BT,
            )
        if HAS_SEGMENTS and USE_DESCRIPTOR_STREAMS:
            m_i, l_i, acc_cb = _ql_descriptor_stream(
                qr, DPK, DNK, DPV, DNV, DCBASE, DNBASE,
                DQDESC, DDCOUNTS, DCODEBOOK,
                s_id, r, scaling, qmask, m_i, l_i, acc_cb,
                D, NR, DESCRIPTOR_CAPACITY, DESCRIPTOR_STRIDE,
                NS, BM, BT,
            )
        # exact prefix + tail (raw → acc_raw)
        if HAS_EXACT:
            m_i, l_i, acc_cb, acc_raw = _exact(q, EK, EV, tl.load(OFFE + r), tl.load(SLE + r), s_id, r, scaling,
                                               qmask, m_i, l_i, acc_cb, acc_raw, D, NS, BM, BT)
        if HAS_SEGMENTS:
            if USE_DESCRIPTOR_STREAMS:
                m_i, l_i, acc_cb, acc_raw = _exact_descriptor_stream(
                    q, DEK, DEV, DEBASE, DEDESC, DDCOUNTS,
                    s_id, r, scaling, qmask, m_i, l_i, acc_cb, acc_raw,
                    D, NR, DESCRIPTOR_CAPACITY, DESCRIPTOR_STRIDE,
                    NS, BM, BT,
                )
            else:
                m_i, l_i, acc_cb, acc_raw = _exact_segments(
                    q, DEK, DEV, DEBASE, DCOUNTS,
                    s_id, r, scaling, qmask, m_i, l_i, acc_cb, acc_raw,
                    D, NR, MAX_SEGMENTS, NS, BM, BT,
                )
        if HAS_TAIL:
            m_i, l_i, acc_cb, acc_raw = _exact(q, TK, TV, tl.load(OFFT + r), tl.load(SLT + r), s_id, r, scaling,
                                               qmask, m_i, l_i, acc_cb, acc_raw, D, NS, BM, BT)
        # Only ``G`` of the ``BM`` tile rows are real GQA queries (BM is padded
        # up to the 16-row tl.dot minimum).  Striding the split-K scratch by SM
        # instead of BM keeps the fp32 partials 4x smaller for GQA, which is what
        # made wider split-K a wash: the extra parallelism was paying for extra
        # scratch traffic.  Out-of-range lanes are masked and never dereferenced.
        po = (r * NS + s_id) * SM + offs_m
        tl.store(M_OUT + po, m_i, mask=qmask)
        tl.store(L_OUT + po, l_i, mask=qmask)
        base = po[:, None] * D + d[None, :]
        tl.store(CB_OUT + base, acc_cb, mask=qmask[:, None])
        tl.store(RAW_OUT + base, acc_raw, mask=qmask[:, None])


    @triton.jit
    def _combine_kernel(
        M, L, CB, RAW, CB_O, RAW_O, L_O,
        NS: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, G: tl.constexpr,
        SM: tl.constexpr,
    ):
        """Cross-split online-softmax combine, folded into ONE launch (grid NR).

        Replaces the ~10 host reduction ops (amax/exp/sub/sum×3) that small-batch
        decode pays per layer when split-K is active. Emits the combined value
        accumulators (still in codebook space) + softmax denominator; the caller
        does the ``@ pi_v`` rotation as a batched tensor-core matmul, which a
        per-row in-kernel dot cannot match."""
        r = tl.program_id(0)
        offs_m = tl.arange(0, BM)
        d = tl.arange(0, D)
        qmask = offs_m < G
        m = tl.full([BM], -float("inf"), tl.float32)
        for s in range(NS):
            ms = tl.load(M + (r * NS + s) * SM + offs_m, mask=qmask, other=-float("inf"))
            m = tl.maximum(m, ms)
        msafe = tl.where(m == -float("inf"), 0.0, m)
        l = tl.zeros([BM], tl.float32)
        acc_cb = tl.zeros([BM, D], tl.float32)
        acc_raw = tl.zeros([BM, D], tl.float32)
        for s in range(NS):
            ms = tl.load(M + (r * NS + s) * SM + offs_m, mask=qmask, other=-float("inf"))
            w = tl.where(ms == -float("inf"), 0.0, tl.exp(ms - msafe))
            ls = tl.load(L + (r * NS + s) * SM + offs_m, mask=qmask, other=0.0)
            l += ls * w
            base = ((r * NS + s) * SM + offs_m)[:, None] * D + d[None, :]
            acc_cb += tl.load(CB + base, mask=qmask[:, None], other=0.0) * w[:, None]
            acc_raw += tl.load(RAW + base, mask=qmask[:, None], other=0.0) * w[:, None]
        obase = (r * G + offs_m)[:, None] * D + d[None, :]
        tl.store(CB_O + obase, acc_cb, mask=qmask[:, None])
        tl.store(RAW_O + obase, acc_raw, mask=qmask[:, None])
        tl.store(L_O + r * G + offs_m, l, mask=qmask)


    @triton.jit
    def _project_value_epilogue_kernel(
        CB,
        ROTATION,
        RAW,
        DENOMINATOR,
        OUTPUT,
        M: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        cb_stride_r: tl.constexpr,
        cb_stride_g: tl.constexpr,
        cb_stride_d: tl.constexpr,
        raw_stride_r: tl.constexpr,
        raw_stride_g: tl.constexpr,
        raw_stride_d: tl.constexpr,
        denominator_stride_r: tl.constexpr,
        denominator_stride_g: tl.constexpr,
        rotation_stride_k: tl.constexpr,
        rotation_stride_n: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        """Flattened pi_v projection + raw add + normalization + BF16 store.

        Flattening all live GQA rows gives tensor cores enough M occupancy and
        replaces the previous cuBLAS matmul plus add/clamp/divide/cast launch
        chain. Explicit source strides also cover the padded NS==1 views.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k = tl.arange(0, D)
        row = m // G
        group = m - row * G
        live_m = m < M
        cb = tl.load(
            CB
            + row[:, None] * cb_stride_r
            + group[:, None] * cb_stride_g
            + k[None, :] * cb_stride_d,
            mask=live_m[:, None],
            other=0.0,
        ).to(tl.float32)
        rotation = tl.load(
            ROTATION
            + k[:, None] * rotation_stride_k
            + n[None, :] * rotation_stride_n,
            mask=n[None, :] < D,
            other=0.0,
        ).to(tl.float32)
        if IEEE:
            projected = tl.dot(cb, rotation, input_precision="ieee")
        else:
            projected = tl.dot(cb, rotation, input_precision="tf32")
        live = live_m[:, None] & (n[None, :] < D)
        raw = tl.load(
            RAW
            + row[:, None] * raw_stride_r
            + group[:, None] * raw_stride_g
            + n[None, :] * raw_stride_d,
            mask=live,
            other=0.0,
        ).to(tl.float32)
        denominator = tl.load(
            DENOMINATOR
            + row * denominator_stride_r
            + group * denominator_stride_g,
            mask=live_m,
            other=1.0,
        ).to(tl.float32)
        output = (projected + raw) / tl.maximum(
            denominator, 1.0e-20,
        )[:, None]
        tl.store(
            OUTPUT + m[:, None] * D + n[None, :],
            output,
            mask=live,
        )


    # ── V1 single-level kernel (kept for the single-bank unit test) ──
    @triton.jit
    def _fused_decode_kernel(
        QR, Q, PK, NK, PV, NV, CK, CV, BASE, OFFQ, SLQ, EK, EV, OFFE, SLE,
        ACC_CB, ACC_RAW, LOUT, scaling,
        G: tl.constexpr, D: tl.constexpr, BYTES: tl.constexpr,
        VPB: tl.constexpr, EFF_BITS: tl.constexpr, MASK: tl.constexpr,
        TRUE3: tl.constexpr,
        BM: tl.constexpr, BT: tl.constexpr,
    ):
        r = tl.program_id(0)
        offs_m = tl.arange(0, BM)
        d = tl.arange(0, D)
        qmask = offs_m < G
        qr = tl.load(QR + (r * G + offs_m)[:, None] * D + d[None, :],
                     mask=qmask[:, None], other=0.0).to(tl.bfloat16)
        q = tl.load(Q + (r * G + offs_m)[:, None] * D + d[None, :],
                    mask=qmask[:, None], other=0.0).to(tl.bfloat16)
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc_cb = tl.zeros([BM, D], tl.float32)
        acc_raw = tl.zeros([BM, D], tl.float32)
        if TRUE3:
            m_i, l_i, acc_cb = _ql3(
                qr, PK, NK, PV, NV, CK, CV, BASE, BASE,
                tl.load(OFFQ + r), tl.load(SLQ + r), 0,
                r, scaling, qmask, m_i, l_i, acc_cb,
                D, BYTES, 1, BM, BT,
            )
        else:
            m_i, l_i, acc_cb = _ql(qr, PK, NK, PV, NV, CK, CV, BASE, BASE,
                                   tl.load(OFFQ + r), tl.load(SLQ + r), 0,
                                   r, scaling, qmask, m_i, l_i, acc_cb, D, BYTES, VPB, EFF_BITS,
                                   MASK, 1, BM, BT, False)
        m_i, l_i, acc_cb, acc_raw = _exact(q, EK, EV, tl.load(OFFE + r), tl.load(SLE + r), 0, r, scaling,
                                           qmask, m_i, l_i, acc_cb, acc_raw, D, 1, BM, BT)
        base = (r * BM + offs_m)[:, None] * D + d[None, :]
        tl.store(ACC_CB + base, acc_cb, mask=qmask[:, None])
        tl.store(ACC_RAW + base, acc_raw, mask=qmask[:, None])
        tl.store(LOUT + r * BM + offs_m, l_i, mask=qmask)


def _empty_exact(NR, D, dev):
    return (torch.zeros(NR, 1, D, device=dev, dtype=torch.bfloat16),
            torch.zeros(NR, 1, D, device=dev, dtype=torch.bfloat16),
            torch.zeros(NR, device=dev, dtype=torch.int32))


def _project_value_epilogue(
    cb: torch.Tensor,
    raw: torch.Tensor,
    denominator: torch.Tensor,
    rotation: torch.Tensor,
    *,
    input_precision: str = "tf32",
) -> torch.Tensor:
    """Fused value-domain projection and softmax normalization.

    ``cb``/``raw`` may be non-contiguous views of the ``NS == 1`` split
    workspace.  Passing their real strides is essential: materializing these
    views would add two full FP32 copies to every layer and decode token.
    """
    if not _HAS_TRITON or not cb.is_cuda:
        raise RuntimeError("fused value epilogue requires CUDA and Triton")
    if cb.ndim != 3 or raw.shape != cb.shape:
        raise ValueError("expected matching cb/raw tensors with shape [NR,G,D]")
    nr, groups, head_dim = cb.shape
    if denominator.shape != (nr, groups):
        raise ValueError(
            f"expected denominator {(nr, groups)}, got {tuple(denominator.shape)}"
        )
    if rotation.shape != (head_dim, head_dim):
        raise ValueError(
            f"expected rotation {(head_dim, head_dim)}, got {tuple(rotation.shape)}"
        )
    if head_dim not in (64, 128):
        raise ValueError(f"native decode supports head_dim 64 or 128, got {head_dim}")
    if not (cb.device == raw.device == denominator.device == rotation.device):
        raise ValueError("epilogue inputs must share one CUDA device")
    input_precision = input_precision.strip().lower()
    if input_precision not in {"tf32", "ieee"}:
        raise ValueError(
            "value epilogue input_precision must be 'tf32' or 'ieee', "
            f"got {input_precision!r}"
        )

    output = torch.empty(
        (nr, groups, head_dim),
        device=cb.device,
        dtype=torch.bfloat16,
    )
    live_rows = nr * groups
    block_m, block_n = 16, 64
    _project_value_epilogue_kernel[
        (triton.cdiv(live_rows, block_m), triton.cdiv(head_dim, block_n))
    ](
        cb,
        rotation,
        raw,
        denominator,
        output,
        M=live_rows,
        G=groups,
        D=head_dim,
        cb_stride_r=cb.stride(0),
        cb_stride_g=cb.stride(1),
        cb_stride_d=cb.stride(2),
        raw_stride_r=raw.stride(0),
        raw_stride_g=raw.stride(1),
        raw_stride_d=raw.stride(2),
        denominator_stride_r=denominator.stride(0),
        denominator_stride_g=denominator.stride(1),
        rotation_stride_k=rotation.stride(0),
        rotation_stride_n=rotation.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IEEE=input_precision == "ieee",
        num_warps=4,
        num_stages=1,
    )
    return output


_ZERO_BASE_CACHE: dict = {}
_ZERO_BASE_LOCK = threading.Lock()


def _zero_base(dev):
    with _ZERO_BASE_LOCK:
        cached = _ZERO_BASE_CACHE.get(dev)
        if cached is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "fused decode descriptor cache must be warmed before CUDA capture"
                )
            value = torch.zeros(1, device=dev, dtype=torch.int32)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(dev))
            # This is a one-time descriptor-cache miss outside capture. Retire
            # the publication event before exposing the entry so later CUDA
            # graph captures never inherit a dependency on uncaptured work.
            ready.synchronize()
            cached = value
            _ZERO_BASE_CACHE[dev] = cached
    return cached


def fused_decode_one_level(
    Q, quant_bank, exact_k, exact_v, exact_seqlen, pi_k, pi_v, scaling,
    BT: int = 64, num_warps: int = 4, num_stages: int = 2,
):
    """One quant bank + one exact bank, single kernel (V1, no split-K)."""
    NR, G, D = Q.shape
    dev = Q.device
    BM = max(16, triton.next_power_of_2(G))
    qr = (Q.float() @ pi_k.float().T).contiguous()
    Qc = Q.float().contiguous()
    bits = quant_bank["bits"]
    physical_bits = int(quant_bank.get("physical_bits", 4 if bits == 3 else bits))
    true3 = bits == 3 and physical_bits == 3
    eff, vpb, mask, BYTES = _pp(bits, D, physical_bits)
    pk = quant_bank["packed_k"]; TQ = pk.shape[1]
    slq = quant_bank.get("seqlen")
    slq = (slq if slq is not None else torch.full((NR,), TQ, device=dev)).to(dev, torch.int32).contiguous()
    if exact_k is None:
        exact_k, exact_v, exact_seqlen = _empty_exact(NR, D, dev)
    TE = exact_k.shape[1]
    sle = (exact_seqlen if exact_seqlen is not None
           else torch.full((NR,), TE, device=dev)).to(dev, torch.int32).contiguous()
    acc_cb = torch.empty(NR, BM, D, device=dev, dtype=torch.float32)
    acc_raw = torch.empty(NR, BM, D, device=dev, dtype=torch.float32)
    lout = torch.empty(NR, BM, device=dev, dtype=torch.float32)
    offq = (torch.arange(NR, device=dev, dtype=torch.int32) * TQ).contiguous()
    offe = (torch.arange(NR, device=dev, dtype=torch.int32) * TE).contiguous()
    base = _zero_base(dev)
    _fused_decode_kernel[(NR,)](
        qr, Qc, pk.contiguous(), quant_bank["norms_k"].float().contiguous(),
        quant_bank["packed_v"].contiguous(), quant_bank["norms_v"].float().contiguous(),
        quant_bank["cent_k"].float().contiguous(), quant_bank["cent_v"].float().contiguous(),
        base, offq, slq,
        exact_k.float().contiguous(), exact_v.float().contiguous(), offe, sle,
        acc_cb, acc_raw, lout, scaling,
        G=G, D=D, BYTES=BYTES, VPB=vpb, EFF_BITS=eff, MASK=mask, TRUE3=true3,
        BM=BM, BT=BT, num_warps=num_warps, num_stages=num_stages)
    acc_cb = acc_cb[:, :G]; acc_raw = acc_raw[:, :G]; lout = lout[:, :G]
    return (acc_cb @ pi_v.float() + acc_raw) / lout.clamp(min=1e-20).unsqueeze(-1)


def _slot(banks, bits, NR, D, dev):
    """Package ``(pk,nk,pv,nv,ck,cv,code_base,norm_base,off,seqlen)``.

    ``off`` is the per-row start offset into a flat token array. Padded banks
    ([NR,T,·], contiguous) reuse the same kernel via ``off = r*T`` (raw pointer
    arithmetic is flat regardless of tensor shape); ragged banks (flat [N,·]) pass
    their real ``cumsum`` offsets and carry zero padding."""
    for b in banks:
        if b["bits"] == bits:
            physical_bits = int(b.get("physical_bits", 4 if bits == 3 else bits))
            _, _, _, BYTES = _pp(bits, D, physical_bits)
            sl = b["seqlen"].to(dev, torch.int32).contiguous()
            if "offset" in b:                                   # ragged: flat [N, BYTES]
                off = b["offset"].to(dev, torch.int32).contiguous()
            else:                                               # padded [NR, T, BYTES]
                T = b["packed_k"].shape[1]
                off = (torch.arange(NR, device=dev, dtype=torch.int32) * T).contiguous()
            # ``dict.get(key, expression)`` eagerly evaluates the default.
            # Avoid creating/publishing a dummy scalar for shared-v2 banks that
            # already carry real device bases.
            code_base = b["code_base"] if "code_base" in b else _zero_base(dev)
            norm_base = b["norm_base"] if "norm_base" in b else _zero_base(dev)
            code_base = code_base.to(dev, torch.int32).contiguous()
            norm_base = norm_base.to(dev, torch.int32).contiguous()
            return (b["packed_k"].contiguous(), b["norms_k"].float().contiguous(),
                    b["packed_v"].contiguous(), b["norms_v"].float().contiguous(),
                    b["cent_k"].float().contiguous(), b["cent_v"].float().contiguous(),
                    code_base, norm_base, off, sl)
    _, _, _, BYTES = _pp(bits, D)
    return _empty_slot(bits, BYTES, NR, dev)


def _physical_bits(banks, bits):
    for bank in banks:
        if bank["bits"] == bits:
            return int(bank.get("physical_bits", 4 if bits == 3 else bits))
    return 4 if bits == 3 else bits


_EMPTY_SLOT_CACHE: dict = {}
_EMPTY_SLOT_LOCK = threading.Lock()


def _empty_slot(bits, BYTES, NR, dev):
    """Dummy slot for an ABSENT bit level — cached and shared (all read-only,
    sl=0 ⇒ never read). Previously `_slot` allocated 4 zeros + 2 clones on every
    fused_decode call; at high eviction some banks are always absent, so those
    allocations + zero-fills were captured into the CUDA graph and replayed each
    step. Caching them (stable addresses, graph-safe) removes that per-step cost."""
    key = (bits, BYTES, NR, dev)
    with _EMPTY_SLOT_LOCK:
        entry = _EMPTY_SLOT_CACHE.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "fused decode empty-bank cache must be warmed before CUDA capture"
                )
            z = torch.zeros(1, BYTES, device=dev, dtype=torch.uint8)
            zn = torch.zeros(1, device=dev, dtype=torch.float32)
            zc = torch.zeros(2 ** bits, device=dev, dtype=torch.float32)
            zo = torch.zeros(NR, device=dev, dtype=torch.int32)
            zb = _zero_base(dev)
            payload = (z, zn, z, zn, zc, zc, zb, zb, zo, zo)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(dev))
            ready.synchronize()
            entry = payload
            _EMPTY_SLOT_CACHE[key] = entry
    return entry


def _slot_tmax(banks, bits):
    """Max per-row token count for the NS heuristic — a Python int (no device
    sync, so it is graph-capture safe). Ragged banks precompute ``T_max``."""
    for b in banks:
        if b["bits"] == bits:
            if "T_max" in b:
                return int(b["T_max"])
            return int(b["packed_k"].shape[1])
    return 0


def fused_decode(Q, quant_banks, exact_k, exact_v, exact_seqlen, pi_k, pi_v, scaling,
                 tail_k=None, tail_v=None, tail_seqlen=None,
                 tail_offset=None,
                 decode_arena=None,
                 BT: int = 32, num_warps: int = 4, num_stages: int = 1):
    """Full mixed-precision decode over all packed, exact, and tail sources.

    ``Q`` is ``[NR,G,D]`` and the returned attention output is contiguous BF16
    ``[NR,G,D]``.  The production attention caller consumes BF16 directly.
    """
    import os as _os
    BT = int(_os.environ.get("R2_BT", BT))
    num_warps = int(_os.environ.get("R2_WARPS", num_warps))
    num_stages = int(_os.environ.get("R2_STAGES", num_stages))
    NR, G, D = Q.shape
    dev = Q.device
    BM = max(16, triton.next_power_of_2(G))
    bf = torch.bfloat16
    Qb = Q.to(bf)
    qr = torch.matmul(Qb, pi_k.to(bf).transpose(-1, -2)).contiguous()
    Qc = Qb.contiguous()
    s0 = _slot(quant_banks, 2, NR, D, dev)
    s1 = _slot(quant_banks, 3, NR, D, dev)
    s2 = _slot(quant_banks, 4, NR, D, dev)
    s3 = _slot(quant_banks, 8, NR, D, dev)
    has_exact = exact_k is not None
    if not has_exact:
        exact_k, exact_v, exact_seqlen = _empty_exact(NR, D, dev)
    # Legacy exact storage is padded [NR,TE,D]. Native packing supplies compact
    # [N,D] CSR plus device offsets/seqlens in a metadata dict. The Triton source
    # already consumes an arbitrary offset, so no padding/copy is required.
    if isinstance(exact_seqlen, dict):
        exact_meta = exact_seqlen
        off_e = exact_meta["offset"].to(dev, torch.int32).contiguous()
        sle = exact_meta["seqlen"].to(dev, torch.int32).contiguous()
        TE = int(exact_meta.get("T_max", 0)) if has_exact else 0
    else:
        TE = exact_k.shape[1] if has_exact else 0
        sle = (exact_seqlen if exact_seqlen is not None
               else torch.full((NR,), TE, device=dev)).to(dev, torch.int32).contiguous()
        # Dummy storage is [NR,1,D], but an absent source keeps logical bound
        # zero and therefore never enters the split-K source loop.
        exact_stride = exact_k.shape[1]
        off_e = (torch.arange(NR, device=dev, dtype=torch.int32) * exact_stride).contiguous()
    has_tail = tail_k is not None
    if not has_tail:
        tail_k, tail_v, tail_seqlen = _empty_exact(NR, D, dev)
    TT = tail_k.shape[1] if has_tail else 0
    slt = (tail_seqlen if tail_seqlen is not None
           else torch.full((NR,), TT, device=dev)).to(dev, torch.int32).contiguous()
    # split-K: fill the GPU (~256 programs) without over-splitting short T.
    #
    # The cap used to be 16, which silently defeated that goal at NR=8 (batch 1,
    # 8 KV heads): it produced 128 programs, one per SM, four warps each, and
    # every dependent LUT-gather latency was exposed.  Measured per-kernel at
    # B=1/C=8192: the split kernel goes 1262.7 -> 872.5 us/step when NS rises
    # 16 -> 32 (-31%).  The earlier conclusion that finer splits "don't help"
    # came from reading end-to-end TPOT only, where the combine kernel's own
    # linear growth in NS (135 -> 247 us) hid the win.
    #
    # A cap of 32 makes the heuristic actually hit ~256 programs for every batch
    # from 1 to 8 (NR = 8, 16, 32, 64 -> NS = 32, 16, 8, 4).
    #
    # Occupancy alone is still not enough, because it pins the program count at
    # ~256 whatever Tmax is: a large batch drives NS to 1, and then one program
    # walks an entire row of Tmax/BT tiles serially with nothing to overlap the
    # gather latency against.  Forcing NS on H20 (uniform 2-bit, C=8192, TPOT ms)
    # showed how much that costs:
    #     B=16 (auto NS=2): 40.40 -> 39.27 (NS=4) -> 36.54 (8) -> 36.23 (16)
    #     B=32 (auto NS=1): 71.68 -> 63.89 (NS=4) -> 63.19 (8) -> 62.82 (16)
    # -10.3% and -12.4%.  Mixed precision gains more, because a tile there runs
    # several dependent LUT gathers rather than one:
    #     B=32 target=2 (auto NS=1): 77.32 -> 60.58 (4) -> 58.67 (8) -> 58.99 (16)
    # -24.1%, and the turn upward at 16 is the combine kernel -- it is grid (NR,)
    # and loops NS serially, so splitting past the point where latency is hidden
    # just moves the cost.  Bounding a program to <= 32 token tiles puts NS at 9
    # for C=8192, the measured optimum for mixed and within 0.5% for uniform, at
    # half the scratch and combine work of a tile bound of 16.  Take whichever of
    # the two requirements is wider, capped at 32.
    has_segments = decode_arena is not None
    use_descriptor_streams = (
        has_segments
        and _os.environ.get("R2_DECODE_DESCRIPTOR_STREAMS", "1") != "0"
    )
    segmented_capacity = (
        int(decode_arena.capacity_per_row) if has_segments else 0
    )
    Tmax = max(_slot_tmax(quant_banks, 2), _slot_tmax(quant_banks, 3),
               _slot_tmax(quant_banks, 4), _slot_tmax(quant_banks, 8), TE, TT,
               segmented_capacity)
    present = tuple(_slot_tmax(quant_banks, bits) > 0 for bits in (2, 3, 4, 8))
    ns_occupancy = -(-256 // NR)
    ns_latency = -(-Tmax // (32 * BT))
    NS = max(1, min(32, max(ns_occupancy, ns_latency)))
    NS = min(NS, max(1, Tmax // 128))
    _forced_ns = _os.environ.get("R2_NS")
    if _forced_ns and int(_forced_ns) > 0:
        NS = int(_forced_ns)
    _, _, _, B2 = _pp(2, D)
    _, _, _, B3 = _pp(3, D, 3)
    _, _, _, B4 = _pp(4, D)
    _, _, _, B8 = _pp(8, D)
    true3 = _physical_bits(quant_banks, 3) == 3
    if has_segments:
        segmented = (
            decode_arena.code_arena_k,
            decode_arena.norm_arena_k,
            decode_arena.code_arena_v,
            decode_arena.norm_arena_v,
            decode_arena.code_row_base,
            decode_arena.norm_row_base,
            decode_arena.counts,
            decode_arena.exact_k,
            decode_arena.exact_v,
            decode_arena.exact_row_base,
            decode_arena.descriptors,
            decode_arena.descriptors,
            decode_arena.descriptor_counts,
            decode_arena.unified_codebook,
        )
        max_segments = int(decode_arena.max_segments)
        descriptor_capacity = int(decode_arena.capacity_per_row)
        descriptor_stride = int(decode_arena.buffer_size)
    else:
        # Compile-time HAS_SEGMENTS=False removes every access. Reuse existing
        # non-empty descriptors so this path introduces no per-step allocation.
        segmented = (
            s0[0], s0[1], s0[2], s0[3], s0[6], s0[7], s0[9],
            exact_k, exact_v, off_e,
            s0[9], off_e, s0[9], s0[4],
        )
        max_segments = 1
        descriptor_capacity = 1
        descriptor_stride = 1
    # Split-K partials only ever hold ``G`` real rows; ``BM`` is padding for the
    # 16-row tl.dot minimum.  ``R2_SPLIT_SCRATCH=pad`` restores the archived
    # BM-strided layout for A/B.
    SM = BM if _os.environ.get("R2_SPLIT_SCRATCH", "compact") == "pad" else G
    M = torch.empty(NR, NS, SM, device=dev, dtype=torch.float32)
    L = torch.empty(NR, NS, SM, device=dev, dtype=torch.float32)
    CB = torch.empty(NR, NS, SM, D, device=dev, dtype=torch.float32)
    RAW = torch.empty(NR, NS, SM, D, device=dev, dtype=torch.float32)
    off_t = (
        tail_offset
        if tail_offset is not None
        else torch.arange(NR, device=dev, dtype=torch.int32) * TT
    ).to(dev, torch.int32).contiguous()
    _fused_split_kernel[(NR, NS)](
        qr, Qc, *s0, *s1, *s2, *s3,
        exact_k.to(bf).contiguous(), exact_v.to(bf).contiguous(), off_e, sle,
        tail_k.to(bf).contiguous(), tail_v.to(bf).contiguous(), off_t, slt,
        *segmented,
        M, L, CB, RAW, scaling,
        G=G, D=D, B2=B2, B3=B3, B4=B4, B8=B8, TRUE3=true3,
        NS=NS, BM=BM, BT=BT, SM=SM, WIDE=_WIDE_LOAD,
        HAS0=present[0], HAS1=present[1], HAS2=present[2], HAS3=present[3],
        HAS_EXACT=has_exact and TE > 0, HAS_TAIL=has_tail and TT > 0,
        HAS_SEGMENTS=has_segments, USE_DESCRIPTOR_STREAMS=use_descriptor_streams,
        NR=NR, MAX_SEGMENTS=max_segments,
        DESCRIPTOR_CAPACITY=descriptor_capacity,
        DESCRIPTOR_STRIDE=descriptor_stride,
        num_warps=num_warps, num_stages=num_stages)
    # combine split-K partials (cross-split online softmax, dual accumulators).
    # NS==1 (large batch) skips the reduction; NS>1 folds it into one kernel so
    # small-batch decode doesn't pay ~10 host reduction launches per layer, then
    # one flattened tensor-core epilogue then projects every live GQA row,
    # adds the raw-value accumulator, normalizes, and writes BF16 in one launch.
    if NS == 1:
        cb, raw, l = CB[:, 0, :G], RAW[:, 0, :G], L[:, 0, :G]
    else:
        cb = torch.empty(NR, G, D, device=dev, dtype=torch.float32)
        raw = torch.empty(NR, G, D, device=dev, dtype=torch.float32)
        l = torch.empty(NR, G, device=dev, dtype=torch.float32)
        _combine_kernel[(NR,)](M, L, CB, RAW, cb, raw, l,
                               NS=NS, D=D, BM=BM, G=G, SM=SM, num_warps=4)
    if _os.environ.get("R2_LEGACY_EPILOGUE", "0") == "1":
        # Diagnostic rollback/A-B path. Production defaults to the fused
        # epilogue; the cast mirrors the dtype seen by the attention caller.
        return (
            (cb @ pi_v.float() + raw) / l.clamp(min=1e-20).unsqueeze(-1)
        ).to(torch.bfloat16)
    return _project_value_epilogue(
        cb,
        raw,
        l,
        pi_v,
        input_precision=_os.environ.get("R2_EPILOGUE_PRECISION", "tf32"),
    )
