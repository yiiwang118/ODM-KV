"""Stage C: fused 2-bit flash-decode-LUT attention kernel.

De-risks the whole system's *speed* claim: decode attention is memory-bound,
so reading 2-bit packed KV (8x less HBM than fp16) should be FASTER than an
fp16 SDPA decode step — if the LUT decode + online softmax overhead stays
small. This kernel measures exactly that.

Layout (single kv-head, its GQA query group of G heads, T cached tokens,
head_dim D, D4 = D // 4 packed bytes):

  * TurboQuant-MSE 2-bit key/value: ``packed`` [T, D4] uint8, 4 codebook
    indices per byte (idx0 | idx1<<2 | idx2<<4 | idx3<<6); byte p holds dims
    4p..4p+3. ``norms`` [T] fp32. ``centroids`` [4] fp32 (shared by all banks
    of this bit-width). Rotations ``pi_k`` / ``pi_v`` [D, D] shared by all banks.

Identities:
    q . k_hat = ||k|| * ( centroids[idx] . (pi_k @ q) )
    sum_i w_i v_hat_i = ( sum_i w_i ||v_i|| centroids[idx_i] ) @ pi_v

So we rotate Q once (q' = pi_k @ q), permute q' into (sub, byte) order to
match the packing, run flash-decode with a nested-``where`` centroid decode
(no gather — only 4 levels), accumulate the value output in codebook space,
and rotate it out once with pi_v.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = torch.cuda.is_available()
except Exception:  # pragma: no cover
    _HAS_TRITON = False


# ─────────────────────────── PyTorch reference ───────────────────────────

def decode_lut_2bit_torch(
    Q: torch.Tensor,            # [G, D]  raw query rows for one kv-head group
    packed_k: torch.Tensor,     # [T, D4] uint8
    norms_k: torch.Tensor,      # [T]
    cent_k: torch.Tensor,       # [4]
    packed_v: torch.Tensor,     # [T, D4] uint8
    norms_v: torch.Tensor,      # [T]
    cent_v: torch.Tensor,       # [4]
    pi_k: torch.Tensor,         # [D, D]
    pi_v: torch.Tensor,         # [D, D]
    scaling: float,
) -> torch.Tensor:
    """Vectorised reference (correct, materialises khat/vhat in bf16-space)."""
    G, D = Q.shape
    qrot = Q.float() @ pi_k.float().T                 # [G, D]  q' = pi_k @ q per row
    # decode key codebook -> khat [T, D]
    idx_k = torch.stack([(packed_k >> (2 * s)) & 3 for s in range(4)], dim=-1)  # [T,D4,4]
    idx_k = idx_k.reshape(packed_k.shape[0], D)        # [T, D] row-major -> dim = 4*byte+sub
    khat = cent_k[idx_k.long()] * norms_k.float().unsqueeze(-1)  # [T, D]
    scores = (qrot @ khat.T) * scaling                # [G, T]
    w = torch.softmax(scores, dim=-1)                 # [G, T]
    idx_v = torch.stack([(packed_v >> (2 * s)) & 3 for s in range(4)], dim=-1)
    idx_v = idx_v.reshape(packed_v.shape[0], D)
    vhat_cb = cent_v[idx_v.long()] * norms_v.float().unsqueeze(-1)  # [T, D] (codebook space)
    out_cb = w @ vhat_cb                              # [G, D]
    return out_cb @ pi_v.float()                      # [G, D]  rotate value out


# ─────────────────────────── Triton kernel ───────────────────────────

if _HAS_TRITON:

    @triton.jit
    def _flash_decode_lut_kernel(
        QR,                 # [NR, G, D] rotated query q' = pi_k @ q (natural dim order)
        PK, NK,             # packed_k [NR, T, BYTES] uint8, norms_k [NR, T]
        PV, NV,             # packed_v [NR, T, BYTES] uint8, norms_v [NR, T]
        CK, CV,             # centroid tables [2**bits] fp32 (key / value)
        SEQLEN,             # [NR] int32 real token count per head (<= T padded)
        M_OUT, L_OUT, ACC_OUT,   # [NR,NS,BM], [NR,NS,BM], [NR,NS,BM,D] partials
        T, TS, scaling,
        G: tl.constexpr, D: tl.constexpr, BYTES: tl.constexpr, NS: tl.constexpr,
        VPB: tl.constexpr, EFF_BITS: tl.constexpr, MASK: tl.constexpr,
        BM: tl.constexpr, BT: tl.constexpr,
    ):
        r = tl.program_id(0)                          # kv-head problem index
        s_id = tl.program_id(1)                       # split-K chunk index
        offs_m = tl.arange(0, BM)                     # query rows (padded to BM >= G)
        d = tl.arange(0, D)                           # [D]
        byte = d // VPB                               # [D] which packed byte holds dim d
        shift = (d % VPB) * EFF_BITS                  # [D] bit shift within that byte
        qmask = offs_m < G
        pbase = r * T * BYTES
        nbase = r * T
        seqlen = tl.load(SEQLEN + r)                   # real token count for this head
        # rotated query tile [BM, D] (invalid rows zeroed; masked to -inf in scores)
        qr = tl.load(QR + (r * G + offs_m)[:, None] * D + d[None, :],
                     mask=qmask[:, None], other=0.0).to(tl.bfloat16)

        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)

        t_lo = s_id * TS
        for t0 in range(t_lo, t_lo + TS, BT):
            t = t0 + tl.arange(0, BT)                 # [BT]
            valid = t < seqlen                        # per-head length mask (padding excluded)
            # load key TRANSPOSED [D, BT] (byte as row) so no tl.trans is needed
            pkT = tl.load(PK + pbase + t[None, :] * BYTES + byte[:, None],
                          mask=valid[None, :], other=0)                # [D, BT]
            ikT = (pkT >> shift[:, None]) & MASK                       # [D, BT]
            nk = tl.load(NK + nbase + t, mask=valid, other=0.0).to(tl.float32)
            khatT = (tl.load(CK + ikT) * nk[None, :]).to(tl.bfloat16)  # [D, BT] decoded key
            # scores via tensor cores: [BM,D] @ [D,BT] -> [BM,BT]
            s = tl.dot(qr, khatT) * scaling
            s = tl.where(valid[None, :] & qmask[:, None], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(s == -float("inf"), 0.0, tl.exp(s - m_new[:, None]))   # [BM, BT]
            l_i = l_i * alpha + tl.sum(p, axis=1)

            pv = tl.load(PV + pbase + t[:, None] * BYTES + byte[None, :],
                         mask=valid[:, None], other=0)
            iv = (pv >> shift[None, :]) & MASK
            nv = tl.load(NV + nbase + t, mask=valid, other=0.0).to(tl.float32)
            vhat = (tl.load(CV + iv) * nv[:, None]).to(tl.bfloat16)   # [BT, D] decoded value
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vhat)
            m_i = m_new

        # store UN-normalised partials for this split; host combines across NS
        po = (r * NS + s_id) * BM + offs_m
        tl.store(M_OUT + po, m_i)
        tl.store(L_OUT + po, l_i)
        tl.store(ACC_OUT + po[:, None] * D + d[None, :], acc)


def _pack_params(bits: int, D: int):
    """(eff_bits, vals_per_byte, mask, bytes_per_token) — mirrors _pack_indices."""
    if bits == 1:
        eff = 1
    elif bits == 2:
        eff = 2
    elif bits <= 4:
        eff = 4
    else:
        eff = 8
    vpb = 8 // eff
    return eff, vpb, (1 << eff) - 1, (D + vpb - 1) // vpb


def _bank_partials(
    qr, packed_k, norms_k, cent_k, packed_v, norms_v, cent_v, bits, scaling,
    seqlen=None, BT=64, num_warps=2, num_stages=1,
):
    """Run split-K flash-decode for ONE bit-level bank; return online-softmax
    partials sliced to the real G rows: m[NR,G], l[NR,G], acc[NR,G,D] (acc in
    VALUE codebook space; caller applies pi_v once). ``seqlen`` [NR] gives the
    real token count per head when banks are padded to a common T; None = full."""
    NR, G, D = qr.shape
    eff, vpb, mask, BYTES = _pack_params(bits, D)
    T = packed_k.shape[1]
    if seqlen is None:
        seqlen = torch.full((NR,), T, device=qr.device, dtype=torch.int32)
    else:
        seqlen = seqlen.to(device=qr.device, dtype=torch.int32).contiguous()
    BM = max(16, triton.next_power_of_2(G))
    # Adaptive split-K: enough programs to fill the GPU (~256) WITHOUT paying an
    # unnecessary host-side cross-split combine. When grid=(NR,1) already gives
    # plenty of programs, use one split and skip the combine entirely.
    n_split = max(1, min(32, -(-256 // NR)))           # ceil(256/NR), capped 32
    n_split = min(n_split, max(1, T // 256))           # don't over-split short T
    TS = ((T + n_split - 1) // n_split + BT - 1) // BT * BT
    n_split = (T + TS - 1) // TS
    M = torch.empty(NR, n_split, BM, device=qr.device, dtype=torch.float32)
    L = torch.empty(NR, n_split, BM, device=qr.device, dtype=torch.float32)
    ACC = torch.empty(NR, n_split, BM, D, device=qr.device, dtype=torch.float32)
    _flash_decode_lut_kernel[(NR, n_split)](
        qr.contiguous(), packed_k.contiguous(), norms_k.float().contiguous(),
        packed_v.contiguous(), norms_v.float().contiguous(),
        cent_k.float().contiguous(), cent_v.float().contiguous(),
        seqlen, M, L, ACC, T, TS, scaling,
        G=G, D=D, BYTES=BYTES, NS=n_split, VPB=vpb, EFF_BITS=eff, MASK=mask,
        BM=BM, BT=BT, num_warps=num_warps, num_stages=num_stages,
    )
    if n_split == 1:                                   # no combine needed
        return M[:, 0, :G], L[:, 0, :G], ACC[:, 0, :G]
    m = M.amax(dim=1)                                  # [NR,BM] true running max (-inf if empty)
    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    scale = torch.exp(M - m_safe.unsqueeze(1))         # [NR,NS,BM]
    l = (L * scale).sum(dim=1)                         # [NR,BM]
    acc = (ACC * scale.unsqueeze(-1)).sum(dim=1)       # [NR,BM,D]
    return m[:, :G], l[:, :G], acc[:, :G]              # [NR,G], [NR,G], [NR,G,D]


def decode_lut_2bit_triton(
    Q, packed_k, norms_k, cent_k, packed_v, norms_v, cent_v, pi_k, pi_v, scaling,
    BT: int = 64, num_warps: int = 2, num_stages: int = 1,
):
    """Single 2-bit bank convenience wrapper (used by the kernel unit tests).
    Q [NR,G,D] or [G,D]; returns matching shape."""
    squeeze = Q.dim() == 2
    if squeeze:
        Q = Q.unsqueeze(0)
        packed_k = packed_k.unsqueeze(0); norms_k = norms_k.unsqueeze(0)
        packed_v = packed_v.unsqueeze(0); norms_v = norms_v.unsqueeze(0)
    qr = (Q.float() @ pi_k.float().T)                  # [NR,G,D]
    m, l, acc = _bank_partials(qr, packed_k, norms_k, cent_k, packed_v, norms_v,
                               cent_v, 2, scaling, None, BT, num_warps, num_stages)
    out = (acc / l.clamp(min=1e-20).unsqueeze(-1)) @ pi_v.float()   # [NR,G,D]
    return out.squeeze(0) if squeeze else out


def banked_decode_attention(
    Q, quant_banks, exact_k, exact_v, pi_k, pi_v, scaling,
    exact_seqlen=None, BT: int = 64, num_warps: int = 2, num_stages: int = 1,
    extra_exact_banks=None,
):
    """Full mixed-precision decode attention over all banks, no materialise.

    Q [NR,G,D] raw query rows per kv-head. ``quant_banks``: list of dicts
    ``{bits, packed_k, norms_k, cent_k, packed_v, norms_v, cent_v, seqlen?}``
    (one per bit level present, 2..8; optional per-head ``seqlen`` for padded
    banks). ``exact_k/exact_v`` [NR,Ne,D] bf16/fp32 or None (16-bit + decode
    tail); ``exact_seqlen`` [NR] masks padded exact rows.
    ``extra_exact_banks`` optionally contains additional ``(K,V,seqlen)``
    tuples.  The system uses this for a reshape-only mutable decode tail while
    keeping the static 16-bit prefix cached. Evicted (0-bit) tokens are simply
    absent. All quant banks share pi_k (query fold) and pi_v (value rotate-out).
    Returns [NR,G,D].
    """
    squeeze = Q.dim() == 2
    if squeeze:
        Q = Q.unsqueeze(0)
        exact_k = None if exact_k is None else exact_k.unsqueeze(0)
        exact_v = None if exact_v is None else exact_v.unsqueeze(0)
        if extra_exact_banks:
            extra_exact_banks = [
                (
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    None if seqlen is None else seqlen.reshape(1),
                )
                for k, v, seqlen in extra_exact_banks
            ]
    NR, G, D = Q.shape
    Qf = Q.float()
    qr = Qf @ pi_k.float().T                            # [NR,G,D] for quant banks

    ms, ls, quant_accs = [], [], []
    for bk in quant_banks:
        if bk["packed_k"].shape[1] == 0:
            continue
        m, l, acc = _bank_partials(
            qr, bk["packed_k"], bk["norms_k"], bk["cent_k"],
            bk["packed_v"], bk["norms_v"], bk["cent_v"], bk["bits"], scaling,
            bk.get("seqlen"), BT, num_warps, num_stages)
        ms.append(m); ls.append(l); quant_accs.append(acc)   # acc in codebook space

    exact_inputs = []
    if exact_k is not None and exact_k.shape[1] > 0:
        exact_inputs.append((exact_k, exact_v, exact_seqlen))
    if extra_exact_banks:
        exact_inputs.extend(
            (k, v, seqlen)
            for k, v, seqlen in extra_exact_banks
            if k is not None and k.shape[1] > 0
        )

    exact_partials = []
    for bank_k, bank_v, bank_seqlen in exact_inputs:
        se = (Qf @ bank_k.float().transpose(-1, -2)) * scaling   # [NR,G,Ne]
        if bank_seqlen is not None:                              # mask padded rows
            Ne = bank_k.shape[1]
            col = torch.arange(Ne, device=Q.device)
            em = col[None, :] < bank_seqlen.to(Q.device)[:, None]   # [NR,Ne]
            se = torch.where(em[:, None, :], se, torch.full_like(se, -float("inf")))
        exact_m = se.amax(dim=-1)                                 # [NR,G]
        m_e_safe = torch.where(torch.isinf(exact_m), torch.zeros_like(exact_m), exact_m)
        pe = torch.where(torch.isinf(se), torch.zeros_like(se),
                         torch.exp(se - m_e_safe.unsqueeze(-1)))  # nan-safe
        exact_l = pe.sum(dim=-1)                                  # [NR,G]
        exact_acc = pe @ bank_v.float()                           # raw value space
        exact_partials.append((exact_m, exact_l, exact_acc))

    # cross-bank online-softmax combine
    all_m = torch.stack(ms + [part[0] for part in exact_partials], dim=0)
    m_global = all_m.amax(dim=0)                                 # [NR,G]
    m_safe = torch.where(torch.isinf(m_global), torch.zeros_like(m_global), m_global)

    l_global = torch.zeros(NR, G, device=Q.device, dtype=torch.float32)
    quant_num = torch.zeros(NR, G, D, device=Q.device, dtype=torch.float32)
    for m, l, acc in zip(ms, ls, quant_accs):
        w = torch.exp(m - m_safe)                                # [NR,G]
        l_global = l_global + w * l
        quant_num = quant_num + w.unsqueeze(-1) * acc            # codebook space
    num = quant_num @ pi_v.float()                               # rotate quant part once
    for exact_m, exact_l, exact_acc in exact_partials:
        we = torch.exp(exact_m - m_safe)
        l_global = l_global + we * exact_l
        num = num + we.unsqueeze(-1) * exact_acc                 # raw value part
    out = num / l_global.clamp(min=1e-20).unsqueeze(-1)          # [NR,G,D]
    return out.squeeze(0) if squeeze else out
