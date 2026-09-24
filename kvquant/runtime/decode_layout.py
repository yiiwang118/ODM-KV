"""Build the per-head, decode-ready bank layout the kernel consumes.

The storage banks (``TurboQuantAdaptiveKVCacheState``) keep tokens as a flat
list keyed by ``(b, h, s)`` position — append-optimised but not attention-
friendly. The decode kernel wants, per (head, bit-level), a contiguous
padded ``[H_kv, T_max, BYTES]`` block + a per-head ``seqlen``.

Split so decode is cheap:
  * ``build_quant_ready`` — the big quantised banks; STATIC between flushes,
    so the caller caches it and only rebuilds on commit/flush.
  * ``build_exact`` — the small 16-bit ExactBank + the bf16 decode tail;
    rebuilt every step (tail grows by one token per step).

Only the config path (key=mse, value=mse, no OCS) is handled; anything else
returns ``None`` so the caller can fall back to the materialise path.
"""
from __future__ import annotations

from typing import Optional

import torch


def _packed_bytes(bits: int, head_dim: int) -> int:
    effective_bits = 2 if bits == 2 else 4 if bits <= 4 else 8
    return (head_dim + (8 // effective_bits) - 1) // (8 // effective_bits)


def _supported(state) -> bool:
    for bank in state._quantized_banks.values():
        if bank.size == 0:
            continue
        if bank.key_quantizer_type != "mse" or bank.value_quantizer != "mse":
            return False
        if bank.outlier_indices is not None:
            return False
    return True


def _regroup_by_row(pk, nk, pv, nv, rows, num_rows):
    """Flat per-token tensors -> padded [num_rows, T_max, ...].

    A row is one ``(batch, kv_head)`` pair.  Keeping batch in the row index
    lets the Triton grid cover real serving batches instead of only the first
    sample.
    """
    BYTES = pk.shape[1]
    counts = torch.bincount(rows, minlength=num_rows)            # [B*H_kv]
    T_max = int(counts.max().item()) if counts.numel() else 0
    dev = pk.device
    PK = torch.zeros(num_rows, T_max, BYTES, device=dev, dtype=pk.dtype)
    PV = torch.zeros(num_rows, T_max, BYTES, device=dev, dtype=pv.dtype)
    NK = torch.zeros(num_rows, T_max, device=dev, dtype=torch.float32)
    NV = torch.zeros(num_rows, T_max, device=dev, dtype=torch.float32)
    if pk.shape[0] == 0 or T_max == 0:
        return PK, NK, PV, NV, counts.to(torch.int32)
    # Vectorized scatter (replaces a per-row Python loop that launched num_rows
    # slice-copies per bank per flush — ~10 ms/layer at 4k context). Each token's
    # destination is row*T_max + its rank within the row; one index_copy_ fills
    # the whole padded block.
    order = torch.argsort(rows, stable=True)
    sorted_rows = rows[order]
    starts = torch.zeros(num_rows, device=dev, dtype=torch.long)
    starts[1:] = counts.cumsum(0)[:-1]
    within = torch.arange(pk.shape[0], device=dev) - starts[sorted_rows]
    dest = sorted_rows * T_max + within                          # flat idx
    PK.view(num_rows * T_max, BYTES).index_copy_(0, dest, pk[order])
    PV.view(num_rows * T_max, BYTES).index_copy_(0, dest, pv[order])
    NK.view(num_rows * T_max).index_copy_(0, dest, nk[order].float())
    NV.view(num_rows * T_max).index_copy_(0, dest, nv[order].float())
    return PK, NK, PV, NV, counts.to(torch.int32)


def build_quant_ready(state, H_kv: int, b: Optional[int] = None):
    """Return ``(quant_banks, pi_k, pi_v)`` or ``None`` if unsupported.

    ``b=None`` (the model fast path) flattens all ``B*H_kv`` rows into one
    layout/kernel grid.  Passing an integer keeps the single-sample diagnostic
    behavior used by older tests.
    """
    if not _supported(state):
        return None
    batch_size = max(int(state.batch_size), 1)
    num_rows = H_kv if b is not None else batch_size * H_kv
    quant_banks = []
    pi_k = pi_v = None
    for level, bank in state._quantized_banks.items():
        if bank.size == 0:
            continue
        pos = bank.positions                                     # [N,3] (b,h,s)
        sel = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device) \
            if b is None else pos[:, 0] == b
        if not bool(sel.any()):
            continue
        selected_pos = pos[sel]
        rows = selected_pos[:, 1] if b is not None else \
            selected_pos[:, 0] * H_kv + selected_pos[:, 1]
        PK, NK, PV, NV, counts = _regroup_by_row(
            bank.key_quantized.indices[sel], bank.key_quantized.norms[sel],
            bank.value_quantized.indices[sel], bank.value_quantized.norms[sel],
            rows, num_rows)
        pi_k = bank.key_quantizer.pi
        pi_v = bank.value_mse_quantizer.pi
        quant_banks.append(dict(
            bits=level, packed_k=PK, norms_k=NK, cent_k=bank.key_quantizer.centroids,
            packed_v=PV, norms_v=NV, cent_v=bank.value_mse_quantizer.centroids,
            seqlen=counts))
    # No occupied quant banks (short prefill: ctx ≤ buffer_size ⇒ every prefill
    # token is protected fp16). Still expose the rotations so the caller takes the
    # fused path (empty bank slots + exact prefix + tail) rather than the
    # materialise fallback — which in graph mode reads the cat tail, not the ring,
    # and would silently drop the decode tokens. Every bit-level QuantizedBank is
    # pre-constructed (holds the shared pi) even at size 0; with empty banks the
    # accumulator is zero so pi's *value* is irrelevant, only its shape is used.
    if pi_k is None and state._quantized_banks:
        any_bank = next(iter(state._quantized_banks.values()))
        pi_k = any_bank.key_quantizer.pi
        pi_v = any_bank.value_mse_quantizer.pi
    return quant_banks, pi_k, pi_v


def build_quant_ready_ragged(state, H_kv: int, b: Optional[int] = None):
    """Ragged (compact) variant of ``build_quant_ready`` for the fused kernel.

    Instead of scattering tokens into a padded ``[num_rows, T_max, BYTES]`` block
    (4× byte inflation when row counts are uneven — see layout_audit), it sorts
    the flat per-token banks by row and returns them contiguous with a per-row
    ``offset`` (CSR-style). The kernel reads ``packed[offset[r] : offset[r]+
    count[r]]`` — no padding in storage *or* reads. ``T_max`` (max count) is
    precomputed here (outside graph capture) for the NS heuristic."""
    if not _supported(state):
        return None
    batch_size = max(int(state.batch_size), 1)
    num_rows = H_kv if b is not None else batch_size * H_kv
    dev = state.device
    quant_banks = []
    pi_k = pi_v = None
    for level, bank in state._quantized_banks.items():
        if bank.size == 0:
            continue
        pos = bank.positions
        sel = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device) \
            if b is None else pos[:, 0] == b
        if not bool(sel.any()):
            continue
        selected_pos = pos[sel]
        rows = selected_pos[:, 1] if b is not None else \
            selected_pos[:, 0] * H_kv + selected_pos[:, 1]
        counts = torch.bincount(rows, minlength=num_rows)
        order = torch.argsort(rows, stable=True)
        offset = torch.zeros(num_rows, device=dev, dtype=torch.int32)
        offset[1:] = counts.cumsum(0)[:-1].to(torch.int32)       # CSR row starts
        pi_k = bank.key_quantizer.pi
        pi_v = bank.value_mse_quantizer.pi
        quant_banks.append(dict(
            bits=level,
            packed_k=bank.key_quantized.indices[sel][order].contiguous(),
            norms_k=bank.key_quantized.norms[sel][order].contiguous(),
            cent_k=bank.key_quantizer.centroids,
            packed_v=bank.value_quantized.indices[sel][order].contiguous(),
            norms_v=bank.value_quantized.norms[sel][order].contiguous(),
            cent_v=bank.value_mse_quantizer.centroids,
            offset=offset, seqlen=counts.to(torch.int32),
            T_max=int(counts.max().item()) if counts.numel() else 0))
    if pi_k is None and state._quantized_banks:
        any_bank = next(iter(state._quantized_banks.values()))
        pi_k = any_bank.key_quantizer.pi
        pi_v = any_bank.value_mse_quantizer.pi
    return quant_banks, pi_k, pi_v


def build_quant_ready_stable(
    state, H_kv: int, reserve_per_row: int, b: Optional[int] = None,
):
    """Build fixed-address CSR banks with room for future decode flushes.

    The normal ragged layout is compact, but every append reallocates/sorts it;
    its new pointers and shapes force a CUDA-graph recapture.  Graph generation
    instead reserves capacity inside each CSR row and appends future compressed
    tokens in place. ``seqlen`` remains a device tensor, so the already-captured
    Triton kernel observes newly written tokens without changing its arguments.
    """
    if not _supported(state):
        return None
    batch_size = max(int(state.batch_size), 1)
    num_rows = H_kv if b is not None else batch_size * H_kv
    dev = state.device
    quant_banks = []
    pi_k = pi_v = None
    reserve_per_row = max(0, int(reserve_per_row))

    # Include empty configured levels too: a level absent after prefill may be
    # selected by a later decode allocation, and a captured graph cannot gain a
    # new pointer argument at that point.
    for level, bank in state._quantized_banks.items():
        BYTES = _packed_bytes(level, state.head_dim)
        if bank.size > 0:
            pos = bank.positions
            sel = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device) \
                if b is None else pos[:, 0] == b
            selected_pos = pos[sel]
            rows = selected_pos[:, 1] if b is not None else \
                selected_pos[:, 0] * H_kv + selected_pos[:, 1]
            counts = torch.bincount(rows, minlength=num_rows).to(torch.int32)
            order = torch.argsort(rows, stable=True)
            sorted_rows = rows[order]
            starts = torch.zeros(num_rows, device=dev, dtype=torch.long)
            starts[1:] = counts.to(torch.long).cumsum(0)[:-1]
            within = torch.arange(rows.numel(), device=dev) - starts[sorted_rows]
            src_pk = bank.key_quantized.indices[sel][order]
            src_pv = bank.value_quantized.indices[sel][order]
            src_nk = bank.key_quantized.norms[sel][order].float()
            src_nv = bank.value_quantized.norms[sel][order].float()
        else:
            counts = torch.zeros(num_rows, dtype=torch.int32, device=dev)
            sorted_rows = torch.empty(0, dtype=torch.long, device=dev)
            within = torch.empty(0, dtype=torch.long, device=dev)
            src_pk = torch.empty(0, BYTES, dtype=torch.uint8, device=dev)
            src_pv = torch.empty_like(src_pk)
            src_nk = torch.empty(0, dtype=torch.float32, device=dev)
            src_nv = torch.empty_like(src_nk)

        # CSR with fixed capacity per row: compact initial contents plus an
        # equal decode reserve for each request/head.  This keeps graph pointers
        # stable without regressing to max-row rectangular padding.
        row_capacity = counts.to(torch.long) + max(1, reserve_per_row)
        offset = torch.zeros(num_rows, device=dev, dtype=torch.int32)
        offset[1:] = row_capacity.cumsum(0)[:-1].to(torch.int32)
        total_capacity = int(row_capacity.sum().item())
        PK = torch.zeros(total_capacity, BYTES, dtype=torch.uint8, device=dev)
        PV = torch.zeros_like(PK)
        NK = torch.zeros(total_capacity, dtype=torch.float32, device=dev)
        NV = torch.zeros_like(NK)
        if sorted_rows.numel():
            dest = offset[sorted_rows].to(torch.long) + within
            PK.index_copy_(0, dest, src_pk)
            PV.index_copy_(0, dest, src_pv)
            NK.index_copy_(0, dest, src_nk)
            NV.index_copy_(0, dest, src_nv)

        pi_k = bank.key_quantizer.pi
        pi_v = bank.value_mse_quantizer.pi
        quant_banks.append(dict(
            bits=level, packed_k=PK, norms_k=NK,
            cent_k=bank.key_quantizer.centroids,
            packed_v=PV, norms_v=NV,
            cent_v=bank.value_mse_quantizer.centroids,
            offset=offset, seqlen=counts.to(torch.int32),
            T_max=int(row_capacity.max().item()),
            stable_row_capacity=row_capacity.to(torch.int32),
        ))
    return quant_banks, pi_k, pi_v


def build_exact(state, tail_k: Optional[torch.Tensor], tail_v: Optional[torch.Tensor],
                H_kv: int, b: Optional[int] = None):
    """Return ``(exact_k, exact_v, exact_seqlen)`` folding the 16-bit ExactBank
    AND the not-yet-flushed bf16 decode tail, per head, padded. Cheap; rebuild
    every step."""
    dev = state.device
    D = state.head_dim
    batch_size = tail_k.shape[0] if tail_k is not None else max(int(state.batch_size), 1)
    num_rows = H_kv if b is not None else batch_size * H_kv
    ex = state._exact_bank
    T_tail = tail_k.shape[2] if (tail_k is not None and tail_k.shape[2] > 0) else 0
    if ex.size > 0 and ex.keys is not None:
        epos = ex.positions
        esel = torch.ones(epos.shape[0], dtype=torch.bool, device=epos.device) \
            if b is None else epos[:, 0] == b
        selected_pos = epos[esel]
        erows = selected_pos[:, 1] if b is not None else \
            selected_pos[:, 0] * H_kv + selected_pos[:, 1]
        ekeys = ex.keys[esel]
        evals = ex.values[esel]
        ecounts = torch.bincount(erows, minlength=num_rows)
    else:
        erows = torch.empty(0, device=dev, dtype=torch.long)
        ekeys = torch.zeros(0, D, device=dev, dtype=torch.float32)
        evals = torch.zeros(0, D, device=dev, dtype=torch.float32)
        ecounts = torch.zeros(num_rows, device=dev, dtype=torch.long)

    Ne_max = (int(ecounts.max().item()) if ecounts.numel() else 0) + T_tail
    if Ne_max == 0:
        return None, None, None
    exact_k = torch.zeros(num_rows, Ne_max, D, device=dev, dtype=torch.bfloat16)
    exact_v = torch.zeros(num_rows, Ne_max, D, device=dev, dtype=torch.bfloat16)
    exact_seqlen = torch.zeros(num_rows, device=dev, dtype=torch.int32)
    # Vectorized prefix scatter (was a per-row Python loop, ~3 ms/layer at 4k).
    if erows.numel():
        order = torch.argsort(erows, stable=True)
        srows = erows[order]
        starts = torch.zeros(num_rows, device=dev, dtype=torch.long)
        starts[1:] = ecounts.cumsum(0)[:-1]
        within = torch.arange(erows.shape[0], device=dev) - starts[srows]
        dest = srows * Ne_max + within
        exact_k.view(num_rows * Ne_max, D).index_copy_(0, dest, ekeys[order].to(torch.bfloat16))
        exact_v.view(num_rows * Ne_max, D).index_copy_(0, dest, evals[order].to(torch.bfloat16))
    exact_seqlen = ecounts.to(torch.int32)
    if T_tail > 0:
        # tail appended after each row's prefix (rare: old build_decode_ready path)
        for row in range(num_rows):
            n = int(ecounts[row].item())
            batch_idx, head_idx = (b, row) if b is not None else divmod(row, H_kv)
            exact_k[row, n:n + T_tail] = tail_k[batch_idx, head_idx].float()
            exact_v[row, n:n + T_tail] = tail_v[batch_idx, head_idx].float()
        exact_seqlen = exact_seqlen + T_tail
    return exact_k, exact_v, exact_seqlen


def build_exact_stable(
    state, H_kv: int, reserve_per_row: int, b: Optional[int] = None,
):
    """Fixed-address exact prefix with per-row decode reserve."""
    exact_k, exact_v, seqlen = build_exact(state, None, None, H_kv, b)
    batch_size = max(int(state.batch_size), 1)
    num_rows = H_kv if b is not None else batch_size * H_kv
    initial_width = 0 if exact_k is None else exact_k.shape[1]
    capacity = max(1, initial_width + max(0, int(reserve_per_row)))
    K = torch.zeros(
        num_rows, capacity, state.head_dim,
        device=state.device, dtype=torch.bfloat16,
    )
    V = torch.zeros_like(K)
    if initial_width:
        K[:, :initial_width].copy_(exact_k)
        V[:, :initial_width].copy_(exact_v)
    if seqlen is None:
        seqlen = torch.zeros(num_rows, device=state.device, dtype=torch.int32)
    return K, V, seqlen.to(torch.int32)


def build_decode_ready(state, tail_k, tail_v, H_kv: int, b: Optional[int] = None):
    """Convenience: quant + exact in one call (used by the layout unit test)."""
    q = build_quant_ready(state, H_kv, b)
    if q is None:
        return None
    quant_banks, pi_k, pi_v = q
    exact_k, exact_v, exact_seqlen = build_exact(state, tail_k, tail_v, H_kv, b)
    return quant_banks, exact_k, exact_v, exact_seqlen, pi_k, pi_v
