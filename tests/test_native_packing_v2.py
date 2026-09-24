"""Correctness and storage contract for fixed-arena native packing v2."""
from __future__ import annotations

import inspect

import pytest
import torch

import kvquant.runtime.native_packing_v2 as packing_v2
from kvquant.tq_backend import _unpack_indices
from kvquant.runtime.native_packing import pack_native_prefill
from kvquant.runtime.native_packing_v2 import (
    NativePackingCapacity,
    NativeSharedPackingCapacity,
    QUANTIZED_LEVELS,
    pack_native_prefill_v2,
    pack_native_prefill_shared_v2,
)
from kvquant.runtime.kernels.fused_decode import _HAS_TRITON as _HAS_FUSED_DECODE, fused_decode
from kvquant.runtime.kernels.native_flush_v2 import NativeFlushWorkspace
from kvquant.runtime.storage.cache import ODMCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native packing v2 is a CUDA path"
)


def _flush_workspace(packed, max_tokens: int) -> NativeFlushWorkspace:
    return NativeFlushWorkspace.allocate(
        rows=packed.num_rows,
        max_tokens=max_tokens,
        head_dim=packed.head_dim,
        device=packed.tags.device,
    )


def _inputs(batch: int, *, tokens: int = 257, dim: int = 64):
    torch.manual_seed(1099 + batch + tokens)
    heads = 4
    keys = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    lut = torch.tensor([0, 2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    tags = lut[
        torch.arange(batch * heads * tokens, device="cuda").reshape(batch, heads, tokens)
        % len(lut)
    ]
    total = batch * heads * tokens
    counts = tuple(sum(index % 6 == level for index in range(total)) for level in range(6))
    capacity = NativePackingCapacity(
        level_slots=counts[1:5],
        exact_slots=counts[5],
    )
    return keys, values, tags, capacity


def _live_indices(bank, rows, width):
    column = torch.arange(width, device="cuda")
    live = column.view(1, -1) < bank["seqlen"].view(rows, 1)
    return (bank["offset"].long().view(rows, 1) + column.view(1, -1))[live]


@pytest.mark.parametrize("batch", [1, 4])
def test_v2_matches_v1_mixed_level_payload_and_materialize(batch):
    keys, values, tags, capacity = _inputs(batch)
    reference = pack_native_prefill(keys, values, tags)
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    packed.validate_capacity()

    rows = keys.shape[0] * keys.shape[1]
    for bits in QUANTIZED_LEVELS:
        got = packed.bank(bits)
        ref = reference.bank(bits)
        live = _live_indices(got, rows, keys.shape[2])
        torch.testing.assert_close(got["offset"], ref["offset"], rtol=0, atol=0)
        torch.testing.assert_close(got["seqlen"], ref["seqlen"], rtol=0, atol=0)
        torch.testing.assert_close(got["norms_k"][live], ref["norms_k"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(got["norms_v"][live], ref["norms_v"], rtol=1e-6, atol=1e-6)
        # A larger fixed GEMM can change the last centroid decision at a tiny
        # number of exact boundaries.  This is the same bounded behavior
        # already accepted for v1's B>1 global GEMM.
        for field in ("packed_k", "packed_v"):
            if bits == 3:
                got_codes = packing_v2._unpack_indices_true3(
                    got[field][live], keys.shape[-1],
                )
                ref_codes = _unpack_indices(ref[field], bits, keys.shape[-1])
                unequal = torch.count_nonzero(got_codes != ref_codes)
                compared = ref_codes.numel()
            else:
                unequal = torch.count_nonzero(got[field][live] != ref[field])
                compared = ref[field].numel()
            allowed = 0 if batch == 1 else max(1, compared // 2500)
            assert unequal <= allowed

    exact_live = _live_indices(packed.exact, rows, keys.shape[2])
    torch.testing.assert_close(packed.exact["keys"][exact_live], reference.exact["keys"])
    torch.testing.assert_close(packed.exact["values"][exact_live], reference.exact["values"])

    got_k, got_v = packed.materialize()
    ref_k, ref_v = reference.materialize()
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)


def test_zero_bit_nan_writes_no_payload():
    keys, values, tags, capacity = _inputs(1)
    zero = tags == 0
    keys[zero.unsqueeze(-1).expand_as(keys)] = torch.nan
    values[zero.unsqueeze(-1).expand_as(values)] = torch.nan
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    packed.validate_capacity()
    for bits in QUANTIZED_LEVELS:
        bank = packed.bank(bits)
        live = _live_indices(bank, packed.num_rows, keys.shape[2])
        assert torch.isfinite(bank["norms_k"][live]).all()
        assert torch.isfinite(bank["norms_v"][live]).all()
    exact_live = _live_indices(packed.exact, packed.num_rows, keys.shape[2])
    assert torch.isfinite(packed.exact["keys"][exact_live]).all()
    got_k, got_v = packed.materialize()
    assert torch.count_nonzero(got_k[zero]) == 0
    assert torch.count_nonzero(got_v[zero]) == 0


def test_payload_views_share_four_arenas_and_source_has_no_dynamic_selection():
    keys, values, tags, capacity = _inputs(1)
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    for bank in packed.quant_banks:
        assert bank["packed_k"].untyped_storage().data_ptr() == packed.code_arena_k.untyped_storage().data_ptr()
        assert bank["packed_v"].untyped_storage().data_ptr() == packed.code_arena_v.untyped_storage().data_ptr()
        assert bank["norms_k"].untyped_storage().data_ptr() == packed.norm_arena_k.untyped_storage().data_ptr()
        assert bank["norms_v"].untyped_storage().data_ptr() == packed.norm_arena_v.untyped_storage().data_ptr()

    for packer in (
        packing_v2.pack_native_prefill_v2,
        packing_v2.pack_native_prefill_shared_v2,
    ):
        source = inspect.getsource(packer)
        assert "torch.nonzero" not in source
        assert ".nonzero(" not in source
        assert "index_select" not in source
        assert "masked_select" not in source
        assert ".item(" not in source

    # The legacy packer stays fully host-free.  The shared packer deliberately
    # reads its already-computed device requirement back once, to size arenas
    # byte-exactly instead of provisioning every level for the whole nominal
    # budget; that readback is a sizing decision, never data-dependent
    # selection, and it must remain a single call on the layer's cold path.
    legacy = inspect.getsource(packing_v2.pack_native_prefill_v2)
    assert ".tolist(" not in legacy
    shared = inspect.getsource(packing_v2.pack_native_prefill_shared_v2)
    assert shared.count(".tolist(") == 1


def test_overflow_is_device_flagged_and_all_stores_are_bounded():
    keys, values, tags, _ = _inputs(1, tokens=65)
    undersized = NativePackingCapacity((1, 1, 1, 1), exact_slots=1)
    packed = pack_native_prefill_v2(keys, values, tags, capacity=undersized)
    torch.cuda.synchronize()  # make any out-of-bounds kernel fault visible
    assert int(packed.overflow_flag.cpu()[0]) == 1
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        packed.validate_capacity()


def test_fixed_2bit_decode_reserve_keeps_payload_addresses_and_semantics():
    keys, values, tags, base_capacity = _inputs(2, tokens=65)
    reserve = 8
    capacity = NativePackingCapacity(
        base_capacity.level_slots,
        base_capacity.exact_slots,
        decode_2bit_reserve_per_row=reserve,
    )
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    workspace = _flush_workspace(packed, reserve)
    pointers = tuple(
        tensor.data_ptr()
        for bank in packed.quant_banks
        for tensor in (bank["packed_k"], bank["packed_v"], bank["norms_k"], bank["norms_v"])
    )
    initial = packed.bank(2)["seqlen"].clone()
    decode_k = []
    decode_v = []
    for step in range(2):
        torch.manual_seed(1400 + step)
        dk = torch.randn(2, 4, 4, 64, device="cuda", dtype=torch.bfloat16)
        dv = torch.randn_like(dk)
        decode_k.append(dk)
        decode_v.append(dv)
        packed.append_decode_2bit(dk, dv, workspace=workspace)
        assert pointers == tuple(
            tensor.data_ptr()
            for bank in packed.quant_banks
            for tensor in (bank["packed_k"], bank["packed_v"], bank["norms_k"], bank["norms_v"])
        )
    torch.testing.assert_close(packed.bank(2)["seqlen"], initial + reserve, rtol=0, atol=0)

    combined_k = torch.cat((keys, *decode_k), dim=2)
    combined_v = torch.cat((values, *decode_v), dim=2)
    combined_tags = torch.cat(
        (tags, torch.full((*tags.shape[:2], reserve), 2, device="cuda", dtype=tags.dtype)),
        dim=2,
    )
    oracle = pack_native_prefill(combined_k, combined_v, combined_tags)
    got_k, got_v = packed.materialize()
    ref_k, ref_v = oracle.materialize()
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)


def test_tight_arena_is_real_physical_compression_not_live_byte_accounting():
    keys, values, tags, capacity = _inputs(8, tokens=2048, dim=128)
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    stats = packed.reserved_storage_stats()
    # This includes unused capacity, tags, CSR, rotations, and codebooks.  It is
    # deliberately not a logical-live-only ratio.
    assert stats["reserved_ratio_vs_dense"] < 0.5


def test_true3_is_physically_dense_and_code_equivalent_to_legacy_nibbles():
    B, H, T, D = 1, 2, 19, 128
    torch.manual_seed(1703)
    keys = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    tags = torch.full((B, H, T), 3, device="cuda", dtype=torch.int32)
    capacity = NativeSharedPackingCapacity.from_histogram(
        counts={3: B * H * T}, head_dim=D, num_rows=B * H,
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    legacy = pack_native_prefill(keys, values, tags)
    bank = packed.bank(3)
    legacy_bank = legacy.bank(3)

    assert bank["physical_bits"] == 3
    assert bank["packed_width"] == 48
    assert legacy_bank["physical_bits"] == 4
    assert legacy_bank["packed_k"].shape[-1] == 64

    live = _live_indices(bank, packed.num_rows, T)
    byte = torch.arange(bank["packed_width"], device="cuda")
    address = bank["code_base"].long() + live[:, None] * bank["packed_width"] + byte
    dense_codes = packing_v2._unpack_indices_true3(
        bank["packed_k"][address], D,
    )
    legacy_codes = _unpack_indices(legacy_bank["packed_k"], 3, D)
    torch.testing.assert_close(dense_codes, legacy_codes, rtol=0, atol=0)

    live_stats = packed.live_storage_stats()
    assert live_stats["live_code_bytes"] == 2 * B * H * T * 48


def test_policy_capacity_constructor_is_host_only_and_conservative_for_budget_fixture():
    plan = NativePackingCapacity.from_policy_budget(
        batch_size=8,
        num_heads=8,
        seq_len=2048,
        target_avg_bits=1.0,
        sink_tokens=4,
        tail_tokens=128,
        decode_2bit_reserve_per_row=300,
    )
    assert plan.level_slots[0] >= (8 * 8 * (2048 - 4)) // 2
    assert plan.exact_slots >= 8 * 8 * (4 + 128)
    assert plan.decode_2bit_reserve_per_row == 300


@pytest.mark.skipif(not _HAS_FUSED_DECODE, reason="fused decode requires Triton")
def test_v2_ragged_views_feed_fused_decode_without_repacking():
    keys, values, tags, capacity = _inputs(1, tokens=257)
    reference = pack_native_prefill(keys, values, tags)
    packed = pack_native_prefill_v2(keys, values, tags, capacity=capacity)
    query = torch.randn(
        keys.shape[0] * keys.shape[1], 4, keys.shape[-1],
        device="cuda", dtype=torch.bfloat16,
    )
    exact_v2 = {
        "offset": packed.exact["offset"],
        "seqlen": packed.exact["seqlen"],
        "T_max": packed.exact["T_max"],
    }
    exact_v1 = {
        "offset": reference.exact["offset"],
        "seqlen": reference.exact["seqlen"],
        "T_max": reference.exact["T_max"],
    }
    got = fused_decode(
        query,
        list(packed.quant_banks),
        packed.exact["keys"],
        packed.exact["values"],
        exact_v2,
        packed.pi_k,
        packed.pi_v,
        keys.shape[-1] ** -0.5,
    )
    expected = fused_decode(
        query,
        list(reference.quant_banks),
        reference.exact["keys"],
        reference.exact["values"],
        exact_v1,
        reference.pi_k,
        reference.pi_v,
        keys.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


def _histogram_capacity(tags_cpu: torch.Tensor, head_dim: int, rows: int, reserve: int = 0):
    histogram = torch.bincount(tags_cpu.reshape(-1), minlength=17)
    counts = {bits: int(histogram[bits]) for bits in (0, 2, 3, 4, 8, 16)}
    return NativeSharedPackingCapacity.from_histogram(
        counts=counts,
        head_dim=head_dim,
        num_rows=rows,
        decode_2bit_reserve_per_row=reserve,
    )


@pytest.mark.parametrize("batch", [1, 4])
def test_shared_budget_arena_matches_v1_mixed_payload(batch):
    keys, values, tags, _ = _inputs(batch)
    capacity = _histogram_capacity(
        tags.cpu(), keys.shape[-1], keys.shape[0] * keys.shape[1]
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    reference = pack_native_prefill(keys, values, tags)
    packed.validate_capacity()
    got_k, got_v = packed.materialize()
    ref_k, ref_v = reference.materialize()
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)
    # Every level references the same two byte arenas and two norm arenas; only
    # its device base differs.
    for bank in packed.quant_banks:
        assert bank["packed_k"].data_ptr() == packed.code_arena_k.data_ptr()
        assert bank["packed_v"].data_ptr() == packed.code_arena_v.data_ptr()
        assert bank["norms_k"].data_ptr() == packed.norm_arena_k.data_ptr()
        assert bank["norms_v"].data_ptr() == packed.norm_arena_v.data_ptr()
        assert bank["code_base"].is_cuda and bank["norm_base"].is_cuda


@pytest.mark.skipif(not _HAS_FUSED_DECODE, reason="fused decode requires Triton")
def test_shared_device_bases_feed_fused_decode_exactly():
    keys, values, tags, _ = _inputs(1)
    capacity = _histogram_capacity(tags.cpu(), keys.shape[-1], keys.shape[1])
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    reference = pack_native_prefill(keys, values, tags)
    query = torch.randn(keys.shape[1], 4, keys.shape[-1], device="cuda", dtype=torch.bfloat16)

    def exact_meta(state):
        return {
            "offset": state.exact["offset"],
            "seqlen": state.exact["seqlen"],
            "T_max": state.exact["T_max"],
        }

    got = fused_decode(
        query, list(packed.quant_banks), packed.exact["keys"], packed.exact["values"],
        exact_meta(packed), packed.pi_k, packed.pi_v, keys.shape[-1] ** -0.5,
    )
    expected = fused_decode(
        query, list(reference.quant_banks), reference.exact["keys"], reference.exact["values"],
        exact_meta(reference), reference.pi_k, reference.pi_v, keys.shape[-1] ** -0.5,
    )
    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)


def test_shared_policy_capacity_is_tight_and_guaranteed_for_production_fixture():
    B, H, T, D = 8, 8, 2048, 128
    torch.manual_seed(1911)
    keys = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    tags = torch.zeros(B, H, T, device="cuda", dtype=torch.int32)
    tags[..., 4:-128:2] = 2
    tags[..., :4] = 16
    tags[..., -128:] = 16
    capacity = NativeSharedPackingCapacity.from_policy_budget(
        batch_size=B,
        num_heads=H,
        seq_len=T,
        head_dim=D,
        target_avg_bits=1.0,
        sink_tokens=4,
        tail_tokens=128,
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    packed.validate_capacity()
    live = packed.live_storage_stats()
    reserved = packed.reserved_storage_stats()
    assert live["live_payload_ratio_vs_dense"] < 0.14
    assert reserved["reserved_ratio_vs_dense"] < 0.24
    # The old independent policy arenas reserve about 42% on this exact shape.
    assert reserved["reserved_ratio_vs_dense"] < 0.6 * 0.4208


def test_shared_2bit_reserve_is_pointer_stable_across_multiple_flushes():
    keys, values, tags, _ = _inputs(2, tokens=65)
    reserve = 12
    capacity = _histogram_capacity(
        tags.cpu(), keys.shape[-1], keys.shape[0] * keys.shape[1], reserve
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    workspace = _flush_workspace(packed, reserve)
    pointers = (
        packed.code_arena_k.data_ptr(),
        packed.code_arena_v.data_ptr(),
        packed.norm_arena_k.data_ptr(),
        packed.norm_arena_v.data_ptr(),
    )
    decode_k = []
    decode_v = []
    for step in range(3):
        torch.manual_seed(2200 + step)
        dk = torch.randn(2, 4, 4, 64, device="cuda", dtype=torch.bfloat16)
        dv = torch.randn_like(dk)
        decode_k.append(dk)
        decode_v.append(dv)
        packed.append_decode_2bit(dk, dv, workspace=workspace)
        assert pointers == (
            packed.code_arena_k.data_ptr(),
            packed.code_arena_v.data_ptr(),
            packed.norm_arena_k.data_ptr(),
            packed.norm_arena_v.data_ptr(),
        )
    combined_k = torch.cat((keys, *decode_k), dim=2)
    combined_v = torch.cat((values, *decode_v), dim=2)
    combined_tags = torch.cat(
        (tags, torch.full((*tags.shape[:2], reserve), 2, device="cuda", dtype=tags.dtype)),
        dim=2,
    )
    oracle = pack_native_prefill(combined_k, combined_v, combined_tags)
    got_k, got_v = packed.materialize()
    ref_k, ref_v = oracle.materialize()
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)


def test_shared_decode_reserve_does_not_expand_prefill_gemm(monkeypatch):
    keys, values, tags, _ = _inputs(2, tokens=65)
    rows = keys.shape[0] * keys.shape[1]
    capacities = (
        _histogram_capacity(tags.cpu(), keys.shape[-1], rows, reserve=0),
        _histogram_capacity(tags.cpu(), keys.shape[-1], rows, reserve=24),
    )
    observed_rows = []
    original_mm = torch.mm

    def record_mm(left, right, *, out=None):
        observed_rows.append(left.shape[0])
        return original_mm(left, right, out=out)

    monkeypatch.setattr(torch, "mm", record_mm)
    for capacity in capacities:
        packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
        packed.validate_capacity()

    # K and V each execute one GEMM.  Physical suffix reserve changes final
    # arena offsets but must never become normalization/rotation/encode work.
    assert observed_rows[:2] == observed_rows[2:]
    assert observed_rows[0] == sum(capacities[0].level_launch_slots)


@pytest.mark.skipif(not _HAS_FUSED_DECODE, reason="fused decode requires Triton")
def test_shared_strided_suffix_matches_contiguous_v1_and_fused_decode():
    """HF-style transposed/sliced K/V must not require a dense copy."""
    torch.manual_seed(2718)
    B, H, source_tokens, T, D = 4, 2, 23, 17, 64
    source_k = torch.randn(
        B, source_tokens, H, D, device="cuda", dtype=torch.bfloat16,
    )
    source_v = torch.randn_like(source_k)
    # Transpose reproduces the usual projection view; slicing adds a nonzero
    # storage offset and leaves the head stride tied to the original width.
    keys = source_k.transpose(1, 2)[..., -T:, :]
    values = source_v.transpose(1, 2)[..., -T:, :]
    assert not keys.is_contiguous() and keys.storage_offset() != 0

    lengths = (17, 3, 1, 17)
    tags = torch.zeros(B, H, T, device="cuda", dtype=torch.int32)
    for row, length in enumerate(lengths):
        tags[row, :, -length:] = 16
    # Exercise normalization/encoding while deliberately leaving 3/4/8 empty.
    tags[0, :, -9:-4:2] = 2
    tags[3, :, -10:-4:2] = 2
    reserve = 10
    capacity = _histogram_capacity(tags.cpu(), D, B * H, reserve)

    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    contiguous = pack_native_prefill_shared_v2(
        keys.contiguous(), values.contiguous(), tags, capacity=capacity,
    )
    reference = pack_native_prefill(
        keys,
        values,
        tags,
    )
    packed.validate_capacity()
    contiguous.validate_capacity()

    got_k, got_v = packed.materialize()
    contiguous_k, contiguous_v = contiguous.materialize()
    torch.testing.assert_close(got_k, contiguous_k, rtol=0, atol=0)
    torch.testing.assert_close(got_v, contiguous_v, rtol=0, atol=0)

    query = torch.randn(B * H, 4, D, device="cuda", dtype=torch.bfloat16)

    def exact_meta(state):
        return {
            "offset": state.exact["offset"],
            "seqlen": state.exact["seqlen"],
            "T_max": state.exact["T_max"],
        }

    got = fused_decode(
        query,
        list(packed.quant_banks),
        packed.exact["keys"],
        packed.exact["values"],
        exact_meta(packed),
        packed.pi_k,
        packed.pi_v,
        D ** -0.5,
    )
    expected = fused_decode(
        query,
        list(reference.quant_banks),
        reference.exact["keys"],
        reference.exact["values"],
        exact_meta(reference),
        reference.pi_k,
        reference.pi_v,
        D ** -0.5,
    )
    torch.testing.assert_close(got, expected, rtol=2e-3, atol=2e-3)

    pack_source = inspect.getsource(packing_v2.pack_native_prefill_shared_v2)
    assert "keys.contiguous" not in pack_source
    assert "values.contiguous" not in pack_source


def test_shared_code_and_norm_overflow_is_flagged_and_store_bounded():
    keys, values, tags, _ = _inputs(1, tokens=65)
    # Keep launch grids large enough to exercise every destination guard while
    # making all three physical arenas intentionally too small.
    capacity = NativeSharedPackingCapacity(
        quant_slots=1,
        code_bytes_per_tensor=1,
        exact_slots=1,
        level_launch_slots=(65 * 4,) * 4,
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=capacity)
    torch.cuda.synchronize()  # any out-of-bounds kernel fault is visible here
    assert int(packed.overflow_flag.cpu()[0]) == 1
    with pytest.raises(RuntimeError, match="shared arena capacity exceeded"):
        packed.validate_capacity()

    # Launch bounds are part of the physical contract too: a large arena with
    # an undersized grid must not silently truncate a level.
    launch_short = NativeSharedPackingCapacity(
        quant_slots=tags.numel(),
        code_bytes_per_tensor=tags.numel() * keys.shape[-1],
        exact_slots=tags.numel(),
        level_launch_slots=(1, 1, 1, 1),
    )
    packed = pack_native_prefill_shared_v2(keys, values, tags, capacity=launch_short)
    assert int(packed.overflow_flag.cpu()[0]) == 1
    with pytest.raises(RuntimeError, match="launch need"):
        packed.validate_capacity()


def test_cache_capacity_fail_fast_is_device_only_and_policy_is_immutable():
    source = inspect.getsource(ODMCache.commit_prefill)
    assert "_assert_async" in source
    assert "overflow_flag == 0" in source
    assert ".item(" not in source
    cache = ODMCache(1, 2, 8, "cpu")
    cache.configure_native_packing_policy(
        target_avg_bits=1.0, sink_tokens=4, tail_tokens=8,
    )
    # Repeating the same immutable model/press policy is idempotent.
    cache.configure_native_packing_policy(
        target_avg_bits=1.0, sink_tokens=4, tail_tokens=8,
    )
    with pytest.raises(RuntimeError, match="policy is immutable"):
        cache.configure_native_packing_policy(
            target_avg_bits=2.0, sink_tokens=4, tail_tokens=8,
        )
