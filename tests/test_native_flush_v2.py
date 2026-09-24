"""Correctness, lifetime, and admission gates for native shared-v2 flush."""
from __future__ import annotations

import inspect

import pytest
import torch

from kvquant.tq_backend import _pack_indices
from kvquant.tq_triton import mse_nearest_centroid
from kvquant.runtime.kernels.fused_decode import fused_decode
from kvquant.runtime.kernels import native_flush_v2 as native_flush
from kvquant.runtime.kernels.native_flush_v2 import (
    NativeFlushWorkspace,
    append_decode_2bit_workspace,
    append_decode_mixed_workspace,
)
from kvquant.runtime.native_packing_v2 import (
    NativePackedKVV2,
    NativeSharedPackingCapacity,
    pack_native_prefill_shared_v2,
)
from kvquant.runtime.storage.cache import ODMCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native flush v2 requires CUDA and Triton",
)


def _build(*, batch=2, heads=2, tokens=11, dim=64, reserve=16, seed=42):
    torch.manual_seed(31000 + reserve + batch)
    keys = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    tags = torch.full((batch, heads, tokens), 2, device="cuda", dtype=torch.int32)
    tags[..., 0] = 16
    capacity = NativeSharedPackingCapacity.from_histogram(
        counts={2: batch * heads * (tokens - 1), 16: batch * heads},
        head_dim=dim,
        num_rows=batch * heads,
        decode_2bit_reserve_per_row=reserve,
    )
    return pack_native_prefill_shared_v2(
        keys, values, tags, capacity=capacity, seed=seed,
    )


def _workspace(packed, max_tokens: int):
    return NativeFlushWorkspace.allocate(
        rows=packed.num_rows,
        max_tokens=max_tokens,
        head_dim=packed.head_dim,
        device=packed.tags.device,
    )


@torch.no_grad()
def _reference_append(packed, keys, values) -> None:
    """Frozen pre-native implementation used only as the numerical oracle."""
    _, _, token_count, dim = keys.shape
    if token_count == 0:
        return
    bank = packed.bank(2)
    flat_k = keys.reshape(-1, dim).float()
    flat_v = values.reshape(-1, dim).float()
    norms_k = flat_k.norm(dim=-1)
    norms_v = flat_v.norm(dim=-1)
    rotated_k = (flat_k / (norms_k[:, None] + 1.0e-10)) @ packed.pi_k.T
    rotated_v = (flat_v / (norms_v[:, None] + 1.0e-10)) @ packed.pi_v.T
    codes_k = _pack_indices(mse_nearest_centroid(rotated_k, bank["cent_k"]), 2)
    codes_v = _pack_indices(mse_nearest_centroid(rotated_v, bank["cent_v"]), 2)
    rows = torch.arange(packed.num_rows, device=keys.device).repeat_interleave(token_count)
    within = torch.arange(token_count, device=keys.device).repeat(packed.num_rows)
    destination = bank["offset"][rows].long() + bank["seqlen"][rows].long() + within
    width = int(bank["packed_width"])
    byte = torch.arange(width, device=keys.device)
    code_destination = bank["code_base"].long() + destination[:, None] * width + byte
    bank["packed_k"].scatter_(0, code_destination.reshape(-1), codes_k.reshape(-1))
    bank["packed_v"].scatter_(0, code_destination.reshape(-1), codes_v.reshape(-1))
    norm_destination = bank["norm_base"].long() + destination
    bank["norms_k"].index_copy_(0, norm_destination, norms_k)
    bank["norms_v"].index_copy_(0, norm_destination, norms_v)
    bank["seqlen"].add_(token_count)
    packed.decode_tags[..., packed.decode_len : packed.decode_len + token_count].fill_(2)
    packed.decode_len += int(token_count)


def _arena_pointers(packed):
    return (
        packed.code_arena_k.data_ptr(),
        packed.code_arena_v.data_ptr(),
        packed.norm_arena_k.data_ptr(),
        packed.norm_arena_v.data_ptr(),
        packed.bank(2)["offset"].data_ptr(),
        packed.bank(2)["seqlen"].data_ptr(),
    )


def _workspace_pointers(workspace):
    return tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.normalized_k,
            workspace.normalized_v,
            workspace.rotated_k,
            workspace.rotated_v,
            workspace.ranks,
            workspace.counts,
            workspace.status,
        )
    )


def _build_mixed(*, batch=2, heads=2, tokens=11, dim=64, reserve=10, seed=42):
    torch.manual_seed(31500 + reserve + batch)
    keys = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    levels = torch.tensor([2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    tags = levels[torch.arange(batch * heads * tokens, device="cuda") % 5].reshape(
        batch, heads, tokens,
    )
    counts = {level: int((tags == level).sum().item()) for level in (2, 3, 4, 8, 16)}
    capacity = NativeSharedPackingCapacity.from_histogram(
        counts=counts,
        head_dim=dim,
        num_rows=batch * heads,
        decode_reserve_per_row_by_level=(reserve,) * 5,
    )
    packed = pack_native_prefill_shared_v2(
        keys, values, tags, capacity=capacity, seed=seed,
    )
    return packed, keys, values, tags


def _build_segmented(
    *, batch=2, heads=2, tokens=11, dim=64, reserve=12, seed=42,
    arena_target=8.0, buffer_size=6,
):
    torch.manual_seed(31600 + reserve + batch)
    keys = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    levels = torch.tensor([0, 2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    tags = levels[torch.arange(batch * heads * tokens, device="cuda") % 6].reshape(
        batch, heads, tokens,
    )
    counts = {level: int((tags == level).sum().item()) for level in levels.tolist()}
    packed = pack_native_prefill_shared_v2(
        keys,
        values,
        tags,
        capacity=NativeSharedPackingCapacity.from_histogram(
            counts=counts, head_dim=dim, num_rows=batch * heads,
        ),
        seed=seed,
    )
    packed.enable_segmented_decode(
        capacity_per_row=reserve,
        buffer_size=buffer_size,
        target_avg_bits=arena_target,
        bit_levels=(0, 2, 3, 4, 8, 16),
    )
    return packed, keys, values, tags


def test_one_hundred_noncontiguous_flushes_match_oracle_and_plateau():
    reserve = 100
    packed = _build(reserve=reserve)
    reference = _build(reserve=reserve)
    workspace = _workspace(packed, 1)
    torch.manual_seed(32001)
    # Transposed [B,H,T,D] keeps a genuine row/time stride at every one-token view.
    key_store = torch.randn(2, reserve, 2, 64, device="cuda", dtype=torch.bfloat16)
    value_store = torch.randn_like(key_store)
    keys = key_store.transpose(1, 2)
    values = value_store.transpose(1, 2)
    assert not keys.is_contiguous() and not values.is_contiguous()

    arena_ptrs = _arena_pointers(packed)
    work_ptrs = _workspace_pointers(workspace)
    allocated = []
    for token in range(reserve):
        key = keys[..., token : token + 1, :]
        value = values[..., token : token + 1, :]
        packed.append_decode_2bit(key, value, workspace=workspace)
        _reference_append(reference, key.contiguous(), value.contiguous())
        torch.cuda.synchronize()
        if token >= 1:  # first append warms any library-internal setup.
            allocated.append(torch.cuda.memory_allocated())

    assert packed.decode_len == reserve
    assert _arena_pointers(packed) == arena_ptrs
    assert _workspace_pointers(workspace) == work_ptrs
    assert max(allocated) - min(allocated) <= 1024 * 1024
    torch.testing.assert_close(packed.required_slots, reference.required_slots)
    torch.testing.assert_close(packed.code_arena_k, reference.code_arena_k, rtol=0, atol=0)
    torch.testing.assert_close(packed.code_arena_v, reference.code_arena_v, rtol=0, atol=0)
    torch.testing.assert_close(packed.norm_arena_k, reference.norm_arena_k, rtol=0, atol=2e-6)
    torch.testing.assert_close(packed.norm_arena_v, reference.norm_arena_v, rtol=0, atol=2e-6)

    snapshot = tuple(tensor.clone() for tensor in (
        packed.code_arena_k,
        packed.code_arena_v,
        packed.norm_arena_k,
        packed.norm_arena_v,
        packed.bank(2)["seqlen"],
    ))
    with pytest.raises(RuntimeError, match="reserve exceeded"):
        packed.append_decode_2bit(
            keys[..., :1, :], values[..., :1, :], workspace=workspace,
        )
    torch.cuda.synchronize()
    for got, expected in zip(
        (
            packed.code_arena_k,
            packed.code_arena_v,
            packed.norm_arena_k,
            packed.norm_arena_v,
            packed.bank(2)["seqlen"],
        ),
        snapshot,
    ):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("chunks", [(0, 3, 5), (8,)])
def test_empty_partial_and_full_strided_prefixes_match_reference(chunks):
    reserve = 8
    packed = _build(reserve=reserve)
    reference = _build(reserve=reserve)
    workspace = _workspace(packed, reserve)
    torch.manual_seed(33008)
    key_store = torch.randn(2, 2, reserve * 2, 64, device="cuda", dtype=torch.bfloat16)
    value_store = torch.randn_like(key_store)
    keys = key_store[..., ::2, :]
    values = value_store[..., 1::2, :]
    assert not keys.is_contiguous() and not values.is_contiguous()
    cursor = 0
    pointers = _arena_pointers(packed)
    for size in chunks:
        key = keys[..., cursor : cursor + size, :]
        value = values[..., cursor : cursor + size, :]
        packed.append_decode_2bit(key, value, workspace=workspace)
        _reference_append(reference, key.contiguous(), value.contiguous())
        cursor += size
    assert cursor == reserve
    assert _arena_pointers(packed) == pointers
    torch.cuda.synchronize()
    torch.testing.assert_close(packed.code_arena_k, reference.code_arena_k, rtol=0, atol=0)
    torch.testing.assert_close(packed.code_arena_v, reference.code_arena_v, rtol=0, atol=0)
    torch.testing.assert_close(packed.bank(2)["seqlen"], reference.bank(2)["seqlen"])


def test_cache_owns_one_workspace_reused_by_all_layers_and_reports_its_bytes():
    reserve, valid = 8, 5
    first = _build(reserve=reserve)
    second = _build(reserve=reserve, seed=49)
    cache = ODMCache(
        num_layers=2,
        num_heads_kv=2,
        head_dim=64,
        device=torch.device("cuda"),
    )
    cache.enable_graph_tail(reserve, num_rows=first.num_rows, max_decode_tokens=reserve)
    cache._native_decode_level = 2
    cache._native_packed = {0: first, 1: second}
    cache._ensure_native_flush_workspace(first, reserve)
    torch.manual_seed(34008)
    for layer in range(2):
        k = torch.randn(2, 2, reserve, 64, device="cuda", dtype=torch.bfloat16)
        cache._tail_buf_k[layer] = k
        cache._tail_buf_v[layer] = torch.randn_like(k)

    cache.flush_graph_tail(0, None, valid_tokens=valid)
    workspace = cache._native_flush_workspace
    assert workspace is not None
    pointers = _workspace_pointers(workspace)
    cache.flush_graph_tail(1, None, valid_tokens=valid)
    torch.cuda.synchronize()
    assert cache._native_flush_workspace is workspace
    assert _workspace_pointers(workspace) == pointers
    expected = (
        4 * first.num_rows * reserve * first.head_dim * 4
        + first.num_rows * reserve * 4
        + 5 * first.num_rows * 4
        + 2 * 4
    )
    assert workspace.reserved_bytes == expected
    assert first.decode_len == valid and second.decode_len == valid


def test_hot_path_source_has_no_dynamic_selection_sync_or_materialization():
    append_source = inspect.getsource(append_decode_2bit_workspace)
    method_source = inspect.getsource(NativePackedKVV2.append_decode_2bit)
    cache_source = inspect.getsource(ODMCache.flush_graph_tail)
    for source in (append_source, method_source):
        for forbidden in (
            "torch.cat", ".item(", ".tolist(", "materialize", "torch.nonzero",
            "index_select", "masked_select", ".contiguous(",
        ):
            assert forbidden not in source
    # The legacy state backend below this branch is allowed to compact; the
    # production native branch must forward the real ring strides unchanged.
    native_branch = cache_source.split("state = self._states", 1)[0]
    assert "buf_k[:, :, :pos].contiguous()" not in native_branch
    assert "self._tail_buf_v[layer_idx][:, :, :pos].contiguous()" not in native_branch


def test_mutable_ring_offsets_are_never_triton_specialization_keys():
    """One compiled kernel must serve every decode segment and ring offset."""
    expected = {
        "_normalize_kv_ring_kernel": {"decode_start"},
        "_encode_rotated_kv_2bit_kernel": {"decode_start"},
        "_rotate_encode_2bit_kernel": {"decode_start"},
        "_publish_decode_suffix_kernel": {"decode_start"},
        "_allocate_decode_segment_kernel": {"segment"},
        "_publish_decode_descriptors_kernel": {"decode_start", "segment"},
        "_scatter_segment_norm_kernel": {"segment"},
        "_scatter_segment_exact_kernel": {"segment"},
        "_prepare_segment_payload_kernel": {"segment"},
        "_encode_segment_mixed_kernel": {"segment"},
        "_encode_segment_3bit_kernel": {"segment"},
        "_publish_mixed_decode_kernel": {"decode_start"},
    }
    for name, dynamic_arguments in expected.items():
        kernel = getattr(native_flush, name)
        assert set(kernel.do_not_specialize) == dynamic_arguments
        assert set(kernel.do_not_specialize_on_alignment) == dynamic_arguments


def test_flush_raises_no_cuda_sync_debug_error():
    packed = _build(reserve=2)
    workspace = _workspace(packed, 1)
    keys = torch.randn(2, 2, 1, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    # Compile/warm on a separate state before making hidden synchronizations fatal.
    warm = _build(reserve=1)
    warm.append_decode_2bit(keys, values, workspace=_workspace(warm, 1))
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        packed.append_decode_2bit(keys, values, workspace=workspace)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.cuda.synchronize()


def test_mixed_bit_append_matches_repack_oracle_across_flushes_and_keeps_pointers():
    reserve = 10
    packed, keys, values, tags = _build_mixed(reserve=reserve)
    workspace = _workspace(packed, 6)
    pointers = _arena_pointers(packed) + tuple(
        tensor.data_ptr()
        for bank in (*packed.quant_banks, packed.exact)
        for tensor in (
            bank["offset"], bank["seqlen"], bank["stable_row_capacity"],
        )
    )
    torch.manual_seed(35010)
    decode_k = torch.randn(2, 2, reserve, 64, device="cuda", dtype=torch.bfloat16)
    decode_v = torch.randn_like(decode_k)
    base = torch.tensor([2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    decode_tags = torch.stack(
        [base.repeat(2).roll(row) for row in range(packed.num_rows)], dim=0,
    ).reshape(2, 2, reserve)

    packed.append_decode(
        decode_k[..., :4, :], decode_v[..., :4, :], decode_tags[..., :4],
        workspace=workspace,
    )
    packed.append_decode(
        decode_k[..., 4:, :], decode_v[..., 4:, :], decode_tags[..., 4:],
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert packed.decode_len == reserve
    assert torch.equal(packed.decode_tags[..., :reserve], decode_tags.to(torch.uint8))
    assert pointers == _arena_pointers(packed) + tuple(
        tensor.data_ptr()
        for bank in (*packed.quant_banks, packed.exact)
        for tensor in (
            bank["offset"], bank["seqlen"], bank["stable_row_capacity"],
        )
    )

    combined_k = torch.cat((keys, decode_k), dim=2)
    combined_v = torch.cat((values, decode_v), dim=2)
    combined_tags = torch.cat((tags, decode_tags), dim=2)
    counts = {
        level: int((combined_tags == level).sum().item())
        for level in (2, 3, 4, 8, 16)
    }
    reference = pack_native_prefill_shared_v2(
        combined_k,
        combined_v,
        combined_tags,
        capacity=NativeSharedPackingCapacity.from_histogram(
            counts=counts,
            head_dim=packed.head_dim,
            num_rows=packed.num_rows,
        ),
        seed=42,
    )
    got_k, got_v = packed.materialize()
    ref_k, ref_v = reference.materialize()
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)
    for level in (2, 3, 4, 8):
        torch.testing.assert_close(
            packed.bank(level)["seqlen"], reference.bank(level)["seqlen"],
            rtol=0, atol=0,
        )
    torch.testing.assert_close(
        packed.exact["seqlen"], reference.exact["seqlen"], rtol=0, atol=0,
    )


@pytest.mark.parametrize("level", [2, 3, 4, 8, 16])
def test_each_decode_bit_has_a_native_reserved_append_path(level):
    packed, _, _, _ = _build_mixed(reserve=3)
    workspace = _workspace(packed, 3)
    key = torch.randn(2, 2, 3, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    bits = torch.full(key.shape[:3], level, device="cuda", dtype=torch.int32)
    before = (
        packed.bank(level)["seqlen"].clone()
        if level != 16 else packed.exact["seqlen"].clone()
    )
    append_decode_mixed_workspace(packed, key, value, bits, workspace)
    torch.cuda.synchronize()
    after = packed.bank(level)["seqlen"] if level != 16 else packed.exact["seqlen"]
    torch.testing.assert_close(after, before + 3, rtol=0, atol=0)


def test_zero_bit_decode_advances_logical_history_without_payload_writes():
    packed, _, _, _ = _build_mixed(reserve=3)
    workspace = _workspace(packed, 3)
    key = torch.full(
        (2, 2, 3, 64), float("nan"), device="cuda", dtype=torch.bfloat16,
    )
    value = torch.full_like(key, float("nan"))
    bits = torch.zeros(key.shape[:3], device="cuda", dtype=torch.int32)
    payload_before = tuple(
        tensor.clone()
        for tensor in (
            packed.code_arena_k,
            packed.code_arena_v,
            packed.norm_arena_k,
            packed.norm_arena_v,
            packed.exact["keys"],
            packed.exact["values"],
        )
    )
    lengths_before = tuple(
        bank["seqlen"].clone() for bank in (*packed.quant_banks, packed.exact)
    )

    append_decode_mixed_workspace(packed, key, value, bits, workspace)
    torch.cuda.synchronize()

    assert packed.decode_len == 3
    assert torch.count_nonzero(packed.decode_tags[..., :3]) == 0
    for bank, before in zip((*packed.quant_banks, packed.exact), lengths_before):
        torch.testing.assert_close(bank["seqlen"], before, rtol=0, atol=0)
    for got, before in zip(
        (
            packed.code_arena_k,
            packed.code_arena_v,
            packed.norm_arena_k,
            packed.norm_arena_v,
            packed.exact["keys"],
            packed.exact["values"],
        ),
        payload_before,
    ):
        torch.testing.assert_close(got, before, rtol=0, atol=0, equal_nan=True)


def test_segmented_zero_bit_decode_publishes_no_payload_or_descriptor():
    packed, _, _, _ = _build_segmented(reserve=6)
    arena = packed.decode_arena
    workspace = _workspace(packed, 6)
    key = torch.full(
        (2, 2, 6, 64), float("nan"), device="cuda", dtype=torch.bfloat16,
    )
    value = torch.full_like(key, float("nan"))
    bits = torch.zeros(key.shape[:3], device="cuda", dtype=torch.int32)
    payload_before = tuple(
        tensor.clone()
        for tensor in (
            arena.code_arena_k,
            arena.code_arena_v,
            arena.norm_arena_k,
            arena.norm_arena_v,
            arena.exact_k,
            arena.exact_v,
        )
    )

    packed.append_decode(key, value, bits, workspace=workspace)
    torch.cuda.synchronize()

    assert packed.decode_len == 6
    assert arena.segment_count == 1
    assert torch.count_nonzero(packed.decode_tags[..., :6]) == 0
    assert torch.count_nonzero(arena.counts) == 0
    assert torch.count_nonzero(arena.descriptor_counts) == 0
    assert torch.count_nonzero(arena.bump) == 0
    for got, before in zip(
        (
            arena.code_arena_k,
            arena.code_arena_v,
            arena.norm_arena_k,
            arena.norm_arena_v,
            arena.exact_k,
            arena.exact_v,
        ),
        payload_before,
    ):
        torch.testing.assert_close(got, before, rtol=0, atol=0, equal_nan=True)


def test_segmented_decode_rejects_nonterminal_partial_flush_before_publication():
    packed, _, _, _ = _build_segmented(reserve=12)
    arena = packed.decode_arena
    workspace = _workspace(packed, 3)
    keys = torch.randn(2, 2, 3, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    bits = torch.full(keys.shape[:3], 2, device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="full-buffer flushes"):
        packed.append_decode(keys, values, bits, workspace=workspace)

    assert packed.decode_len == 0
    assert arena.segment_count == 0
    assert not arena.segment_starts
    assert not arena.segment_lengths
    assert torch.count_nonzero(arena.descriptor_counts) == 0


def test_segmented_decode_accepts_one_terminal_partial_flush():
    packed, _, _, _ = _build_segmented(reserve=9)
    arena = packed.decode_arena
    workspace = _workspace(packed, 6)
    keys = torch.randn(2, 2, 9, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    bits = torch.full(keys.shape[:3], 2, device="cuda", dtype=torch.int32)

    packed.append_decode(
        keys[..., :6, :], values[..., :6, :], bits[..., :6], workspace=workspace,
    )
    packed.append_decode(
        keys[..., 6:, :], values[..., 6:, :], bits[..., 6:], workspace=workspace,
    )
    torch.cuda.synchronize()

    assert packed.decode_len == 9
    assert arena.segment_count == 2
    assert arena.segment_lengths == [6, 3]
    assert torch.equal(
        arena.descriptor_counts[0],
        torch.full_like(arena.descriptor_counts[0], 9),
    )


@pytest.mark.parametrize("level", [2, 3, 4, 8, 16])
def test_each_segmented_bit_publishes_exact_dense_descriptor_stream(level):
    # Size this adversarial fixture for the all-exact case; production arenas
    # use their real target and separately test the budget proof.
    packed, keys, values, tags = _build_segmented(
        reserve=6, dim=128, arena_target=16.0,
    )
    arena = packed.decode_arena
    workspace = _workspace(packed, 6)
    torch.manual_seed(36600 + level)
    decode_k = torch.randn(
        2, 2, 6, packed.head_dim, device="cuda", dtype=torch.bfloat16,
    )
    decode_v = torch.randn_like(decode_k)
    decode_tags = torch.full(
        decode_k.shape[:3], level, device="cuda", dtype=torch.int32,
    )

    packed.append_decode(decode_k, decode_v, decode_tags, workspace=workspace)
    torch.cuda.synchronize()

    quant_count = 0 if level == 16 else 6
    exact_count = 6 if level == 16 else 0
    assert torch.equal(
        arena.descriptor_counts[0],
        torch.full_like(arena.descriptor_counts[0], quant_count),
    )
    assert torch.equal(
        arena.descriptor_counts[1],
        torch.full_like(arena.descriptor_counts[1], exact_count),
    )
    level_index = (2, 3, 4, 8, 16).index(level)
    expected = [
        (level_index if level != 16 else 0) * arena.buffer_size + rank
        for rank in range(6)
    ]
    for row in range(packed.num_rows):
        if level == 16:
            got = arena.descriptors[row, -6:].flip(0).cpu().tolist()
        else:
            got = arena.descriptors[row, :6].cpu().tolist()
        assert got == expected

    combined_tags = torch.cat((tags, decode_tags), dim=2)
    reference = pack_native_prefill_shared_v2(
        torch.cat((keys, decode_k), dim=2),
        torch.cat((values, decode_v), dim=2),
        combined_tags,
        capacity=NativeSharedPackingCapacity.from_histogram(
            counts={
                bit: int((combined_tags == bit).sum().item())
                for bit in (0, 2, 3, 4, 8, 16)
            },
            head_dim=packed.head_dim,
            num_rows=packed.num_rows,
        ),
        seed=42,
    )
    got_k, got_v = packed.materialize(dtype=torch.float32)
    ref_k, ref_v = reference.materialize(dtype=torch.float32)
    # Repacking a larger M dimension can move a centroid tie at TF32 rounding
    # boundaries even though the per-token transform is identical.  The byte-
    # exact same-shape path is covered below; this full-repack oracle uses the
    # established one-quantization-step materialization tolerance.
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=1 / 64)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=1 / 64)
    query = torch.randn(
        packed.num_rows, 2, packed.head_dim,
        device="cuda", dtype=torch.bfloat16,
    )

    def exact_meta(state):
        return {
            "offset": state.exact["offset"],
            "seqlen": state.exact["seqlen"],
            "T_max": state.exact["T_max"],
        }

    got_attention = fused_decode(
        query,
        list(packed.quant_banks),
        packed.exact["keys"],
        packed.exact["values"],
        exact_meta(packed),
        packed.pi_k,
        packed.pi_v,
        packed.head_dim ** -0.5,
        decode_arena=arena,
    )
    ref_attention = fused_decode(
        query,
        list(reference.quant_banks),
        reference.exact["keys"],
        reference.exact["values"],
        exact_meta(reference),
        reference.pi_k,
        reference.pi_v,
        reference.head_dim ** -0.5,
    )
    torch.testing.assert_close(
        got_attention, ref_attention, rtol=1e-2, atol=1 / 256,
    )


def test_tag_aware_fused_prepare_is_byte_exact_to_three_kernel_oracle(monkeypatch):
    optimized, _, _, _ = _build_segmented(reserve=6)
    oracle, _, _, _ = _build_segmented(reserve=6)
    optimized_workspace = _workspace(optimized, 6)
    oracle_workspace = _workspace(oracle, 6)
    torch.manual_seed(36606)
    keys = torch.randn(2, 2, 6, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    levels = torch.tensor([0, 2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    tags = torch.stack(
        [levels.roll(row) for row in range(optimized.num_rows)], dim=0,
    ).reshape(2, 2, 6)

    monkeypatch.setenv("R2_FLUSH_FUSED_PREPARE", "1")
    optimized.append_decode(keys, values, tags, workspace=optimized_workspace)
    monkeypatch.setenv("R2_FLUSH_FUSED_PREPARE", "0")
    oracle.append_decode(keys, values, tags, workspace=oracle_workspace)
    torch.cuda.synchronize()

    opt_arena = optimized.decode_arena
    ref_arena = oracle.decode_arena
    torch.testing.assert_close(opt_arena.bump, ref_arena.bump, rtol=0, atol=0)
    torch.testing.assert_close(opt_arena.counts, ref_arena.counts, rtol=0, atol=0)
    torch.testing.assert_close(
        opt_arena.descriptor_counts, ref_arena.descriptor_counts, rtol=0, atol=0,
    )
    for row in range(optimized.num_rows):
        quant_count = int(opt_arena.descriptor_counts[0, row].item())
        exact_count = int(opt_arena.descriptor_counts[1, row].item())
        torch.testing.assert_close(
            opt_arena.descriptors[row, :quant_count],
            ref_arena.descriptors[row, :quant_count],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            opt_arena.descriptors[row, -exact_count:],
            ref_arena.descriptors[row, -exact_count:],
            rtol=0,
            atol=0,
        )
    code_bytes, norm_slots, exact_slots = opt_arena.bump.cpu().tolist()
    # Codes and 16-bit slots share one pool: codes occupy the low bytes, exact
    # rows the top ``exact_slots``.  Compare the live regions, not [0:n).
    exact_lo = opt_arena.exact_capacity - exact_slots
    for got, expected in (
        (opt_arena.code_arena_k[:code_bytes], ref_arena.code_arena_k[:code_bytes]),
        (opt_arena.code_arena_v[:code_bytes], ref_arena.code_arena_v[:code_bytes]),
        (opt_arena.norm_arena_k[:norm_slots], ref_arena.norm_arena_k[:norm_slots]),
        (opt_arena.norm_arena_v[:norm_slots], ref_arena.norm_arena_v[:norm_slots]),
        (opt_arena.exact_k[exact_lo:], ref_arena.exact_k[exact_lo:]),
        (opt_arena.exact_v[exact_lo:], ref_arena.exact_v[exact_lo:]),
    ):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_unified_mixed_encoder_is_byte_exact_across_two_segments(monkeypatch):
    optimized, _, _, _ = _build_segmented(
        reserve=12, dim=128, arena_target=16.0, buffer_size=6,
    )
    oracle, _, _, _ = _build_segmented(
        reserve=12, dim=128, arena_target=16.0, buffer_size=6,
    )
    optimized_workspace = _workspace(optimized, 6)
    oracle_workspace = _workspace(oracle, 6)
    torch.manual_seed(36611)
    keys = torch.randn(2, 2, 12, 128, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    levels = torch.tensor([0, 2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    tags = torch.stack(
        [levels.repeat(2).roll(row) for row in range(optimized.num_rows)], dim=0,
    ).reshape(2, 2, 12)

    for start in (0, 6):
        stop = start + 6
        monkeypatch.setenv("R2_FLUSH_UNIFIED_ENCODE", "1")
        optimized.append_decode(
            keys[..., start:stop, :],
            values[..., start:stop, :],
            tags[..., start:stop],
            workspace=optimized_workspace,
        )
        monkeypatch.setenv("R2_FLUSH_UNIFIED_ENCODE", "0")
        oracle.append_decode(
            keys[..., start:stop, :],
            values[..., start:stop, :],
            tags[..., start:stop],
            workspace=oracle_workspace,
        )
    torch.cuda.synchronize()

    got = optimized.decode_arena
    expected = oracle.decode_arena
    assert got.segment_count == expected.segment_count == 2
    torch.testing.assert_close(got.bump, expected.bump, rtol=0, atol=0)
    torch.testing.assert_close(got.counts, expected.counts, rtol=0, atol=0)
    torch.testing.assert_close(got.code_row_base, expected.code_row_base, rtol=0, atol=0)
    torch.testing.assert_close(got.norm_row_base, expected.norm_row_base, rtol=0, atol=0)
    torch.testing.assert_close(got.exact_row_base, expected.exact_row_base, rtol=0, atol=0)
    torch.testing.assert_close(
        got.descriptor_counts, expected.descriptor_counts, rtol=0, atol=0,
    )
    torch.testing.assert_close(
        optimized.decode_tags, oracle.decode_tags, rtol=0, atol=0,
    )
    code_bytes, norm_slots, exact_slots = got.bump.cpu().tolist()
    for actual, reference in (
        (got.code_arena_k[:code_bytes], expected.code_arena_k[:code_bytes]),
        (got.code_arena_v[:code_bytes], expected.code_arena_v[:code_bytes]),
        (got.norm_arena_k[:norm_slots], expected.norm_arena_k[:norm_slots]),
        (got.norm_arena_v[:norm_slots], expected.norm_arena_v[:norm_slots]),
        (got.exact_k[got.exact_capacity - exact_slots:],
         expected.exact_k[expected.exact_capacity - exact_slots:]),
        (got.exact_v[got.exact_capacity - exact_slots:],
         expected.exact_v[expected.exact_capacity - exact_slots:]),
    ):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for row in range(optimized.num_rows):
        quant_count = int(got.descriptor_counts[0, row].item())
        exact_count = int(got.descriptor_counts[1, row].item())
        torch.testing.assert_close(
            got.descriptors[row, :quant_count],
            expected.descriptors[row, :quant_count],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            got.descriptors[row, -exact_count:],
            expected.descriptors[row, -exact_count:],
            rtol=0,
            atol=0,
        )


def test_segmented_shared_arena_matches_full_repack_across_two_flushes(monkeypatch):
    packed, keys, values, tags = _build_segmented()
    workspace = _workspace(packed, 6)
    torch.manual_seed(36612)
    decode_k = torch.randn(2, 2, 12, 64, device="cuda", dtype=torch.bfloat16)
    decode_v = torch.randn_like(decode_k)
    levels = torch.tensor([0, 2, 3, 4, 8, 16], device="cuda", dtype=torch.int32)
    decode_tags = torch.stack(
        [levels.repeat(2).roll(row) for row in range(packed.num_rows)], dim=0,
    ).reshape(2, 2, 12)
    arena = packed.decode_arena
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            arena.code_arena_k, arena.code_arena_v,
            arena.norm_arena_k, arena.norm_arena_v,
            arena.exact_k, arena.exact_v,
            arena.code_row_base, arena.norm_row_base,
            arena.exact_row_base, arena.counts,
            arena.descriptors,
            arena.descriptor_counts, arena.unified_codebook,
        )
    )

    packed.append_decode(
        decode_k[..., :6, :], decode_v[..., :6, :], decode_tags[..., :6],
        workspace=workspace,
    )
    packed.append_decode(
        decode_k[..., 6:, :], decode_v[..., 6:, :], decode_tags[..., 6:],
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert arena.segment_count == 2
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            arena.code_arena_k, arena.code_arena_v,
            arena.norm_arena_k, arena.norm_arena_v,
            arena.exact_k, arena.exact_v,
            arena.code_row_base, arena.norm_row_base,
            arena.exact_row_base, arena.counts,
            arena.descriptors,
            arena.descriptor_counts, arena.unified_codebook,
        )
    )
    assert torch.equal(packed.decode_tags[..., :12], decode_tags.to(torch.uint8))

    flat_tags = decode_tags.reshape(packed.num_rows, 12)
    descriptor_counts = arena.descriptor_counts.cpu()
    for row in range(packed.num_rows):
        expected_quant = []
        expected_exact = []
        for segment in range(2):
            segment_tags = flat_tags[row, segment * 6 : (segment + 1) * 6]
            for level_index, level in enumerate((2, 3, 4, 8, 16)):
                count = int((segment_tags == level).sum().item())
                encoded = [
                    (
                        segment * (4 if level != 16 else 1)
                        + (level_index if level != 16 else 0)
                    ) * arena.buffer_size + rank
                    for rank in range(count)
                ]
                (expected_exact if level == 16 else expected_quant).extend(encoded)
        assert descriptor_counts[0, row].item() == len(expected_quant)
        assert descriptor_counts[1, row].item() == len(expected_exact)
        assert (
            arena.descriptors[row, : len(expected_quant)].cpu().tolist()
            == expected_quant
        )
        assert (
            arena.descriptors[
                row, arena.capacity_per_row - len(expected_exact) :
            ].flip(0).cpu().tolist()
            == expected_exact
        )

    combined_k = torch.cat((keys, decode_k), dim=2)
    combined_v = torch.cat((values, decode_v), dim=2)
    combined_tags = torch.cat((tags, decode_tags), dim=2)
    histogram = {
        level: int((combined_tags == level).sum().item())
        for level in (0, 2, 3, 4, 8, 16)
    }
    reference = pack_native_prefill_shared_v2(
        combined_k,
        combined_v,
        combined_tags,
        capacity=NativeSharedPackingCapacity.from_histogram(
            counts=histogram, head_dim=64, num_rows=packed.num_rows,
        ),
        seed=42,
    )
    got_k, got_v = packed.materialize(dtype=torch.float32)
    ref_k, ref_v = reference.materialize(dtype=torch.float32)
    torch.testing.assert_close(got_k, ref_k, rtol=0, atol=2e-6)
    torch.testing.assert_close(got_v, ref_v, rtol=0, atol=2e-6)

    def exact_meta(state):
        return {
            "offset": state.exact["offset"],
            "seqlen": state.exact["seqlen"],
            "T_max": state.exact["T_max"],
        }

    query = torch.randn(
        packed.num_rows, 2, packed.head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    got_attention = fused_decode(
        query,
        list(packed.quant_banks),
        packed.exact["keys"],
        packed.exact["values"],
        exact_meta(packed),
        packed.pi_k,
        packed.pi_v,
        packed.head_dim ** -0.5,
        decode_arena=arena,
    )
    monkeypatch.setenv("R2_DECODE_DESCRIPTOR_STREAMS", "0")
    segmented_attention = fused_decode(
        query,
        list(packed.quant_banks),
        packed.exact["keys"],
        packed.exact["values"],
        exact_meta(packed),
        packed.pi_k,
        packed.pi_v,
        packed.head_dim ** -0.5,
        decode_arena=arena,
    )
    ref_attention = fused_decode(
        query,
        list(reference.quant_banks),
        reference.exact["keys"],
        reference.exact["values"],
        exact_meta(reference),
        reference.pi_k,
        reference.pi_v,
        reference.head_dim ** -0.5,
    )
    # Segment boundaries alter only the fp32 online-softmax reduction order;
    # storage above is exactly identical and BF16 output may move by one ULP.
    torch.testing.assert_close(
        got_attention, ref_attention, rtol=1e-2, atol=1 / 256,
    )
    torch.testing.assert_close(
        got_attention, segmented_attention, rtol=1e-2, atol=1 / 256,
    )

    # Aggregate capacity is materially below five independent full suffixes.
    independent_payload = packed.num_rows * 12 * (
        2 * sum((16, 24, 32, 64)) + 2 * 4 * 4 + 2 * 64 * 2
    )
    assert arena.reserved_bytes < independent_payload


def test_segmented_descriptor_storage_is_in_byte_exact_reserved_accounting():
    packed, _, _, _ = _build_segmented(reserve=12)
    arena = packed.decode_arena
    stats = packed.reserved_storage_stats()
    descriptor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            arena.descriptors,
            arena.descriptor_counts,
        )
    )
    unified_bytes = (
        arena.unified_codebook.numel() * arena.unified_codebook.element_size()
    )
    assert stats["metadata_bytes"] >= descriptor_bytes
    assert stats["shared_codebook_bytes"] >= unified_bytes
    assert stats["total_reserved_bytes"] == (
        stats["reserved_payload_bytes"]
        + stats["metadata_bytes"]
        + stats["shared_rotation_codebook_bytes"]
    )
