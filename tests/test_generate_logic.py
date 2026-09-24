import torch
import pytest
from types import SimpleNamespace
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from kvquant.runtime.generate import (
    _canonicalize_prefill_attention_mask,
    _truncate_at_first_all_eos,
)
from kvquant.runtime.graph_decode import GraphDecoder
from kvquant.runtime.storage.cache import _ODMLayer
from kvquant.runtime.storage.cache import ODMCache


def test_realsys2_implements_generic_cache_without_dynamic_kv_storage():
    cache = ODMCache(1, 2, 8, "cpu")
    assert isinstance(cache, Cache)
    assert not isinstance(cache, DynamicCache)
    assert len(cache.layers) == 1
    assert not isinstance(cache.layers[0], DynamicLayer)
    assert cache.layers[0].keys is None
    assert cache.layers[0].values is None
    with pytest.raises(RuntimeError, match="metadata-layer update is prohibited"):
        cache.layers[0].update(
            torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8),
        )


def test_truncate_speculative_pinned_eos_suffix():
    tokens = torch.tensor([
        [1, 9, 9, 9, 9],
        [2, 3, 9, 9, 9],
    ])
    got = _truncate_at_first_all_eos(tokens, eos_token_id=9)
    assert torch.equal(got, tokens[:, :3])


def test_no_all_eos_column_keeps_full_output():
    tokens = torch.tensor([
        [1, 9, 9],
        [2, 3, 4],
    ])
    got = _truncate_at_first_all_eos(tokens, eos_token_id=9)
    assert torch.equal(got, tokens)


def test_graph_mask_sizes_use_host_tail_mirror_without_device_read():
    class Parent:
        _graph_tail = True
        _graph_mask_tail_pos = 7
        _states = {0: SimpleNamespace(seq_len=64)}

        def get_seq_length(self, _layer_idx):
            raise AssertionError("graph mask construction must not read tail_pos.item()")

    layer = _ODMLayer(Parent(), 0)
    assert layer.get_mask_sizes(torch.tensor([71])) == (72, 0)


def test_native_graph_rejects_non_left_padding_before_model_execution():
    ids = torch.ones(2, 5, dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[0, -1] = 0
    with pytest.raises(ValueError, match="strict left padding"):
        _canonicalize_prefill_attention_mask(ids, mask)


def test_all_valid_mask_is_canonicalized_to_none():
    ids = torch.ones(2, 5, dtype=torch.long)
    assert _canonicalize_prefill_attention_mask(ids, torch.ones_like(ids)) is None


def test_empty_graph_tail_descriptor_keeps_four_item_abi():
    cache = ODMCache(1, 2, 8, "cpu")
    cache.enable_graph_tail(8, num_rows=2, max_decode_tokens=8)
    assert cache.get_tail_for_decode(0, 2) == (None, None, None, None)


def test_direct_graph_decoder_never_inflates_capacity_after_prefill_layout():
    cache = SimpleNamespace(
        _graph_buf_size=8,
        num_layers=1,
        _graph_decode_capacity=4,
        _native_packed={0: object()},
        _states={},
        _committed={0: True},
        _prefill_kv={},
    )
    decoder = GraphDecoder(None, cache, 1, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="reserve is already fixed by prefill"):
        decoder.generate(
            torch.zeros(1, 1, dtype=torch.long),
            start_pos=8,
            n_new=9,
            lm_head=None,
            bits_fn=None,
        )
    assert cache._graph_decode_capacity == 4
