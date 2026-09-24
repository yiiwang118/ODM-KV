"""Public API, analytical score, and real model/cache integration checks."""
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from kvquant import ODMPress, ODMScorer, make_press
from kvquant.attention_patch import reference_attention
from kvquant.runtime.generate import _truncate_at_first_all_eos
from kvquant.runtime.native_allocator import BatchedNativeAllocator
from kvquant.tq_backend import TurboQuantMSE


def tiny_model(family="llama"):
    cls, config = (LlamaForCausalLM, LlamaConfig) if family == "llama" else (Qwen3ForCausalLM, Qwen3Config)
    return cls(config(
        vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )).eval()


def test_joint_score_matches_direct_output_distortion_expression():
    torch.manual_seed(11)
    k, v = torch.randn(3, 17, 8), torch.randn(3, 17, 8)
    p = torch.softmax(torch.randn(3, 17), dim=-1)
    output = (p[..., None] * v).sum(dim=1)
    expected = (p + 0.01).square() * (
        v.square().sum(-1) + k.square().sum(-1) / 8 * (v - output[:, None]).square().sum(-1)
    )
    actual = ODMScorer().score_with_attn(k, v, p)
    torch.testing.assert_close(actual, expected)


def test_missing_prefill_statistics_is_an_error():
    with pytest.raises(ValueError, match="requires the attention module"):
        ODMScorer().score_prefill(torch.randn(2, 12, 8), torch.randn(2, 12, 8))


@pytest.mark.parametrize("config", [
    {"scorer": "random"}, {"scorer": "risk"}, {"ratios": [0.5, 0.5]},
    {"target_avg_bits": float("nan")}, {"mode": "unknown"}, {"layerwise": False},
    {"allow_decode_eviction": True}, {"mode": "native", "n_outlier_channels": 3},
])
def test_factory_rejects_alternative_methods_and_invalid_settings(config):
    with pytest.raises(ValueError):
        make_press(config)


def test_default_factory_uses_odm():
    press = make_press()
    assert isinstance(press, ODMPress)
    assert isinstance(press.scorer, ODMScorer)
    assert press.target_avg_bits == 2.0


def test_eviction_is_exact_for_opposing_gqa_queries():
    # No fake key can suppress both q and -q; a head-wise -inf mask can.
    q = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
    k, v = torch.randn(1, 1, 4, 2), torch.randn(1, 1, 4, 2)
    before = k.clone()
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True,
        masked_key_indices=(torch.tensor([0]), torch.tensor([0]), torch.tensor([1])))
    actual, _ = reference_attention(module, q, k, v, None)
    keep = torch.tensor([0, 2, 3])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k[:, :, keep].repeat_interleave(2, 1), v[:, :, keep].repeat_interleave(2, 1),
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(k, before)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("target", [1.0, 3.0, 16.0])
def test_tiny_model_prefill_and_multiple_decode_flushes(family, target):
    torch.manual_seed(5)
    model = tiny_model(family)
    old_impl = model.config._attn_implementation
    ids = torch.randint(3, 48, (2, 24))
    press = ODMPress(target_avg_bits=target, sink_tokens=2, buffer_size=4)
    with torch.inference_mode(), press(model):
        tokens = model.generate(ids, max_new_tokens=11, do_sample=False, eos_token_id=None)
        assert tokens.shape == (2, 35)
        assert press._prefill_hooks_fired == 2
        for state in press._states.values():
            assert state.seq_len == 34  # Last sampled token is not fed back.
            masked = state.masked_key_indices()
            if target < 2:
                assert masked is not None and masked[2].numel() > 0
            if masked is not None:
                assert (masked[2] >= 2).all()
                assert (masked[2] < 20).all()  # Sinks, tail, and new tokens survive.
            if target < 16:
                assert state._quantized_banks[2].size > 0
    assert model.config._attn_implementation == old_impl
    assert not press._states
    assert all(not block.self_attn._forward_hooks for block in model.model.layers)


def test_full_precision_budget_matches_uncompressed_model():
    torch.manual_seed(13)
    model = tiny_model()
    ids = torch.randint(3, 48, (1, 24))
    with torch.inference_mode():
        expected = model.generate(ids, max_new_tokens=11, do_sample=False, eos_token_id=None)
        with ODMPress(target_avg_bits=16, buffer_size=4)(model):
            actual = model.generate(ids, max_new_tokens=11, do_sample=False, eos_token_id=None)
    assert torch.equal(actual, expected)


def test_context_restores_model_after_exception():
    model, press = tiny_model(), ODMPress()
    original = model.config._attn_implementation
    with pytest.raises(RuntimeError, match="test failure"):
        with press(model):
            raise RuntimeError("test failure")
    assert model.config._attn_implementation == original
    assert not press._active
    assert all(not block.self_attn._forward_hooks for block in model.model.layers)


def test_reference_rejects_padded_scoring():
    model = tiny_model()
    ids = torch.ones(2, 12, dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    with torch.inference_mode(), ODMPress()(model), pytest.raises(ValueError, match="unpadded"):
        model(ids, attention_mask=mask)


def test_eval_rejects_native_cache_protocol():
    from benchmark.core.press_factory import make_press as make_eval_press
    with pytest.raises(ValueError, match="graph_generate"):
        make_eval_press({"mode": "native"})


def test_evaluation_rewinds_cache_between_questions():
    from benchmark.core.pipeline import KVPressTextGenerationRunner

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return ",".join(str(int(x)) for x in tokens)

    torch.manual_seed(41)
    model = tiny_model()
    model.generation_config.eos_token_id = None
    runner = KVPressTextGenerationRunner(model, Tokenizer())
    context = torch.randint(3, 48, (1, 24))
    question = torch.randint(3, 48, (1, 3))
    answers = runner._run(
        {"context_ids": context, "questions_ids": [question, question]},
        ODMPress(target_avg_bits=3, buffer_size=4), 11, None,
    )
    assert answers[0] == answers[1]


def test_reference_and_native_decode_allocate_same_budget():
    torch.manual_seed(19)
    press = ODMPress(target_avg_bits=3.0)
    scores = torch.rand(2, 5, 128)
    expected = torch.stack([press._allocate(s, 8, decode=True) for s in scores])
    allocator = BatchedNativeAllocator(
        bit_levels=tuple(b for b in press.bit_levels if b), target_avg_bits=3.0,
        epsilon=press._epsilon[8], sink_tokens=0, tail_tokens=0,
    )
    assert torch.equal(expected, allocator.allocate(scores))


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_mse_quantizer_roundtrip(bits):
    torch.manual_seed(37)
    x = torch.randn(32, 8)
    quantizer = TurboQuantMSE(8, bits, device=torch.device("cpu"), dtype=torch.float32)
    reconstructed = quantizer.dequantize(quantizer.quantize(x))
    assert reconstructed.shape == x.shape
    assert torch.isfinite(reconstructed).all()
    assert (x - reconstructed).square().mean() < x.square().mean() * 0.2


def test_generation_accepts_multiple_eos_ids():
    tokens = torch.tensor([[1, 8, 8, 8], [2, 3, 9, 9]])
    assert torch.equal(_truncate_at_first_all_eos(tokens, [8, 9]), tokens[:, :3])
