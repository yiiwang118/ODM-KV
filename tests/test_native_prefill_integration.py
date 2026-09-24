"""Integration gates for the native Realsys2 prefill path.

These tests deliberately cover contracts at the Transformers boundary rather
than only testing the scorer's algebra in isolation:

* the Realsys2 prefill wrapper must execute the same FlashAttention-2 path as
  Hugging Face, including a two-dimensional padding mask;
* inverse RoPE must remain correct when the rotary implementation applies an
  attention scaling factor (the transform is then scaled-orthogonal, not
  strictly orthogonal);
* current Llama/Qwen3 attention layers do not own ``rotary_emb`` -- it lives on
  the model -- so a native scorer must receive the root rotary module explicitly;
* the custom attention backend must register FlashAttention's mask factory as
  well as its attention function.

No pretrained weights are downloaded.  The real-model ownership checks use
tiny randomly initialized configs, and the numerical FA2 checks are skipped on
hosts without CUDA + flash-attn.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kvquant.attention_utils import (
    _apply_rotary_pos_emb_q,
    _invert_rotary_pos_emb_q,
)
from kvquant.runtime.attn_impl import (
    _flashattention2_prefill,
    register_odmkv_attention,
)


try:
    import flash_attn  # noqa: F401

    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False


_CUDA_FA2 = torch.cuda.is_available() and _HAS_FLASH_ATTN
cuda_fa2_only = pytest.mark.skipif(
    not _CUDA_FA2,
    reason="native prefill output parity requires CUDA + flash-attn",
)


class _AttentionShell(nn.Module):
    """Small module exposing the fields used by HF's FA2 integration."""

    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.layer_idx = 0
        self.is_causal = True
        self.sliding_window = None


def _qkv(device: torch.device):
    torch.manual_seed(123)
    batch, seq, h_q, h_kv, dim = 2, 64, 4, 2, 64
    q = torch.randn(batch, h_q, seq, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, h_kv, seq, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    return q, k, v


@cuda_fa2_only
@pytest.mark.parametrize("with_padding", [False, True])
def test_native_prefill_output_matches_hf_flashattention2(with_padding):
    """The model output must come from exactly the same FA2 integration.

    The padded case forces HF's varlen/unpad path, so this catches wrappers that
    call only ``flash_attn_func(causal=True)`` and silently ignore request masks.
    """
    from transformers.integrations.flash_attention import flash_attention_forward

    device = torch.device("cuda")
    q, k, v = _qkv(device)
    mask = None
    if with_padding:
        mask = torch.ones(q.shape[0], q.shape[2], dtype=torch.bool, device=device)
        mask[0, :11] = False
        mask[1, :3] = False

    scaling = 1.0 / math.sqrt(q.shape[-1])
    official_module = _AttentionShell("flash_attention_2").to(device)
    native_module = _AttentionShell("odmkv_pertok").to(device)

    expected, _ = flash_attention_forward(
        official_module,
        q,
        k,
        v,
        mask,
        dropout=0.0,
        scaling=scaling,
    )
    actual, _ = _flashattention2_prefill(
        native_module,
        q,
        k,
        v,
        mask,
        scaling=scaling,
        dropout=0.0,
    )

    assert actual.shape == expected.shape == (
        q.shape[0], q.shape[2], q.shape[1], q.shape[-1]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_native_prefill_forwards_same_mask_contract_as_hf(monkeypatch):
    """Compare wrapper arguments without requiring a GPU installation.

    For the supported full-causal contract, both wrappers must pass the original
    2-D padding mask and the same causal, length, scaling, dropout, and softcap
    policy to HF's shared dispatcher. Layout is also checked: the private
    dispatcher receives ``[B,T,H,D]``.
    """
    import transformers.integrations.flash_attention as hf_flash
    import transformers.modeling_flash_attention_utils as hf_utils

    calls: dict[str, dict] = {}

    def make_fake(label):
        def fake(q, k, v, attention_mask, **kwargs):
            calls[label] = {
                "q_shape": tuple(q.shape),
                "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "mask": attention_mask,
                **kwargs,
            }
            return q

        return fake

    official_fake = make_fake("official")
    native_fake = make_fake("native")
    monkeypatch.setattr(hf_flash, "_flash_attention_forward", official_fake)
    monkeypatch.setattr(hf_utils, "_flash_attention_forward", native_fake)

    q = torch.randn(2, 4, 9, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 2, 9, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    scaling = 1.0 / math.sqrt(q.shape[-1])

    from transformers.integrations.flash_attention import flash_attention_forward

    flash_attention_forward(
        _AttentionShell("flash_attention_2"),
        q,
        k,
        v,
        mask,
        dropout=0.125,
        scaling=scaling,
        sliding_window=None,
        softcap=3.0,
    )
    _flashattention2_prefill(
        _AttentionShell("odmkv_pertok"),
        q,
        k,
        v,
        mask,
        scaling=scaling,
        dropout=0.125,
        sliding_window=None,
        softcap=3.0,
    )

    official = calls["official"]
    native = calls["native"]
    assert official["mask"] is mask
    assert native["mask"] is mask
    assert native["q_shape"] == official["q_shape"] == (2, 9, 4, 8)
    assert native["k_shape"] == official["k_shape"] == (2, 9, 2, 8)
    assert native["v_shape"] == official["v_shape"] == (2, 9, 2, 8)
    for key in (
        "query_length",
        "is_causal",
        "dropout",
        "softmax_scale",
        "sliding_window",
        "softcap",
        "use_top_left_mask",
        "target_dtype",
        "attn_implementation",
        "layer_idx",
    ):
        assert native[key] == official[key], key


def test_native_prefill_rejects_sliding_window_before_dispatch(monkeypatch):
    """Decode has no logical-position filter, so windowed prefill must fail."""
    import transformers.modeling_flash_attention_utils as hf_utils

    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("windowed request reached FA2 dispatcher")

    monkeypatch.setattr(hf_utils, "_flash_attention_forward", forbidden)
    q = torch.randn(1, 4, 9, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 9, 8, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    with pytest.raises(RuntimeError, match="sliding-window"):
        _flashattention2_prefill(
            _AttentionShell("odmkv_pertok"),
            q,
            k,
            v,
            None,
            scaling=1.0 / math.sqrt(q.shape[-1]),
            dropout=0.0,
            sliding_window=7,
        )
    assert not called


@pytest.mark.parametrize("attention_scaling", [0.625, 1.75])
def test_bf16_inverse_rope_handles_attention_scaling(attention_scaling):
    """Scaled RoPE needs division by ``cos^2 + sin^2`` on inversion."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    batch, heads, seq, dim = 2, 4, 37, 64
    q = torch.randn(batch, heads, seq, dim, device=device, dtype=torch.bfloat16)

    half = dim // 2
    theta_half = torch.randn(batch, seq, half, device=device, dtype=torch.float32)
    theta = torch.cat((theta_half, theta_half), dim=-1)
    cos = (theta.cos() * attention_scaling).to(torch.bfloat16)
    sin = (theta.sin() * attention_scaling).to(torch.bfloat16)

    q_rotated = _apply_rotary_pos_emb_q(q, cos, sin)
    recovered = _invert_rotary_pos_emb_q(q_rotated, cos, sin)

    # The implementation may intentionally keep the inverse in fp32 to avoid a
    # second BF16 rounding/division error; the contract is accuracy from BF16
    # attention inputs, not a forced output dtype.
    assert recovered.dtype in (torch.bfloat16, torch.float32)
    assert torch.isfinite(recovered).all()
    torch.testing.assert_close(
        recovered.float(), q.float(), rtol=3e-2, atol=4e-2,
    )


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_rotary_embedding_is_owned_by_root_model_not_attention_layer(family):
    """Use real HF classes with tiny configs; no model download is involved."""
    if family == "llama":
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        config = LlamaConfig(
            vocab_size=257,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
        )
        model = LlamaModel(config)
    else:
        from transformers import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        config = Qwen3Config(
            vocab_size=257,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
            use_sliding_window=False,
            layer_types=["full_attention"],
        )
        model = Qwen3Model(config)

    attention = model.layers[0].self_attn
    assert hasattr(model, "rotary_emb")
    assert not hasattr(attention, "rotary_emb"), (
        "The native scorer must be passed the root model's rotary embedding; "
        "it must not assume that the attention layer owns one."
    )

    positions = torch.arange(13).unsqueeze(0)
    probe = torch.zeros(1, 13, config.hidden_size)
    cos, sin = model.rotary_emb(probe, positions)
    assert cos.shape == sin.shape == (1, 13, config.head_dim)


def test_realsys2_registers_flashattention_mask_factory():
    """Attention and mask dispatch are separate registries in Transformers."""
    from transformers.masking_utils import (
        ALL_MASK_ATTENTION_FUNCTIONS,
        flash_attention_mask,
    )

    register_odmkv_attention()
    mapping = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    assert "odmkv_pertok" in mapping
    assert mapping["odmkv_pertok"] is flash_attention_mask

    mask_factory = ALL_MASK_ATTENTION_FUNCTIONS["odmkv_pertok"]
    cache_position = torch.arange(6)
    all_valid = torch.ones(2, 6, dtype=torch.bool)
    padded = all_valid.clone()
    padded[0, :2] = False

    # Same optimization/contract as FA2: all-valid -> no explicit mask, while
    # a real padding pattern remains two-dimensional for the varlen unpad path.
    assert mask_factory(
        batch_size=2,
        q_length=6,
        cache_position=cache_position,
        kv_length=6,
        attention_mask=all_valid,
    ) is None
    got = mask_factory(
        batch_size=2,
        q_length=6,
        cache_position=cache_position,
        kv_length=6,
        attention_mask=padded,
    )
    assert got.shape == (2, 6)
    assert torch.equal(got, padded)
