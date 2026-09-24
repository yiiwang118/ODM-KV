"""Correctness gates for Q/K/V-reuse native prefill scoring."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kvquant.attention_utils import _apply_avg_rope, _apply_rotary_pos_emb_q
from kvquant.attention_utils import _invert_rotary_pos_emb_q
from kvquant.scorer import ODMScorer
from kvquant.runtime.kernels.prefill_score import (
    gqa_query_key_logits,
    inverse_rope_suffix_mean,
    joint_mse_score,
    weighted_value_sum,
)


class _CountingLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return super().forward(x)


class _Rotary(nn.Module):
    def __init__(self, dim, attention_scaling=1.0):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.attention_scaling = float(attention_scaling)

    def forward(self, x, position_ids):
        freq = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freq, freq), dim=-1)
        scale = self.attention_scaling
        return (emb.cos() * scale).to(x.dtype), (emb.sin() * scale).to(x.dtype)


class _Attention(nn.Module):
    def __init__(self, hidden, h_q, dim):
        super().__init__()
        self.num_heads = h_q
        self.head_dim = dim
        self.q_proj = _CountingLinear(hidden, h_q * dim, bias=False)
        self.rotary_emb = _Rotary(dim)
        self.config = SimpleNamespace(num_attention_heads=h_q, head_dim=dim)


@pytest.mark.parametrize("grain", ["global"])
@pytest.mark.parametrize("attention_scaling", [1.0, 1.37])
def test_native_batched_score_matches_legacy_qproj_path(grain, attention_scaling):
    torch.manual_seed(7)
    batch, seq, hidden, h_q, h_kv, dim = 3, 19, 32, 4, 2, 8
    module = _Attention(hidden, h_q, dim)
    module.rotary_emb = _Rotary(dim, attention_scaling=attention_scaling)
    hidden_states = torch.randn(batch, seq, hidden)
    keys = torch.randn(batch, h_kv, seq, dim)
    values = torch.randn_like(keys)

    q_pre = module.q_proj(hidden_states).view(batch, seq, h_q, dim).transpose(1, 2)
    positions = torch.arange(seq).unsqueeze(0)
    cos, sin = module.rotary_emb(q_pre, positions)
    q_post = _apply_rotary_pos_emb_q(q_pre, cos, sin)

    scorer = ODMScorer(
        n_future_positions=23, n_sink=4, epsilon=1e-2,
    )
    scorer.normalize_grain = grain
    expected = torch.stack([
        scorer.score_prefill(
            keys[b], values[b], 0,
            module=module, hidden_states=hidden_states[b:b + 1],
        )
        for b in range(batch)
    ])

    module.q_proj.calls = 0
    actual = scorer.score_prefill_from_qkv(
        q_post, keys, values, 0,
        module=module, cos=cos, sin=sin,
    )

    # The native path must consume attention's Q, never replay q_proj.
    assert module.q_proj.calls == 0
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_native_score_rejects_incompatible_qkv_shapes():
    scorer = ODMScorer(n_future_positions=4, n_sink=1)
    module = _Attention(hidden=8, h_q=2, dim=4)
    q = torch.randn(2, 2, 5, 4)
    k = torch.randn(2, 1, 5, 4)
    v = torch.randn(2, 1, 4, 4)
    cos, sin = module.rotary_emb(q, torch.arange(5).unsqueeze(0))
    with pytest.raises(ValueError, match="incompatible native Q/K/V"):
        scorer.score_prefill_from_qkv(
            q, k, v, 0, module=module, cos=cos, sin=sin,
        )


def test_direct_average_rope_mean_matches_rotation_matrix_definition():
    """The allocation scorer must not build [future,D,D] just to rotate mu."""
    torch.manual_seed(31)
    batch, heads, dim = 3, 4, 8
    q_len, n_future = 19, 23
    module = _Attention(hidden=32, h_q=heads, dim=dim)
    mu = torch.randn(batch, heads, dim)

    actual, cov = _apply_avg_rope(module, mu, None, q_len, n_future)
    assert cov is None

    pos = torch.arange(q_len, q_len + n_future).unsqueeze(0)
    cos, sin = module.rotary_emb(mu, pos)
    half = dim // 2
    eye = torch.eye(dim)
    perm = torch.zeros(dim, dim)
    perm[half:, :half] = torch.eye(half)
    perm[:half, half:] = -torch.eye(half)
    rotations = cos[0].unsqueeze(-1) * eye + sin[0].unsqueeze(-1) * perm
    expected = torch.matmul(mu, rotations.mean(0).T)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native reduction requires CUDA")
@pytest.mark.parametrize("seq", [5, 129, 2048])
def test_native_inverse_rope_mean_matches_materialized_oracle(seq):
    torch.manual_seed(47 + seq)
    device = torch.device("cuda")
    batch, heads, dim, n_sink = 2, 4, 64, 4
    q_pre = torch.randn(
        batch, heads, seq, dim, device=device, dtype=torch.bfloat16,
    )
    rotary = _Rotary(dim, attention_scaling=1.37).to(device)
    positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    cos, sin = rotary(q_pre, positions)
    q_post = _apply_rotary_pos_emb_q(q_pre, cos, sin)

    expected = _invert_rotary_pos_emb_q(q_post, cos, sin)[:, :, n_sink:].mean(2)
    actual = inverse_rope_suffix_mean(q_post, cos, sin, n_sink)

    assert actual.shape == (batch, heads, dim)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scorer ops require CUDA")
def test_native_joint_score_ops_match_fp32_oracle():
    torch.manual_seed(93)
    device = torch.device("cuda")
    batch, h_q, h_kv, seq, dim = 2, 8, 2, 65, 64
    groups = h_q // h_kv
    mean_q = torch.randn(batch, h_q, dim, device=device)
    keys = torch.randn(batch, h_kv, seq, dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    scale = dim ** -0.5

    logits = gqa_query_key_logits(mean_q, keys, scale)
    ref_logits = torch.einsum(
        "bhgd,bhtd->bhgt",
        mean_q.reshape(batch, h_kv, groups, dim),
        keys.float(),
    ) * scale
    torch.testing.assert_close(logits, ref_logits, rtol=2e-5, atol=2e-5)

    probability = torch.softmax(logits, -1).mean(2)
    out = weighted_value_sum(probability, values)
    ref_out = torch.einsum("bht,bhtd->bhd", probability, values.float())
    torch.testing.assert_close(out, ref_out, rtol=2e-5, atol=2e-5)

    score = joint_mse_score(probability, keys, values, out, epsilon=1e-2)
    kf, vf = keys.float(), values.float()
    vn2 = vf.square().sum(-1)
    vo = torch.einsum("bhtd,bhd->bht", vf, ref_out)
    irrep = (vn2 - 2 * vo + ref_out.square().sum(-1, keepdim=True)).clamp_min(0)
    ref_score = (probability + 1e-2).square() * (
        vn2 + (kf.square().sum(-1) / dim) * irrep
    )
    torch.testing.assert_close(score, ref_score, rtol=3e-5, atol=3e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="multi-device Triton launch regression requires two CUDA devices",
)
def test_native_prefill_ops_launch_on_tensor_device_not_current_device():
    """A device-mapped layer may not live on torch.cuda.current_device()."""
    torch.manual_seed(101)
    torch.cuda.set_device(0)
    current = torch.cuda.current_device()
    device = torch.device("cuda", 1)
    batch, h_q, h_kv, seq, dim = 1, 4, 2, 17, 64

    q_pre = torch.randn(
        batch, h_q, seq, dim, device=device, dtype=torch.bfloat16,
    )
    rotary = _Rotary(dim).to(device)
    positions = torch.arange(seq, device=device).unsqueeze(0)
    cos, sin = rotary(q_pre, positions)
    q_post = _apply_rotary_pos_emb_q(q_pre, cos, sin)
    mean_q = inverse_rope_suffix_mean(q_post, cos, sin, n_sink=4)

    keys = torch.randn(
        batch, h_kv, seq, dim, device=device, dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    logits = gqa_query_key_logits(mean_q, keys, dim ** -0.5)
    probability = torch.softmax(logits, dim=-1).mean(dim=2)
    expected_value = weighted_value_sum(probability, values)
    score = joint_mse_score(
        probability, keys, values, expected_value, epsilon=1e-2,
    )

    assert score.device == device
    assert score.shape == (batch, h_kv, seq)
    assert torch.cuda.current_device() == current


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native scorer ops require CUDA")
def test_native_decode_qkv_score_matches_existing_fp32_formula():
    torch.manual_seed(109)
    device = torch.device("cuda")
    batch, h_q, h_kv, seq, dim = 2, 8, 2, 128, 64
    groups = h_q // h_kv
    query = torch.randn(
        batch, h_q, dim, device=device, dtype=torch.bfloat16,
    )
    keys = torch.randn(
        batch, h_kv, seq, dim, device=device, dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    scorer = ODMScorer(epsilon=1e-2)

    grouped_query = query.reshape(batch, h_kv, groups, dim).float()
    attention = torch.softmax(
        torch.matmul(grouped_query, keys.float().transpose(-1, -2))
        * (dim ** -0.5),
        dim=-1,
    ).mean(dim=2)
    expected = scorer.score_with_attn_batched(keys, values, attention)
    actual = scorer.score_decode_from_qkv_batched(query, keys, values)

    torch.testing.assert_close(actual, expected, rtol=5e-5, atol=5e-5)
