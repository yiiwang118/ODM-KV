"""Attention / RoPE / GQA helpers shared by scorers and press implementations.

Previously these lived as private ``_`` helpers inside :mod:`kvquant.scorer`,
which caused :mod:`kvquant.adaptive_backend_press` to reach into another module's
private namespace. They are attention-mechanism utilities, not scorer-specific,
so they belong in their own module.

:mod:`kvquant.scorer` re-exports all of these under the same ``_`` names for
backwards compatibility, so ``from kvquant.scorer import _repeat_kv`` still works.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim,
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _get_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.shape
    cfg = getattr(module, "config", None)
    num_heads = getattr(module, "num_heads", None)
    if num_heads is None and cfg is not None:
        num_heads = getattr(cfg, "num_attention_heads", None)
    if num_heads is None:
        raise AttributeError(f"Cannot determine num_heads from {module.__class__.__name__}")
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None and cfg is not None:
        head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_states.shape[-1] // num_heads

    if hasattr(module, "qkv_proj"):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"ExpectedAttentionScorer does not support {module.__class__.__name__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    return query_states


def _apply_avg_rope(
    module: nn.Module,
    mu: torch.Tensor,          # [..., d]
    cov: Optional[torch.Tensor],  # [..., d, d] or None
    q_len: int,
    n_future_positions: int,
    n_samples: int = 8,         # kept for API compat, no longer used
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply averaged RoPE to query mean (and covariance) over future positions.

    Aligned with kvpress's ExpectedAttentionPress.apply_avg_rope: builds the
    rotation matrix R(p) for each future position p in [q_len, q_len+n_future),
    averages them element-wise, and applies the averaged matrix to mu/cov.

    Geometric interpretation: high-freq RoPE components oscillate over the
    n_future window, so their cos/sin averages collapse toward 0 — these
    components are filtered out (low-pass effect). Low-freq components are
    preserved. This is a designed feature, not a bug.

    History:
    - commit 3510445: original implementation (matched kvpress, correct)
    - commit ce8daf3: rewrote to K=8 Monte Carlo + var_mu correction. The
      rewrite assumed matrix-averaging was wrong (non-orthogonal), but it
      is intentionally non-orthogonal as a low-pass filter.
    - commit 2026-04-27: revert to kvpress-aligned matrix averaging.
    """
    rotary_emb = getattr(module, "rotary_emb", None)
    if rotary_emb is None or n_future_positions <= 0:
        return mu, cov

    position_ids = torch.arange(
        q_len, q_len + n_future_positions, device=mu.device,
    ).unsqueeze(0)
    cos, sin = _aligned_rotary_outputs(rotary_emb, mu, position_ids)
    cos = cos[0]  # [n_future, d]
    sin = sin[0]  # [n_future, d]

    head_dim = mu.shape[-1]
    half = head_dim // 2

    eye = torch.eye(head_dim, device=mu.device, dtype=mu.dtype)
    perm = torch.zeros((head_dim, head_dim), device=mu.device, dtype=mu.dtype)
    perm[half:, :half] = torch.eye(half, device=mu.device, dtype=mu.dtype)
    perm[:half, half:] = -torch.eye(half, device=mu.device, dtype=mu.dtype)

    # R[p] = diag(cos_p) @ eye + diag(sin_p) @ perm  for each p
    # Average element-wise across positions → R_avg [d, d]
    R = cos.unsqueeze(-1) * eye + sin.unsqueeze(-1) * perm   # [n_future, d, d]
    R = R.mean(dim=0)                                          # [d, d]

    # Apply: μ' = R · μ. For [..., d] input, mu @ R.T == R @ mu.T (transpose).
    mu_out = torch.matmul(mu, R.T)

    if cov is None:
        return mu_out, None

    # Σ' = R · Σ · R.T
    cov_out = torch.matmul(R, torch.matmul(cov, R.T))
    return mu_out, cov_out


def _aligned_rotary_outputs(
    rotary_emb,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RoPE cos/sin aligned to the caller's device and dtype.

    In transformers, `rotary_emb` is usually safe to call with an input tensor on
    the target device. In this project we sometimes attach the root `lm.rotary_emb`
    object onto sharded attention layers under `device_map="auto"`, so making the
    alignment explicit avoids cross-device failures inside custom scorers.
    """
    cos, sin = rotary_emb(x, position_ids)
    target = {"device": x.device, "dtype": x.dtype}
    if cos.device != x.device or cos.dtype != x.dtype:
        cos = cos.to(**target)
    if sin.device != x.device or sin.dtype != x.dtype:
        sin = sin.to(**target)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the head dim: [x1, x2] → [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_q(
    q: torch.Tensor,       # [bsz, H, T, d]
    cos: torch.Tensor,     # [bsz, T, d] or [T, d]
    sin: torch.Tensor,     # same
) -> torch.Tensor:
    """Apply rotary embeddings to query states only."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)   # [bsz, 1, T, d]
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin)


def _chunked_attention_qk(
    query_states: torch.Tensor,   # [1, H_q, T_q, d]
    expanded_keys_t: torch.Tensor,  # [1, H_q, d, T_kv]
    T_kv: int,
    H_kv: int,
    n_groups: int,
    scale: float,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention statistics per KV position, chunked over queries.

    Returns (sum_attn, max_attn, sum_sq_attn) each of shape [H_kv, T_kv].
    Divide sum_attn by the per-position query count to get the mean.
    sum_sq_attn = Σ_t a_{t,i}² is the L2 norm of the attention column,
    used by EvictionAwareScorer to estimate eviction risk.
    """
    T_q = query_states.shape[2]
    sum_attn = torch.zeros(H_kv, T_kv, device=device, dtype=dtype)
    max_attn = torch.zeros(H_kv, T_kv, device=device, dtype=dtype)
    sum_sq_attn = torch.zeros(H_kv, T_kv, device=device, dtype=dtype)

    for start in range(0, T_q, chunk_size):
        end = min(start + chunk_size, T_q)
        q_chunk = query_states[:, :, start:end, :]    # [1, H_q, chunk, d]

        attn_logits = torch.matmul(q_chunk, expanded_keys_t) / scale  # [1, H_q, chunk, T_kv]

        # Causal mask
        q_pos = torch.arange(start, end, device=device)
        k_pos = torch.arange(T_kv, device=device)
        causal = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)            # [chunk, T_kv]
        attn_logits.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_w = torch.softmax(attn_logits, dim=-1)                 # [1, H_q, chunk, T_kv]

        # Reduce over GQA groups → [1, H_kv, chunk, T_kv]
        attn_kv = attn_w.reshape(1, H_kv, n_groups, end - start, T_kv).mean(dim=2)

        chunk_kv = attn_kv.squeeze(0)                                # [H_kv, chunk, T_kv]

        # Accumulate sum over query positions → [H_kv, T_kv]
        sum_attn += chunk_kv.sum(dim=1)

        # Accumulate max over query positions → [H_kv, T_kv]
        max_attn = torch.max(max_attn, chunk_kv.amax(dim=1))

        # Accumulate sum of squares → [H_kv, T_kv]
        sum_sq_attn += chunk_kv.pow(2).sum(dim=1)

    return sum_attn, max_attn, sum_sq_attn
