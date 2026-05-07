"""KV-unit scoring interface.

A scorer assigns an importance score in [0, 1] to each KV unit (layer, head,
token). Higher score means the token should receive more bits.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

# Attention / RoPE / GQA helpers live in a dedicated module, but are re-exported
# here so ``from kvquant.scorer import _aligned_rotary_outputs`` (and friends)
# continues to work for existing callers (adaptive_backend_press, tests, etc.).
from kvquant.attention_utils import (  # noqa: F401 — re-export for back-compat
    _aligned_rotary_outputs,
    _apply_avg_rope,
    _apply_rotary_pos_emb_q,
    _chunked_attention_qk,
    _get_query_states,
    _repeat_kv,
    _rotate_half,
)


class BaseScorer(ABC):
    """Interface every scorer must implement."""

    # Normalization granularity for final scores. YAML config knob; default
    # is "layer" which is the historical behavior. See `_normalize_scores`
    # for semantics of each mode.
    #   "layer"  — one min-max across [H, T] of the current layer (default)
    #   "head"   — per-row min-max on [H, T]; every head gets its own [0,1]
    #   "global" — return raw (no normalize). Because press.py's global
    #              Lagrangian is invariant to monotone cross-tensor
    #              transforms, this is equivalent to one cross-layer
    #              concat-then-minmax and was already empirically tested
    #              (commit 79acd78 → reverted). Provided as a diagnostic.
    normalize_grain: str = "layer"

    def _maybe_normalize(self, raw: torch.Tensor) -> torch.Tensor:
        return _normalize_scores(raw, grain=self.normalize_grain)

    @abstractmethod
    def score_prefill(
        self,
        keys: torch.Tensor,    # [H, T, d]  — current layer's keys
        values: torch.Tensor,  # [H, T, d]  — current layer's values
        layer_idx: int,
        *,
        module: nn.Module | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores in [0, 1] of shape [H, T] for this layer's tokens."""

    def score_decode(
        self,
        key: torch.Tensor,    # [H, d]  — single new token
        value: torch.Tensor,  # [H, d]
        layer_idx: int,
        *,
        module: nn.Module | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scores in [0, 1] of shape [H] for a single decode token.

        Default: return 1.0 (keep at highest precision) for all decode tokens.
        """
        return torch.ones(key.shape[0], device=key.device)

    def score_with_attn(
        self,
        keys: torch.Tensor,           # [H, T, d]  — buffer keys
        values: torch.Tensor,         # [H, T, d]  — buffer values
        attn_weights: torch.Tensor,   # [H, T]     — per-token attention given current query
        *,
        module: nn.Module | None = None,
    ) -> torch.Tensor:
        """Score buffer tokens given pre-computed attention weights.

        Used by decode-buffer flush in adaptive backend, where the current
        decode query has already produced an attention pattern over buffered
        tokens.  The default implementation falls back to NormScorer-equivalent
        ``p²·‖v‖²`` to remain safe; **scorers should override this to apply
        the same formula they use in score_prefill** so paper claims about
        the score formula hold for the entire prefill+decode pipeline.

        Returns: [H, T] raw (un-normalised) scores.  Caller is responsible for
        any subsequent normalisation.
        """
        # Generic fallback: p² · ‖v‖² (matches mu_only_alpha2)
        eps = float(getattr(self, "epsilon", 0.0) or 0.0)
        vn2 = values.float().pow(2).sum(dim=-1)
        return (attn_weights + eps).pow(2) * vn2


def _normalize_scores(raw: torch.Tensor, *, grain: str = "layer") -> torch.Tensor:
    """Min-max normalize a raw-score tensor.

    grain:
        "layer"  — one min/max across the whole tensor (legacy default).
        "head"   — per-row min/max on [H, T] (raw.dim() >= 2). Each head
                   gets its own [0, 1]. Rows with zero-variance fall back
                   to 0.5, matching the legacy behavior. For 1-D inputs
                   (decode path [H]) degenerates to "layer".
        "global" — no-op. The press's layerwise=False Lagrangian is
                   invariant to monotone cross-tensor transforms, so a
                   true cross-layer min-max would not change assignments.
                   Leaving raw through means the caller (Lagrangian) sees
                   the native magnitude, which is what "global" semantics
                   reduce to in this pipeline.
    """
    if raw.numel() == 0:
        return raw
    if grain == "global":
        return raw
    if grain == "head" and raw.dim() >= 2:
        lo = raw.amin(dim=-1, keepdim=True)
        hi = raw.amax(dim=-1, keepdim=True)
        denom = hi - lo
        safe = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        out = (raw - lo) / safe
        dead = (denom.squeeze(-1) < 1e-8)  # [..., H]
        if dead.any():
            out[dead] = 0.5
        return out
    # grain == "layer", or "head" fallback for 1-D (decode) input.
    lo = raw.min()
    hi = raw.max()
    if (hi - lo) < 1e-8:
        return torch.full_like(raw, 0.5)
    return (raw - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Norm fallback (used when expected-attention statistics are unavailable)
# ---------------------------------------------------------------------------

class _NormFallback(BaseScorer):
    """L2-norm scorer used as defensive fallback inside ExpectedAttentionScorer."""

    def score_prefill(self, keys, values, layer_idx, *, module=None, hidden_states=None):
        raw = keys.norm(dim=-1) + values.norm(dim=-1)
        return self._maybe_normalize(raw)

    def score_decode(self, key, value, layer_idx, *, module=None, hidden_states=None):
        raw = key.norm(dim=-1) + value.norm(dim=-1)
        return self._maybe_normalize(raw)


class ExpectedAttentionScorer(BaseScorer):
    """Expected-attention scorer adapted into the local kvquant pipeline.

    The score estimates how much future queries are expected to attend to each
    cached token. It models the distribution of future pre-RoPE queries, applies
    average future RoPE, and scores each KV slot with:

        softmax(E[q]^T k / sqrt(d) + 1/2 k^T Cov[q] k / d) * ||v||

    Sink tokens can still be protected separately by the allocator; internally
    ``n_sink`` is used only to exclude early hidden states from the query
    statistics and to keep those tokens at the top of the ranking.
    """

    def __init__(
        self,
        *,
        n_future_positions: int = 512,
        n_sink: int = 4,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
    ):
        self.n_future_positions = n_future_positions
        self.n_sink = n_sink
        self.use_covariance = use_covariance
        self.use_vnorm = use_vnorm
        self.epsilon = epsilon
        self._fallback = _NormFallback()

    def _query_statistics(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_len = hidden_states.shape[1]
        if q_len <= self.n_sink:
            raise ValueError("not enough tokens to estimate expected attention")

        query_input = hidden_states[:, self.n_sink :]
        query_states = _get_query_states(module, query_input)   # [1, H, T, d]
        mu = query_states.mean(dim=2)                           # [1, H, d]

        cov = None
        if self.use_covariance:
            centered = query_states - mu.unsqueeze(2)
            cov = torch.einsum("bhsd,bhse->bhde", centered, centered) / query_input.shape[1]

        return _apply_avg_rope(module, mu, cov, q_len, self.n_future_positions)

    def score_prefill(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        *,
        module: nn.Module | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if module is None or hidden_states is None:
            return self._fallback.score_prefill(keys, values, layer_idx)

        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)

        if hidden_states.shape[0] != 1:
            raise ValueError("ExpectedAttentionScorer expects a single-sample hidden-state slice.")

        if keys.shape[1] <= self.n_sink:
            return torch.ones(keys.shape[:2], device=keys.device, dtype=keys.dtype)

        try:
            mean_query, cov_query = self._query_statistics(module, hidden_states)
        except (AttributeError, NotImplementedError, ValueError):
            return self._fallback.score_prefill(keys, values, layer_idx)

        keys_core = keys[:, self.n_sink :]
        values_core = values[:, self.n_sink :]
        if keys_core.numel() == 0:
            return torch.ones(keys.shape[:2], device=keys.device, dtype=keys.dtype)

        bsz = 1
        num_key_value_heads, core_len, head_dim = keys_core.shape
        cfg = getattr(module, "config", None)
        num_attention_heads = getattr(module, "num_heads", None)
        if num_attention_heads is None and cfg is not None:
            num_attention_heads = getattr(cfg, "num_attention_heads", None)
        if num_attention_heads is None:
            num_attention_heads = num_key_value_heads
        num_key_value_groups = num_attention_heads // num_key_value_heads

        expanded_keys = _repeat_kv(keys_core.unsqueeze(0), num_key_value_groups).transpose(2, 3)
        raw = torch.matmul(mean_query.unsqueeze(2), expanded_keys).squeeze(2) / math.sqrt(head_dim)
        if cov_query is not None:
            raw = raw + torch.einsum("bhdt,bhde,bhet->bht", expanded_keys, cov_query, expanded_keys) / head_dim / 2
        raw = torch.softmax(raw, dim=-1)

        raw = raw.reshape(bsz, num_key_value_heads, num_key_value_groups, core_len).mean(dim=2)
        if self.use_vnorm:
            raw = (raw + self.epsilon) * values_core.unsqueeze(0).norm(dim=-1)

        sink_value = float(raw.max().item()) if raw.numel() else 1.0
        raw = torch.nn.functional.pad(raw, (self.n_sink, 0), value=sink_value)
        return self._maybe_normalize(raw.squeeze(0))

    def score_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        *,
        module: nn.Module | None = None,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fallback.score_decode(
            key,
            value,
            layer_idx,
            module=module,
            hidden_states=hidden_states,
        )


# ---------------------------------------------------------------------------
# RoPE helpers for KVZip (apply to queries so they match cached post-RoPE keys)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KVZip scorer
# ---------------------------------------------------------------------------

def _expected_core_internals(self, keys, values, module, hidden_states):
    """Shared internals: compute raw softmax-attention [B, H_kv, T_core]
    (core = excluding sinks) using μ, Σ.

    Returns (raw_attn, mean_query, cov_query, keys_core, values_core,
            num_key_value_heads, core_len, head_dim, num_key_value_groups,
            expanded_keys).
    Caller can decide how to combine raw_attn with norms/extras.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    mean_query, cov_query = self._query_statistics(module, hidden_states)
    keys_core = keys[:, self.n_sink:]
    values_core = values[:, self.n_sink:]

    num_key_value_heads, core_len, head_dim = keys_core.shape
    cfg = getattr(module, "config", None)
    num_attention_heads = getattr(module, "num_heads", None)
    if num_attention_heads is None and cfg is not None:
        num_attention_heads = getattr(cfg, "num_attention_heads", None)
    if num_attention_heads is None:
        num_attention_heads = num_key_value_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads

    expanded_keys = _repeat_kv(keys_core.unsqueeze(0), num_key_value_groups).transpose(2, 3)
    raw = torch.matmul(mean_query.unsqueeze(2), expanded_keys).squeeze(2) / math.sqrt(head_dim)
    if cov_query is not None:
        raw = raw + torch.einsum("bhdt,bhde,bhet->bht",
                                 expanded_keys, cov_query, expanded_keys) / head_dim / 2
    raw = torch.softmax(raw, dim=-1)
    raw = raw.reshape(1, num_key_value_heads, num_key_value_groups, core_len).mean(dim=2)

    return (raw, mean_query, cov_query, keys_core, values_core,
            num_key_value_heads, core_len, head_dim, num_key_value_groups)


def _pad_sinks_and_normalize(self, raw_core, n_sink):
    """Pad sinks to top-value and apply min-max per layer."""
    sink_value = float(raw_core.max().item()) if raw_core.numel() else 1.0
    padded = torch.nn.functional.pad(raw_core, (n_sink, 0), value=sink_value)
    return self._maybe_normalize(padded.squeeze(0))


# ── R1: last-K queries only ────────────────────────────────────────────

def _expected_core_with_raw_logits(self, keys, values, module, hidden_states, *, temperature=1.0):
    """Like _expected_core_internals but returns both PRE-softmax logits and
    the core tensors. Temperature applied to the logits before softmax.
    """
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    mean_query, cov_query = self._query_statistics(module, hidden_states)
    keys_core = keys[:, self.n_sink:]
    values_core = values[:, self.n_sink:]

    num_key_value_heads, core_len, head_dim = keys_core.shape
    cfg = getattr(module, "config", None)
    num_attention_heads = getattr(module, "num_heads", None)
    if num_attention_heads is None and cfg is not None:
        num_attention_heads = getattr(cfg, "num_attention_heads", None)
    if num_attention_heads is None:
        num_attention_heads = num_key_value_heads
    num_key_value_groups = num_attention_heads // num_key_value_heads

    expanded_keys = _repeat_kv(keys_core.unsqueeze(0), num_key_value_groups).transpose(2, 3)
    raw_logits = torch.matmul(mean_query.unsqueeze(2), expanded_keys).squeeze(2) / math.sqrt(head_dim)
    if cov_query is not None:
        raw_logits = raw_logits + torch.einsum(
            "bhdt,bhde,bhet->bht",
            expanded_keys, cov_query, expanded_keys) / head_dim / 2
    raw_logits = raw_logits / temperature
    raw = torch.softmax(raw_logits, dim=-1)
    raw = raw.reshape(1, num_key_value_heads, num_key_value_groups, core_len).mean(dim=2)
    return raw, raw_logits, keys_core, values_core, mean_query, num_key_value_heads, num_key_value_groups


# ── R3.1: q·v direct inner product ───────────────────────────────────

class RiskScorer(ExpectedAttentionScorer):
    """s = p² · [‖v‖² + (‖k‖²/d) · ‖v − o‖²].

    First-order joint MSE of attention output under independent V + K
    quantization noise:

        E‖Δo‖² ≈ Σⱼ ε(bⱼ)·pⱼ²·[‖vⱼ‖² + (‖kⱼ‖²/d)·‖vⱼ − o‖²]

    Two complementary terms:
      - V error: pⱼ²·‖vⱼ‖² — protects high-attention high-norm tokens
        (e.g. BOS sink, where the first term dominates even though
        ‖v − o‖² collapses).
      - K-induced p-shift: pⱼ²·(‖kⱼ‖²/d)·‖vⱼ − o‖² — protects unique
        content tokens whose deletion would shift the output direction.

    Cross-term vanishes because V and K quantization noise are independent.
    Multiplicative form (1 + ‖k‖²/d) is wrong; this additive form is right.
    """
    def __init__(self, **kw):
        kw["use_covariance"] = False
        super().__init__(**kw)

    def score_prefill(self, keys, values, layer_idx, *, module=None, hidden_states=None):
        if module is None or hidden_states is None:
            return self._fallback.score_prefill(keys, values, layer_idx)
        if keys.shape[1] <= self.n_sink:
            return torch.ones(keys.shape[:2], device=keys.device, dtype=keys.dtype)
        try:
            tup = _expected_core_internals(self, keys, values, module, hidden_states)
        except (AttributeError, NotImplementedError, ValueError):
            return self._fallback.score_prefill(keys, values, layer_idx)
        raw, keys_core, values_core, head_dim = tup[0], tup[3], tup[4], tup[7]
        p = raw[0].float()
        p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        o = torch.einsum('ht,htd->hd', p_norm, values_core.float())
        v_minus_o = values_core.float() - o.unsqueeze(1)
        irrep = v_minus_o.pow(2).sum(dim=-1)
        kn2 = keys_core.unsqueeze(0).float().pow(2).sum(dim=-1)   # [1, H, T]
        vn2 = values_core.unsqueeze(0).float().pow(2).sum(dim=-1)
        k_term = (kn2 / float(head_dim)) * irrep.unsqueeze(0)
        scored = (raw + self.epsilon).pow(2) * (vn2 + k_term)
        return _pad_sinks_and_normalize(self, scored, self.n_sink)

    def score_with_attn(self, keys, values, attn_weights, *, module=None):
        # Joint MSE formula given pre-computed attention weights (decode buffer flush).
        p = attn_weights.float()
        p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        o = torch.einsum('ht,htd->hd', p_norm, values.float())
        v_minus_o = values.float() - o.unsqueeze(1)
        irrep = v_minus_o.pow(2).sum(dim=-1)
        kn2 = keys.float().pow(2).sum(dim=-1)
        vn2 = values.float().pow(2).sum(dim=-1)
        head_dim = keys.shape[-1]
        return (p + self.epsilon).pow(2) * (vn2 + (kn2 / float(head_dim)) * irrep)


