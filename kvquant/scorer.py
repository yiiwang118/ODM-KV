"""ODMScorer: first-order attention-output distortion for one KV entry.

Both reference and native backends use the same score and raw score scale.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from kvquant.attention_utils import _apply_avg_rope, _get_query_states, _repeat_kv


class ODMScorer:
    """Estimate joint K/V sensitivity for every (KV head, token).

    p is obtained from the mean pre-RoPE query and averaged future RoPE.
    Decode flush uses the current query's attention over the buffered tokens.
    Missing attention statistics are errors; no alternative score is used.
    """

    normalize_grain = "global"  # Preserve cardinal sensitivity, not just rank.

    def __init__(self, *, n_future_positions=512, n_sink=4, epsilon=1e-2):
        if n_future_positions < 0 or n_sink < 0:
            raise ValueError("future positions and sink count must be nonnegative")
        if not math.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative")
        self.n_future_positions = int(n_future_positions)
        self.n_sink = int(n_sink)
        self.epsilon = float(epsilon)

    def score_prefill(self, keys, values, layer_idx=0, *, module=None, hidden_states=None):
        if keys.ndim != 3 or keys.shape != values.shape:
            raise ValueError("prefill expects matching [H,T,D] keys and values")
        if keys.shape[1] <= self.n_sink:
            return torch.ones(keys.shape[:2], device=keys.device, dtype=torch.float32)
        if module is None or hidden_states is None:
            raise ValueError("ODMScorer requires the attention module and prefill hidden states")
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.shape[:2] != (1, keys.shape[1]):
            raise ValueError("hidden states must contain the same single-request prefill span")
        query = _get_query_states(module, hidden_states[:, self.n_sink:]).float()
        mean_query, _ = _apply_avg_rope(
            module, query.mean(dim=2), None, keys.shape[1], self.n_future_positions,
        )
        h_kv, _, dim = keys.shape
        h_q = mean_query.shape[1]
        if h_q % h_kv:
            raise ValueError("query heads must be divisible by KV heads")
        groups = h_q // h_kv
        core_k, core_v = keys[:, self.n_sink:], values[:, self.n_sink:]
        expanded = _repeat_kv(core_k.unsqueeze(0).float(), groups)
        logits = torch.matmul(mean_query.unsqueeze(2), expanded.transpose(2, 3)).squeeze(2)
        attention = torch.softmax(logits / math.sqrt(dim), dim=-1)
        attention = attention.reshape(h_kv, groups, -1).mean(dim=1)
        score = self.score_with_attn(core_k, core_v, attention, module=module)
        return torch.cat((score.amax().expand(h_kv, self.n_sink), score), dim=-1)

    def score_prefill_from_qkv(
        self,
        query: torch.Tensor,   # [B, H_q, T, D], post-RoPE attention input
        keys: torch.Tensor,    # [B, H_kv, T, D], post-RoPE
        values: torch.Tensor,  # [B, H_kv, T, D]
        layer_idx: int,
        *,
        module: nn.Module,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary_emb: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Batched native-prefill score from attention's existing Q/K/V.

        This is algebraically the same ODM score as
        :meth:`score_prefill`, but it inverts RoPE on the already-projected Q
        instead of replaying q_proj for each batch item.  It returns
        ``[B,H_kv,T]`` and contains no host synchronization or per-sample loop.
        """
        if query.dim() != 4 or keys.dim() != 4 or values.dim() != 4:
            raise ValueError("native prefill scorer expects Q/K/V shaped [B,H,T,D]")
        B, H_q, T, D = query.shape
        if keys.shape[0] != B or values.shape != keys.shape or keys.shape[2:] != (T, D):
            raise ValueError(
                f"incompatible native Q/K/V shapes: Q={tuple(query.shape)} "
                f"K={tuple(keys.shape)} V={tuple(values.shape)}"
            )
        H_kv = keys.shape[1]
        if H_q % H_kv != 0:
            raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
        if T <= self.n_sink:
            return torch.ones(B, H_kv, T, device=keys.device, dtype=keys.dtype)

        # Fuse scaled-RoPE inversion with the suffix reduction.  Only [B,H,D]
        # is written; a naive fp32 q_pre_rope temporary is about 2 GiB for
        # Llama B8/T2k and would overlap the layer's fresh Q/K/V.
        from kvquant.runtime.kernels.prefill_score import (
            gqa_query_key_logits,
            inverse_rope_suffix_mean,
            joint_mse_score,
            weighted_value_sum,
        )

        mu = inverse_rope_suffix_mean(query, cos, sin, self.n_sink)
        mean_query, _ = _apply_avg_rope(
            module, mu, None, T, self.n_future_positions,
            rotary_emb=rotary_emb,
        )

        keys_core = keys[:, :, self.n_sink:]
        values_core = values[:, :, self.n_sink:]
        n_groups = H_q // H_kv

        # Compute per-query-head logits without repeat_kv materialization, then
        # retain the original semantics: softmax per Q head, mean over GQA group.
        logits = gqa_query_key_logits(mean_query, keys_core, 1.0 / math.sqrt(D))
        raw = torch.softmax(logits, dim=-1).mean(dim=2)  # [B,H_kv,T_core]

        p_norm = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        out_est = weighted_value_sum(p_norm, values_core)
        score_core = joint_mse_score(
            raw, keys_core, values_core, out_est, self.epsilon,
        )

        # The legacy per-sample scorer pads each sink with that sample's maximum
        # core score before normalization. Preserve it without max().item().
        sink_value = score_core.amax(dim=(1, 2), keepdim=True)
        sink = sink_value.expand(B, H_kv, self.n_sink)
        scored = torch.cat((sink, score_core), dim=-1)

        grain = getattr(self, "normalize_grain", "layer")
        if grain == "global":
            return scored
        if grain == "head":
            lo = scored.amin(dim=-1, keepdim=True)
            hi = scored.amax(dim=-1, keepdim=True)
        else:  # historical "layer": independently normalize each request
            lo = scored.amin(dim=(1, 2), keepdim=True)
            hi = scored.amax(dim=(1, 2), keepdim=True)
        denom = hi - lo
        safe = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
        normalized = (scored - lo) / safe
        return torch.where(denom < 1e-8, torch.full_like(normalized, 0.5), normalized)


    def score_with_attn(self, keys, values, attn_weights, *, module=None):
        # Joint MSE formula given pre-computed attention weights (decode buffer flush).
        p = attn_weights.float()
        p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vc = values.float()
        o = torch.einsum('ht,htd->hd', p_norm, vc)
        vn2 = vc.pow(2).sum(dim=-1)
        vo = torch.einsum('htd,hd->ht', vc, o)
        on2 = o.pow(2).sum(dim=-1, keepdim=True)
        irrep = (vn2 - 2.0 * vo + on2).clamp_(min=0.0)
        kn2 = keys.float().pow(2).sum(dim=-1)
        head_dim = keys.shape[-1]
        return (p + self.epsilon).pow(2) * (vn2 + (kn2 / float(head_dim)) * irrep)


    def score_with_attn_batched(self, keys, values, attn_weights, *, module=None):
        """Batched decode-buffer form of :meth:`score_with_attn`.

        Shapes are ``keys/values=[B,H,T,D]`` and ``attn=[B,H,T]``. Keeping
        the batch axis native avoids a Python request/head loop in the periodic
        system flush while preserving the exact joint-MSE expression.
        """
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("batched joint-MSE expects matching [B,H,T,D] K/V")
        if attn_weights.shape != keys.shape[:-1]:
            raise ValueError(
                f"batched attention shape {tuple(attn_weights.shape)} does not "
                f"match K/V prefix {tuple(keys.shape[:-1])}"
            )
        p = attn_weights.float()
        p_norm = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vc = values.float()
        out = torch.einsum("bht,bhtd->bhd", p_norm, vc)
        vn2 = vc.pow(2).sum(dim=-1)
        vo = torch.einsum("bhtd,bhd->bht", vc, out)
        on2 = out.pow(2).sum(dim=-1, keepdim=True)
        irrep = (vn2 - 2.0 * vo + on2).clamp_(min=0.0)
        kn2 = keys.float().pow(2).sum(dim=-1)
        return (p + self.epsilon).pow(2) * (
            vn2 + (kn2 / float(keys.shape[-1])) * irrep
        )


    def score_decode_from_qkv_batched(self, query, keys, values, *, module=None):
        """Native decode-ring score from the model's existing post-RoPE query.

        ``query=[B,H_q,D]`` and ``keys/values=[B,H_kv,T,D]``.  This preserves
        :meth:`score_with_attn_batched`'s formula while avoiding full-size fp32
        K/V casts and its chain of elementwise/einsum temporaries.  K/V are
        converted only in registers by the same Triton primitives used by the
        native prefill scorer.  The CPU branch deliberately retains the exact
        PyTorch oracle arithmetic.
        """
        if query.ndim != 3 or keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError(
                "native decode joint-MSE expects Q=[B,Hq,D] and matching "
                "K/V=[B,Hkv,T,D]"
            )
        batch, h_q, dim = query.shape
        if keys.shape[0] != batch or keys.shape[-1] != dim or h_q % keys.shape[1]:
            raise ValueError(
                f"incompatible decode Q/K/V shapes: Q={tuple(query.shape)} "
                f"K={tuple(keys.shape)} V={tuple(values.shape)}"
            )
        from kvquant.runtime.kernels.prefill_score import (
            gqa_query_key_logits,
            joint_mse_score,
            weighted_value_sum,
        )

        logits = gqa_query_key_logits(query, keys, 1.0 / math.sqrt(dim))
        attention = torch.softmax(logits, dim=-1).mean(dim=2)
        probability = attention / attention.sum(
            dim=-1, keepdim=True,
        ).clamp(min=1e-8)
        expected_value = weighted_value_sum(probability, values)
        return joint_mse_score(
            attention, keys, values, expected_value, self.epsilon,
        )

