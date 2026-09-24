"""Exact head-wise eviction masks for the dense reference backend."""
from __future__ import annotations

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def reference_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    batch, q_heads, q_len, _ = query.shape
    kv_heads, k_len = key.shape[1:3]
    if q_len == k_len:
        module.masked_key_indices = None
    masked = getattr(module, "masked_key_indices", None)
    if masked is not None:
        # A true -inf mask works for every query, including opposing GQA heads.
        # It does not modify cached keys or approximate eviction with fake keys.
        bias = torch.zeros(batch, kv_heads, 1, k_len, device=query.device, dtype=query.dtype)
        b, h, t = masked
        bias[b, h, 0, t] = float("-inf")
        bias = bias.repeat_interleave(q_heads // kv_heads, dim=1)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                bias = bias.masked_fill(~attention_mask, float("-inf"))
            else:
                bias = bias + attention_mask
        elif q_len > 1:
            q_pos = torch.arange(k_len - q_len, k_len, device=query.device)
            k_pos = torch.arange(k_len, device=query.device)
            bias = bias.expand(batch, q_heads, q_len, k_len).clone()
            bias.masked_fill_(k_pos[None, :] > q_pos[:, None], float("-inf"))
        attention_mask = bias
    return sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs,
    )


def register_reference_attention():
    ALL_ATTENTION_FUNCTIONS.register("odmkv_reference", reference_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        "odmkv_reference", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"],
    )
