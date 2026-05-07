"""Attention-function patch for length-preserving head-wise KV masking."""
from __future__ import annotations

import functools

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


_PATCH_ATTR = "_kvtoken_quant_attention_patched"


def search_hyperplane(x: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
    """Find a fake key k such that exp(<q, k>) ~= 0 for every query q in x.

    Matches kvpress/kvpress/attention_patch.py: returns -1e5 * Y / ||Y||^2.
    """
    orig_dtype = x.dtype
    x = x.float()
    y = x.mean(dim=1)
    for _ in range(max_iter):
        mask = torch.bmm(x, y.unsqueeze(-1)) <= 0
        if not mask.any():
            result = -1e5 * y / y.norm(dim=-1, keepdim=True) ** 2
            return result.to(orig_dtype)
        y = y + (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError("Could not find fake keys such that exp(<q, k>) ~= 0")


def _attention_patch(func):
    if getattr(func, _PATCH_ATTR, False):
        return func

    @functools.wraps(func)
    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            fake_keys = search_hyperplane(q).view(bsz, num_key_value_heads, head_dim)

            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = fake_keys[batch_indices, head_indices]

        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    setattr(wrapper, _PATCH_ATTR, True)
    return wrapper


def patch_attention_functions() -> None:
    for name, func in list(ALL_ATTENTION_FUNCTIONS.items()):
        if getattr(func, _PATCH_ATTR, False):
            continue
        ALL_ATTENTION_FUNCTIONS[name] = _attention_patch(func)
