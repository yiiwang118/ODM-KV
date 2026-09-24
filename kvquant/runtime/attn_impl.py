"""Realsys2 native compressed-attention dispatch.

Prefill delegates the model output to FlashAttention-2 while handing the
already-projected Q/K/V to the native scorer/packer. The validated q_len=1 path
reads mixed-bit packed banks directly through one fused Triton attention
source. Attention dispatch never materializes the compressed cache and never
falls back to SDPA: multi-token/speculative decode and unsupported masks fail
explicitly at this boundary.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import contextvars

from kvquant.runtime.storage.cache import ODMCache


# Which ODMCache the HF attention function reads from. A ContextVar (not a
# module global) so concurrent requests / threads / async tasks each see their
# own cache instead of clobbering a shared global.
_ACTIVE_CACHE: "contextvars.ContextVar[Optional[ODMCache]]" = \
    contextvars.ContextVar("odmkv_active_cache", default=None)


def _env_on(name: str) -> bool:
    """Truthy env parse — ``"0"``/``"false"``/empty are OFF (``bool("0")`` is
    True, which silently enabled the fused/materialize paths)."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def set_active_cache(cache: ODMCache):
    """Bind the active cache; returns a token to restore the PREVIOUS binding
    (so nested ``with press(model)`` contexts restore the outer cache on exit
    instead of clobbering it to None)."""
    return _ACTIVE_CACHE.set(cache)


def reset_active_cache(token) -> None:
    _ACTIVE_CACHE.reset(token)


def clear_active_cache() -> None:
    _ACTIVE_CACHE.set(None)


def _flashattention2_prefill(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    scaling: float,
    dropout: float,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Run the exact HF FlashAttention-2 integration under our custom backend.

    Calling the public wrapper directly would feed it
    ``module.config._attn_implementation == 'odmkv_pertok'`` and make HF's
    lazy loader resolve the wrong kernel.  Invoke the shared FA2 dispatcher with
    an explicit implementation instead; this preserves padding/varlen support
    without a config mutation or SDPA fallback.
    """
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
        from transformers.integrations.flash_attention import (
            _use_top_left_mask,
            get_target_dtype,
        )
    except ImportError as exc:  # pragma: no cover - deployment configuration
        raise RuntimeError(
            "odmkv native prefill requires Transformers FlashAttention support"
        ) from exc

    flash_kwargs = dict(kwargs)
    sliding_window = flash_kwargs.pop(
        "sliding_window", getattr(module, "sliding_window", None),
    )
    if sliding_window not in (None, 0):
        raise RuntimeError(
            "odmkv native compressed decode cannot preserve sliding-window "
            f"semantics (requested window={sliding_window}); packed bit banks "
            "do not retain global token positions"
        )
    softcap = flash_kwargs.pop("softcap", None)
    is_causal = flash_kwargs.pop("is_causal", getattr(module, "is_causal", True))
    out = _flash_attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention_mask,
        query_length=query.shape[2],
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        use_top_left_mask=_use_top_left_mask,
        softcap=softcap,
        target_dtype=get_target_dtype(query.transpose(1, 2), module),
        attn_implementation="flash_attention_2",
        layer_idx=getattr(module, "layer_idx", None),
        **flash_kwargs,
    )
    return out, None


def odmkv_attention_forward(
    module,
    query: torch.Tensor,            # [B, H_q, q_len, D]
    key: torch.Tensor,              # [B, H_kv, q_len, D] — just the new K
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    B, H_q, q_len, D = query.shape
    H_kv = key.shape[1]
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    layer_idx = int(module.layer_idx) if hasattr(module, "layer_idx") else 0
    cache = _ACTIVE_CACHE.get()

    if cache is None or not cache._committed.get(layer_idx, False):
        # Native prefill: FlashAttention-2 produces the ordinary model output;
        # the same post-RoPE Q/K/V feed the scorer and are compressed only after
        # this layer forward completes.
        if cache is not None:
            # ``graph_generate`` configures this before any Q/K/V allocation.
            # Keep this idempotent call as a correctness gate for direct press
            # users, so a padded request can never be packed as if pad tokens
            # were real KV cells.
            cache.configure_prefill_layout(
                attention_mask,
                batch_size=B,
                seq_len=q_len,
            )
        result = _flashattention2_prefill(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )
        if cache is not None:
            cache.run_native_prefill_callback(
                module, layer_idx, query, key, value,
            )
        return result

    n_groups = H_q // H_kv

    # ── Fast path: fused mixed-precision decode straight from the packed
    # banks (no materialise). q_len=1 and supported state (mse/mse, no OCS).
    # Batch and KV-head are flattened into one kernel grid. Anything else
    # falls through to the materialise path below.
    graph_tail = getattr(cache, "_graph_tail", False)
    if (
        q_len == 1
        and attention_mask is None
    ):
        # Graph-tail mode REQUIRES the fused kernel: the decode tokens live in the
        # fixed ring (_tail_buf_k), which only the fused path reads — the banked
        # path reads the (empty) cat tail and would silently drop them. So the
        # ring forces fused regardless of the env var, and a public API that
        # enabled the ring (graph_generate) no longer depends on a hidden env.
        use_fused = _env_on("REALSYS2_FUSED_KERNEL") or graph_tail
        # The fused kernel reads the ragged (un-padded) layout; the banked
        # fallback keeps the padded one.
        qready = (cache.get_quant_ready_ragged(layer_idx, H_kv) if use_fused
                  else cache.get_quant_ready(layer_idx, H_kv))
        if qready is not None and qready[1] is not None:
            quant_banks, pi_k, pi_v = qready
            cache.record_decode_query(layer_idx, query[:, :, 0, :])
            Qd = query[:, :, 0, :].reshape(B * H_kv, n_groups, D)   # bf16; kernels convert
            if use_fused:
                # Single split-K fused kernel: all bit levels + cached exact
                # prefix + tail-view in one launch (no per-step build_exact).
                from kvquant.runtime.kernels.fused_decode import fused_decode
                exact_k, exact_v, exact_seqlen = cache.get_exact_ready(layer_idx, H_kv)
                tk, tv, tsl, toff = cache.get_tail_for_decode(layer_idx, H_kv)
                decode_arena = cache.get_native_decode_arena(layer_idx)
                # ``device_map`` moves Q and the cache shard together but does
                # not necessarily update the process-global current device.
                # Bind the complete fused dispatcher so all nested Triton
                # launches target the layer that owns these tensors.
                with torch.cuda.device(query.device):
                    out = fused_decode(
                        Qd, quant_banks, exact_k, exact_v, exact_seqlen,
                        pi_k, pi_v, scaling, tail_k=tk, tail_v=tv,
                        tail_seqlen=tsl, tail_offset=toff,
                        decode_arena=decode_arena,
                    )
                return out.reshape(B, 1, H_q, D).to(query.dtype), None
            from kvquant.runtime.kernels.per_bit_decode import banked_decode_attention
            exact_k, exact_v, exact_seqlen = cache.get_exact_ready(layer_idx, H_kv)
            tail_k = cache._tail_k.get(layer_idx)
            tail_v = cache._tail_v.get(layer_idx)
            extra_exact_banks = None
            if tail_k is not None and tail_k.shape[2] > 0:
                tail_len = tail_k.shape[2]
                extra_exact_banks = [(tail_k.reshape(B * H_kv, tail_len, D),
                                      tail_v.reshape(B * H_kv, tail_len, D), None)]
            out = banked_decode_attention(
                Qd, quant_banks, exact_k, exact_v, pi_k, pi_v, scaling,
                exact_seqlen=exact_seqlen,
                extra_exact_banks=extra_exact_banks)                   # [B*H_kv,G,D]
            return out.reshape(B, 1, H_q, D).to(query.dtype), None

    reason = (
        f"q_len={q_len} (only q_len=1 decode is native)"
        if q_len != 1 else
        "a non-None decode attention_mask"
        if attention_mask is not None else
        "missing a supported native packed-bank descriptor"
    )
    raise RuntimeError(
        "odmkv compressed attention has no materialize/SDPA fallback: "
        f"{reason}. Use graph_generate with fixed-batch greedy q_len=1 decode."
    )


def register_odmkv_attention() -> None:
    """Register attention and its matching upstream mask policy.

    The prefill operator is FlashAttention-2, so Transformers must construct the
    same 2-D padding/packed-sequence mask that it would construct for the
    official ``flash_attention_2`` backend. Registering only the attention
    callable leaves the custom backend outside the mask registry and can route
    it through the generic external-backend policy.
    """
    if "odmkv_pertok" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["odmkv_pertok"] = odmkv_attention_forward
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    except ImportError as exc:  # pragma: no cover - supported HF has registry
        raise RuntimeError(
            "odmkv requires Transformers attention-mask registry support"
        ) from exc
    # AttentionMaskInterface instances keep a local mapping, while model mask
    # construction may instantiate a fresh interface. ``register`` updates the
    # global mapping used by all instances; plain item assignment would make
    # this function appear registered here but remain invisible upstream.
    global_mapping = getattr(ALL_MASK_ATTENTION_FUNCTIONS, "_global_mapping", {})
    if "odmkv_pertok" not in global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            "odmkv_pertok",
            ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"],
        )
