# Adapted from autokv/kvpress/presses/base_press.py
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel


logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _is_quantized(layer: Any) -> bool:
    return hasattr(layer, "_quantized_keys") and hasattr(layer, "_dequantize")


def extract_keys_and_values(cache: Any, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.layers[layer_idx]
    if _is_quantized(layer):
        return layer._dequantize(layer._quantized_keys), layer._dequantize(layer._quantized_values)
    return layer.keys, layer.values


def set_keys_and_values(cache: Any, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    layer = cache.layers[layer_idx]
    if _is_quantized(layer) and hasattr(layer, "_quantize"):
        layer._quantized_keys   = layer._quantize(keys,   axis=getattr(layer, "axis_key",   -1))
        layer._quantized_values = layer._quantize(values, axis=getattr(layer, "axis_value", -1))
        layer.keys   = torch.zeros(0, dtype=keys.dtype,   device=keys.device)
        layer.values = torch.zeros(0, dtype=values.dtype, device=values.device)
        if hasattr(layer, "cumulative_length"):
            layer.cumulative_length = keys.shape[2]
        return
    layer.keys   = keys
    layer.values = values


# ── Base class ────────────────────────────────────────────────────────────────

@dataclass
class BasePress:
    """Base class for KV-cache compression methods.

    Subclasses implement compress(); the hook machinery here handles
    cache extraction / write-back and prefill-only gating.

    Parameters
    ----------
    decode_quant : bool, default False
        If True, also quantise newly generated tokens during decode.
        Only the q_len tokens added in each decode step are compressed;
        previously cached tokens are never re-quantised.
    """

    decode_quant: bool = False

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        """Optional: initialise from model config before hooks are attached."""

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_hook(self, module: nn.Module, _input: tuple, kwargs: dict, output: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and _input:
            hidden_states = _input[0]

        cache = kwargs.get("past_key_values")
        layer_idx = getattr(module, "layer_idx", None)

        if hidden_states is None or cache is None or layer_idx is None:
            return output

        layer = cache.layers[layer_idx]

        # Skip decode / question-encoding steps — only compress during prefill.
        # We use cache length vs current input length instead of cache_position,
        # because transformers >= 5.4 no longer passes cache_position to attention
        # hooks when position_ids is explicitly provided by the caller.
        # A fresh prefill has cache_len == q_len (cache was just populated by this
        # call). Any call with past context has cache_len > q_len.
        keys, values = extract_keys_and_values(cache, layer_idx)
        q_len        = hidden_states.shape[1]
        cache_len    = keys.shape[2]
        attentions   = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 else None

        if cache_len == q_len:
            # Fresh prefill: compress all tokens.
            hook_kwargs = dict(kwargs)
            hook_kwargs["_kvpress_cache_length"] = cache_len
            hook_kwargs["_kvpress_new_length"] = q_len
            hook_kwargs["_kvpress_slice_start"] = 0
            keys, values = self.compress(module, hidden_states, keys, values, attentions, hook_kwargs)
            set_keys_and_values(cache, layer_idx, keys, values)

        elif self.decode_quant:
            # Decode / question-encoding with decode_quant enabled:
            # compress only the q_len newly added tokens (never re-touch old ones).
            new_k = keys[:, :, -q_len:, :].clone()
            new_v = values[:, :, -q_len:, :].clone()
            hook_kwargs = dict(kwargs)
            hook_kwargs["_kvpress_cache_length"] = cache_len
            hook_kwargs["_kvpress_new_length"] = q_len
            hook_kwargs["_kvpress_slice_start"] = cache_len - q_len
            new_k, new_v = self.compress(module, hidden_states, new_k, new_v, attentions, hook_kwargs)
            keys[:, :, -q_len:, :]   = new_k   # in-place: safe across all transformers versions
            values[:, :, -q_len:, :] = new_v

            # Quantized cache backends expose dequantized copies via
            # extract_keys_and_values(), so the in-place slice update above only
            # mutates local tensors. Re-serialize the full cache to persist the
            # updated tail back into the backend representation.
            if _is_quantized(layer):
                set_keys_and_values(cache, layer_idx, keys, values)

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        self.post_init_from_model(model)

        # Unwrap to the bare transformer stack
        lm = model
        if hasattr(lm, "model"):          lm = lm.model
        if hasattr(lm, "language_model"): lm = lm.language_model

        if not hasattr(lm, "layers"):
            raise ValueError(f"Cannot find transformer layers in {type(model)}")

        hooks: list = []
        try:
            for i, layer in enumerate(lm.layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None or getattr(attn, "is_sliding", False):
                    continue
                if not hasattr(attn, "layer_idx"):
                    attn.layer_idx = i
                if hasattr(lm, "rotary_emb"):
                    try: attn.rotary_emb = lm.rotary_emb
                    except Exception: pass
                hooks.append(attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for h in hooks:
                h.remove()
