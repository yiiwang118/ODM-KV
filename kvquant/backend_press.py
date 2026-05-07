from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from kvquant.base_press import BasePress
from kvquant.tq_backend import TurboQuantKVCacheState


@dataclass
class TurboQuantBackendPress(BasePress):
    """TurboQuant-style backend for HF evaluation.

    Stores compressed historical KV externally per layer, keeps a small recent
    exact buffer, and materializes the full dequantized cache back into the HF
    cache tensors after each hook so the standard attention path can run
    unchanged on subsequent calls.
    """

    key_bits: int = 3
    value_bits: int = 2
    key_quantizer: str = "mse"
    value_quantizer: str = "minmax"
    value_group_size: int = 32
    buffer_size: int = 128
    initial_layers_count: int = 0
    initial_layers_key_bits: Optional[int] = None
    seed: int = 42
    decode_quant: bool = True

    _states: dict[int, TurboQuantKVCacheState] = field(default_factory=dict, init=False, repr=False)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return keys, values

    def _layer_key_bits(self, layer_idx: int) -> int:
        if self.initial_layers_count > 0 and self.initial_layers_key_bits is not None and layer_idx < self.initial_layers_count:
            return self.initial_layers_key_bits
        return self.key_bits

    @staticmethod
    def _get_cache_tensors(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values

    @staticmethod
    def _set_cache_tensors(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
            cache.key_cache[layer_idx] = keys
            cache.value_cache[layer_idx] = values
            return
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values

    def _get_or_create_state(self, layer_idx: int, keys: torch.Tensor) -> TurboQuantKVCacheState:
        state = self._states.get(layer_idx)
        if state is not None:
            return state
        state = TurboQuantKVCacheState(
            head_dim=keys.shape[-1],
            key_bits=self._layer_key_bits(layer_idx),
            value_bits=self.value_bits,
            value_quantizer=self.value_quantizer,
            key_quantizer_type=self.key_quantizer,
            value_group_size=self.value_group_size,
            buffer_size=self.buffer_size,
            device=keys.device,
            dtype=keys.dtype,
            seed=self.seed + layer_idx * 7,
        )
        self._states[layer_idx] = state
        return state

    def forward_hook(self, module: nn.Module, _input: tuple, kwargs: dict, output: Any) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and _input:
            hidden_states = _input[0]

        cache = kwargs.get("past_key_values")
        layer_idx = getattr(module, "layer_idx", None)
        if hidden_states is None or cache is None or layer_idx is None:
            return output

        keys, values = self._get_cache_tensors(cache, int(layer_idx))
        q_len = hidden_states.shape[1]
        cache_len = keys.shape[2]
        state = self._get_or_create_state(int(layer_idx), keys)

        if cache_len == q_len:
            state.prefill(keys, values)
            materialized_k, materialized_v = state.materialize()
            self._set_cache_tensors(cache, int(layer_idx), materialized_k, materialized_v)
        else:
            new_k = keys[:, :, -q_len:, :]
            new_v = values[:, :, -q_len:, :]
            state.append(new_k, new_v)
            # Incremental update: if buffer flushed, patch only the flushed
            # positions instead of re-dequantizing the entire cache.
            flush_info = getattr(state, "_last_flush_dequant", None)
            if flush_info is not None:
                pos, flushed_k, flushed_v = flush_info
                n = flushed_k.shape[-2]
                keys[..., pos : pos + n, :] = flushed_k
                values[..., pos : pos + n, :] = flushed_v
            # New fp16 token is already in HF cache from model forward.

        if hasattr(cache, "layers"):
            cache.layers[int(layer_idx)]._tq_backend_state = state
        return output

    @contextlib.contextmanager
    def __call__(self, model: PreTrainedModel):
        lm = model
        if hasattr(lm, "model"):
            lm = lm.model
        if hasattr(lm, "language_model"):
            lm = lm.language_model

        self._states = {}
        hooks = []
        try:
            for i, layer in enumerate(lm.layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None or getattr(attn, "is_sliding", False):
                    continue
                if not hasattr(attn, "layer_idx"):
                    attn.layer_idx = i
                hooks.append(attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for h in hooks:
                h.remove()
            self._states = {}
