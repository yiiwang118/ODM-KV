"""benchmark/core eval-pipeline adapter for the new per-token ODMNativePress."""
from __future__ import annotations

from typing import Optional

import torch

from kvquant.runtime.press import ODMNativePress
from kvquant.runtime.storage.cache import ODMCache


class ODMNativeBackend:
    def __init__(
        self,
        *,
        scorer,
        bit_levels: tuple[int, ...] = (0, 2, 3, 4, 8, 16),
        target_avg_bits: float = 2.0,
        eviction_cost: float = 0.5,
        above_target_alpha: float = 1.0,
        n_outlier_channels: int = 0,
        outlier_min_bits: int = 3,
        sink_tokens: int = 4,
        score_sink_tokens: int = 4,
        buffer_size: int = 128,
        layerwise: bool = True,
        decode_quant: bool = True,
        allow_decode_eviction: bool = False,
        initial_layers_fp16: int = 0,
        key_quantizer: str = "mse",
        value_quantizer: str = "mse",
        value_group_size: int = 32,
        seed: int = 42,
    ):
        self.scorer = scorer
        self.bit_levels = tuple(bit_levels)
        self.target_avg_bits = target_avg_bits
        self.eviction_cost = eviction_cost
        self.above_target_alpha = above_target_alpha
        self.n_outlier_channels = n_outlier_channels
        self.outlier_min_bits = outlier_min_bits
        self.sink_tokens = sink_tokens
        self.score_sink_tokens = score_sink_tokens
        self.buffer_size = buffer_size
        self.layerwise = layerwise
        self.decode_quant = bool(decode_quant)
        self.allow_decode_eviction = bool(allow_decode_eviction)
        self.initial_layers_fp16 = int(initial_layers_fp16)
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
        self.value_group_size = value_group_size
        self.seed = seed
        self.cache: Optional[ODMCache] = None

    def make_compressed_cache(self, model) -> ODMCache:
        cfg = model.config
        num_layers = cfg.num_hidden_layers
        H_kv = cfg.num_key_value_heads
        # Prefer the explicit head_dim (some models, e.g. Qwen3, set head_dim
        # independent of hidden_size / num_attention_heads).
        D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        device = next(model.parameters()).device
        self.cache = ODMCache(
            num_layers=num_layers,
            num_heads_kv=H_kv,
            head_dim=D,
            device=device,
            bit_levels=self.bit_levels,
            key_quantizer=self.key_quantizer,
            value_quantizer=self.value_quantizer,
            value_group_size=self.value_group_size,
            outlier_min_bits=self.outlier_min_bits,
            seed=self.seed,
        )
        return self.cache

    def __call__(self, model):
        if self.cache is None:
            self.make_compressed_cache(model)
        inner = ODMNativePress(
            cache=self.cache,
            scorer=self.scorer,
            bit_levels=self.bit_levels,
            target_avg_bits=self.target_avg_bits,
            eviction_cost=self.eviction_cost,
            above_target_alpha=self.above_target_alpha,
            n_outlier_channels=self.n_outlier_channels,
            outlier_min_bits=self.outlier_min_bits,
            sink_tokens=self.sink_tokens,
            score_sink_tokens=self.score_sink_tokens,
            buffer_size=self.buffer_size,
            layerwise=self.layerwise,
            decode_quant=self.decode_quant,
            allow_decode_eviction=self.allow_decode_eviction,
            initial_layers_fp16=self.initial_layers_fp16,
            seed=self.seed,
        )
        # Expose the active inner press (holds the hooks + compute_flush_bits) so
        # a CUDA-graph decoder can flush the fixed ring through identical logic.
        self._inner = inner
        return inner(model)
