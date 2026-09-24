"""Readable, dense-cache reference implementation of ODM-KV.

The reference reconstructs quantized K/V for ordinary attention. Physical
compressed-cache inference lives in kvquant.runtime.
"""
from __future__ import annotations

import contextlib
import math

import torch

from kvquant.allocator import DEFAULT_BIT_LEVELS, calibrate_epsilon, optimal_scores_to_bits
from kvquant.attention_patch import register_reference_attention
from kvquant.attention_utils import _aligned_rotary_outputs, _apply_rotary_pos_emb_q, _get_query_states
from kvquant.scorer import ODMScorer
from kvquant.tq_adaptive_backend import TurboQuantAdaptiveKVCacheState


class ODMPress:
    """Compress each layer after prefill and each full decode buffer.

    The bit budget is per request and per layer. Sinks and the prefill tail
    stay exact. Generated tokens cannot be evicted; each flush solves a new
    allocation over the nonzero levels, as in the native backend.
    """

    decode_quant = True
    layerwise = True

    def __init__(
        self, *, target_avg_bits=2.0, scorer=None, bit_levels=DEFAULT_BIT_LEVELS,
        sink_tokens=4, buffer_size=128, eviction_cost=0.5, seed=42,
        n_outlier_channels=0, outlier_min_bits=3, value_group_size=32,
    ):
        if not math.isfinite(target_avg_bits) or not 0 <= target_avg_bits <= 16:
            raise ValueError("target_avg_bits must be finite and in [0,16]")
        if sink_tokens < 0 or buffer_size < 1:
            raise ValueError("invalid sink or decode-buffer size")
        if tuple(bit_levels) != DEFAULT_BIT_LEVELS:
            raise ValueError(f"ODM-KV uses bit_levels={DEFAULT_BIT_LEVELS}")
        if n_outlier_channels < 0 or outlier_min_bits not in (2, 3, 4, 8):
            raise ValueError("invalid outlier-channel settings")
        self.target_avg_bits = float(target_avg_bits)
        self.bit_levels = tuple(bit_levels)
        self.sink_tokens = int(sink_tokens)
        self.buffer_size = int(buffer_size)
        self.eviction_cost = float(eviction_cost)
        self.seed = int(seed)
        self.n_outlier_channels = int(n_outlier_channels)
        self.outlier_min_bits = int(outlier_min_bits)
        self.value_group_size = int(value_group_size)
        self.scorer = scorer or ODMScorer(n_sink=self.sink_tokens)
        if not isinstance(self.scorer, ODMScorer) or self.scorer.n_sink != self.sink_tokens:
            raise ValueError("ODMPress requires ODMScorer with the same sink count")
        self._states = {}
        self._buffer_start = {}
        self._epsilon = {}
        self._active = False
        self._prefill_hooks_fired = 0

    def _allocate(self, scores, dim, *, decode=False):
        levels = tuple(b for b in self.bit_levels if b) if decode else self.bit_levels
        if decode and self.target_avg_bits <= levels[0]:
            return torch.full_like(scores, levels[0], dtype=torch.int32)
        if dim not in self._epsilon:
            self._epsilon[dim] = calibrate_epsilon(
                dim, self.bit_levels, seed=self.seed, device=scores.device,
                value_quantizer="mse", value_group_size=self.value_group_size,
                n_outlier=self.n_outlier_channels, outlier_min_bits=self.outlier_min_bits,
            )
        sink = None
        if not decode and self.sink_tokens:
            sink = torch.zeros_like(scores, dtype=torch.bool)
            sink[..., :self.sink_tokens] = True
        return optimal_scores_to_bits(
            scores, levels, self.target_avg_bits, self._epsilon[dim], sink,
            eviction_cost=self.eviction_cost, n_outlier=self.n_outlier_channels,
            head_dim=dim, outlier_min_bits=self.outlier_min_bits,
        )

    def _new_state(self, keys, layer_idx):
        dim = keys.shape[-1]
        outliers = regular = None
        if self.n_outlier_channels:
            if self.n_outlier_channels >= dim:
                raise ValueError("n_outlier_channels must be smaller than head_dim")
            sampled = keys[:, :, ::max(1, keys.shape[2] // 512)].float()
            variance = sampled.var(dim=2, unbiased=False).mean(dim=(0, 1))
            outliers = variance.topk(self.n_outlier_channels).indices.sort().values
            mask = torch.ones(dim, device=keys.device, dtype=torch.bool)
            mask[outliers] = False
            regular = mask.nonzero(as_tuple=True)[0]
        return TurboQuantAdaptiveKVCacheState(
            head_dim=dim, bit_levels=self.bit_levels, key_quantizer="mse",
            value_quantizer="mse", value_group_size=self.value_group_size,
            device=keys.device, dtype=keys.dtype, seed=self.seed + layer_idx * 7,
            outlier_indices=outliers, regular_indices=regular,
            outlier_min_bits=self.outlier_min_bits,
        )

    @staticmethod
    def _sync(layer, module, state):
        state.materialize_quantized_into(layer.keys, layer.values)
        module.masked_key_indices = state.masked_key_indices()
        layer._tq_backend_state = state

    def _prefill(self, module, hidden, layer, layer_idx):
        keys, values = layer.keys, layer.values
        scores = [
            self.scorer.score_prefill(k, v, layer_idx, module=module, hidden_states=hidden[b:b+1])
            for b, (k, v) in enumerate(zip(keys, values))
        ]
        bits = torch.stack([self._allocate(s, keys.shape[-1]) for s in scores])
        bits[..., -min(self.buffer_size, keys.shape[2]):] = 16
        state = self._new_state(keys, layer_idx)
        state.prefill(keys, values, bits)
        self._states[layer_idx] = state
        self._buffer_start[layer_idx] = state.seq_len
        self._sync(layer, module, state)
        self._prefill_hooks_fired += 1

    def _flush(self, module, hidden, layer, state, layer_idx, position_embeddings=None):
        positions, keys, values = state.get_buffer_tokens(self._buffer_start[layer_idx])
        if not positions.numel():
            return
        if self.target_avg_bits <= 2:
            bits = torch.full((len(positions),), 2, dtype=torch.int32, device=keys.device)
        else:
            query = _get_query_states(module, hidden[:, -1:])
            if position_embeddings is None:
                pos = torch.tensor([[state.seq_len - 1]], device=query.device)
                cos, sin = _aligned_rotary_outputs(module.rotary_emb, query, pos)
            else:
                cos, sin = position_embeddings
                cos, sin = cos[:, -1:], sin[:, -1:]
            query = _apply_rotary_pos_emb_q(query, cos, sin).float()[:, :, 0]
            groups = query.shape[1] // state.num_heads
            bits = torch.empty(len(positions), dtype=torch.int32, device=keys.device)
            # Preserve softmax per Q head before averaging the GQA group.
            for b in range(state.batch_size):
                scores = torch.zeros(len(positions), dtype=torch.float32, device=keys.device)
                for h in range(state.num_heads):
                    selected = (positions[:, 0] == b) & (positions[:, 1] == h)
                    if not selected.any():
                        continue
                    q = query[b, h * groups:(h + 1) * groups]
                    logits = q @ keys[selected].float().T / math.sqrt(keys.shape[-1])
                    p = torch.softmax(logits, dim=-1).mean(dim=0)
                    scores[selected] = self.scorer.score_with_attn(
                        keys[selected].unsqueeze(0), values[selected].unsqueeze(0), p.unsqueeze(0),
                    )[0]
                request = positions[:, 0] == b
                bits[request] = self._allocate(scores[request], keys.shape[-1], decode=True)
        state.flush_buffer(self._buffer_start[layer_idx], bits)
        self._buffer_start[layer_idx] = state.seq_len
        self._sync(layer, module, state)

    @torch.no_grad()
    def _forward_hook(self, module, args, kwargs, output):
        cache = kwargs.get("past_key_values")
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if cache is None or hidden is None:
            return output
        idx = int(module.layer_idx)
        layer = cache.layers[idx]
        q_len = hidden.shape[1]
        if layer.keys.shape[2] == q_len:
            self._prefill(module, hidden, layer, idx)
            return output
        state = self._states.get(idx)
        if state is None:
            raise RuntimeError("ODMPress must cover both prefill and decode for the same cache")
        # An evaluation runner may rewind the cache between questions.
        previous_len = layer.keys.shape[2] - q_len
        if state.seq_len > previous_len:
            state.truncate(previous_len)
        if self._buffer_start[idx] > previous_len:
            self._buffer_start[idx] = previous_len
        state.append_all_exact(layer.keys[:, :, -q_len:], layer.values[:, :, -q_len:])
        if q_len > 1:
            # The evaluation question is appended as an exact prompt suffix.
            self._buffer_start[idx] = state.seq_len
        elif state.seq_len - self._buffer_start[idx] >= self.buffer_size:
            self._flush(module, hidden, layer, state, idx, kwargs.get("position_embeddings"))
        module.masked_key_indices = state.masked_key_indices()
        layer._tq_backend_state = state
        return output

    @staticmethod
    def _validate_inputs(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is not None and (mask.ndim != 2 or not bool((mask == 1).all())):
            raise ValueError("reference ODMPress requires unpadded input; use the native backend for left padding")

    @contextlib.contextmanager
    def __call__(self, model):
        """Apply ODM-KV for a complete prefill + generation request."""
        if self._active:
            raise RuntimeError("an ODMPress instance cannot serve overlapping requests")
        backbone = getattr(model, "model", model)
        if not hasattr(backbone, "layers") or not hasattr(backbone, "rotary_emb"):
            raise ValueError("ODM-KV requires a Llama/Qwen-style decoder with rotary embeddings")
        register_reference_attention()
        old_impl = model.config._attn_implementation
        hooks, saved = [], []
        self._states, self._buffer_start = {}, {}
        self._active = True
        try:
            model.config._attn_implementation = "odmkv_reference"
            hooks.append(backbone.register_forward_pre_hook(self._validate_inputs, with_kwargs=True))
            for idx, block in enumerate(backbone.layers):
                module = block.self_attn
                window = getattr(module, "sliding_window", None)
                if window not in (None, 0):
                    raise ValueError("sliding-window attention is not supported")
                saved.append((module, getattr(module, "rotary_emb", None), hasattr(module, "rotary_emb")))
                module.rotary_emb = backbone.rotary_emb
                module.masked_key_indices = None
                hooks.append(module.register_forward_hook(self._forward_hook, with_kwargs=True))
            yield self
        finally:
            for handle in hooks:
                handle.remove()
            for module, rotary, existed in saved:
                if existed:
                    module.rotary_emb = rotary
                else:
                    del module.rotary_emb
                module.masked_key_indices = None
            model.config._attn_implementation = old_impl
            self._states, self._buffer_start = {}, {}
            self._active = False
