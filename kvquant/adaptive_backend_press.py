from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from kvquant.allocator import ratio_scores_to_bits, scores_to_bits, optimal_scores_to_bits, calibrate_epsilon
from kvquant.attention_utils import (
    _aligned_rotary_outputs,
    _apply_rotary_pos_emb_q,
    _get_query_states,
    _repeat_kv,
)
from kvquant.press import TurboQuantPerTokenPress
from kvquant.tq_adaptive_backend import TurboQuantAdaptiveKVCacheState


class TurboQuantPerTokenBackendPress(TurboQuantPerTokenPress):
    """Adaptive backend that stores per-token mixed-precision KV externally.

    This keeps the scorer/allocator semantics from ``TurboQuantPerTokenPress``,
    but writes the assigned tokens into a backend state instead of quantizing the
    HF cache in place. The current HF integration still materializes the full
    dequantized cache after each hook so standard attention can run unchanged.
    """

    def __init__(
        self,
        *,
        value_group_size: int = 32,
        key_quantizer: str = "prod",
        value_quantizer: str = "minmax",
        scorer=None,
        bit_levels: tuple[int, ...] = (0, 2, 4, 8, 16),
        ratios: Optional[tuple[float, ...]] = None,
        target_avg_bits: Optional[float] = None,
        eviction_cost: float = 5.0,
        above_target_alpha: float = 1.0,
        n_outlier_channels: int = 0,
        outlier_min_bits: int = 4,
        seed: int = 42,
        decode_quant: bool = True,
        sink_tokens: int = 0,
        layerwise: bool = False,
        buffer_size: int = 128,
        initial_layers_fp16: int = 0,
        allow_decode_eviction: bool = False,
    ):
        min_bits = 2 if key_quantizer in ("prod", "turboquant_prod", "turboquantprod") else 1
        invalid = [bit for bit in bit_levels if bit not in (0, 16) and bit < min_bits]
        if invalid:
            raise ValueError(
                f"TurboQuantPerTokenBackendPress does not support key bit-levels below "
                f"{min_bits} with key_quantizer={key_quantizer!r}: {invalid}"
            )
        super().__init__(
            scorer=scorer,
            bit_levels=bit_levels,
            ratios=ratios,
            target_avg_bits=target_avg_bits,
            eviction_cost=eviction_cost,
            above_target_alpha=above_target_alpha,
            n_outlier_channels=n_outlier_channels,
            outlier_min_bits=outlier_min_bits,
            seed=seed,
            decode_quant=decode_quant,
            sink_tokens=sink_tokens,
            layerwise=layerwise,
            quantizer="turboquant",
            value_quantizer=value_quantizer,
            value_group_size=value_group_size,
            allow_decode_eviction=allow_decode_eviction,
        )
        self.key_quantizer = key_quantizer
        self.outlier_min_bits = outlier_min_bits
        self.buffer_size = buffer_size
        self.initial_layers_fp16 = initial_layers_fp16
        self._states: dict[int, TurboQuantAdaptiveKVCacheState] = {}
        self._buffer_counts: dict[int, int] = {}
        self._buffer_start_seqs: dict[int, int] = {}
        # OCS: per-layer outlier indices, detected once during first prefill
        self._outlier_indices: Optional[dict[int, torch.Tensor]] = None
        self._regular_indices: Optional[dict[int, torch.Tensor]] = None
        # Cached per-(layer, head) observed bit ratios from prefill Lagrangian.
        # Used during decode buffer flush to skip re-solving Lagrangian.
        # Key: (layer_idx, head_idx) → tuple of (bit_level, fraction).
        self._observed_ratios: dict[tuple[int, int], tuple[tuple[int, float], ...]] = {}

    def _prewarm_quantizers(self, model: PreTrainedModel) -> None:
        # The adaptive backend creates local quantizers lazily inside its state.
        return

    def _get_or_create_state(
        self, layer_idx: int, keys: torch.Tensor,
    ) -> TurboQuantAdaptiveKVCacheState:
        state = self._states.get(layer_idx)
        if state is not None:
            return state
        # Ensure bit_levels includes 16 when buffer_size > 0, since buffer
        # tokens are assigned bit=16 and the state iterates only over known levels
        levels = self.bit_levels
        if self.buffer_size > 0 and 16 not in levels:
            levels = tuple(sorted(set(levels) | {16}))
        # OCS: get outlier indices for this layer (if detected)
        out_idx = None
        reg_idx = None
        if self._outlier_indices is not None and layer_idx in self._outlier_indices:
            out_idx = self._outlier_indices[layer_idx]
            reg_idx = self._regular_indices[layer_idx]

        state = TurboQuantAdaptiveKVCacheState(
            head_dim=keys.shape[-1],
            bit_levels=levels,
            key_quantizer=self.key_quantizer,
            value_quantizer=self.value_quantizer,
            value_group_size=self.value_group_size,
            device=keys.device,
            dtype=keys.dtype,
            seed=self.seed + layer_idx * 7,
            outlier_indices=out_idx,
            regular_indices=reg_idx,
            outlier_min_bits=self.outlier_min_bits,
        )
        self._states[layer_idx] = state
        return state

    def _sync_cache_from_state(
        self,
        cache,
        layer_idx: int,
        module: nn.Module,
        state: TurboQuantAdaptiveKVCacheState,
        *,
        prefill_fast: bool = False,
    ) -> None:
        if prefill_fast:
            # HF cache already has fp16 data — only overwrite quantized positions
            keys, values = self._get_cache_tensors(cache, layer_idx)
            state.materialize_quantized_into(keys, values)
        else:
            keys, values = state.materialize()
            self._set_cache_tensors(cache, layer_idx, keys, values)
        module.masked_key_indices = state.masked_key_indices()
        if hasattr(cache, "layers"):
            cache.layers[layer_idx]._tq_backend_state = state

    @torch.no_grad()
    def _detect_outlier_channels(self, cache) -> None:
        """Detect outlier channels by variance on a sampled subset of tokens.

        Samples at most 512 tokens per layer (sufficient for stable variance
        estimates) and only inspects 4 evenly-spaced layers, sharing the
        detected indices across all layers.  This reduces detection cost
        from O(L*T*d) to O(4*512*d).
        """
        if self.n_outlier_channels <= 0:
            return
        self._outlier_indices = {}
        self._regular_indices = {}

        all_layers = sorted(self._prefill_scores)
        # Sample 4 representative layers
        sample_layers = [all_layers[i] for i in
                         [0, len(all_layers)//3, 2*len(all_layers)//3, -1]]
        sample_layers = sorted(set(sample_layers))

        var_accum = None
        for layer_idx in sample_layers:
            keys, _ = self._get_cache_tensors(cache, layer_idx)  # [B, H, T, d]
            T = keys.shape[2]
            # Sample at most 512 tokens
            if T > 512:
                idx = torch.linspace(0, T - 1, 512, device=keys.device).long()
                keys_sample = keys[:, :, idx, :]
            else:
                keys_sample = keys
            var = keys_sample.float().var(dim=2).mean(dim=(0, 1)).cpu()  # [d] on CPU
            var_accum = var if var_accum is None else var_accum + var

        d = var_accum.shape[0]
        n_out = min(self.n_outlier_channels, d // 2)
        out_idx_cpu = var_accum.topk(n_out).indices.sort().values
        mask = torch.ones(d, dtype=torch.bool)
        mask[out_idx_cpu] = False
        reg_idx_cpu = torch.where(mask)[0]

        # Share same outlier indices across all layers, on each layer's device
        for layer_idx in all_layers:
            layer_keys, _ = self._get_cache_tensors(cache, layer_idx)
            dev = layer_keys.device
            self._outlier_indices[layer_idx] = out_idx_cpu.to(dev)
            self._regular_indices[layer_idx] = reg_idx_cpu.to(dev)

    def _allocate_bits_via_cached_ratios(
        self,
        scores: torch.Tensor,      # [N_buf]
        positions: torch.Tensor,    # [N_buf, 3]
        num_heads: int,
        layer_idx: int,
        allowed_levels: set[int],
    ) -> Optional[torch.Tensor]:
        """Apply per-head observed ratios from prefill. Returns None if any
        head lacks a cached distribution (triggers Lagrangian fallback).

        Multi-GPU safe: all tensors (bits, sub_scores, etc.) inherit
        ``scores.device``; no cross-device transfers.
        """
        # Need cache for every head to avoid partial allocation
        if not all((layer_idx, h) in self._observed_ratios for h in range(num_heads)):
            return None

        bits = torch.empty_like(scores, dtype=torch.int32)
        head_indices = positions[:, 1]

        for h in range(num_heads):
            cached = self._observed_ratios[(layer_idx, h)]
            # Filter ratios to levels allowed at decode time. When
            # allow_decode_eviction=False this drops the 0-bit bucket; the
            # remaining fractions are renormalised.
            filtered = [(b, r) for b, r in cached if b in allowed_levels and r > 0]
            if not filtered:
                # This head would produce an empty ratio set. Bail to Lagrangian.
                return None
            total = sum(r for _, r in filtered)
            norm_pairs = [(b, r / total) for b, r in filtered]
            flush_levels, flush_ratios = zip(*norm_pairs)

            mask = head_indices == h
            if not mask.any():
                continue
            sub_scores = scores[mask]
            sub_bits = ratio_scores_to_bits(
                sub_scores, tuple(flush_levels), tuple(flush_ratios),
            )
            bits[mask] = sub_bits.to(dtype=torch.int32)

        return bits

    def _record_observed_ratios(self, layer_idx: int, bits: torch.Tensor) -> None:
        """Cache per-(layer, head) observed bit distribution from the Lagrangian
        output so decode buffer flush can reuse it instead of re-solving the
        optimization problem every 128 tokens.

        Parameters
        ----------
        layer_idx : int
        bits : [B, H_kv, T]  — Lagrangian allocation BEFORE the window-retention
            override. We slice out the sink prefix and the buffer protection
            window so the cached distribution reflects the *bulk* prefill
            allocation, not tail positions that the override forced to 16.
            This avoids biasing flush-time allocation toward high bits.
        """
        if bits.dim() != 3:
            return   # unexpected shape; skip
        B, H_kv, T = bits.shape
        if T == 0:
            return

        # Trim sink (head) and protection window (tail) — these positions
        # don't reflect the Lagrangian's natural distribution over regular
        # context tokens. Flush will allocate bits to *decode* tokens whose
        # role mirrors bulk context, not sink or tail.
        start = min(int(self.sink_tokens), T) if self.sink_tokens > 0 else 0
        end = T - min(int(self.buffer_size), T) if self.buffer_size > 0 else T
        if end <= start:
            # Whole prefill is sink + protection window → fall back to full
            start, end = 0, T
        bulk = bits[..., start:end]

        T_bulk = bulk.shape[-1]
        if T_bulk == 0:
            return

        flat = bulk.reshape(B * H_kv, T_bulk)
        for row_idx in range(B * H_kv):
            h = row_idx % H_kv
            row = flat[row_idx]
            uniq, counts = torch.unique(row, return_counts=True)
            total = counts.sum().item()
            if total == 0:
                continue
            fracs = (counts.float() / total).cpu().tolist()
            pairs = tuple(sorted(zip(uniq.cpu().tolist(), fracs)))
            # Keyed by (layer, head); overwrite on new prefill. Multi-GPU safe
            # because we only store Python scalars, not device tensors.
            self._observed_ratios[(layer_idx, h)] = pairs

    def _finalize_prefill(self, cache) -> None:
        # Detect outlier channels before creating states (one-time)
        if self._outlier_indices is None and self.n_outlier_channels > 0:
            self._detect_outlier_channels(cache)

        bits_by_layer = self._allocate_bits_for_prefill()

        for layer_idx, bits in bits_by_layer.items():
            keys, values = self._get_cache_tensors(cache, layer_idx)

            # Capture per-(layer, head) bit-ratios BEFORE the window override,
            # so decode flush can reuse the Lagrangian-solved distribution
            # without re-running the solver. Source of truth = the Lagrangian
            # output over the full prefill span (excluding sink / window).
            self._record_observed_ratios(layer_idx, bits)

            if layer_idx < self.initial_layers_fp16:
                bits = torch.full_like(bits, 16)
            elif self.buffer_size > 0:
                seq_len = bits.shape[-1]
                n_protect = min(self.buffer_size, seq_len)
                bits = bits.clone()
                bits[..., -n_protect:] = 16
            state = self._get_or_create_state(layer_idx, keys)
            state.prefill(keys, values, bits.to(dtype=torch.int32, device=keys.device))
            module = self._layer_modules.get(layer_idx)
            if module is not None:
                self._sync_cache_from_state(cache, layer_idx, module, state,
                                            prefill_fast=True)
            if self.buffer_size > 0:
                self._buffer_start_seqs[layer_idx] = int(keys.shape[2])
                self._buffer_counts[layer_idx] = 0

        self._reset_prefill_state()

    def _forward_hook(self, module: nn.Module, args, kwargs, output):
        cache = kwargs.get("past_key_values")
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if cache is None or hidden_states is None:
            return output

        layer_idx = int(module.layer_idx)
        self._layer_modules[layer_idx] = module
        keys, values = self._get_cache_tensors(cache, layer_idx)
        q_len = hidden_states.shape[1]
        cache_len = keys.shape[2]
        is_prefill = cache_len == q_len

        if is_prefill:
            # Fused-FA2 path (optional): per-layer col_sum was stashed by the
            # attention kernel; pop it and forward to the scorer. Only valid
            # for B=1 — fused impl skips col_sum capture otherwise.
            col_sum_layer = None
            if keys.shape[0] == 1:
                try:
                    from kvquant.fused_attention_patch import get_col_sum_buffer
                    col_sum_layer = get_col_sum_buffer().pop(layer_idx, None)
                except ImportError:
                    pass

            layer_scores: list[torch.Tensor] = []
            for batch_idx in range(keys.shape[0]):
                k_layer = keys[batch_idx]
                v_layer = values[batch_idx]
                h_batch = hidden_states[batch_idx : batch_idx + 1]
                scores = self._score_prefill(
                    k_layer,
                    v_layer,
                    layer_idx,
                    module=module,
                    hidden_states=h_batch,
                    col_sum=col_sum_layer,
                )
                layer_scores.append(scores)
                self._prefill_hooks_fired += 1

            self._prefill_scores[layer_idx] = torch.stack(layer_scores, dim=0)
            self._prefill_sink_masks[layer_idx] = self._build_sink_mask(
                keys.shape[1], keys.shape[2], keys.device,
            )

            should_finalize = (
                self._last_hooked_layer_idx is None
                or layer_idx == self._last_hooked_layer_idx
            )
            if should_finalize:
                self._finalize_prefill(cache)
            return output

        if not self.decode_quant:
            return output

        if layer_idx < self.initial_layers_fp16:
            return output

        state = self._get_or_create_state(layer_idx, keys)

        if self.buffer_size > 0:
            buf_start = self._buffer_start_seqs.get(layer_idx)
            if buf_start is not None and buf_start > state.seq_len:
                self._buffer_start_seqs[layer_idx] = state.seq_len
                self._buffer_counts[layer_idx] = 0

        new_k = keys[:, :, -q_len:, :].contiguous()
        new_v = values[:, :, -q_len:, :].contiguous()

        if self.buffer_size > 0:
            # Fast path: every new decode token is pinned to the exact_bank
            # until the buffer fills and flushes. Skip the per-level loop in
            # state.append (5× mask.any syncs + nonzero + fancy-indexing) and
            # route directly — byte-identical to append(..., bits=full(16)).
            state.append_all_exact(new_k, new_v)

            # Multi-token non-prefill call (e.g. question-encoding pass in the
            # pipeline: cache already holds context, now the question ids are
            # forwarded in one shot). Semantically these are *part of the
            # prompt*, not generated output. Advance buffer_start past them so
            # the flush-buffer re-quantizer cannot reach them — they stay fp16
            # in ExactBank, like the prefill protection window.
            if q_len > 1:
                self._buffer_start_seqs[layer_idx] = state.seq_len
                module.masked_key_indices = state.masked_key_indices()
                if hasattr(cache, "layers"):
                    cache.layers[layer_idx]._tq_backend_state = state
                return output
        else:
            batch_bits: list[torch.Tensor] = []
            for batch_idx in range(keys.shape[0]):
                h_batch = hidden_states[batch_idx : batch_idx + 1]
                if q_len == 1:
                    scores = self._score_decode(
                        new_k[batch_idx, :, 0, :],
                        new_v[batch_idx, :, 0, :],
                        layer_idx,
                        module=module,
                        hidden_states=h_batch,
                    ).unsqueeze(-1)
                else:
                    scores = self._score_prefill(
                        new_k[batch_idx],
                        new_v[batch_idx],
                        layer_idx,
                        module=module,
                        hidden_states=h_batch,
                    )
                batch_bits.append(scores_to_bits(scores, self._decode_bit_levels))
            bits = torch.stack(batch_bits, dim=0).to(dtype=torch.int32, device=keys.device)
            state.append(new_k, new_v, bits)

        # Buffer-flush
        if self.buffer_size > 0:
            self._buffer_counts[layer_idx] = self._buffer_counts.get(layer_idx, 0) + q_len
            if self._buffer_counts[layer_idx] >= self.buffer_size:
                self._flush_decode_buffer(
                    state, layer_idx, cache, module, hidden_states,
                )
                return output

            module.masked_key_indices = state.masked_key_indices()
            if hasattr(cache, "layers"):
                cache.layers[layer_idx]._tq_backend_state = state
            return output

        self._sync_cache_from_state(cache, layer_idx, module, state, prefill_fast=True)
        return output

    # ------------------------------------------------------------------
    # Decode buffer flush
    # ------------------------------------------------------------------

    def _flush_decode_buffer(
        self,
        state: TurboQuantAdaptiveKVCacheState,
        layer_idx: int,
        cache,
        module: nn.Module,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        """Score buffer tokens and re-quantize them to meet the bit budget.

        Uses the current decode query to compute attention weights over the
        buffer tokens (cost: one q_proj + one small matmul per flush).
        Falls back to norm-based scoring when hidden_states is unavailable.
        """
        buffer_start = self._buffer_start_seqs.get(layer_idx)
        if buffer_start is None:
            self._sync_cache_from_state(cache, layer_idx, module, state, prefill_fast=True)
            return

        positions, buf_keys, buf_values = state.get_buffer_tokens(buffer_start)
        n_buf = positions.shape[0]
        if n_buf == 0:
            # Nothing to flush — reset tracking so next cycle starts clean.
            self._buffer_start_seqs[layer_idx] = state.seq_len
            self._buffer_counts[layer_idx] = 0
            self._sync_cache_from_state(cache, layer_idx, module, state, prefill_fast=True)
            return

        # ── Score buffer tokens ──────────────────────────────────────
        # Use the configured scorer's score_with_attn so prefill and decode
        # apply the SAME formula. Falls back to a generic NormScorer-style
        # ‖v‖² when no scorer is set (e.g. baseline runs).

        # Attention-based scoring (cheap: single query against buffer keys)
        attn_weights = self._flush_attention(
            module, hidden_states, buf_keys, positions, state,
        )

        if attn_weights is not None and self.scorer is not None:
            # Score per (batch, head) group: scorer expects [H, T, d] but
            # buffer tokens may have variable T per head. Flatten into [N]
            # then dispatch group-by-group.
            batch_indices = positions[:, 0]
            head_indices = positions[:, 1]
            B = int(batch_indices.max().item()) + 1 if positions.numel() > 0 else 1
            H_kv = state.num_heads

            scores = torch.zeros_like(attn_weights, dtype=torch.float32)
            for b in range(B):
                for h in range(H_kv):
                    bh_mask = (batch_indices == b) & (head_indices == h)
                    if not bh_mask.any():
                        continue
                    k_bh = buf_keys[bh_mask].unsqueeze(0)        # [1, T_bh, d]
                    v_bh = buf_values[bh_mask].unsqueeze(0)
                    a_bh = attn_weights[bh_mask].unsqueeze(0)    # [1, T_bh]
                    s_bh = self.scorer.score_with_attn(
                        k_bh, v_bh, a_bh, module=module,
                    )                                             # [1, T_bh]
                    scores[bh_mask] = s_bh.squeeze(0).float()
        elif attn_weights is not None:
            # No scorer available: simple p² · ‖v‖² fallback (matches mu_only_alpha2)
            scores = attn_weights.pow(2) * buf_values.pow(2).sum(-1)
        else:
            # Norm-only fallback (no current attention available)
            scores = buf_keys.pow(2).sum(-1) + buf_values.pow(2).sum(-1)

        # Normalize to [0, 1]
        lo, hi = scores.min(), scores.max()
        if hi - lo > 1e-8:
            scores = (scores - lo) / (hi - lo)
        else:
            scores.fill_(0.5)

        # ── Allocate bits. Priority order:
        #    1. fixed ratios (YAML-configured) — unchanged
        #    2. cached per-(layer, head) ratios from prefill Lagrangian  ← NEW
        #    3. re-solve Lagrangian (fallback, expensive)
        #    4. uniform partition
        #
        # Whether 0-bit is included in the candidate pool is controlled by
        # self._decode_bit_levels, which respects the allow_decode_eviction flag.
        levels = self._decode_bit_levels
        allowed_levels = set(levels)
        bits = None

        if self.ratios is not None:
            pairs = [(b, r) for b, r in zip(self.bit_levels, self.ratios)
                     if b in allowed_levels]
            if pairs:
                flush_levels, flush_ratios = zip(*pairs)
                bits = ratio_scores_to_bits(
                    scores, tuple(flush_levels), tuple(flush_ratios),
                )

        elif self.target_avg_bits is not None:
            # Try cached per-head ratios first — skip Lagrangian re-solve.
            bits = self._allocate_bits_via_cached_ratios(
                scores, positions, state.num_heads, layer_idx, allowed_levels,
            )
            if bits is None:
                # Fallback: Lagrangian (rare — only if no prefill cache yet)
                if self._epsilon is None:
                    d = self._head_dim or 128
                    self._epsilon = calibrate_epsilon(
                        d, self.bit_levels, seed=self.seed,
                        device=scores.device,
                        value_group_size=self.value_group_size,
                        n_outlier=self.n_outlier_channels,
                        outlier_min_bits=self.outlier_min_bits,
                        value_quantizer=self.value_quantizer,
                    )
                flush_eps = {b: self._epsilon.get(b, 0.0) for b in levels}
                bits = optimal_scores_to_bits(
                    scores, levels, self.target_avg_bits, flush_eps,
                    eviction_cost=self.eviction_cost,
                    n_outlier=self.n_outlier_channels,
                    head_dim=self._head_dim or 128,
                    outlier_min_bits=self.outlier_min_bits,
                    above_target_alpha=self.above_target_alpha,
                )

        if bits is None:
            bits = scores_to_bits(scores, levels)

        state.flush_buffer(buffer_start, bits)

        # Advance buffer window
        self._buffer_start_seqs[layer_idx] = state.seq_len
        self._buffer_counts[layer_idx] = 0

        self._sync_cache_from_state(cache, layer_idx, module, state, prefill_fast=True)

    # ------------------------------------------------------------------
    # Flush-time attention helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _flush_attention(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor | None,
        buf_keys: torch.Tensor,
        positions: torch.Tensor,
        state: TurboQuantAdaptiveKVCacheState,
    ) -> torch.Tensor | None:
        """Compute attention of the current query over buffer tokens.

        Returns flat ``[N_buf]`` attention weights (after softmax over buffer
        entries only), or *None* if attention cannot be computed.

        Cost: one ``q_proj`` + one ``[H_q, 1, N_buf]`` matmul per layer.
        """
        if hidden_states is None or not hasattr(module, "layer_idx"):
            return None

        try:
            query_states = _get_query_states(module, hidden_states)  # [B, H_q, q, d]
        except (AttributeError, NotImplementedError, ValueError):
            return None

        # Apply RoPE to query
        rotary_emb = getattr(module, "rotary_emb", None)
        if rotary_emb is not None:
            seq_len = state.seq_len
            pos_ids = torch.arange(
                seq_len - 1, seq_len, device=query_states.device,
            ).unsqueeze(0)
            cos, sin = _aligned_rotary_outputs(rotary_emb, query_states, pos_ids)
            query_states = _apply_rotary_pos_emb_q(query_states, cos, sin)

        # query_states: [B, H_q, q, d] — use last query token per batch
        B = query_states.shape[0]
        H_q = query_states.shape[1]
        H_kv = state.num_heads
        n_groups = H_q // H_kv
        d = query_states.shape[-1]
        scale = math.sqrt(d)

        # Buffer entries are flat [N, d] with positions [N, 3] = (batch, head, seq).
        batch_indices = positions[:, 0]  # [N]
        head_indices = positions[:, 1]   # [N]

        # Per-batch query: [B, H_q, d] → GQA mean → [B, H_kv, d]
        q_last = query_states[:, :, -1, :]  # [B, H_q, d]
        q_kv = q_last.reshape(B, H_kv, n_groups, d).mean(dim=2)  # [B, H_kv, d]

        # Gather the matching query for each buffer entry using its batch+head
        q_per_entry = q_kv[batch_indices, head_indices]  # [N, d]

        # Dot product: [N]
        logits = (q_per_entry * buf_keys).sum(-1) / scale

        # Softmax per (batch, head) — standard attention normalisation
        attn = torch.zeros_like(logits)
        for b in range(B):
            for h in range(H_kv):
                bh_mask = (batch_indices == b) & (head_indices == h)
                if bh_mask.any():
                    attn[bh_mask] = torch.softmax(logits[bh_mask], dim=0)

        return attn

    @contextlib.contextmanager
    def __call__(self, model: PreTrainedModel):
        self._states = {}
        self._buffer_counts = {}
        self._buffer_start_seqs = {}
        self._outlier_indices = None
        self._regular_indices = None
        try:
            with super().__call__(model):
                yield
        finally:
            self._states = {}
            self._buffer_counts = {}
            self._buffer_start_seqs = {}
            self._outlier_indices = None
            self._regular_indices = None
