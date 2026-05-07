"""TurboQuantPerTokenPress — per-(layer, head, token) mixed-precision quantisation.

Pipeline
--------
Prefill  (hook fires once per layer, after that layer's KV is written)
  scorer.score_prefill(keys_l, values_l, layer_idx) → scores [H, T]
  collect scores for every hooked layer
  last hooked layer finalises prefill:
    allocator maps scores → bits either
      - globally across all layers (kvpress-style default), or
      - independently per layer (layerwise=True)
    _apply_bits(): quantise non-zero, non-16 assignments
    0-bit positions are stored as head-wise masks; cache length is unchanged

Decode  (hook fires once per layer after new KV is appended)
  q_len == 1:
    scorer.score_decode(key, value, layer_idx)       → scores [H]
    allocator.scores_to_bits(scores.unsqueeze(-1))   → bits   [H, 1]
  q_len > 1:
    scorer.score_prefill(k_new, v_new, layer_idx)    → scores [H, q_len]
    allocator.scores_to_bits(scores)                 → bits   [H, q_len]
  _apply_bits(): quantise the full appended tail per head
"""
from __future__ import annotations

import contextlib
import inspect
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from kvquant.allocator import DEFAULT_BIT_LEVELS, scores_to_bits, ratio_scores_to_bits, optimal_scores_to_bits, calibrate_epsilon
from kvquant.attention_patch import patch_attention_functions
from kvquant.scorer import BaseScorer
from kvquant.tq_backend import normalize_value_quantizer
from kvquant.tq_backend import TurboQuantMSE


# ---------------------------------------------------------------------------
# Quant-dequant helpers
# ---------------------------------------------------------------------------

def _qd_keys(kq: TurboQuantMSE, x: torch.Tensor) -> torch.Tensor:
    """Quant(MSE) → dequant.  x: [N, d]

    Uses TurboQuantMSE (Lloyd-Max codebook) for minimum reconstruction error.
    TurboQuantMSE.quantize internally normalises, stores norms, and
    dequantize restores the scale — no external normalisation needed.
    """
    idx = kq.quantize(x)
    return kq.dequantize(idx)


def _qd_values_group(x: torch.Tensor, b: int, group_size: int = 32) -> torch.Tensor:
    """Per-group asymmetric scalar quantization for values.  x: [N, d]

    Follows the design in turboquant/kv_cache.py: divide each vector into
    groups, compute per-group min/max, quantize to b bits, dequantize.
    """
    N, d = x.shape
    n_groups = d // group_size
    remainder = d % group_size

    if remainder != 0 or n_groups == 0:
        # Fallback to per-vector scalar when d is not divisible by group_size
        return _qd_scalar(x, b)

    n_levels = 2 ** b - 1
    x_grouped = x.reshape(N, n_groups, group_size)

    v_min = x_grouped.min(dim=-1, keepdim=True).values
    v_max = x_grouped.max(dim=-1, keepdim=True).values
    scale = (v_max - v_min) / n_levels
    scale = scale.clamp(min=1e-10)

    x_q = ((x_grouped - v_min) / scale).round().clamp(0, n_levels)
    x_deq = x_q * scale + v_min
    return x_deq.reshape(N, d)


def _qd_values_mse(vq: TurboQuantMSE, x: torch.Tensor) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    idx = vq.quantize(x / norms)
    return vq.dequantize(idx) * norms


def _qd_scalar(x: torch.Tensor, b: int) -> torch.Tensor:
    """Per-vector asymmetric scalar quantization (RTN).  x: [N, d]"""
    n_levels = 2 ** b
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = scale.clamp(min=1e-8)
    x_q = ((x - x_min) / scale).round().clamp(0, n_levels - 1)
    return x_q * scale + x_min


# ---------------------------------------------------------------------------
# Press
# ---------------------------------------------------------------------------

class TurboQuantPerTokenPress:
    """Per-(layer, head, token) mixed-precision TurboQuant press.

    Parameters
    ----------
    scorer:
        Implements BaseScorer.  Defaults to RandomScorer.
    bit_levels:
        Discrete bit-width candidates, e.g. (0, 1, 2, 4, 8).
        0 = head-wise masked eviction (cache length unchanged).
        16 = keep fp16 (skip quantisation).
    seed:
        Seed for TurboQuant quantiser initialisation.

    Usage::

        press = TurboQuantPerTokenPress()
        # or with a custom scorer / bit levels:
        press = TurboQuantPerTokenPress(scorer=MyScorer(), bit_levels=(0, 2, 4, 8))

        with press(model):
            model.model(input_ids=ctx_ids, past_key_values=cache)

        # decode step (outside or inside the context manager):
        for q_ids in questions:
            with press(model):
                model(input_ids=q_ids, past_key_values=cache, ...)
    """

    def __init__(
        self,
        scorer: Optional[BaseScorer] = None,
        bit_levels: tuple[int, ...] = DEFAULT_BIT_LEVELS,
        ratios: Optional[tuple[float, ...]] = None,
        target_avg_bits: Optional[float] = None,
        eviction_cost: float = 5.0,
        above_target_alpha: float = 1.0,
        n_outlier_channels: int = 0,
        outlier_min_bits: int = 4,
        seed: int = 42,
        decode_quant: bool = False,
        sink_tokens: int = 0,
        layerwise: bool = False,
        quantizer: str = "turboquant",
        value_quantizer: str = "minmax",
        value_group_size: int = 32,
        allow_decode_eviction: bool = False,
    ):
        if scorer is None:
            raise ValueError("TurboQuantPerTokenPress requires an explicit scorer.")
        self.scorer          = scorer
        self.bit_levels      = bit_levels
        self.ratios          = ratios
        self.target_avg_bits = target_avg_bits
        self.eviction_cost   = eviction_cost
        self.above_target_alpha = above_target_alpha
        self.n_outlier_channels = n_outlier_channels
        self.outlier_min_bits = outlier_min_bits
        self.seed            = seed
        self.decode_quant    = decode_quant
        self.sink_tokens     = sink_tokens
        self.layerwise       = layerwise
        self.quantizer       = quantizer
        self.value_quantizer = normalize_value_quantizer(value_quantizer)
        self.value_group_size = value_group_size
        self.allow_decode_eviction = bool(allow_decode_eviction)

        # Bit levels used during decode.
        # - Default (allow_decode_eviction=False): exclude 0 — never evict newly
        #   generated tokens (they can't be recovered from the prompt).
        # - allow_decode_eviction=True: include 0 — decode tokens compete for the
        #   target avg bit budget on equal footing with prefill tokens.
        if self.allow_decode_eviction:
            self._decode_bit_levels = tuple(bit_levels)
        else:
            self._decode_bit_levels = tuple(b for b in bit_levels if b != 0) or (16,)

        # Epsilon cache for optimal allocation (calibrated lazily)
        self._epsilon: Optional[dict[int, float]] = None
        self._head_dim: Optional[int] = None

        # quantiser cache keyed by (d, b, device, dtype)
        self._key_qs: dict = {}
        self._value_qs: dict = {}

        # diagnostic counter — check this is > 0 after a forward pass
        self._prefill_hooks_fired: int = 0
        self._score_prefill_supports_kwargs = self._supports_kwargs(self.scorer.score_prefill)
        self._score_decode_supports_kwargs = self._supports_kwargs(self.scorer.score_decode)
        self._hooked_layer_indices: list[int] = []
        self._last_hooked_layer_idx: Optional[int] = None
        self._layer_modules: dict[int, nn.Module] = {}
        self._reset_prefill_state()

    # ------------------------------------------------------------------
    # Quantiser cache
    # ------------------------------------------------------------------

    def _key_q(self, d: int, b: int, device, dtype) -> TurboQuantMSE:
        k = (d, b, device, dtype)
        if k not in self._key_qs:
            self._key_qs[k] = TurboQuantMSE(d, b, device=device, dtype=dtype, seed=self.seed)
        return self._key_qs[k]

    def _value_q(self, d: int, b: int, device, dtype) -> TurboQuantMSE:
        k = (d, b, device, dtype)
        if k not in self._value_qs:
            self._value_qs[k] = TurboQuantMSE(d, b, device=device, dtype=dtype, seed=self.seed + 1000)
        return self._value_qs[k]

    # ------------------------------------------------------------------
    # Budget + sink helpers
    # ------------------------------------------------------------------

    def _build_sink_mask(self, H: int, T: int, device: torch.device) -> Optional[torch.Tensor]:
        """Build ``[H, T]`` bool mask where ``True`` = sink token (fp16)."""
        if self.sink_tokens <= 0:
            return None
        mask = torch.zeros(H, T, dtype=torch.bool, device=device)
        mask[:, :min(self.sink_tokens, T)] = True
        return mask

    @staticmethod
    def _supports_kwargs(method) -> bool:
        params = inspect.signature(method).parameters.values()
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params) or any(
            p.name in {"module", "hidden_states"} for p in params
        )

    _alloc_logged: bool = False

    def _compute_bits(
        self,
        scores: torch.Tensor,
        sink_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Map scores → bit levels.

        Dispatch order:
        1. ``ratios`` given → fixed-ratio allocation.
        2. ``target_avg_bits`` given → Lagrangian-optimal allocation.
        3. Neither → uniform-partition fallback.
        """
        if self.ratios is not None:
            bits = ratio_scores_to_bits(
                scores, self.bit_levels, self.ratios, sink_mask,
            )
        elif self.target_avg_bits is not None:
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
            bits = optimal_scores_to_bits(
                scores, self.bit_levels, self.target_avg_bits,
                self._epsilon, sink_mask,
                eviction_cost=self.eviction_cost,
                n_outlier=self.n_outlier_channels,
                head_dim=self._head_dim or 128,
                outlier_min_bits=self.outlier_min_bits,
                above_target_alpha=self.above_target_alpha,
            )
        else:
            bits = scores_to_bits(scores, self.bit_levels)
            if sink_mask is not None:
                bits = bits.clone()
                bits[sink_mask] = 16
            return bits

        # ── Log allocation stats (first sample only) ────────────────
        if not self._alloc_logged:
            self._alloc_logged = True
            import sys
            from kvquant.allocator import effective_bits as _eff_bits
            total = bits.numel()
            avg_nom = bits.float().mean().item()
            n_oc = self.n_outlier_channels
            d = self._head_dim or 128
            avg_eff = sum(
                _eff_bits(b, n_oc, d, self.outlier_min_bits) * int((bits == b).sum().item())
                for b in sorted(set(self.bit_levels) | {16})
            ) / max(total, 1)
            parts = []
            for b in sorted(set(self.bit_levels) | {16}):
                cnt = int((bits == b).sum().item())
                if cnt > 0:
                    parts.append(f"{b}bit:{cnt}({cnt/total*100:.1f}%)")
            print(f"[alloc] n={total} avg_nominal={avg_nom:.3f} avg_effective={avg_eff:.3f}  "
                  f"{' '.join(parts)}", file=sys.stderr, flush=True)

        return bits

    def _reset_prefill_state(self) -> None:
        self._prefill_scores: dict[int, torch.Tensor] = {}
        self._prefill_sink_masks: dict[int, Optional[torch.Tensor]] = {}

    @staticmethod
    def _get_cache_tensors(cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        cache_layer = cache.layers[layer_idx]
        return cache_layer.keys, cache_layer.values

    @staticmethod
    def _set_cache_tensors(cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        if hasattr(cache, "key_cache") and isinstance(cache.key_cache, list):
            cache.key_cache[layer_idx] = keys
            cache.value_cache[layer_idx] = values
            return
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values

    @staticmethod
    def _bits_to_masked_indices(bits: torch.Tensor) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        masked = (bits == 0).nonzero(as_tuple=True)
        if masked[0].numel() == 0:
            return None
        return masked

    def _allocate_bits_for_prefill(self) -> dict[int, torch.Tensor]:
        ordered_layers = sorted(self._prefill_scores)
        if not ordered_layers:
            return {}

        bits_by_layer = {
            layer_idx: torch.empty_like(scores, dtype=torch.int32)
            for layer_idx, scores in self._prefill_scores.items()
        }
        batch_size = self._prefill_scores[ordered_layers[0]].shape[0]

        for batch_idx in range(batch_size):
            if self.layerwise:
                for layer_idx in ordered_layers:
                    scores = self._prefill_scores[layer_idx][batch_idx]
                    sink_mask = self._prefill_sink_masks[layer_idx]
                    bits_by_layer[layer_idx][batch_idx] = self._compute_bits(
                        scores.detach(),
                        None if sink_mask is None else sink_mask.detach(),
                    )
                continue

            flat_scores_parts: list[torch.Tensor] = []
            flat_sink_parts: list[torch.Tensor] = []
            shapes: dict[int, torch.Size] = {}
            any_sink = False
            target_device = self._prefill_scores[ordered_layers[0]][batch_idx].device

            for layer_idx in ordered_layers:
                scores = self._prefill_scores[layer_idx][batch_idx]
                sink_mask = self._prefill_sink_masks[layer_idx]
                flat_scores_parts.append(scores.detach().reshape(-1).to(target_device))
                shapes[layer_idx] = scores.shape
                if sink_mask is None:
                    flat_sink_parts.append(torch.zeros(scores.numel(), dtype=torch.bool, device=target_device))
                else:
                    any_sink = True
                    flat_sink_parts.append(sink_mask.detach().reshape(-1).to(target_device))

            flat_scores = torch.cat(flat_scores_parts, dim=0)
            flat_sink = torch.cat(flat_sink_parts, dim=0) if any_sink else None
            flat_bits = self._compute_bits(flat_scores, flat_sink)

            cursor = 0
            for layer_idx in ordered_layers:
                size = int(self._prefill_scores[layer_idx][batch_idx].numel())
                layer_device = self._prefill_scores[layer_idx].device
                bits_by_layer[layer_idx][batch_idx] = flat_bits[cursor : cursor + size].reshape(
                    shapes[layer_idx]
                ).to(layer_device)
                cursor += size

        return bits_by_layer

    def _score_prefill(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        *,
        module: nn.Module,
        hidden_states: torch.Tensor,
        col_sum: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._score_prefill_supports_kwargs:
            kwargs: dict = {"module": module, "hidden_states": hidden_states}
            if col_sum is not None:
                kwargs["col_sum"] = col_sum
            return self.scorer.score_prefill(keys, values, layer_idx, **kwargs)
        return self.scorer.score_prefill(keys, values, layer_idx)

    def _score_decode(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        *,
        module: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self._score_decode_supports_kwargs:
            return self.scorer.score_decode(
                key,
                value,
                layer_idx,
                module=module,
                hidden_states=hidden_states,
            )
        return self.scorer.score_decode(key, value, layer_idx)

    # ------------------------------------------------------------------
    # Core: apply bit assignments and prefill masks
    # ------------------------------------------------------------------

    def _apply_bits(
        self,
        keys: torch.Tensor,    # [H, T, d]
        values: torch.Tensor,  # [H, T, d]
        bits: torch.Tensor,    # [H, T]  int32
    ) -> None:
        """Quantise in-place for non-zero, non-16 bit levels.

        Keys:   TurboQuantMSE (Lloyd-Max codebook, minimum reconstruction error)
        Values: either per-group min-max or TurboQuantMSE
        """
        H, T, d = keys.shape
        device, dtype = keys.device, keys.dtype
        use_scalar = self.quantizer == "scalar"

        for b in sorted(set(self.bit_levels)):
            if b == 0 or b == 16:
                continue

            mask = (bits == b)               # [H, T]
            if not mask.any():
                continue

            h_idx, t_idx = mask.nonzero(as_tuple=True)

            if use_scalar:
                keys[h_idx, t_idx]   = _qd_scalar(keys[h_idx, t_idx], b)
                values[h_idx, t_idx] = _qd_scalar(values[h_idx, t_idx], b)
            else:
                # Keys: TurboQuantMSE for reconstruction fidelity
                kq = self._key_q(d, b, device, dtype)
                keys[h_idx, t_idx] = _qd_keys(kq, keys[h_idx, t_idx])
                if self.value_quantizer == "mse":
                    vq = self._value_q(d, b, device, dtype)
                    values[h_idx, t_idx] = _qd_values_mse(vq, values[h_idx, t_idx])
                else:
                    values[h_idx, t_idx] = _qd_values_group(values[h_idx, t_idx], b, self.value_group_size)

        # bit_level == 16: no-op, keep fp16 as-is

    def _finalize_prefill(self, cache) -> None:
        bits_by_layer = self._allocate_bits_for_prefill()
        for layer_idx, bits in bits_by_layer.items():
            keys, values = self._get_cache_tensors(cache, layer_idx)
            keys = keys.clone()
            values = values.clone()

            for batch_idx in range(keys.shape[0]):
                self._apply_bits(keys[batch_idx], values[batch_idx], bits[batch_idx])

            self._set_cache_tensors(cache, layer_idx, keys, values)
            module = self._layer_modules.get(layer_idx)
            if module is not None:
                module.masked_key_indices = self._bits_to_masked_indices(bits)

        self._reset_prefill_state()

    # ------------------------------------------------------------------
    # Forward hook
    # ------------------------------------------------------------------

    def _forward_hook(self, module: nn.Module, args, kwargs, output):
        cache = kwargs.get("past_key_values")

        # avoid `or` on Tensor (raises "ambiguous truth value")
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]

        if cache is None or hidden_states is None:
            return output

        layer_idx = int(module.layer_idx)
        self._layer_modules[layer_idx] = module
        # Read KV tensors directly from underlying lists when possible,
        # so write-back is guaranteed to go to the same storage.
        keys, values = self._get_cache_tensors(cache, layer_idx)

        # Capture head_dim for lazy epsilon calibration
        if self._head_dim is None:
            self._head_dim = keys.shape[-1]

        # Detect prefill vs decode using cache length vs input length.
        # This is robust across transformers versions: transformers >= 5.4 no
        # longer passes cache_position when position_ids is explicitly given.
        # Fresh prefill: cache_len == q_len (cache just filled by this call).
        # Decode / question-encoding: cache_len > q_len (past context exists).
        q_len      = hidden_states.shape[1]
        cache_len  = keys.shape[2]
        is_prefill = (cache_len == q_len)

        if is_prefill:
            # Fused-FA2 path (optional): per-layer col_sum was stashed by the
            # attention kernel; pop it here and forward to the scorer. Only
            # valid when B=1 (the fused impl skips B>1).
            col_sum_layer = None
            if keys.shape[0] == 1:
                try:
                    from kvquant.fused_attention_patch import get_col_sum_buffer
                    col_sum_layer = get_col_sum_buffer().pop(layer_idx, None)
                except ImportError:
                    pass

            layer_scores: list[torch.Tensor] = []
            for batch_idx in range(keys.shape[0]):
                k_layer = keys[batch_idx]           # [H, T_ctx, d]
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

        elif self.decode_quant:
            # Decode / question-encoding: compress only the q_len tokens newly
            # appended by this call. For a single new token, keep using the
            # dedicated score_decode() path. For multi-token appends (for
            # example the first question-encoding call), reuse the prefill
            # scorer on just the appended tail so every new token is handled.
            for batch_idx in range(keys.shape[0]):
                k_new = keys[batch_idx, :, -q_len:, :]    # [H, q_len, d]
                v_new = values[batch_idx, :, -q_len:, :]  # [H, q_len, d]
                h_batch = hidden_states[batch_idx : batch_idx + 1]

                if q_len == 1:
                    scores = self._score_decode(
                        k_new[:, 0, :],
                        v_new[:, 0, :],
                        layer_idx,
                        module=module,
                        hidden_states=h_batch,
                    ).unsqueeze(-1)                      # [H, 1]
                else:
                    scores = self._score_prefill(
                        k_new,
                        v_new,
                        layer_idx,
                        module=module,
                        hidden_states=h_batch,
                    )

                # Decode tokens are few — use simple uniform allocation,
                # not budget (which would degenerate on small token counts).
                # Exclude 0-bit level: no eviction during decode.
                bits = scores_to_bits(scores, self._decode_bit_levels)
                self._apply_bits(k_new, v_new, bits)

        return output

    # ------------------------------------------------------------------
    # Quantiser prewarm  (Fix 5)
    # ------------------------------------------------------------------

    def _prewarm_quantizers(self, model: PreTrainedModel) -> None:
        """Pre-compute all codebooks before the first hook fires.

        Lloyd-Max can be expensive for large d; doing it here (once, before
        any hook is registered) avoids latency inside the forward pass.

        Collects unique devices from attention layers to handle device_map="auto".
        Skipped when using the scalar quantizer (no codebooks needed).
        """
        if self.quantizer == "scalar":
            return

        cfg      = model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        dtype    = next(model.parameters()).dtype

        # Collect devices from each attention layer (handles device_map="auto")
        lm = model
        if hasattr(lm, "model"):          lm = lm.model
        if hasattr(lm, "language_model"): lm = lm.language_model

        devices: set = set()
        for layer in lm.layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                p = next(attn.parameters(), None)
                if p is not None:
                    devices.add(p.device)
        if not devices:
            devices = {next(model.parameters()).device}

        for b in sorted(set(self.bit_levels)):
            if b in (0, 16):
                continue
            for device in devices:
                self._key_q(head_dim, b, device, dtype)
                if self.value_quantizer == "mse":
                    self._value_q(head_dim, b, device, dtype)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def __call__(self, model: PreTrainedModel):
        """Pre-warm quantisers, register per-layer hooks, yield, then clean up."""
        lm = model
        if hasattr(lm, "model"):          lm = lm.model
        if hasattr(lm, "language_model"): lm = lm.language_model

        self._prewarm_quantizers(model)   # Fix 5: codebooks ready before first hook
        patch_attention_functions()
        self._reset_prefill_state()
        self._hooked_layer_indices = []
        self._last_hooked_layer_idx = None
        self._layer_modules = {}

        fused_cm = None
        if getattr(self, "fused_col_sum", False):
            from kvquant.fused_attention_patch import fused_col_sum_enabled
            fused_cm = fused_col_sum_enabled(model)
            fused_cm.__enter__()

        hooks = []
        try:
            for i, layer in enumerate(lm.layers):
                attn = getattr(layer, "self_attn", None)
                if attn is None or getattr(attn, "is_sliding", False):
                    continue
                if not hasattr(attn, "layer_idx"):
                    attn.layer_idx = i
                attn.masked_key_indices = None
                if hasattr(lm, "rotary_emb"):
                    try:
                        attn.rotary_emb = lm.rotary_emb
                    except Exception:
                        pass
                self._hooked_layer_indices.append(i)
                hooks.append(
                    attn.register_forward_hook(self._forward_hook, with_kwargs=True)
                )
            self._last_hooked_layer_idx = self._hooked_layer_indices[-1] if self._hooked_layer_indices else None
            yield
        finally:
            for h in hooks:
                h.remove()
            # Intentionally DO NOT clear module.masked_key_indices here.
            # The eviction mask (prefill 0-bit positions) must keep enforcing
            # during the decode phase even when decode_quant=False — otherwise
            # positions the Lagrangian assigned to 0-bit silently fall through
            # to the model as their original bf16 values (no real eviction).
            # attention_patch resets the mask at the next fresh prefill
            # (q_len == k_len), so there's no cross-run leak.
            self._hooked_layer_indices = []
            self._last_hooked_layer_idx = None
            self._layer_modules = {}
            self._reset_prefill_state()
            if fused_cm is not None:
                fused_cm.__exit__(None, None, None)
