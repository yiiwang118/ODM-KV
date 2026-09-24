"""odmkv per-token press — fully aligned with the kvquant simulator yaml.

Knobs supported (one-to-one with simulator's TurboQuantPerTokenBackendPress):
  * scorer (ODMScorer)             — score_prefill / score_with_attn
  * bit_levels                      — list of allowed bit levels
  * target_avg_bits / eviction_cost / above_target_alpha — Lagrangian inputs
  * n_outlier_channels / outlier_min_bits — OCS
  * sink_tokens                     — first-N tokens forced to 16-bit
  * score_sink_tokens               — sink padding for the scorer
  * buffer_size                     — decode tail flush threshold
  * decode_quant                    — keep press hooks active during decode
  * allow_decode_eviction           — whether 0-bit is allowed at decode flush
  * initial_layers_fp16             — first N layers entirely 16-bit
  * layerwise                       — per-layer Lagrangian (we always do this:
                                      hook fires per-layer, no global solve)
  * key_quantizer / value_quantizer / value_group_size — passed through cache
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from kvquant.allocator import calibrate_epsilon
from kvquant.scorer import ODMScorer

from kvquant.runtime.storage.cache import ODMCache
from kvquant.runtime.native_allocator import BatchedNativeAllocator
from kvquant.runtime.model_runtime import exclusive_model_context
from kvquant.runtime.attn_impl import (
    register_odmkv_attention, set_active_cache, reset_active_cache, clear_active_cache,
)


class ODMNativePress:
    def __init__(
        self,
        cache: ODMCache,
        scorer: ODMScorer,
        *,
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
        epsilon: Optional[dict] = None,
        seed: int = 42,
    ):
        self.cache = cache
        self.scorer = scorer
        self.bit_levels = tuple(sorted(set(int(b) for b in bit_levels)))
        self.target_avg_bits = float(target_avg_bits)
        self.eviction_cost = float(eviction_cost)
        self.above_target_alpha = float(above_target_alpha)
        self.n_outlier_channels = int(n_outlier_channels)
        self.outlier_min_bits = int(outlier_min_bits)
        self.sink_tokens = int(sink_tokens)
        self.score_sink_tokens = int(score_sink_tokens)
        self.buffer_size = int(buffer_size)
        self.layerwise = bool(layerwise)
        self.decode_quant = bool(decode_quant)
        self.allow_decode_eviction = bool(allow_decode_eviction)
        self.initial_layers_fp16 = int(initial_layers_fp16)
        self.seed = int(seed)

        # A configuration with one admissible level is a uniform-bit-width
        # baseline, not mixed precision: there is nothing to score, no budget to
        # solve, and no per-entry bit-width to record.  Treating it as such is
        # what makes it a fair same-backend comparison point.
        self.uniform_bit_width = len(self.bit_levels) == 1

        # Decode-time bit pool: drops 0 if eviction not allowed.
        if self.allow_decode_eviction:
            self._decode_bit_levels = self.bit_levels
        else:
            self._decode_bit_levels = tuple(b for b in self.bit_levels if b != 0)

        # Tell the cache both the admissible native banks and whether allocation
        # collapses to one compile-time constant.  Fixed 2-bit keeps its highly
        # optimized append; higher targets use the mixed-bank native append.
        native_decode_level = None
        if self.decode_quant and self._decode_bit_levels:
            lowest = int(self._decode_bit_levels[0])
            if (len(self._decode_bit_levels) == 1
                    or self.target_avg_bits <= float(lowest)):
                native_decode_level = lowest
        cache._native_decode_level = native_decode_level
        cache._native_decode_levels = tuple(int(level) for level in self._decode_bit_levels)
        cache._native_initial_layers_fp16 = self.initial_layers_fp16
        configure_packing = getattr(cache, "configure_native_packing_policy", None)
        if configure_packing is not None:
            configure_packing(
                target_avg_bits=self.target_avg_bits,
                sink_tokens=self.sink_tokens,
                # Prefill allocation protects the current decode buffer tail
                # after solving the nominal bit budget; the capacity proof must
                # mirror that exact post-allocation override.
                tail_tokens=self.buffer_size,
            )

        if epsilon is None:
            try:
                self.epsilon = calibrate_epsilon(
                    cache.head_dim, self.bit_levels, seed=seed,
                    device=cache.device, value_quantizer="mse",
                )
            except Exception as _e:
                # Fail fast: a silent fallback would pass an EMPTY epsilon dict to
                # the allocator (⇒ every quant level looks lossless ⇒ garbage
                # allocation) while looking like a normal run. Surface it.
                raise RuntimeError(
                    f"odmkv: epsilon calibration failed "
                    f"({type(_e).__name__}: {_e}). This is a config error — the "
                    f"Lagrangian cannot price bit levels without it."
                ) from _e
        else:
            self.epsilon = epsilon

        # One immutable allocator per policy, reused across every layer.  It
        # solves all requests in a batch together while keeping an independent
        # lambda/budget per request, with no device-to-host reads.
        #
        # R2_ALLOC_SEARCH_STEPS=15 enables the fast lambda search: a tightened
        # bisection bracket (native_allocator._initial_hi) lets ~15 steps match
        # the 64-step tags. Unset => 64, byte-identical to the archived matrix.
        _alloc_env = os.environ.get("R2_ALLOC_SEARCH_STEPS", "").strip()
        _alloc_search_steps = int(_alloc_env) if _alloc_env else 64
        self._prefill_allocator = BatchedNativeAllocator(
            bit_levels=self.bit_levels,
            search_steps=_alloc_search_steps,
            target_avg_bits=self.target_avg_bits,
            epsilon=self.epsilon or {},
            sink_bits=16,
            sink_tokens=self.sink_tokens,
            tail_tokens=self.buffer_size,
            eviction_cost=self.eviction_cost,
            n_outlier=self.n_outlier_channels,
            head_dim=cache.head_dim,
            outlier_min_bits=self.outlier_min_bits,
            above_target_alpha=self.above_target_alpha,
        )
        self._decode_allocator = BatchedNativeAllocator(
            bit_levels=self._decode_bit_levels,
            search_steps=_alloc_search_steps,
            target_avg_bits=self.target_avg_bits,
            epsilon=self.epsilon or {},
            sink_bits=16,
            sink_tokens=0,
            tail_tokens=0,
            eviction_cost=self.eviction_cost,
            n_outlier=self.n_outlier_channels,
            head_dim=cache.head_dim,
            outlier_min_bits=self.outlier_min_bits,
            above_target_alpha=self.above_target_alpha,
        )

        self._handles: list = []
        # The attention interface computes only the Q-dependent score while Q
        # is live. The module forward hook consumes this small tensor after the
        # attention frame (and its large Q) has returned, then allocates/packs.
        self._native_prefill_scores: dict[int, torch.Tensor] = {}
        # Outlier indices detected once on first prefill, shared across layers.
        self._outlier_indices: Optional[torch.Tensor] = None
        # Per-layer decode tail counter (n tokens since last flush).
        self._tail_count: dict[int, int] = {}
        # Transformers 5.x owns RoPE at the decoder root rather than on every
        # attention module. Resolved for the active model on context entry.
        self._rotary_emb: Optional[nn.Module] = None

    # ── lifecycle ─────────────────────────────────────────────────────

    @contextlib.contextmanager
    def __call__(self, model: PreTrainedModel):
        register_odmkv_attention()
        with exclusive_model_context(model, "ODMNativePress"):
            self._validate_model_contract(model)
            with self._bind_model(model) as active:
                yield active

    @staticmethod
    def _validate_model_contract(model: PreTrainedModel) -> None:
        """Reject attention policies the packed decode cannot represent."""
        layer_idx = 0
        for module in model.modules():
            if module.__class__.__name__ not in (
                "LlamaAttention", "Qwen2Attention", "MistralAttention", "Qwen3Attention",
            ):
                continue
            module_window = getattr(module, "sliding_window", None)
            config = getattr(module, "config", None)
            configured_window = getattr(config, "sliding_window", None)
            use_window = getattr(config, "use_sliding_window", None)
            active_window = module_window
            if active_window is None and use_window is not False:
                active_window = configured_window
            if active_window not in (None, 0):
                raise RuntimeError(
                    "odmkv native compressed decode does not support "
                    f"sliding-window attention (layer={layer_idx}, "
                    f"window={active_window}). The packed banks do not retain "
                    "global token positions, so silently accepting this model "
                    "would make FA2 prefill and decode disagree."
                )
            layer_idx += 1

    @contextlib.contextmanager
    def _bind_model(self, model: PreTrainedModel):
        cache_token = set_active_cache(self.cache)      # restores outer on exit
        scorer = getattr(self, "scorer", None)
        set_prefill_callback = getattr(self.cache, "set_native_prefill_callback", None)
        # A single admissible bit level has nothing to allocate, so scoring is
        # dead work.  Skipping it keeps a uniform-bit-width configuration an
        # honest same-backend baseline instead of one that silently pays this
        # method's scorer.
        needs_scores = (
            getattr(scorer, "score_prefill_from_qkv", None) is not None
            and not self.uniform_bit_width
        )
        previous_prefill_callback = (
            set_prefill_callback(
                self._native_prefill_boundary if needs_scores else None
            )
            if set_prefill_callback is not None else None
        )

        # Prefer the canonical decoder-root attribute; fall back to a named
        # module search for compatible model wrappers. There must be exactly one
        # shared positional embedding in every currently supported family.
        rotary_emb = None
        candidates = [
            getattr(getattr(model, "model", None), "rotary_emb", None),
            getattr(getattr(model, "transformer", None), "rotary_emb", None),
        ]
        rotary_emb = next((candidate for candidate in candidates if candidate is not None), None)
        if rotary_emb is None:
            for name, candidate in model.named_modules():
                if name.endswith("rotary_emb"):
                    rotary_emb = candidate
                    break
        if (getattr(scorer, "score_prefill_from_qkv", None) is not None
                and getattr(scorer, "n_future_positions", 0) > 0
                and rotary_emb is None):
            reset_active_cache(cache_token)
            if set_prefill_callback is not None:
                set_prefill_callback(previous_prefill_callback)
            raise RuntimeError(
                "odmkv native prefill scorer requires the model's root rotary_emb"
            )
        self._rotary_emb = rotary_emb
        # Snapshot every DISTINCT config before mutating any of them. ``model``
        # is itself the first item yielded by model.modules(), and multiple
        # modules commonly share model.config.  Mutating model.config first and
        # then snapshotting the module list therefore records
        # ``odmkv_pertok`` as the supposed original value and leaves the
        # backend poisoned after context exit.
        saved_attn: list[tuple] = []
        seen_configs: set[int] = set()

        def _snapshot(cfg) -> None:
            if (cfg is not None and hasattr(cfg, "_attn_implementation")
                    and id(cfg) not in seen_configs):
                seen_configs.add(id(cfg))
                saved_attn.append((cfg, cfg._attn_implementation))

        _snapshot(getattr(model, "config", None))
        for mod in model.modules():
            _snapshot(getattr(mod, "config", None))
        for cfg, _ in saved_attn:
            cfg._attn_implementation = "odmkv_pertok"

        self._handles = []
        layer_count = 0
        for _, module in model.named_modules():
            if module.__class__.__name__ in (
                "LlamaAttention", "Qwen2Attention", "MistralAttention", "Qwen3Attention",
            ):
                handle = module.register_forward_hook(
                    self._make_hook(layer_count), with_kwargs=True,
                )
                self._handles.append(handle)
                layer_count += 1

        def _restore():
            reset_active_cache(cache_token)
            if set_prefill_callback is not None:
                set_prefill_callback(previous_prefill_callback)
            self._rotary_emb = None
            getattr(self, "_native_prefill_scores", {}).clear()
            for cfg, orig in saved_attn:
                cfg._attn_implementation = orig

        try:
            if layer_count == 0:
                raise RuntimeError(
                    "ODMNativePress: no attention modules found "
                    "(supported: Llama/Qwen2/Qwen3/Mistral)."
                )
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles = []
            _restore()

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs, output):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            # The hook performs allocation and native packing after attention.
            # Under a sharded model those Triton launches must use this layer's
            # device, not the process-global current CUDA device.
            if isinstance(hidden_states, torch.Tensor) and hidden_states.is_cuda:
                with torch.cuda.device(hidden_states.device):
                    return self._on_layer_forward(
                        module, layer_idx, args, kwargs, output,
                    )
            return self._on_layer_forward(module, layer_idx, args, kwargs, output)
        return hook

    @torch.no_grad()
    def _native_prefill_boundary(
        self, module, layer_idx, query, keys, values, rope,
    ) -> None:
        """Compute only the Q-dependent score at the attention boundary.

        Allocation and packing deliberately wait for the module forward hook.
        At that point the attention/dispatcher frames that owned the full Q
        tensor have returned; only the much smaller ``[B,H_kv,T]`` score remains.
        """
        B, H_kv, T, _ = keys.shape
        if layer_idx < self.initial_layers_fp16 or T <= self.sink_tokens + 1:
            return
        native_score = getattr(self.scorer, "score_prefill_from_qkv", None)
        if native_score is None:
            return
        cos, sin = rope
        try:
            buckets = self.cache.prefill_length_buckets()
            if not buckets:
                scores = native_score(
                    query,
                    keys,
                    values,
                    layer_idx,
                    module=module,
                    cos=cos,
                    sin=sin,
                    rotary_emb=self._rotary_emb,
                )
            else:
                # Left-padded rows are compact suffixes.  Group requests with
                # the same valid length so Q/K/V, RoPE inversion and future
                # position estimation see exactly positions [0, L), without a
                # per-request loop or padded-token contribution.
                scores = torch.zeros(
                    B, H_kv, T,
                    dtype=torch.float32,
                    device=keys.device,
                )

                def _rope_bucket(table, rows, start):
                    if table.dim() == 2:
                        return table[start:]
                    if table.shape[0] == 1:
                        return table[:, start:]
                    return table.index_select(0, rows)[:, start:]

                for valid_length, rows in buckets:
                    start = T - valid_length
                    if valid_length <= self.sink_tokens:
                        # Every valid cell in this bucket is a protected sink
                        # (and also within the protected tail).  Allocation will
                        # force it to 16-bit, so invoking an analytical suffix
                        # scorer has no decision value and some scorer families
                        # do not define an empty post-sink core.
                        continue
                    bucket_scores = native_score(
                        query.index_select(0, rows)[:, :, start:],
                        keys.index_select(0, rows)[:, :, start:],
                        values.index_select(0, rows)[:, :, start:],
                        layer_idx,
                        module=module,
                        cos=_rope_bucket(cos, rows, start),
                        sin=_rope_bucket(sin, rows, start),
                        rotary_emb=self._rotary_emb,
                    )
                    padded = torch.nn.functional.pad(
                        bucket_scores.to(torch.float32), (start, 0), value=0.0,
                    )
                    scores.index_copy_(0, rows, padded)
        finally:
            self.cache.release_prefill_native_inputs(layer_idx)
        expected = (B, H_kv, T)
        if tuple(scores.shape) != expected:
            raise RuntimeError(
                f"native scorer returned {tuple(scores.shape)}, expected {expected}"
            )
        self._native_prefill_scores[layer_idx] = scores

    def _force_valid_prefill_bits(
        self,
        batch: int,
        heads: int,
        tokens: int,
        bit: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Fill valid prompt cells with ``bit`` and keep input padding at 0."""
        valid = self.cache._prefill_valid_mask
        if valid is None:
            return torch.full(
                (batch, heads, tokens), bit,
                dtype=torch.int32, device=device,
            )
        return torch.where(
            valid[:, None, :],
            torch.full(
                (batch, heads, tokens), bit,
                dtype=torch.int32, device=device,
            ),
            torch.zeros(
                (batch, heads, tokens),
                dtype=torch.int32, device=device,
            ),
        )

    def _allocate_valid_prefill(self, scores: torch.Tensor) -> torch.Tensor:
        """Allocate only compact valid suffixes and scatter padding as tag 0."""
        buckets = self.cache.prefill_length_buckets()
        if not buckets:
            return self._prefill_allocator.allocate(scores)
        batch, heads, tokens = scores.shape
        bits = torch.zeros(
            batch, heads, tokens,
            dtype=torch.int32, device=scores.device,
        )
        for valid_length, rows in buckets:
            start = tokens - valid_length
            bucket_bits = self._prefill_allocator.allocate(
                scores.index_select(0, rows)[:, :, start:],
            )
            padded = torch.nn.functional.pad(
                bucket_bits, (start, 0), value=0,
            )
            bits.index_copy_(0, rows, padded)
        return bits

    # ── prefill commit ────────────────────────────────────────────────

    @torch.no_grad()
    def _detect_outliers(self, k: torch.Tensor) -> Optional[torch.Tensor]:
        if self.n_outlier_channels <= 0:
            return None
        T = k.shape[2]
        if T > 512:
            idx = torch.linspace(0, T - 1, 512, device=k.device).long()
            sample = k[:, :, idx, :]
        else:
            sample = k
        var = sample.float().var(dim=2).mean(dim=(0, 1))   # [D]
        n_out = min(self.n_outlier_channels, var.shape[0] // 2)
        return var.topk(n_out).indices.sort().values

    @torch.no_grad()
    def _on_layer_forward(self, module, layer_idx, args, kwargs, output):
        cache = self.cache
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]

        # Prefill path: compress this layer's bf16 K, V into per-bit banks.
        if not cache._committed.get(layer_idx, False):
            self._prefill_commit(module, layer_idx, hidden_states)
            return output

        # Decode path: tail flush if buffer reaches threshold.
        if not self.decode_quant or self.buffer_size <= 0:
            return output
        if layer_idx < self.initial_layers_fp16:
            # First N layers stay all-16; tail just accumulates and is
            # appended to ExactBank via state.append on flush. We DO still
            # flush so memory doesn't grow unboundedly — flush with all-16.
            self._maybe_flush_all_fp16(layer_idx)
            return output

        self._maybe_flush_quant(module, layer_idx, hidden_states)
        return output

    def _prefill_commit(
        self,
        module,
        layer_idx,
        hidden_states,
    ) -> None:
        cache = self.cache
        prefill_kv = cache._prefill_kv.get(layer_idx)
        if prefill_kv is None:
            return
        keys, values = prefill_kv                       # [B, H_kv, T, D]
        B, H_kv, T, D = keys.shape

        # initial_layers_fp16: skip Lagrangian; force entire layer to 16-bit.
        if layer_idx < self.initial_layers_fp16:
            bits = self._force_valid_prefill_bits(B, H_kv, T, 16, keys.device)
            cache.commit_prefill(layer_idx, bits, outlier_indices=None)
            self._tail_count[layer_idx] = 0
            return

        # Very short prefill (sink only).
        if T <= self.sink_tokens + 1:
            bits = self._force_valid_prefill_bits(B, H_kv, T, 16, keys.device)
            cache.commit_prefill(layer_idx, bits, outlier_indices=None)
            self._tail_count[layer_idx] = 0
            return

        # Uniform bit width: no scorer, no Lagrangian, no per-entry decision.
        #
        # Two variants, because the recent-tail override cuts both ways as a
        # baseline.  Keeping it (``R2_UNIFORM_TAIL=16``, the default) matches this
        # method on everything except bit allocation, so the gap isolates mixed
        # precision.  Dropping it (``R2_UNIFORM_TAIL=level``) is the faithful
        # uniform-quantizer baseline, which has no full-precision window and
        # therefore also spends less memory than the padded variant.
        if self.uniform_bit_width:
            level = int(self.bit_levels[0])
            bits = self._force_valid_prefill_bits(
                B, H_kv, T, level, keys.device,
            )
            tail_mode = os.environ.get("R2_UNIFORM_TAIL", "16").strip().lower()
            protected = min(self.buffer_size, T) if tail_mode == "16" else 0
            if protected > 0:
                window = bits[..., -protected:]
                bits[..., -protected:] = torch.where(
                    window > 0, torch.full_like(window, 16), window,
                )
            cache.commit_prefill(layer_idx, bits, outlier_indices=None)
            self._tail_count[layer_idx] = 0
            return

        # Detect outlier channels once (first prefill layer).
        if self._outlier_indices is None and self.n_outlier_channels > 0:
            self._outlier_indices = self._detect_outliers(keys)

        # Production odm consumes the Q/K/V already computed by the
        # native attention forward.  No hidden-state hook replay, no B-times
        # q_proj, and no host sync. Generic/diagnostic scorers retain the legacy
        # per-sample interface until they receive their own native operator.
        native_score = getattr(self.scorer, "score_prefill_from_qkv", None)
        if native_score is not None:
            scores = self._native_prefill_scores.pop(layer_idx, None)
            if scores is None:
                raise RuntimeError(
                    "odmkv native prefill scorer did not publish scores at the "
                    "attention boundary; refusing to replay q_proj or silently fall back"
                )
        else:
            per_batch = []
            for b in range(B):
                hb = hidden_states[b:b + 1] if hidden_states is not None else None
                try:
                    sc = self.scorer.score_prefill(
                        keys[b], values[b], layer_idx, module=module, hidden_states=hb,
                    )
                except (TypeError, AttributeError):
                    sc = self.scorer.score_prefill(keys[b], values[b], layer_idx)
                per_batch.append(sc[0] if sc.dim() == 3 else sc)
            scores = torch.stack(per_batch, dim=0)

        # Batched native allocation preserves the simulator's exact ordering:
        # solve one lambda per request across that request's heads/tokens,
        # exclude/force sinks, then override the recent decode buffer to 16 bit.
        with torch.autograd.profiler.record_function("odmkv::prefill_allocate"):
            bits = self._allocate_valid_prefill(scores)
        with torch.autograd.profiler.record_function("odmkv::prefill_pack"):
            cache.commit_prefill(
                layer_idx, bits, outlier_indices=self._outlier_indices,
            )
        self._tail_count[layer_idx] = 0

    # ── decode flush ──────────────────────────────────────────────────

    def _maybe_flush_all_fp16(self, layer_idx: int) -> None:
        """initial_layers_fp16 path — when tail fills, push as all-16-bit."""
        cache = self.cache
        tail_k = cache._tail_k.get(layer_idx)
        if tail_k is None:
            return
        tail_len = tail_k.shape[2]
        if tail_len < self.buffer_size:
            return
        B, H_kv, T_tail, _ = tail_k.shape
        bits = torch.full((B, H_kv, T_tail), 16, dtype=torch.int32, device=tail_k.device)
        cache.flush_decode_tail(layer_idx, bits)
        self._tail_count[layer_idx] = 0

    def _maybe_flush_quant(self, module, layer_idx: int, hidden_states) -> None:
        cache = self.cache
        tail_k = cache._tail_k.get(layer_idx)
        tail_v = cache._tail_v.get(layer_idx)
        if tail_k is None:
            return
        T_tail = tail_k.shape[2]
        if T_tail < self.buffer_size:
            return
        bits = self.compute_flush_bits(module, layer_idx, tail_k, tail_v, hidden_states)
        cache.flush_decode_tail(layer_idx, bits)
        self._tail_count[layer_idx] = 0

    def compute_flush_bits(self, module, layer_idx: int, tail_k, tail_v,
                           hidden_states) -> torch.Tensor:
        """Per-(batch,head,token) bit allocation for a decode tail about to be
        flushed into the banks. Same scorer + Lagrangian path as prefill, exposed
        so the CUDA-graph decoder can flush its fixed-ring tail through identical
        logic (guaranteeing graph and eager flush produce the same banks)."""
        B, H_kv, T_tail, D = tail_k.shape

        # Exact fast path for the production decode policy.  Eviction is
        # disabled, so the lowest candidate is 2 bits while target_avg_bits is
        # 1.0.  optimal_scores_to_bits clamps that infeasible target to the
        # lowest candidate and therefore returns all-2 regardless of scores.
        # Avoid scoring B requests and launching the 64-step Lagrangian search
        # when its result is mathematically predetermined.
        levels = self._decode_bit_levels
        if len(levels) == 1 or self.target_avg_bits <= float(levels[0]):
            return torch.full(
                (B, H_kv, T_tail), int(levels[0]),
                dtype=torch.int32, device=tail_k.device,
            )

        # The attention boundary records the exact post-RoPE query already
        # produced by the model. Reusing it here avoids both q_proj replay and
        # the former hidden_len=1 -> norm-based scoring semantic degradation.
        query = self.cache.get_last_decode_query(layer_idx)
        if query is None:
            if self.cache._graph_tail:
                raise RuntimeError(
                    "native decode flush has no recorded post-RoPE query; "
                    "refusing to silently replace odm with norm scoring"
                )
            # Non-graph diagnostic path: retain the generic scorer contract,
            # but never catch and relabel an arbitrary failure as a valid score.
            per_batch = []
            for b in range(B):
                tail_hidden = (
                    hidden_states[b:b + 1] if hidden_states is not None else None
                )
                score = self.scorer.score_prefill(
                    tail_k[b], tail_v[b], layer_idx,
                    module=module, hidden_states=tail_hidden,
                )
                per_batch.append(score[0] if score.dim() == 3 else score)
            scores = torch.stack(per_batch, dim=0)
        else:
            if query.shape[0] != B or query.shape[-1] != D:
                raise RuntimeError(
                    f"recorded query {tuple(query.shape)} is incompatible with "
                    f"decode ring {tuple(tail_k.shape)}"
                )
            H_q = query.shape[1]
            if H_q % H_kv != 0:
                raise RuntimeError(f"H_q={H_q} is not divisible by H_kv={H_kv}")
            native_decode_score = getattr(
                self.scorer, "score_decode_from_qkv_batched", None,
            )
            use_native_decode_score = (
                callable(native_decode_score)
                and query.is_cuda
                and os.environ.get("R2_DECODE_NATIVE_SCORE", "1") != "0"
            )
            if use_native_decode_score:
                return self._decode_allocator.allocate(
                    native_decode_score(
                        query, tail_k, tail_v, module=module,
                    )
                )
            groups = H_q // H_kv
            grouped_q = query.reshape(B, H_kv, groups, D).float()
            logits = torch.matmul(
                grouped_q, tail_k.float().transpose(-1, -2),
            ) * (D ** -0.5)
            attention = torch.softmax(logits, dim=-1).mean(dim=2)
            batched_score = getattr(self.scorer, "score_with_attn_batched", None)
            if batched_score is not None:
                scores = batched_score(
                    tail_k, tail_v, attention, module=module,
                )
            else:
                scores = torch.stack(
                    tuple(
                        self.scorer.score_with_attn(
                            tail_k[b], tail_v[b], attention[b], module=module,
                        )
                        for b in range(B)
                    ),
                    dim=0,
                )

        return self._decode_allocator.allocate(scores)

    @torch.no_grad()
    def compute_flush_bits_all_layers(
        self,
        modules,
        layer_indices,
        tail_keys,
        tail_values,
    ) -> torch.Tensor:
        """Allocate every flushed layer in one device batch.

        Scores retain the exact single-layer arithmetic.  Only after all layers
        have produced their scores is the leading ``layer`` axis folded into
        the allocator's request axis, so each ``(layer, request)`` still receives
        an independent lambda and budget.  This is exactly the configured
        *layerwise* policy; it removes repeated allocator dispatch without
        changing scorer GEMM shapes or pooling layers.

        Returns int32 tags shaped ``[L,B,H_kv,T]``.  Graph-tail callers always
        have a recorded post-RoPE query for every layer, hence this method does
        not provide the diagnostic norm-based scoring fallback used by the eager
        single-layer boundary.
        """
        modules = tuple(modules)
        layer_indices = tuple(int(index) for index in layer_indices)
        tail_keys = tuple(tail_keys)
        tail_values = tuple(tail_values)
        layers = len(layer_indices)
        if not (
            len(modules) == layers
            and len(tail_keys) == layers
            and len(tail_values) == layers
        ):
            raise ValueError("batched decode flush inputs must have equal layer counts")
        if layers == 0:
            raise ValueError("batched decode flush requires at least one layer")

        shape = tuple(tail_keys[0].shape)
        if len(shape) != 4:
            raise ValueError(f"decode ring must be [B,H,T,D], got {shape}")
        if any(tuple(key.shape) != shape for key in tail_keys):
            raise ValueError("all decode K rings must share one fixed shape")
        if any(tuple(value.shape) != shape for value in tail_values):
            raise ValueError("all decode V rings must match the K ring shape")
        if any(key.device != tail_keys[0].device for key in tail_keys + tail_values):
            raise ValueError("all decode flush tensors must share one CUDA device")

        B, H_kv, T_tail, D = shape
        levels = self._decode_bit_levels
        if len(levels) == 1 or self.target_avg_bits <= float(levels[0]):
            return torch.full(
                (layers, B, H_kv, T_tail), int(levels[0]),
                dtype=torch.int32, device=tail_keys[0].device,
            )

        scores_by_layer = []
        batched_score = getattr(self.scorer, "score_with_attn_batched", None)
        if batched_score is None:
            raise RuntimeError(
                "native all-layer decode flush requires a batched scorer; "
                f"{type(self.scorer).__name__} does not provide one"
            )
        for module, layer_idx, key, value in zip(
            modules, layer_indices, tail_keys, tail_values, strict=True,
        ):
            query = self.cache.get_last_decode_query(layer_idx)
            if query is None:
                raise RuntimeError(
                    f"native batched decode flush has no post-RoPE query for "
                    f"layer {layer_idx}"
                )
            if query.shape[0] != B or query.shape[-1] != D:
                raise RuntimeError(
                    f"recorded layer-{layer_idx} query {tuple(query.shape)} is "
                    f"incompatible with decode ring {shape}"
                )
            H_q = int(query.shape[1])
            if H_q % H_kv != 0:
                raise RuntimeError(f"H_q={H_q} is not divisible by H_kv={H_kv}")
            groups = H_q // H_kv
            native_decode_score = getattr(
                self.scorer, "score_decode_from_qkv_batched", None,
            )
            if (
                callable(native_decode_score)
                and query.is_cuda
                and os.environ.get("R2_DECODE_NATIVE_SCORE", "1") != "0"
            ):
                scores_by_layer.append(
                    native_decode_score(query, key, value, module=module)
                )
                continue
            grouped_query = query.reshape(B, H_kv, groups, D).float()
            # Preserve the exact single-layer scorer arithmetic and GEMM shape.
            # Only the already-computed score tensors are combined below; this
            # avoids layer-batched GEMM reduction changes at allocation ties.
            logits = torch.matmul(
                grouped_query, key.float().transpose(-1, -2),
            ) * (D ** -0.5)
            attention = torch.softmax(logits, dim=-1).mean(dim=2)
            scores_by_layer.append(
                batched_score(key, value, attention, module=module)
            )

        # L*B is only the allocator batch axis.  Its persistent kernel owns one
        # independent CTA/lambda per row, so no cross-layer budget is possible.
        scores = torch.stack(scores_by_layer, dim=0).reshape(
            layers * B, H_kv, T_tail,
        )
        bits = self._decode_allocator.allocate(scores)
        return bits.reshape(layers, B, H_kv, T_tail)
