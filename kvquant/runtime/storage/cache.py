"""odmkv per-token compressed KV cache.

Wraps ``kvquant.tq_adaptive_backend.TurboQuantAdaptiveKVCacheState`` per
layer. The state holds:
  - QuantizedBank per bit level (TurboQuantMSE / Prod for K, MinMax/MSE
    for V), with optional OCS outlier-channel separation.
  - ExactBank for 16-bit (uncompressed bf16).
  - Masked positions for 0-bit eviction.

After prefill commit, ``cache.update()`` appends only to the small exact decode
tail.  The custom attention path reads packed banks directly; its correctness
fallback materializes on demand, while persistent storage remains compressed.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from kvquant.tq_adaptive_backend import TurboQuantAdaptiveKVCacheState


class _ODMLayer(CacheLayerMixin):
    """Metadata-only HF cache layer; it never owns or appends K/V tensors."""

    def __init__(self, parent: "ODMCache", layer_idx: int):
        super().__init__()
        self.parent = parent
        self.layer_idx = layer_idx

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor,
    ) -> None:
        raise RuntimeError(
            "Realsys2 metadata layers cannot initialize dense K/V storage"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "ODMCache.update owns the native compressed write path; "
            "metadata-layer update is prohibited"
        )

    def get_seq_length(self, *_args, **_kwargs) -> int:
        return self.parent.get_seq_length(self.layer_idx)

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Graph-safe HF mask metadata.

        ``DynamicLayer.get_mask_sizes`` calls ``get_seq_length``.  Our graph
        tail length is intentionally a device scalar, so reading it with
        ``.item()`` during CUDA capture is illegal.  The graph driver maintains
        an exact host mirror for Python-side mask construction; graph replay
        itself executes no Python and the FA2/no-padding mask is ``None``.
        """
        if self.parent._graph_tail:
            state = self.parent._states.get(self.layer_idx)
            native = getattr(self.parent, "_native_packed", {}).get(self.layer_idx)
            prefix = (
                native.total_seq_len if native is not None
                else (int(state.seq_len) if state is not None else 0)
            )
            return prefix + self.parent._graph_mask_tail_pos + cache_position.shape[0], 0
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_max_cache_shape(self) -> int:
        return -1


class ODMCache(Cache):
    """Per-token compressed KV cache implementing Transformers' cache API.

    This intentionally derives from the abstract :class:`Cache` container, not
    :class:`DynamicCache`.  The ``_ODMLayer`` objects supply HF's mask
    metadata contract only; no production K/V tensor is owned or appended by a
    Transformers dynamic cache layer.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads_kv: int,
        head_dim: int,
        device: torch.device | str,
        bit_levels: tuple[int, ...] = (0, 2, 3, 4, 8, 16),
        key_quantizer: str = "mse",
        value_quantizer: str = "mse",
        value_group_size: int = 32,
        outlier_min_bits: int = 3,
        seed: int = 42,
    ):
        # Register as a generic HF Cache while keeping storage entirely in the
        # native compressed arenas below.  An empty explicit layer list avoids
        # Cache's lazy DynamicLayer replication contract; our metadata layers
        # are installed once their parent object is initialized.
        super().__init__(layers=[])
        self.num_layers = num_layers
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.bit_levels = tuple(sorted(set(int(b) for b in bit_levels)))
        # The fused kernel has fixed slots for quant levels {2,3,4,8} + exact(16)
        # + evict(0); any other bit width would be silently dropped, so reject it
        # up front instead of losing tokens at decode.
        _allowed = {0, 2, 3, 4, 8, 16}
        _bad = sorted(set(self.bit_levels) - _allowed)
        if _bad:
            raise ValueError(
                f"ODMCache: bit_levels {_bad} are not supported by the fused "
                f"decode kernel (allowed: {sorted(_allowed)}).")
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
        self.value_group_size = value_group_size
        self.outlier_min_bits = outlier_min_bits
        self.seed = seed
        self.batch_size = 0

        self.layers = [_ODMLayer(self, i) for i in range(num_layers)]
        # Per-layer adaptive state (banks + exact + masked positions).
        # Built lazily on first commit_prefill (need outlier_indices then).
        self._states: dict[int, TurboQuantAdaptiveKVCacheState] = {}
        # Production native layout: direct packed CSR with no intermediate
        # TurboQuantAdaptiveKVCacheState or position triples.
        self._native_packed: dict[int, object] = {}
        self._native_decode_level: Optional[int] = None
        self._native_decode_levels: tuple[int, ...] = ()
        self._native_initial_layers_fp16 = 0
        # Frozen allocator facts used to prove a sync-free shared-arena bound.
        # ODMNativePress configures this once; reset() preserves policy because
        # it is model/press configuration rather than request state.
        self._native_packing_policy = None
        # One reusable fp32 scratch set serves every shared-v2 layer serially
        # during a periodic fixed or mixed-bit flush. It is intentionally cache-wide,
        # not per packed layer: B8/Hkv8/D128/buffer128 would otherwise retain
        # roughly 512 MiB of duplicate scratch across 32 layers.
        self._native_flush_workspace = None
        # Stash bf16 prefill K, V for the press to compress.
        self._prefill_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Native-prefill side data. HF computes RoPE before cache.update(); keep
        # only its tables until the attention-boundary scorer consumes them.
        # Q is passed directly through the callback and is never retained here.
        self._prefill_rope: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Request-local native pipeline entry.  The custom FA2 attention calls
        # this immediately after producing its output, before its Q/K/V frame
        # is released. Generic diagnostic scorers leave it unset and use the
        # legacy hook contract.
        self._native_prefill_callback: Optional[Callable] = None
        # Optional request-level left-padding layout for native varlen prefill.
        # ``_prefill_valid_mask`` is the 2-D keep mask [B,T].  Buckets group
        # requests by equal valid length so scoring/allocation can operate on
        # compact suffixes without a Python loop per request.  The mask remains
        # available after commit to distinguish input padding from algorithmic
        # 0-bit eviction in diagnostics; 0-tag padding is still an attention
        # mask and is physically absent from every packed payload bank.
        self._prefill_layout_configured = False
        self._prefill_valid_mask: Optional[torch.Tensor] = None
        self._prefill_valid_lengths: Optional[torch.Tensor] = None
        self._prefill_length_buckets: tuple[tuple[int, torch.Tensor], ...] = ()
        self._committed: dict[int, bool] = {}
        # Tail buffer for post-commit decode tokens not yet flushed into banks.
        self._tail_k: dict[int, torch.Tensor] = {}
        self._tail_v: dict[int, torch.Tensor] = {}
        # Decode-ready per-head quant-bank layout, cached per layer. Static
        # between flushes; invalidated on commit_prefill / flush_decode_tail.
        self._quant_ready: dict[int, tuple] = {}
        # Ragged (compact, un-padded) variant used by the fused kernel.
        self._quant_ready_ragged: dict[int, tuple] = {}
        # The 16-bit prefix bank is also static between flushes.  Cache its
        # padded row layout separately so token-by-token decode only exposes a
        # cheap view of the small mutable tail instead of copying the full
        # exact prefix every layer and every step.
        self._exact_ready: dict[int, tuple] = {}
        # Last post-RoPE decode query per layer. The attention boundary copies
        # into these fixed buffers during graph replay; periodic buffer scoring
        # then reuses the model-native query instead of replaying q_proj or
        # falling back to a norm proxy.
        self._last_decode_query: dict[int, torch.Tensor] = {}

        # ── Fixed-buffer decode tail (CUDA-graph mode) ────────────────────
        # The default tail (``_tail_k``) grows via ``torch.cat`` — a fresh
        # allocation every step, which is illegal inside a captured CUDA graph
        # (the graph records fixed addresses). When ``enable_graph_tail`` is on
        # the tail is a pre-allocated ``[B,H_kv,buf,D]`` ring: new tokens are
        # written in place with ``index_copy_`` at a *device* position scalar,
        # and the kernel masks the unwritten slots via a *device* seqlen. Both
        # update in place, so one captured decode step replays across the whole
        # request. Production shared-v2 flushes into pre-reserved CSR suffixes
        # without changing graph-visible addresses; only the retained legacy
        # layout can change bank shapes and force a re-capture.
        self._graph_tail = False
        self._graph_buf_size = 0
        self._graph_decode_capacity = 0
        self._graph_stable_banks = False
        self._tail_buf_k: dict[int, torch.Tensor] = {}
        self._tail_buf_v: dict[int, torch.Tensor] = {}
        self._tail_pos: Optional[torch.Tensor] = None       # [1] long, write index
        self._tail_len_rows: Optional[torch.Tensor] = None  # [NR] int32, kernel seqlen
        self._tail_offset_rows: Optional[torch.Tensor] = None
        # Python-side mirror used only while Transformers constructs masks
        # before a CUDA graph is recorded. Never read from the device here.
        self._graph_mask_tail_pos = 0

    def _make_state(
        self, layer_idx: int, outlier_indices: Optional[torch.Tensor],
    ) -> TurboQuantAdaptiveKVCacheState:
        regular_indices = None
        if outlier_indices is not None and outlier_indices.numel() > 0:
            all_idx = torch.arange(self.head_dim, device=self.device)
            mask = torch.ones(self.head_dim, dtype=torch.bool, device=self.device)
            mask[outlier_indices] = False
            regular_indices = all_idx[mask]
        return TurboQuantAdaptiveKVCacheState(
            head_dim=self.head_dim,
            bit_levels=self.bit_levels,
            value_group_size=self.value_group_size,
            device=self.device,
            dtype=torch.bfloat16,
            key_quantizer=self.key_quantizer,
            value_quantizer=self.value_quantizer,
            # Per-layer rotation seed. NOTE: this does NOT bit-match the
            # simulator (which uses a fixed key seed 42 / value seed 42+1000 for
            # every layer); the rotations differ but are equal-MSE, so the
            # dequantized reconstruction matches (verified in alloc_compare.py:
            # ‖K_orig−K_deq‖ identical to ~1e-3) and RULER accuracy is at parity.
            seed=self.seed + layer_idx * 7,
            outlier_indices=outlier_indices,
            regular_indices=regular_indices,
            outlier_min_bits=self.outlier_min_bits,
        )

    # ── HF-facing API ─────────────────────────────────────────────────

    def update(
        self,
        key_states: torch.Tensor,    # [B, H_kv, q_len, D]
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_size == 0:
            self.batch_size = key_states.shape[0]

        if not self._committed.get(layer_idx, False):
            # Prefill: stash the fresh, post-RoPE K/V for the native scorer and
            # direct packer.  The attention interface receives and uses these
            # exact tensors before commit changes persistent storage.
            self._prefill_kv[layer_idx] = (key_states, value_states)
            if cache_kwargs is not None:
                cos = cache_kwargs.get("cos")
                sin = cache_kwargs.get("sin")
                if cos is not None and sin is not None:
                    self._prefill_rope[layer_idx] = (cos, sin)
            return key_states, value_states

        if self._graph_tail:
            return self._update_graph_tail(key_states, value_states, layer_idx)

        # Post-commit decode append. Do NOT materialise — the fused decode
        # attention reads the packed banks + this tail directly (memory stays
        # compressed). HF's attention gets the tail here but our custom
        # attn_impl ignores the passed K, V (fast path) or re-materialises on
        # demand (fallback). Returning the tail keeps the interface happy.
        prev_k = self._tail_k.get(layer_idx)
        if prev_k is None:
            # Llama/Qwen commonly produce K/V via transpose.  Make the small
            # decode tail contiguous once on insertion so flattening B×H for
            # the fused grid remains a view rather than a hidden per-layer
            # copy on every attention call.
            self._tail_k[layer_idx] = key_states.contiguous()
            self._tail_v[layer_idx] = value_states.contiguous()
        else:
            self._tail_k[layer_idx] = torch.cat([prev_k, key_states], dim=2)
            self._tail_v[layer_idx] = torch.cat([self._tail_v[layer_idx], value_states], dim=2)
        return self._tail_k[layer_idx], self._tail_v[layer_idx]

    def set_native_prefill_callback(self, callback: Optional[Callable]):
        previous = self._native_prefill_callback
        self._native_prefill_callback = callback
        return previous

    def configure_native_packing_policy(
        self, *, target_avg_bits: float, sink_tokens: int, tail_tokens: int,
    ) -> None:
        """Set immutable capacity facts for production native packing."""
        from kvquant.runtime.native_packing_v2 import NativePackingPolicy

        policy = NativePackingPolicy(
            target_avg_bits=float(target_avg_bits),
            sink_tokens=int(sink_tokens),
            tail_tokens=int(tail_tokens),
        )
        previous = self._native_packing_policy
        if previous is not None and previous != policy:
            raise RuntimeError(
                f"native packing policy is immutable: existing={previous}, new={policy}"
            )
        self._native_packing_policy = policy

    def configure_prefill_layout(
        self,
        attention_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
    ) -> None:
        """Freeze the request's equal-length or strict left-padded layout.

        Native varlen support intentionally accepts only a two-dimensional
        boolean/0-1 keep mask whose rows are ``0* 1+``.  Validation happens once
        before prefill (``graph_generate`` calls this explicitly); subsequent
        attention layers reuse device index tensors grouped by unique valid
        length.  An all-valid mask is canonicalized to the original ``None``
        fast path.
        """
        expected = (int(batch_size), int(seq_len))
        if self._prefill_layout_configured:
            if self._prefill_valid_mask is None:
                if attention_mask is None:
                    return
                candidate = attention_mask
                if tuple(candidate.shape) == expected and bool((candidate != 0).all()):
                    return
            elif attention_mask is self._prefill_valid_mask:
                return
            elif attention_mask is not None and tuple(attention_mask.shape) == expected:
                candidate = attention_mask.to(
                    device=self.device, dtype=torch.bool,
                )
                if torch.equal(candidate, self._prefill_valid_mask):
                    return
            raise RuntimeError("prefill attention layout changed within one cache request")

        if attention_mask is None:
            self._prefill_layout_configured = True
            return
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != expected:
            raise ValueError(
                f"native varlen prefill requires attention_mask [B,T]={expected}, "
                f"got {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype != torch.bool:
            binary = (attention_mask == 0) | (attention_mask == 1)
            if not bool(binary.all()):
                raise ValueError("native varlen attention_mask values must be exactly 0 or 1")
        keep = attention_mask.to(device=self.device, dtype=torch.bool).contiguous()
        if not bool(keep.any(dim=1).all()):
            raise ValueError("native varlen prefill requires at least one valid token per row")
        # Strict left padding: after the first valid cell no zero may appear.
        if keep.shape[1] > 1 and bool((keep[:, :-1] & ~keep[:, 1:]).any()):
            raise ValueError("native varlen prefill supports strict left padding only (0* 1+)")

        self._prefill_layout_configured = True
        if bool(keep.all()):
            # Preserve the exact equal-length path: no saved mask, no buckets,
            # no scorer/allocation slicing.
            return

        lengths = keep.sum(dim=1, dtype=torch.long)
        # One intentional setup sync creates reusable request metadata.  It is
        # outside every layer/scorer hot path and loops over unique lengths,
        # never individual requests.
        length_values = tuple(int(length) for length in lengths.tolist())
        unique_lengths = sorted(set(length_values))
        buckets = []
        for length in unique_lengths:
            rows = torch.tensor(
                [row for row, value in enumerate(length_values) if value == length],
                dtype=torch.long,
                device=self.device,
            )
            buckets.append((length, rows))
        self._prefill_valid_mask = keep
        self._prefill_valid_lengths = lengths
        self._prefill_length_buckets = tuple(buckets)

    def prefill_length_buckets(self) -> tuple[tuple[int, torch.Tensor], ...]:
        """Return immutable ``(valid_length, batch_rows)`` request buckets."""
        return self._prefill_length_buckets

    def prefill_padding_mask(self) -> Optional[torch.Tensor]:
        """Return ``True`` at input-padding cells, or ``None`` if unpadded."""
        if self._prefill_valid_mask is None:
            return None
        return ~self._prefill_valid_mask

    def run_native_prefill_callback(
        self,
        module,
        layer_idx: int,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        """Run the Q-dependent scoring stage at the attention boundary."""
        callback = self._native_prefill_callback
        if callback is None or self._committed.get(layer_idx, False):
            return False
        rope = self._prefill_rope.get(layer_idx)
        if rope is None:
            raise RuntimeError(
                f"native prefill layer {layer_idx} did not receive RoPE tables"
            )
        # Accelerate's ``device_map`` moves layer tensors but does not promise
        # to change ``torch.cuda.current_device()``.  Native scorer wrappers
        # launch several Triton kernels, so bind their launch context to the
        # layer's actual Q/K/V device and restore the caller's device on exit.
        if query.is_cuda:
            with torch.cuda.device(query.device):
                callback(module, layer_idx, query, keys, values, rope)
        else:
            callback(module, layer_idx, query, keys, values, rope)
        return True

    def release_prefill_native_inputs(self, layer_idx: int) -> None:
        self._prefill_rope.pop(layer_idx, None)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx in self._prefill_kv:
            return self._prefill_kv[layer_idx][0].shape[2]
        if not self._committed.get(layer_idx, False):
            return 0
        native = self._native_packed.get(layer_idx)
        if native is not None:
            if self._graph_tail:
                return native.total_seq_len + self.graph_tail_pos()
            tail = self._tail_k.get(layer_idx)
            return native.total_seq_len + (tail.shape[2] if tail is not None else 0)
        state = self._states.get(layer_idx)
        s_len = int(state.seq_len) if state is not None else 0
        if self._graph_tail:
            return s_len + self.graph_tail_pos()
        tail = self._tail_k.get(layer_idx)
        return s_len + (tail.shape[2] if tail is not None else 0)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    # ── Fixed-buffer decode tail (CUDA-graph mode) ────────────────────

    def enable_graph_tail(
        self, buffer_size: int, num_rows: int,
        max_decode_tokens: Optional[int] = None,
    ) -> None:
        """Switch the decode tail to a pre-allocated ring so decode steps are
        CUDA-graph capturable. ``num_rows = B * H_kv`` (the fused-kernel grid).

        Buffers are allocated lazily per layer on first ``update`` (we learn D
        and the exact dtype from the first decode K). The position/seqlen device
        scalars are allocated now so capture sees a stable address."""
        self._graph_tail = True
        self._graph_buf_size = int(buffer_size)
        # Reserve fixed-address decode-ready storage.  With stable pointers and
        # shapes, a flush only fills unused slots + updates device seqlens; the
        # already-captured graph remains valid.  The default covers two complete
        # flushes for direct GraphDecoder users; graph_generate passes its exact
        # generation bound.
        self._graph_decode_capacity = (
            self._graph_buf_size * 2
            if max_decode_tokens is None
            else max(0, int(max_decode_tokens))
        )
        self._graph_stable_banks = True
        self._tail_pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self._tail_len_rows = torch.zeros(num_rows, dtype=torch.int32, device=self.device)
        self._tail_offset_rows = (
            torch.arange(num_rows, dtype=torch.int32, device=self.device)
            * self._graph_buf_size
        ).contiguous()
        self._graph_mask_tail_pos = 0

    def set_graph_mask_tail_pos(self, position: int) -> None:
        """Set the host mirror consumed by ``_ODMLayer.get_mask_sizes``."""
        position = int(position)
        if position < 0 or position > self._graph_buf_size:
            raise ValueError(
                f"graph mask tail position {position} outside [0,{self._graph_buf_size}]"
            )
        self._graph_mask_tail_pos = position

    def _update_graph_tail(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write one decode token into the fixed ring at the device position.

        All layers of a step write the SAME slot (they share ``_tail_pos``); the
        position/seqlen advance exactly once, on the final layer, so the next
        replay sees the incremented offset. Every op here has a fixed address —
        ``index_copy_`` into a pre-allocated buffer and in-place scalar adds —
        which is what makes the enclosing forward safe to capture."""
        if key_states.shape[2] != 1:
            raise ValueError(
                f"graph-tail decode is q_len=1 only; got q_len={key_states.shape[2]}. "
                f"Multi-token forwards (query encoding) must run before "
                f"enable_graph_tail / outside graph mode.")
        buf_k = self._tail_buf_k.get(layer_idx)
        if buf_k is None:
            B, H_kv, _, D = key_states.shape
            buf_k = torch.zeros(B, H_kv, self._graph_buf_size, D,
                                dtype=torch.bfloat16, device=self.device)
            buf_v = torch.zeros_like(buf_k)
            self._tail_buf_k[layer_idx] = buf_k
            self._tail_buf_v[layer_idx] = buf_v
        else:
            buf_v = self._tail_buf_v[layer_idx]
        # Write this step's token at slot ``pos`` (all layers share the slot).
        # Attention this step must SEE the token it just wrote (self position),
        # so the kernel seqlen is ``pos + 1`` for every layer. ``_tail_pos`` only
        # advances after the final layer, so the next replay writes the next
        # slot. copy_ broadcasts the [1] scalar into the [NR] seqlen in place.
        buf_k.index_copy_(2, self._tail_pos, key_states.to(torch.bfloat16))
        buf_v.index_copy_(2, self._tail_pos, value_states.to(torch.bfloat16))
        self._tail_len_rows.copy_((self._tail_pos + 1).to(torch.int32))
        if layer_idx == self.num_layers - 1:
            self._tail_pos.add_(1)
        return key_states, value_states

    def get_tail_for_decode(self, layer_idx: int, num_heads_kv: int):
        """Return ``(tail_k, tail_v, tail_seqlen, tail_offset)`` for decode.

        Normal mode: the grown ``[B,H,T,D]`` tail with a uniform (``None``)
        seqlen. Graph mode: the full fixed ring plus the device seqlen so the
        kernel ignores the not-yet-written slots — both addresses static across
        graph replays."""
        if self._graph_tail:
            buf_k = self._tail_buf_k.get(layer_idx)
            if buf_k is None:
                return None, None, None, None
            B, H_kv, buf, D = buf_k.shape
            return (buf_k.reshape(B * H_kv, buf, D),
                    self._tail_buf_v[layer_idx].reshape(B * H_kv, buf, D),
                    self._tail_len_rows, self._tail_offset_rows)
        tail_k = self._tail_k.get(layer_idx)
        if tail_k is None or tail_k.shape[2] == 0:
            return None, None, None, None
        B, H_kv, tl_len, D = tail_k.shape
        return (tail_k.reshape(B * H_kv, tl_len, D),
                self._tail_v[layer_idx].reshape(B * H_kv, tl_len, D), None, None)

    def record_decode_query(self, layer_idx: int, query: torch.Tensor) -> None:
        """Publish the current post-RoPE ``[B,H_q,D]`` query in place.

        The first warmup forward allocates the buffer. Subsequent warmups,
        capture, and every graph replay execute only ``copy_`` at a stable
        address, so flush-time scoring can consume the exact model query.
        """
        if query.ndim != 3:
            raise ValueError(
                f"decode query must have shape [B,H_q,D], got {tuple(query.shape)}"
            )
        stored = self._last_decode_query.get(layer_idx)
        if stored is None:
            stored = torch.empty_like(query)
            self._last_decode_query[layer_idx] = stored
        elif stored.shape != query.shape or stored.dtype != query.dtype:
            raise RuntimeError(
                "decode query descriptor changed after graph warmup: "
                f"stored={tuple(stored.shape)}/{stored.dtype}, "
                f"new={tuple(query.shape)}/{query.dtype}"
            )
        stored.copy_(query)

    def get_last_decode_query(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._last_decode_query.get(layer_idx)

    def get_native_decode_arena(self, layer_idx: int):
        native = self._native_packed.get(layer_idx)
        return getattr(native, "decode_arena", None) if native is not None else None

    def flush_graph_tail(
        self, layer_idx: int, bits: Optional[torch.Tensor],
        valid_tokens: Optional[int] = None,
    ) -> None:
        """Move the ring's written tokens (``[:, :, :pos]``) into the banks and
        rewind the ring. Stable decode-ready banks are updated in place so the
        captured graph remains valid."""
        buf_k = self._tail_buf_k.get(layer_idx)
        pos = (int(valid_tokens) if valid_tokens is not None else
               (int(self._tail_pos.item()) if self._tail_pos is not None else 0))
        if buf_k is None or pos == 0:
            return
        native = self._native_packed.get(layer_idx)
        if native is not None:
            # Every enabled bank was allocated with its final per-row capacity
            # at prefill commit.  Payload/offset/seqlen tensor addresses never
            # change, so the already captured attention graph remains valid.
            self._append_native_decode(
                native,
                buf_k[:, :, :pos],
                self._tail_buf_v[layer_idx][:, :, :pos],
                bits,
            )
            return
        state = self._states[layer_idx]
        if bits is None:
            raise RuntimeError("legacy graph flush requires explicit bit tags")
        quant_before = {
            level: bank.size for level, bank in state._quantized_banks.items()
        }
        exact_before = state._exact_bank.size
        state.append(buf_k[:, :, :pos].contiguous(),
                     self._tail_buf_v[layer_idx][:, :, :pos].contiguous(),
                     bits.to(dtype=torch.int32))
        if (self._graph_stable_banks and
                layer_idx in self._quant_ready_ragged and
                layer_idx in self._exact_ready):
            self._append_graph_ready(
                layer_idx, quant_before=quant_before, exact_before=exact_before,
            )
        else:
            self._quant_ready.pop(layer_idx, None)
            self._quant_ready_ragged.pop(layer_idx, None)
            self._exact_ready.pop(layer_idx, None)

    @staticmethod
    def _append_rows(
        dst_k: torch.Tensor, dst_v: torch.Tensor,
        dst_nk: Optional[torch.Tensor], dst_nv: Optional[torch.Tensor],
        seqlen: torch.Tensor, rows: torch.Tensor,
        src_k: torch.Tensor, src_v: torch.Tensor,
        src_nk: Optional[torch.Tensor] = None,
        src_nv: Optional[torch.Tensor] = None,
    ) -> None:
        """Append flat tokens into fixed padded rows without reallocating."""
        if rows.numel() == 0:
            return
        num_rows, capacity = dst_k.shape[:2]
        counts = torch.bincount(rows, minlength=num_rows).to(torch.int32)
        needed = seqlen + counts
        max_needed = int(needed.max().item())
        if max_needed > capacity:
            raise RuntimeError(
                f"stable graph bank capacity exceeded: need "
                f"{max_needed}, capacity={capacity}. Pass a larger "
                f"max_decode_tokens to enable_graph_tail().")
        order = torch.argsort(rows, stable=True)
        sorted_rows = rows[order]
        starts = torch.zeros(num_rows, device=rows.device, dtype=torch.long)
        starts[1:] = counts.to(torch.long).cumsum(0)[:-1]
        within = torch.arange(rows.numel(), device=rows.device) - starts[sorted_rows]
        dest = (sorted_rows * capacity + seqlen[sorted_rows].to(torch.long) + within)
        dst_k.view(num_rows * capacity, *dst_k.shape[2:]).index_copy_(0, dest, src_k[order])
        dst_v.view(num_rows * capacity, *dst_v.shape[2:]).index_copy_(0, dest, src_v[order])
        if dst_nk is not None and dst_nv is not None:
            dst_nk.view(-1).index_copy_(0, dest, src_nk[order].float())
            dst_nv.view(-1).index_copy_(0, dest, src_nv[order].float())
        seqlen.add_(counts)

    def _append_graph_ready(
        self, layer_idx: int, quant_before: dict[int, int], exact_before: int,
    ) -> None:
        """Copy only newly appended state entries into stable decode buffers."""
        state = self._states[layer_idx]
        quant_banks, _, _ = self._quant_ready_ragged[layer_idx]
        ready_by_level = {bank["bits"]: bank for bank in quant_banks}
        H_kv = self.num_heads_kv
        for level, bank in state._quantized_banks.items():
            start = quant_before[level]
            if bank.size == start:
                continue
            ready = ready_by_level[level]
            positions = bank.positions[start:]
            rows = positions[:, 0] * H_kv + positions[:, 1]
            self._append_ragged_rows(
                ready["packed_k"], ready["packed_v"],
                ready["norms_k"], ready["norms_v"], ready["seqlen"],
                ready["offset"], ready["stable_row_capacity"], rows,
                bank.key_quantized.indices[start:],
                bank.value_quantized.indices[start:],
                bank.key_quantized.norms[start:],
                bank.value_quantized.norms[start:],
            )

        exact = state._exact_bank
        if exact.size > exact_before:
            exact_k, exact_v, exact_seqlen = self._exact_ready[layer_idx]
            positions = exact.positions[exact_before:]
            rows = positions[:, 0] * H_kv + positions[:, 1]
            self._append_rows(
                exact_k, exact_v, None, None, exact_seqlen, rows,
                exact.keys[exact_before:].to(torch.bfloat16),
                exact.values[exact_before:].to(torch.bfloat16),
            )

    @staticmethod
    def _append_ragged_rows(
        dst_k: torch.Tensor, dst_v: torch.Tensor,
        dst_nk: torch.Tensor, dst_nv: torch.Tensor,
        seqlen: torch.Tensor, offset: torch.Tensor,
        row_capacity: torch.Tensor, rows: torch.Tensor,
        src_k: torch.Tensor, src_v: torch.Tensor,
        src_nk: torch.Tensor, src_nv: torch.Tensor,
    ) -> None:
        """Append flat tokens into fixed-capacity CSR rows in place."""
        if rows.numel() == 0:
            return
        num_rows = seqlen.numel()
        counts = torch.bincount(rows, minlength=num_rows).to(torch.int32)
        needed = seqlen + counts
        overflow = needed > row_capacity
        if bool(overflow.any()):
            row = int(overflow.nonzero()[0].item())
            raise RuntimeError(
                f"stable graph bank capacity exceeded for row {row}: need "
                f"{int(needed[row].item())}, capacity={int(row_capacity[row].item())}. "
                f"Pass a larger max_decode_tokens to enable_graph_tail().")
        order = torch.argsort(rows, stable=True)
        sorted_rows = rows[order]
        starts = torch.zeros(num_rows, device=rows.device, dtype=torch.long)
        starts[1:] = counts.to(torch.long).cumsum(0)[:-1]
        within = torch.arange(rows.numel(), device=rows.device) - starts[sorted_rows]
        dest = (offset[sorted_rows].to(torch.long)
                + seqlen[sorted_rows].to(torch.long) + within)
        dst_k.index_copy_(0, dest, src_k[order])
        dst_v.index_copy_(0, dest, src_v[order])
        dst_nk.index_copy_(0, dest, src_nk[order].float())
        dst_nv.index_copy_(0, dest, src_nv[order].float())
        seqlen.add_(counts)

    def rewind_graph_tail(self) -> None:
        """Zero the ring position/seqlen after all layers have been flushed."""
        if self._tail_pos is not None:
            self._tail_pos.zero_()
            self._tail_len_rows.zero_()
        self._graph_mask_tail_pos = 0

    def graph_tail_pos(self) -> int:
        return int(self._tail_pos.item()) if self._tail_pos is not None else 0

    # ── odmkv API (called by the press) ────────────────────────────

    def _ensure_native_flush_workspace(self, native, max_tokens: int):
        """Return cache-wide shared-v2 flush scratch, growing only on shape change."""
        from kvquant.runtime.kernels.native_flush_v2 import NativeFlushWorkspace

        tokens = int(max_tokens)
        workspace = self._native_flush_workspace
        if (
            workspace is None
            or workspace.rows != native.num_rows
            or workspace.head_dim != native.head_dim
            or workspace.max_tokens < tokens
            or workspace.normalized_k.device != native.tags.device
        ):
            workspace = NativeFlushWorkspace.allocate(
                rows=native.num_rows,
                max_tokens=tokens,
                head_dim=native.head_dim,
                device=native.tags.device,
            )
            self._native_flush_workspace = workspace
        return workspace

    def _append_native_decode(
        self,
        native,
        keys: torch.Tensor,
        values: torch.Tensor,
        bits: Optional[torch.Tensor],
    ) -> None:
        """Dispatch fixed-2 or allocator-selected mixed-bit native flush."""
        from kvquant.runtime.native_packing_v2 import NativePackedKVV2

        if isinstance(native, NativePackedKVV2):
            workspace = self._ensure_native_flush_workspace(native, keys.shape[2])
            if self._native_decode_level == 2:
                native.append_decode_2bit(keys, values, workspace=workspace)
            else:
                if bits is None:
                    raise RuntimeError("native mixed-bit decode flush requires bit tags")
                native.append_decode(keys, values, bits, workspace=workspace)
            return
        # R2_NATIVE_PACKING=v1 is an explicit comparison backend, not the
        # production path. Its independent arena API retains its own append.
        if self._native_decode_level != 2:
            raise RuntimeError("R2_NATIVE_PACKING=v1 supports fixed 2-bit decode only")
        native.append_decode_2bit(keys.contiguous(), values.contiguous())

    def commit_prefill(
        self,
        layer_idx: int,
        bits: torch.Tensor,                                 # int [B, H_kv, T]
        outlier_indices: Optional[torch.Tensor] = None,    # [n_out] long
    ) -> None:
        kv = self._prefill_kv.pop(layer_idx, None)
        if kv is None:
            raise RuntimeError(f"commit_prefill({layer_idx}): no prefill stash")
        k, v = kv                                          # [B, H_kv, T, D]
        native_decode_levels = self._native_decode_levels or (
            (self._native_decode_level,) if self._native_decode_level is not None else ()
        )
        use_native = (
            self._graph_tail
            and bool(native_decode_levels)
            and layer_idx >= self._native_initial_layers_fp16
            and self.key_quantizer == "mse"
            and self.value_quantizer == "mse"
            and (outlier_indices is None or outlier_indices.numel() == 0)
        )
        if use_native:
            backend = os.environ.get("R2_NATIVE_PACKING", "shared").strip().lower()
            if backend not in {"shared", "v1"}:
                raise ValueError(
                    f"R2_NATIVE_PACKING must be 'shared' or 'v1', got {backend!r}"
                )
            if backend == "v1" and self._native_decode_level != 2:
                raise RuntimeError(
                    "R2_NATIVE_PACKING=v1 is a fixed-2 diagnostic backend; "
                    "mixed 2/3/4/8/16 decode requires R2_NATIVE_PACKING=shared"
                )
            native_bits = bits.to(dtype=torch.int32)
            if backend == "v1":
                # Explicit diagnostic fallback for byte-for-byte comparison.
                from kvquant.runtime.native_packing import pack_native_prefill

                packed = pack_native_prefill(
                    k,
                    v,
                    native_bits,
                    seed=self.seed + layer_idx * 7,
                    decode_2bit_reserve_per_row=self._graph_decode_capacity,
                )
            else:
                policy = self._native_packing_policy
                if policy is None:
                    raise RuntimeError(
                        "native shared packing requires configure_native_packing_policy(); "
                        "production ODMNativePress configures it automatically"
                    )
                from kvquant.runtime.native_packing_v2 import (
                    NativeSharedPackingCapacity,
                    pack_native_prefill_shared_v2,
                )

                B, H, T, D = k.shape
                if self._native_decode_level == 2:
                    decode_2bit_reserve = self._graph_decode_capacity
                    decode_reserves = None
                else:
                    # Mixed/DE suffixes use one aggregate segmented arena,
                    # allocated after the compact prefill has been built.
                    decode_2bit_reserve = 0
                    decode_reserves = None
                # Mirror BatchedNativeAllocator's clamp exactly. A config such
                # as levels=(2,4,16), target=1 allocates against target=2; using
                # the raw target here would under-provision the physical arena.
                effective_target = max(
                    float(self.bit_levels[0]),
                    min(float(self.bit_levels[-1]), policy.target_avg_bits),
                )
                capacity = NativeSharedPackingCapacity.from_policy_budget(
                    batch_size=B,
                    num_heads=H,
                    seq_len=T,
                    head_dim=D,
                    target_avg_bits=effective_target,
                    sink_tokens=policy.sink_tokens,
                    tail_tokens=policy.tail_tokens,
                    decode_2bit_reserve_per_row=decode_2bit_reserve,
                    decode_reserve_per_row_by_level=decode_reserves,
                )
                packed = pack_native_prefill_shared_v2(
                    k,
                    v,
                    native_bits,
                    capacity=capacity,
                    seed=self.seed + layer_idx * 7,
                )
                if self._native_decode_level != 2:
                    packed.enable_segmented_decode(
                        capacity_per_row=self._graph_decode_capacity,
                        buffer_size=max(self._graph_buf_size, 1),
                        target_avg_bits=policy.target_avg_bits,
                        bit_levels=native_decode_levels,
                    )
                assert_async = getattr(torch, "_assert_async", None)
                if assert_async is None:
                    raise RuntimeError(
                        "native shared packing requires torch._assert_async for "
                        "device-only capacity fail-fast"
                    )
                assert_async(
                    packed.overflow_flag == 0,
                    "native shared packing capacity proof was violated; payload "
                    "stores were bounded and generation is aborted",
                )
                # Allocate once after the first layer proves the request shape;
                # all remaining layers reuse this same scratch set.  The ring
                # size, rather than the total decode reserve, bounds one flush.
                self._ensure_native_flush_workspace(
                    packed, max(self._graph_buf_size, 1),
                )
            self._native_packed[layer_idx] = packed
            self._committed[layer_idx] = True
            self.release_prefill_native_inputs(layer_idx)
            self._quant_ready.pop(layer_idx, None)
            self._quant_ready_ragged.pop(layer_idx, None)
            self._exact_ready.pop(layer_idx, None)
            return
        state = self._make_state(layer_idx, outlier_indices)
        state.append(k, v, bits.to(dtype=torch.int32))
        self._states[layer_idx] = state
        self._committed[layer_idx] = True
        self.release_prefill_native_inputs(layer_idx)
        self._quant_ready.pop(layer_idx, None)
        self._quant_ready_ragged.pop(layer_idx, None)
        self._exact_ready.pop(layer_idx, None)

    def get_quant_ready(self, layer_idx: int, num_heads_kv: int):
        """Cached per-head decode-ready quant-bank layout (built on first use
        after each commit/flush). Returns ``(quant_banks, pi_k, pi_v)`` or None
        if the state is unsupported (caller falls back to materialize)."""
        cached = self._quant_ready.get(layer_idx)
        if cached is not None:
            return cached
        # Native storage is ragged CSR only; the production fused path consumes
        # it directly. The old padded banked kernel is a diagnostic fallback.
        if layer_idx in self._native_packed:
            return None
        state = self._states.get(layer_idx)
        if state is None:
            return None
        from kvquant.runtime.decode_layout import build_quant_ready
        # Build one layout over all batch×KV-head rows so the fused kernel can
        # expose serving-batch parallelism instead of processing only b=0.
        built = build_quant_ready(state, num_heads_kv, b=None)
        self._quant_ready[layer_idx] = built
        return built

    def get_quant_ready_ragged(self, layer_idx: int, num_heads_kv: int):
        """Ragged (un-padded) quant-bank layout for the fused kernel — same
        contract as ``get_quant_ready`` but compact (no 4× padding). Cached and
        invalidated at the same points."""
        cached = self._quant_ready_ragged.get(layer_idx)
        if cached is not None:
            return cached
        native = self._native_packed.get(layer_idx)
        if native is not None:
            built = (list(native.quant_banks), native.pi_k_decode, native.pi_v)
            self._quant_ready_ragged[layer_idx] = built
            return built
        state = self._states.get(layer_idx)
        if state is None:
            return None
        if self._graph_tail and self._graph_stable_banks:
            from kvquant.runtime.decode_layout import build_quant_ready_stable
            built = build_quant_ready_stable(
                state, num_heads_kv, self._graph_decode_capacity, b=None,
            )
        else:
            from kvquant.runtime.decode_layout import build_quant_ready_ragged
            built = build_quant_ready_ragged(state, num_heads_kv, b=None)
        self._quant_ready_ragged[layer_idx] = built
        return built

    def get_exact_ready(self, layer_idx: int, num_heads_kv: int):
        """Cached ``(K, V, seqlen)`` layout for the static 16-bit prefix.

        The decode tail is deliberately excluded: callers pass its existing
        ``[B,H,T,D]`` storage as a reshape-only second exact bank.
        """
        cached = self._exact_ready.get(layer_idx)
        if cached is not None:
            return cached
        native = self._native_packed.get(layer_idx)
        if native is not None:
            exact = native.exact
            built = (
                exact["keys"],
                exact["values"],
                {
                    "offset": exact["offset"],
                    "seqlen": exact["seqlen"],
                    "T_max": exact["T_max"],
                },
            )
            self._exact_ready[layer_idx] = built
            return built
        state = self._states.get(layer_idx)
        if state is None:
            return None, None, None
        if self._graph_tail and self._graph_stable_banks:
            from kvquant.runtime.decode_layout import build_exact_stable
            built = build_exact_stable(
                state, num_heads_kv, self._graph_decode_capacity, b=None,
            )
        else:
            from kvquant.runtime.decode_layout import build_exact
            built = build_exact(state, None, None, num_heads_kv, b=None)
        self._exact_ready[layer_idx] = built
        return built

    def flush_decode_tail(
        self,
        layer_idx: int,
        bits: torch.Tensor,           # int [B, H_kv, T_tail]
    ) -> None:
        """Move tail buffer tokens into the per-bit banks via the state."""
        tail_k = self._tail_k.pop(layer_idx, None)
        tail_v = self._tail_v.pop(layer_idx, None)
        if tail_k is None or tail_k.shape[2] == 0:
            return
        native = self._native_packed.get(layer_idx)
        if native is not None:
            self._append_native_decode(native, tail_k, tail_v, bits)
            return
        state = self._states[layer_idx]
        state.append(tail_k, tail_v, bits.to(dtype=torch.int32))
        self._quant_ready.pop(layer_idx, None)
        self._quant_ready_ragged.pop(layer_idx, None)
        self._exact_ready.pop(layer_idx, None)

    def materialize_layer(
        self, layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self._prefill_kv:
            return self._prefill_kv[layer_idx]
        native = self._native_packed.get(layer_idx)
        if native is not None:
            K, V = native.materialize()
        else:
            state = self._states.get(layer_idx)
            if state is None:
                raise RuntimeError(f"materialize_layer({layer_idx}): no state")
            K, V = state.materialize()                          # [B, H_kv, T, D]
        if self._graph_tail:
            # Graph mode keeps the decode tail in the fixed ring, not _tail_k;
            # read the written prefix [:, :, :pos] so a materialize fallback (mask
            # / multi-token) doesn't silently drop generated tokens.
            buf_k = self._tail_buf_k.get(layer_idx)
            pos = self.graph_tail_pos()
            if buf_k is not None and pos > 0:
                K = torch.cat([K, buf_k[:, :, :pos]], dim=2)
                V = torch.cat([V, self._tail_buf_v[layer_idx][:, :, :pos]], dim=2)
            return K, V
        tail_k = self._tail_k.get(layer_idx)
        tail_v = self._tail_v.get(layer_idx)
        if tail_k is not None and tail_k.shape[2] > 0:
            K = torch.cat([K, tail_k], dim=2)
            V = torch.cat([V, tail_v], dim=2)
        return K, V

    def native_storage_stats(self, layer_idx: int) -> dict:
        """Return explicit live/reserved accounting for a native packed layer.

        The v1 diagnostic fallback predates separate accounting, so its single
        byte-exact report is exposed as ``combined`` rather than mislabeled as
        live storage. Production shared-v2 always provides both views.
        """
        native = self._native_packed.get(layer_idx)
        if native is None:
            raise RuntimeError(f"native_storage_stats({layer_idx}): no native state")
        live_fn = getattr(native, "live_storage_stats", None)
        reserved_fn = getattr(native, "reserved_storage_stats", None)
        if callable(live_fn) and callable(reserved_fn):
            return {
                "layout": "shared_v2",
                "live": live_fn(),
                "reserved": reserved_fn(),
            }
        combined_fn = getattr(native, "physical_storage_stats", None)
        if callable(combined_fn):
            return {
                "layout": "v1",
                "live": None,
                "reserved": None,
                "combined": combined_fn(),
            }
        raise RuntimeError(
            f"native state {type(native).__name__} exposes no storage accounting"
        )

    def evict_mask_for_layer(
        self, layer_idx: int, b: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Return the bool mask of evicted positions, or ``None``.

        By default the shape is ``[B, H_kv, T_total]`` so a batched fallback
        attention never reuses sample 0's allocation for other samples.
        Passing ``b`` retains the diagnostic single-sample shape
        ``[H_kv, T_total]``.
        """
        state = self._states.get(layer_idx)
        native = self._native_packed.get(layer_idx)
        if native is not None:
            tags = native.tags
            if native.decode_len:
                tags = torch.cat(
                    (tags, native.decode_tags[..., :native.decode_len]), dim=-1,
                )
            evicted = tags == 0
            return evicted[b] if b is not None else evicted
        if state is None or state._masked_positions.numel() == 0:
            return None
        T_state = int(state.seq_len)
        tail = self._tail_k.get(layer_idx)
        T_total = T_state + (tail.shape[2] if tail is not None else 0)
        mp = state._masked_positions                       # [N, 3] (b, h, s)
        if b is not None:
            mask = torch.zeros(
                self.num_heads_kv, T_total,
                dtype=torch.bool, device=self.device,
            )
            sel = mp[:, 0] == b
            if sel.any():
                sub = mp[sel]
                mask[sub[:, 1], sub[:, 2]] = True
            return mask

        batch_size = max(int(state.batch_size), self.batch_size, 1)
        mask = torch.zeros(
            batch_size, self.num_heads_kv, T_total,
            dtype=torch.bool, device=self.device,
        )
        mask[mp[:, 0], mp[:, 1], mp[:, 2]] = True
        return mask

    def algorithmic_evict_mask_for_layer(
        self, layer_idx: int, b: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Return 0-bit allocation decisions, excluding input padding.

        ``evict_mask_for_layer`` is an attention exclusion mask and therefore
        must mark both padded cells and algorithmically evicted real tokens.
        Reporting them as one quantity would inflate eviction for heterogeneous
        batches, so this diagnostic view removes the immutable prefill padding
        while retaining decode positions as valid.
        """
        excluded = self.evict_mask_for_layer(layer_idx, b=None)
        if excluded is None:
            return None
        valid_prefill = self._prefill_valid_mask
        if valid_prefill is None:
            result = excluded
        else:
            valid = valid_prefill[:, None, :].expand(
                -1, self.num_heads_kv, -1,
            )
            if excluded.shape[-1] > valid.shape[-1]:
                decode_valid = torch.ones(
                    *valid.shape[:-1], excluded.shape[-1] - valid.shape[-1],
                    dtype=torch.bool, device=valid.device,
                )
                valid = torch.cat((valid, decode_valid), dim=-1)
            result = excluded & valid[..., : excluded.shape[-1]]
        return result[b] if b is not None else result

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError(
            "Realsys2 native packed cache supports fixed-batch greedy decode; "
            "beam reorder would need to permute every CSR row and is not implemented."
        )

    def batch_repeat_interleave(self, repeats: int) -> None:
        del repeats
        raise NotImplementedError(
            "Realsys2 native packed cache does not support batch expansion; "
            "construct the final fixed batch before prefill."
        )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        del indices
        raise NotImplementedError(
            "Realsys2 native packed cache does not support dynamic slot selection."
        )

    def truncate_layer(self, layer_idx: int, target_seq_len: int) -> None:
        self._quant_ready.pop(layer_idx, None)
        self._quant_ready_ragged.pop(layer_idx, None)
        self._exact_ready.pop(layer_idx, None)
        cur = self.get_seq_length(layer_idx)
        if cur <= target_seq_len:
            return
        if not self._committed.get(layer_idx, False):
            if layer_idx in self._prefill_kv:
                k, v = self._prefill_kv[layer_idx]
                self._prefill_kv[layer_idx] = (
                    k[..., :target_seq_len, :].contiguous(),
                    v[..., :target_seq_len, :].contiguous(),
                )
            return
        if layer_idx in self._native_packed:
            raise NotImplementedError(
                "truncate of native packed prefix is not implemented; use a fresh request cache"
            )
        state = self._states[layer_idx]
        if target_seq_len < state.seq_len:
            state.truncate(target_seq_len)
            self._tail_k.pop(layer_idx, None)
            self._tail_v.pop(layer_idx, None)
            return
        new_tail = target_seq_len - state.seq_len
        if new_tail == 0:
            self._tail_k.pop(layer_idx, None)
            self._tail_v.pop(layer_idx, None)
        else:
            tk = self._tail_k.get(layer_idx)
            tv = self._tail_v.get(layer_idx)
            if tk is not None:
                self._tail_k[layer_idx] = tk[..., :new_tail, :].contiguous()
                self._tail_v[layer_idx] = tv[..., :new_tail, :].contiguous()

    def reset(self) -> None:
        self._states.clear()
        self._native_packed.clear()
        self._tail_k.clear()
        self._tail_v.clear()
        self._prefill_rope.clear()
        self._prefill_kv.clear()
        self._native_prefill_callback = None
        self._prefill_layout_configured = False
        self._prefill_valid_mask = None
        self._prefill_valid_lengths = None
        self._prefill_length_buckets = ()
        self._committed.clear()
        self._quant_ready.clear()
        self._quant_ready_ragged.clear()
        self._exact_ready.clear()
        self._last_decode_query.clear()
        self._tail_buf_k.clear()
        self._tail_buf_v.clear()
        self._graph_mask_tail_pos = 0
        if self._tail_pos is not None:
            self._tail_pos.zero_()
            self._tail_len_rows.zero_()
        self.batch_size = 0
