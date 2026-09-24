"""CUDA-graph decode over fixed-address compressed KV banks.

Input and tail buffers retain stable addresses across replay. The shared native
packer reserves space for future tokens, so flushes append in place without
invalidating a captured graph. This backbone driver also supports the sampling
path; full_graph_decode captures the language-model head for greedy generation.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from kvquant.runtime.graph_runtime import graph_capture_guard, graph_warmup_stream


class GraphDecoder:
    """Capture-and-replay driver for launch-free incremental fused decode.

    ``backbone`` is the model body (``model.model``) whose forward runs the
    odmkv fused attention; ``cache`` is a ``ODMCache`` with the graph
    tail enabled. The driver owns the static input buffers and the captured
    graph. Shared-v2 flushes preserve it; legacy dynamic banks may invalidate it.
    """

    def __init__(
        self,
        backbone,
        cache,
        batch_size: int,
        device: torch.device,
        warmup_iters: int = 3,
    ):
        self.backbone = backbone
        self.cache = cache
        self.B = batch_size
        self.dev = device
        self.warmup_iters = warmup_iters
        # Static decode inputs — updated in place, read by the captured graph.
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.cpos = torch.zeros(1, dtype=torch.long, device=device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_hidden: Optional[torch.Tensor] = None
        self.buffer_size = cache._graph_buf_size
        self.num_layers = cache.num_layers
        # Host mirror of the ring write position.  Reading the device scalar via
        # ``.item()`` in needs_flush() serialized every generated token.  The
        # graph advances exactly once per replay, so the host can track the same
        # value without a GPU->CPU synchronization.
        self._tail_count = 0
        if self.buffer_size < warmup_iters + 1:
            raise ValueError(
                f"GraphDecoder: ring buffer_size {self.buffer_size} must be "
                f">= warmup_iters+1 ({warmup_iters + 1}); capture warmup writes "
                f"{warmup_iters} slots and would run off the ring.")

    def _forward(self):
        return self.backbone(
            input_ids=self.ids,
            past_key_values=self.cache,
            position_ids=self.pos,
            cache_position=self.cpos,
        )

    def _set_inputs(self, tok_ids: torch.Tensor, position: int) -> None:
        self.ids.copy_(tok_ids.reshape(self.B, 1))
        self.pos.fill_(position)
        self.cpos.fill_(position)

    def _capture(self) -> None:
        if self.graph is not None:
            return
        with graph_capture_guard(self.dev):
            if self.graph is None:
                self._capture_locked()

    def _capture_locked(self) -> None:
        """Warm up (compile/autotune the fused kernels), then RECORD one decode
        step into a graph.

        Crucially, ``cudaStreamBeginCapture`` records work without executing it,
        so the ops run inside the ``with`` block produce no valid output and the
        in-place ring/position updates do NOT actually advance. The graph is
        therefore a pure recording: the ring is rewound to the pre-capture
        position and EVERY real step — including the first — is a ``replay`` (an
        actual execution). This is the vLLM/TGI decode-graph discipline; reading
        output straight from capture would return garbage."""
        pre = self._tail_count
        s = graph_warmup_stream(self.dev)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for warmup_idx in range(self.warmup_iters):
                self.cache.set_graph_mask_tail_pos(pre + warmup_idx)
                self._forward()
        torch.cuda.current_stream().wait_stream(s)
        self.cache._tail_pos.fill_(pre)                 # undo warmup advance
        self.cache.set_graph_mask_tail_pos(pre)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            out = self._forward()
        self.static_hidden = out.last_hidden_state
        self.cache._tail_pos.fill_(pre)                 # capture didn't execute → rewind
        self.cache.set_graph_mask_tail_pos(pre)

    def step(self, tok_ids: torch.Tensor, position: int) -> torch.Tensor:
        """Advance one token. Captures on first use (recording only), then the
        SAME call and every subsequent one replay the graph for real.

        Returns the fixed-address last-position hidden state (overwritten next
        step — clone if you need to keep it)."""
        self._set_inputs(tok_ids, position)
        if self.graph is None:
            self._capture()
        self.graph.replay()
        self._tail_count += 1
        self.cache.set_graph_mask_tail_pos(self._tail_count)
        return self.static_hidden[:, -1]

    @torch.inference_mode()
    def generate(self, first_tok, start_pos, n_new, lm_head, bits_fn, greedy=True):
        """Greedy launch-free generation. ``first_tok`` [B,1] is the token that
        follows the prefill; positions run ``start_pos ..`` absolute (RoPE is
        unaffected by ring flushes). Flushes the ring in place when it fills.
        Returns the generated token ids [B, n_new]."""
        # The decode reserve is physical arena storage fixed by prefill packing;
        # changing only ``_graph_decode_capacity`` afterwards would create a
        # false logical bound and fail several flushes later.  Direct callers
        # must therefore size it before prefill, just like ``graph_generate``.
        required_capacity = int(n_new)
        if required_capacity > self.cache._graph_decode_capacity:
            layout_exists = bool(
                self.cache._native_packed
                or self.cache._states
                or self.cache._committed
                or self.cache._prefill_kv
            )
            if self.graph is not None or layout_exists:
                raise RuntimeError(
                    "GraphDecoder decode reserve is already fixed by prefill: "
                    f"need {required_capacity}, allocated "
                    f"{self.cache._graph_decode_capacity}. Call "
                    "enable_graph_tail(..., max_decode_tokens=n_new) before "
                    "the prefill forward."
                )
            self.cache._graph_decode_capacity = required_capacity
        toks, tok = [], first_tok
        for k in range(n_new):
            if self.needs_flush():
                self.flush(bits_fn)
            h = self.step(tok, start_pos + k)              # [B, H]
            logits = lm_head(h)                            # [B, V]
            tok = logits.argmax(-1, keepdim=True)          # greedy [B,1]
            toks.append(tok)
        return torch.cat(toks, dim=1)

    def needs_flush(self) -> bool:
        """True when the ring has no room for another token."""
        return self._tail_count >= self.buffer_size

    def flush(self, bits_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        """Move the full ring into the banks.

        ``bits_fn(layer_idx, ring_k, ring_v) -> int32[B,H_kv,buffer]`` supplies
        the per-token bit allocation for the flushed tail (the caller's decode
        press policy). Stable graph banks are updated in place and reuse the
        existing capture; the legacy dynamic layout re-captures on next step.
        """
        valid_tokens = self._tail_count
        pending = []
        for layer_idx in range(self.num_layers):
            buf_k = self.cache._tail_buf_k.get(layer_idx)
            if buf_k is None:
                continue
            buf_v = self.cache._tail_buf_v[layer_idx]
            pending.append((
                layer_idx,
                buf_k[:, :, :valid_tokens],
                buf_v[:, :, :valid_tokens],
            ))

        if self.cache._native_decode_level == 2:
            allocated = (None,) * len(pending)
        else:
            all_layers = getattr(bits_fn, "all_layers", None)
            if callable(all_layers) and pending:
                batched = all_layers(
                    tuple(item[0] for item in pending),
                    tuple(item[1] for item in pending),
                    tuple(item[2] for item in pending),
                )
                if batched.shape[0] != len(pending):
                    raise RuntimeError(
                        "all-layer decode allocator returned the wrong layer axis: "
                        f"{tuple(batched.shape)} for {len(pending)} layers"
                    )
                allocated = tuple(batched.unbind(0))
            else:
                allocated = tuple(
                    bits_fn(layer_idx, key, value)
                    for layer_idx, key, value in pending
                )

        for (layer_idx, _, _), bits in zip(pending, allocated, strict=True):
            self.cache.flush_graph_tail(
                layer_idx, bits, valid_tokens=valid_tokens,
            )
        self.cache.rewind_graph_tail()
        self._tail_count = 0
        if not self.cache._graph_stable_banks:
            # Legacy dynamic layouts change pointers/shapes and must recapture.
            self.graph = None
            self.static_hidden = None
