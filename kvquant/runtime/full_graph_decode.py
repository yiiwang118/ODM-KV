"""Complete CUDA-graph decode for the compressed Realsys2 system.

One replay performs the full token transition:

``embedding/backbone -> mixed-bit attention -> lm_head -> fp32 argmax ->
EOS state -> token feedback/history -> position updates``.

The fixed KV ring and packed-bank pointers are owned by :class:`ODMCache`.
Default all-2-bit maintenance still occurs once per full ring outside the graph;
it writes fixed addresses and does not invalidate this graph.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from kvquant.runtime.graph_runtime import graph_capture_guard, graph_warmup_stream


class NativeCompressedGraphDecoder:
    """Full fixed-batch greedy graph with per-request logical positions.

    Equal-length requests pass an integer ``start_position`` and retain the
    original path.  A strict left-padded batch passes ``[B,1]`` logical start
    positions while ``cache_start_position`` remains the common physical padded
    width expected by Transformers cache metadata.  Attention itself reads
    compact per-row CSR lengths and never materializes a padding mask at decode.
    """

    def __init__(
        self,
        backbone,
        lm_head,
        cache,
        first_token: torch.Tensor,
        start_position: int | torch.Tensor,
        max_new_tokens: int,
        bits_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        eos_token_ids: Optional[torch.Tensor | list[int] | tuple[int, ...]] = None,
        warmup_iters: int = 3,
        cache_start_position: Optional[int] = None,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if first_token.ndim != 2 or first_token.shape[1] != 1:
            raise ValueError("first_token must have shape [B,1]")
        if cache._graph_buf_size < warmup_iters + 1:
            raise ValueError(
                f"ring buffer {cache._graph_buf_size} must be >= warmup_iters+1 "
                f"({warmup_iters + 1})"
            )
        self.backbone = backbone
        self.lm_head = lm_head
        self.cache = cache
        self.bits_fn = bits_fn
        self.batch_size = int(first_token.shape[0])
        self.device = first_token.device
        self.max_new_tokens = int(max_new_tokens)
        self.warmup_iters = int(warmup_iters)
        self.buffer_size = int(cache._graph_buf_size)
        self.num_layers = int(cache.num_layers)
        self._tail_count = 0
        self._flush_count = 0
        self._real_replays = 0

        self.ids = first_token.detach().clone().reshape(self.batch_size, 1)
        if isinstance(start_position, torch.Tensor):
            logical_start = start_position.to(
                device=self.device, dtype=torch.long,
            )
            if logical_start.numel() == 1:
                logical_start = logical_start.reshape(1, 1).expand(
                    self.batch_size, 1,
                )
            elif logical_start.shape == (self.batch_size,):
                logical_start = logical_start.view(self.batch_size, 1)
            elif logical_start.shape != (self.batch_size, 1):
                raise ValueError(
                    "tensor start_position must be scalar, [B], or [B,1]; "
                    f"got {tuple(logical_start.shape)}"
                )
            self.position_ids = logical_start.contiguous().clone()
            if cache_start_position is None and logical_start.numel() != 1:
                raise ValueError(
                    "cache_start_position is required for per-request logical positions"
                )
            physical_start = (
                int(cache_start_position)
                if cache_start_position is not None else
                int(logical_start.reshape(-1)[0].item())
            )
            self.start_position = self.position_ids.clone()
        else:
            physical_start = (
                int(start_position)
                if cache_start_position is None else int(cache_start_position)
            )
            self.start_position = int(start_position)
            self.position_ids = torch.full(
                (self.batch_size, 1), int(start_position),
                device=self.device, dtype=torch.long,
            )
        self.cache_position = torch.full(
            (1,), physical_start, device=self.device, dtype=torch.long,
        )
        self._history_width = max(self.max_new_tokens, self.warmup_iters + 1)
        self.history = torch.empty(
            self.batch_size, self._history_width,
            device=self.device, dtype=torch.long,
        )
        self.history[:, :1].copy_(self.ids)
        self.history_position = torch.ones(1, device=self.device, dtype=torch.long)

        if eos_token_ids is None:
            self.eos_token_ids = None
            self.done = None
            self.pinned_eos = None
        else:
            eos = torch.as_tensor(
                eos_token_ids, device=self.device, dtype=torch.long,
            ).reshape(-1)
            if eos.numel() == 0:
                raise ValueError("eos_token_ids cannot be empty")
            self.eos_token_ids = eos
            self.pinned_eos = eos[0]
            self.done = (self.ids == eos.view(1, -1)).any(dim=-1, keepdim=True)

        self.static_hidden: Optional[torch.Tensor] = None
        self.static_token: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None

    def _forward(self) -> torch.Tensor:
        out = self.backbone(
            input_ids=self.ids,
            past_key_values=self.cache,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            use_cache=True,
        )
        hidden = out.last_hidden_state[:, -1]
        # Keep identical baseline semantics: BF16/FP32-accum lm_head, explicit
        # fp32 logits for deterministic greedy argmax.
        token = self.lm_head(hidden).float().argmax(-1, keepdim=True)
        if self.eos_token_ids is not None:
            hit = (token == self.eos_token_ids.view(1, -1)).any(
                dim=-1, keepdim=True,
            )
            self.done.logical_or_(hit)
            token = torch.where(self.done, self.pinned_eos, token)
        self.ids.copy_(token)
        self.history.index_copy_(1, self.history_position, token)
        self.history_position.add_(1)
        self.position_ids.add_(1)
        self.cache_position.add_(1)
        self.static_hidden = hidden
        self.static_token = token
        return token

    def _restore(
        self,
        ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        history: torch.Tensor,
        history_position: torch.Tensor,
        done: Optional[torch.Tensor],
        tail_position: int,
    ) -> None:
        self.ids.copy_(ids)
        self.position_ids.copy_(position_ids)
        self.cache_position.copy_(cache_position)
        self.history.copy_(history)
        self.history_position.copy_(history_position)
        if done is not None:
            self.done.copy_(done)
        self.cache._tail_pos.fill_(tail_position)
        if self.cache._tail_len_rows is not None:
            self.cache._tail_len_rows.fill_(tail_position)
        self.cache.set_graph_mask_tail_pos(tail_position)

    def capture(self) -> None:
        if self.graph is not None or self.max_new_tokens <= 1:
            return
        with graph_capture_guard(self.device):
            # Another thread cannot normally share this decoder, but recheck
            # after admission so the guard remains correct for future pools.
            if self.graph is None:
                self._capture_locked()

    def _capture_locked(self) -> None:
        if self.graph is not None or self.max_new_tokens <= 1:
            return
        if self._tail_count + self.warmup_iters >= self.buffer_size:
            raise RuntimeError("not enough empty ring slots for graph warmup")
        ids = self.ids.clone()
        position_ids = self.position_ids.clone()
        cache_position = self.cache_position.clone()
        history = self.history.clone()
        history_position = self.history_position.clone()
        done = self.done.clone() if self.done is not None else None
        pre = self._tail_count

        stream = graph_warmup_stream(self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for warmup_idx in range(self.warmup_iters):
                self.cache.set_graph_mask_tail_pos(pre + warmup_idx)
                self._forward()
        torch.cuda.current_stream().wait_stream(stream)
        self._restore(
            ids, position_ids, cache_position, history, history_position, done, pre,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self._forward()
        self.graph = graph
        # Capture records but does not constitute a real generated token.
        self._restore(
            ids, position_ids, cache_position, history, history_position, done, pre,
        )

    def needs_flush(self) -> bool:
        return self._tail_count >= self.buffer_size

    @torch.inference_mode()
    def flush(self) -> None:
        valid_tokens = self._tail_count
        if valid_tokens == 0:
            return
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

        native_fixed = self.cache._native_decode_level == 2
        if native_fixed:
            allocated = (None,) * len(pending)
        else:
            all_layers = getattr(self.bits_fn, "all_layers", None)
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
                    self.bits_fn(layer_idx, key, value)
                    for layer_idx, key, value in pending
                )

        for (layer_idx, _, _), bits in zip(pending, allocated, strict=True):
            self.cache.flush_graph_tail(
                layer_idx, bits, valid_tokens=valid_tokens,
            )
        self.cache.rewind_graph_tail()
        self._tail_count = 0
        self._flush_count += 1

    def replay(self) -> torch.Tensor:
        if self.graph is None:
            self.capture()
        if self._real_replays >= self.max_new_tokens - 1:
            raise RuntimeError("requested more graph steps than max_new_tokens")
        if self.needs_flush():
            self.flush()
        self.graph.replay()
        self._tail_count += 1
        self._real_replays += 1
        # Host mirror is for future capture/mask diagnostics only. Replay itself
        # contains no Python mask construction.
        self.cache.set_graph_mask_tail_pos(self._tail_count)
        return self.static_token

    def generate(self, *, clone_output: bool = True) -> torch.Tensor:
        """Generate the fixed token count.

        Public callers receive an owning clone by default. Systems benchmarks
        can request the stable history view and account for the same explicit
        post-generation clone as the Full-KV baseline.
        """
        if self.max_new_tokens == 1:
            result = self.history[:, :1]
        else:
            self.capture()
            for _ in range(self.max_new_tokens - 1):
                self.replay()
            result = self.history[:, : self.max_new_tokens]
        return result.clone() if clone_output else result
