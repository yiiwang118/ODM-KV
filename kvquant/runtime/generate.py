"""End-to-end generation on the native compressed-attention system.

The production entry point a serving layer calls: prefill a prompt into the
mixed-precision banks, then decode through a fixed-ring CUDA graph.  Greedy
generation captures the complete token transition (backbone + mixed-bit
attention + LM head + fp32 argmax + feedback/history/positions).  Non-greedy
sampling retains the host sampler around the backbone graph.

``graph_generate`` returns the generated token ids; the decode is bitwise
identical to the eager odmkv fused path (`graph_generate_test.py`), so on real
text it reproduces eager generation exactly while running the compressed,
launch-free kernel.
"""
from __future__ import annotations

from typing import Optional

import torch

from kvquant.factory import make_press
from kvquant.runtime.full_graph_decode import NativeCompressedGraphDecoder
from kvquant.runtime.graph_decode import GraphDecoder
from kvquant.runtime.prefill_memory import chunked_prefill_feed_forward


DEFAULT_PRESS = dict(
    mode="native", scorer="odm", epsilon=1e-2, normalize_grain="global",
    bits=[0, 2, 3, 4, 8, 16], target_avg_bits=2.0,
    eviction_cost=0.5, layerwise=True, outlier_min_bits=3, key_quantizer="mse",
    value_quantizer="mse", n_outlier_channels=0, sink_tokens=4, score_sink_tokens=4,
    buffer_size=128, decode_quant=True, allow_decode_eviction=False,
    initial_layers_fp16=0,
)


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """[B,V] logits → [B,1] next ids. temperature==0 ⇒ greedy (deterministic,
    the token-for-token-vs-eager-verified path). Otherwise temperature + nucleus."""
    if temperature <= 0.0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if 0.0 < top_p < 1.0:
        s, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(s, dim=-1).cumsum(-1)
        drop = cum - torch.softmax(s, dim=-1) > top_p
        s = s.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, s)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


def _truncate_at_first_all_eos(tokens: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Trim speculative pinned-EOS columns with one device→host sync."""
    ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
    all_eos = torch.isin(tokens, torch.tensor(ids, device=tokens.device)).all(dim=0)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    sentinel = torch.full_like(positions, tokens.shape[1])
    first = torch.where(all_eos, positions, sentinel).min()
    end = int(first.item())
    return tokens[:, :end + 1] if end < tokens.shape[1] else tokens


def _canonicalize_prefill_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Validate strict left padding and canonicalize all-valid to ``None``."""
    if attention_mask is None:
        return None
    if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "native varlen prefill requires attention_mask with the same [B,T] "
            f"shape as input_ids; got {tuple(attention_mask.shape)} vs "
            f"{tuple(input_ids.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        binary = (attention_mask == 0) | (attention_mask == 1)
        if not bool(binary.all()):
            raise ValueError("native varlen attention_mask values must be exactly 0 or 1")
    keep = attention_mask.to(device=input_ids.device, dtype=torch.bool).contiguous()
    if not bool(keep.any(dim=1).all()):
        raise ValueError("native varlen prefill requires at least one valid token per row")
    if keep.shape[1] > 1 and bool((keep[:, :-1] & ~keep[:, 1:]).any()):
        raise ValueError("native varlen prefill supports strict left padding only (0* 1+)")
    return None if bool(keep.all()) else keep


def _validate_graph_config(model, cfg: dict, buffer_size: int, warmup_iters: int = 3):
    """Reject configs outside the fused/graph decode's validated envelope up
    front (Llama-like · bf16 · single GPU · MSE/MSE · exact level present · no
    OCS / initial-fp16). Silent fallbacks here corrupt or crash, so fail fast."""
    if model.training:
        raise ValueError(
            "native graph generation is inference-only; call model.eval() first"
        )
    dt = next(model.parameters()).dtype
    if dt != torch.bfloat16:
        raise ValueError(f"graph decode supports bf16 only (cache is bf16); model is {dt}.")
    devs = {p.device for p in model.parameters()}
    if any(dev.type != "cuda" for dev in devs):
        raise ValueError(
            f"graph decode requires every model parameter on CUDA; model spans "
            f"{devs}. CPU/disk offload would invalidate latency measurements and "
            f"is not supported by the single-GPU graph path.")
    if len(devs) != 1:
        raise ValueError(f"graph decode is single-GPU; model spans {devs}. Load on one GPU.")
    if cfg.get("key_quantizer", "mse") != "mse" or cfg.get("value_quantizer", "mse") != "mse":
        raise ValueError("graph decode supports key/value quantizer 'mse' only.")
    levels = set(int(b) for b in cfg.get("bits", []))
    unsupported = levels - {0, 2, 3, 4, 8, 16}
    if unsupported:
        raise ValueError(f"native packed graph has unsupported bit levels: {sorted(unsupported)}")
    if 16 not in levels:
        raise ValueError("bit_levels must include 16 (exact) for sink/recent-buffer protection.")
    nonzero = sorted(levels - {0})
    if not nonzero or nonzero[0] != 2:
        raise ValueError("native decode requires 2 as the lowest non-zero bit level.")
    if not bool(cfg.get("layerwise", True)):
        raise ValueError("native system implements the requested layerwise allocation only.")
    if not bool(cfg.get("decode_quant", True)):
        raise ValueError("native graph requires decode_quant=true.")
    target = float(cfg.get("target_avg_bits", 1.0))
    if target < 0.0 or target > float(nonzero[-1]):
        raise ValueError(
            f"target_avg_bits must lie in [0,{nonzero[-1]}], got {target}"
        )
    if float(cfg.get("eviction_cost", 0.5)) != 0.5:
        raise ValueError("native system contract fixes eviction_cost=0.5.")
    if buffer_size < warmup_iters + 1:
        raise ValueError(f"buffer_size {buffer_size} must be >= {warmup_iters + 1} "
                         f"(graph capture warms up {warmup_iters} ring slots).")
    if int(cfg.get("initial_layers_fp16", 0)) > 0:
        raise ValueError("initial_layers_fp16>0 is not honored by the graph decode path.")
    if int(cfg.get("n_outlier_channels", 0)) > 0:
        raise ValueError("n_outlier_channels>0 (OCS) is not supported by the fused kernel.")

    config = model.config
    h_q = int(config.num_attention_heads)
    h_kv = int(getattr(config, "num_key_value_heads", None) or h_q)
    # Some supported Transformers releases expose ``head_dim`` but leave it
    # as ``None``.  Falling back with ``or`` keeps admission deterministic
    # instead of failing with the opaque ``int(None)`` error.
    head_dim = int(getattr(config, "head_dim", None) or (config.hidden_size // h_q))
    if h_q % h_kv:
        raise ValueError(f"num_attention_heads={h_q} must be divisible by H_kv={h_kv}.")
    if head_dim not in (64, 128):
        raise ValueError(
            f"native fused graph is validated for head_dim 64/128, got {head_dim}"
        )
    attention_modules = [
        module for module in model.modules()
        if module.__class__.__name__ in (
            "LlamaAttention", "Qwen2Attention", "Qwen3Attention",
        )
    ]
    expected_layers = int(config.num_hidden_layers)
    if len(attention_modules) != expected_layers:
        raise ValueError(
            "native graph requires exactly one supported attention module per "
            f"decoder layer; found {len(attention_modules)}, expected {expected_layers}"
        )
    layer_indices = [
        int(getattr(module, "layer_idx", None))
        if getattr(module, "layer_idx", None) is not None else -1
        for module in attention_modules
    ]
    if layer_indices != list(range(expected_layers)):
        raise ValueError(
            f"attention layer_idx sequence must be 0..{expected_layers - 1}, "
            f"got {layer_indices}"
        )


def make_flush_bits_fn(press, backbone, cache):
    """Bit allocation for a ring flush — the press's own scorer+Lagrangian, so a
    graph flush produces exactly the banks the eager path would."""
    layers = backbone.layers

    def bits_fn(layer_idx, ring_k, ring_v):
        return press._inner.compute_flush_bits(
            layers[layer_idx].self_attn, layer_idx,
            ring_k, ring_v, None)

    def all_layers(layer_indices, ring_keys, ring_values):
        indices = tuple(int(index) for index in layer_indices)
        return press._inner.compute_flush_bits_all_layers(
            tuple(layers[index].self_attn for index in indices),
            indices,
            tuple(ring_keys),
            tuple(ring_values),
        )

    # Keep the existing callable contract for diagnostics/custom decoders.
    # Scorers without a batched decode entry retain the exact per-layer path;
    # constant-bit policies need no scorer and may still batch allocation.
    inner = press._inner
    levels = inner._decode_bit_levels
    predetermined = (
        len(levels) == 1
        or inner.target_avg_bits <= float(levels[0])
    )
    if predetermined or callable(
        getattr(inner.scorer, "score_with_attn_batched", None)
    ):
        bits_fn.all_layers = all_layers

    return bits_fn


@torch.inference_mode()
def graph_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    press_cfg: Optional[dict] = None,
    buffer_size: int = 128,
    seed: int = 42,
    eos_token_id: int | list[int] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_check_interval: int = 8,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prefill ``input_ids`` into the compressed cache, then decode up to
    ``max_new_tokens`` via the CUDA-graph decoder. Greedy by default (temperature
    0); pass temperature/top_p for sampling. Per-sequence EOS: once a row emits
    ``eos_token_id`` it is pinned to EOS and generation stops when all rows have.
    EOS completion is polled every ``eos_check_interval`` steps instead of
    synchronizing the GPU on every token; rows remain pinned to EOS and any
    speculative suffix is trimmed exactly before return. Returns
    [B, ≤max_new_tokens].  ``attention_mask`` may describe a strict
    left-padded batch (rows ``0* 1+``).  The padded greedy path keeps FA2 varlen
    prefill and the same q_len=1 native fused graph decode; it is a bounded
    correctness extension whose bucketed prefill is not yet the headline
    launch-optimal equal-length path."""
    if eos_token_id == []:
        eos_token_id = None
    eos_ids = None if eos_token_id is None else ([eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    B, ctx = input_ids.shape
    attention_mask = _canonicalize_prefill_attention_mask(
        input_ids, attention_mask,
    )
    if max_new_tokens <= 0:
        return torch.empty(B, 0, dtype=torch.long, device=input_ids.device)
    if eos_check_interval <= 0:
        raise ValueError("eos_check_interval must be positive")
    cfg = dict(DEFAULT_PRESS)
    cfg.update(press_cfg or {})
    if cfg["mode"] not in {"native", "odmkv"}:
        raise ValueError("graph_generate requires the native backend")
    cfg["buffer_size"] = buffer_size
    _validate_graph_config(model, cfg, buffer_size)
    press = make_press(cfg, seed=seed)
    cache = press.make_compressed_cache(model)
    # Validate/freeze padding before the first layer allocates Q/K/V.  All-one
    # masks canonicalize to None and therefore execute the byte-for-byte
    # equal-length fast path.
    cache.configure_prefill_layout(
        attention_mask,
        batch_size=B,
        seq_len=ctx,
    )
    prefill_mask = cache._prefill_valid_mask
    if prefill_mask is not None and temperature > 0.0:
        raise ValueError(
            "native left-padded batches currently support greedy graph decode "
            "only (temperature must be 0)"
        )
    H_kv = model.config.num_key_value_heads
    cache.enable_graph_tail(
        buffer_size, num_rows=B * H_kv,
        max_decode_tokens=max_new_tokens,
    )
    with press(model):
        bb = model.model
        prefill_kwargs = {}
        if prefill_mask is not None:
            # HF's raw model forward otherwise assigns the physical padded
            # columns 0..T-1 to every row.  Generation semantics require each
            # request's real tokens to occupy logical positions 0..L-1.
            position_ids = prefill_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~prefill_mask, 0)
            prefill_kwargs.update(
                attention_mask=prefill_mask,
                position_ids=position_ids,
            )
        with chunked_prefill_feed_forward(model):
            out = bb(
                input_ids=input_ids,
                past_key_values=cache,
                **prefill_kwargs,
            )
        first = _sample(model.lm_head(out.last_hidden_state[:, -1]).float(), temperature, top_p)
        bits_fn = make_flush_bits_fn(press, bb, cache)

        # Production greedy path: every GPU operation for one generated token,
        # including LM head, argmax, feedback and position updates, is captured
        # in one graph. Ring maintenance writes fixed-address bit-bank reserves in
        # place and therefore does not invalidate or recapture this graph.
        if temperature <= 0.0:
            dec = NativeCompressedGraphDecoder(
                bb,
                model.lm_head,
                cache,
                first,
                (
                    cache._prefill_valid_lengths.view(B, 1)
                    if prefill_mask is not None else ctx
                ),
                max_new_tokens,
                bits_fn,
                eos_token_ids=eos_ids,
                cache_start_position=ctx,
            )
            dec.capture()
            for step in range(max_new_tokens - 1):
                if (dec.done is not None and step % eos_check_interval == 0
                        and bool(dec.done.all())):
                    break
                dec.replay()
            generated = dec.history[:, : 1 + dec._real_replays].clone()
            if eos_token_id is not None:
                generated = _truncate_at_first_all_eos(generated, eos_token_id)
            return generated

        # Sampling path: graph the transformer backbone, then apply temperature
        # and nucleus sampling on the host-controlled loop.
        dec = GraphDecoder(bb, cache, B, input_ids.device)
        done = torch.zeros(B, 1, dtype=torch.bool, device=input_ids.device)
        if eos_token_id is not None:
            done |= torch.isin(first, torch.tensor(eos_ids, device=first.device))
        toks = [first]
        tok = first
        for k in range(max_new_tokens - 1):
            if (eos_token_id is not None and k % eos_check_interval == 0
                    and bool(done.all())):
                break
            if dec.needs_flush():
                dec.flush(bits_fn)
            h = dec.step(tok, ctx + k)
            tok = _sample(model.lm_head(h).float(), temperature, top_p)
            if eos_token_id is not None:
                tok = torch.where(done, torch.full_like(tok, eos_ids[0]), tok)
                done |= torch.isin(tok, torch.tensor(eos_ids, device=tok.device))
            toks.append(tok)
        generated = torch.cat(toks, dim=1)
        if eos_token_id is not None:
            generated = _truncate_at_first_all_eos(generated, eos_token_id)
        return generated


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens: int = 128,
                  temperature: float = 0.0, top_p: float = 1.0,
                  buffer_size: int = 128, chat: bool = True,
                  eos_check_interval: int = 8) -> str:
    """Plainest entry point: text prompt → generated text on the compressed cache.

    ``chat=True`` applies the model's chat template (instruct models). This is the
    one-call path a serving layer or a user calls; the decode underneath is the
    launch-free graph path, token-for-token equal to eager odmkv at
    temperature 0."""
    dev = next(model.parameters()).device
    if chat:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt", return_dict=True)
        ids = enc["input_ids"].to(dev)
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    out = graph_generate(model, ids, max_new_tokens, buffer_size=buffer_size,
                         eos_token_id=model.generation_config.eos_token_id,
                         temperature=temperature, top_p=top_p,
                         eos_check_interval=eos_check_interval)
    return tokenizer.decode(out[0], skip_special_tokens=True)

