# Adapted from autokv/kvpress/pipeline.py
from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

import torch
from transformers import Cache, DynamicCache, PreTrainedModel, PreTrainedTokenizerBase

from benchmark.core.base_press import BasePress

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


class KVPressTextGenerationRunner:
    """Prefill context (optionally with a press), then greedy-decode answers.

    Pattern:  context tokens → model.model() w/ press → DynamicCache
              for each question: decode from cache, then trim cache back
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model     = model
        self.tokenizer = tokenizer

    def _model_input_device(self) -> torch.device:
        try:
            emb = self.model.get_input_embeddings()
        except Exception:
            emb = None
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def __call__(
        self,
        context: str,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        enable_thinking: bool = False,
        use_chat_template: bool = True,
        cache: Optional[Cache] = None,
    ) -> dict[str, str | list[str]]:
        assert not (question and questions), "Provide question or questions, not both."
        single = questions is None
        questions = questions or ([question] if question else [""])

        tokenizer_limit = min(getattr(self.tokenizer, "model_max_length", int(1e10)), int(1e10))
        max_ctx = tokenizer_limit if max_context_length is None else min(max_context_length, tokenizer_limit)
        tensors = self._tokenize(context, questions, answer_prefix or "", max_ctx,
                                 enable_thinking=enable_thinking,
                                 use_chat_template=use_chat_template)
        answers = self._run(tensors, press, max_new_tokens, cache)

        return {"answer": answers[0]} if single else {"answers": answers}

    # ── Tokenisation ──────────────────────────────────────────────────────────

    def _tokenize(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
        enable_thinking: bool = False,
        use_chat_template: bool = True,
    ) -> dict[str, Any]:
        if self.tokenizer.chat_template is None or not use_chat_template:
            ctx_text      = (getattr(self.tokenizer, "bos_token", None) or "") + context
            question_sfx  = "\n"
        else:
            sep = "#" * (len(context) + 10)
            try:
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context + sep}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context + sep}],
                    add_generation_prompt=True, tokenize=False,
                )
            if sep not in templated:
                raise RuntimeError("Separator missing from chat template output.")
            ctx_text, question_sfx = templated.split(sep, maxsplit=1)

        ctx_ids = self.tokenizer.encode(ctx_text, return_tensors="pt", add_special_tokens=False)
        if ctx_ids.shape[1] > max_context_length:
            logger.warning(
                "Context length has been truncated from %s to %s tokens.",
                ctx_ids.shape[1], max_context_length,
            )
            ctx_ids = ctx_ids[:, :max_context_length]

        q_ids = [
            self.tokenizer.encode(q + question_sfx + answer_prefix,
                                  return_tensors="pt", add_special_tokens=False)
            for q in questions
        ]
        return {"context_ids": ctx_ids, "questions_ids": q_ids}

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _run(
        self,
        tensors: dict[str, Any],
        press: Optional[BasePress],
        max_new_tokens: int,
        cache: Optional[Cache],
    ) -> list[str]:
        input_device = self._model_input_device()
        ctx_ids = tensors["context_ids"].to(input_device)
        ctx_len = ctx_ids.shape[1]
        if cache is None:
            cache = DynamicCache()

        # Press context manager wraps the prefill (and decode too if the backend
        # needs hooks during decode, e.g. TurboQuantPerTokenBackendPress with
        # decode_quant=True).
        backbone = self.model.model if hasattr(self.model, "model") else self.model
        needs_decode_hooks = press is not None and getattr(press, "decode_quant", False)

        q_list = tensors["questions_ids"]
        ctx_mgr = press(self.model) if press is not None else contextlib.nullcontext()
        with ctx_mgr:
            backbone(input_ids=ctx_ids, past_key_values=cache)
            if needs_decode_hooks:
                return self._decode_all(q_list, cache, ctx_len, max_new_tokens, input_device)

        # Hooks removed — decode without press
        return self._decode_all(q_list, cache, ctx_len, max_new_tokens, input_device)

    @staticmethod
    def _cache_seq_len(cache: Cache) -> int:
        """Return the actual KV sequence length from the cache tensor."""
        if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
            kc = cache.key_cache[0]
            if kc.numel() > 0:
                return kc.shape[2]
        return cache.get_seq_length(0)

    def _decode_all(self, q_list, cache, ctx_len, max_new_tokens, input_device):
        answers: list[str] = []
        for q_ids in q_list:
            snap = [cache.get_seq_length(i) for i in range(len(cache))]
            answers.append(self._decode(q_ids.to(input_device), cache, ctx_len, max_new_tokens))
            self._trim_cache(cache, snap)
        return answers

    def _decode(
        self,
        question_ids: torch.Tensor,
        cache: Cache,
        context_length: int,
        max_new_tokens: int,
    ) -> str:
        input_device = self._model_input_device()

        if max_new_tokens <= 0:
            return ""

        q_len = question_ids.shape[1]
        start_pos = context_length
        position_ids = torch.arange(
            start_pos, start_pos + q_len, device=input_device,
        ).unsqueeze(0)
        cache_position = torch.arange(
            start_pos, start_pos + q_len, device=input_device,
        )

        out = self.model(
            input_ids=question_ids, past_key_values=cache,
            position_ids=position_ids, cache_position=cache_position,
            num_logits_to_keep=1,
        )

        eos = self.model.generation_config.eos_token_id or []
        if not isinstance(eos, list):
            eos = [eos]

        next_pos = start_pos + q_len
        tokens = [out.logits[0, -1].argmax()]
        for step in range(max_new_tokens - 1):
            cur_pos = next_pos + step
            out = self.model(
                input_ids=tokens[-1].view(1, 1), past_key_values=cache,
                position_ids=torch.tensor([[cur_pos]], device=input_device),
                cache_position=torch.tensor([cur_pos], device=input_device),
            )
            tok = out.logits[0, -1].argmax()
            tokens.append(tok)
            if tok.item() in eos:
                break

        return self.tokenizer.decode(torch.stack(tokens).cpu(), skip_special_tokens=True)

    def _trim_cache(self, cache: Cache, lengths: list[int]) -> None:
        """Rewind cache to lengths captured before the last question was decoded."""
        for i, seq_len in enumerate(lengths):
            layer = cache.layers[i]
            backend_state = getattr(layer, "_tq_backend_state", None)
            if backend_state is not None and hasattr(backend_state, "truncate"):
                backend_state.truncate(seq_len)
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if isinstance(t, torch.Tensor) and t.shape[2] > seq_len:
                    setattr(layer, attr, t[:, :, :seq_len])
            for attr in ("cumulative_length", "seen_tokens", "_seen_tokens"):
                if hasattr(layer, attr):
                    setattr(layer, attr, seq_len)
        max_len = max(lengths, default=0)
        for attr in ("seen_tokens", "_seen_tokens"):
            if hasattr(cache, attr):
                setattr(cache, attr, max_len)
