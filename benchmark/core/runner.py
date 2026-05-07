"""Model loading + runner construction shared by all benchmarks.

Extracted from ``benchmark/longbench/evaluator.py`` so that needle / ruler /
math / longbench can all import the same ``DEFAULT_MODEL`` and ``build_runner``
without going through a benchmark-specific module.
"""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark.core.pipeline import KVPressTextGenerationRunner

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def _resolve_model_source(model_name: str) -> str:
    """Prefer a cached local snapshot for Hub model IDs to avoid metadata lookups."""
    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return model_name


def build_runner(model_name: str) -> tuple[KVPressTextGenerationRunner, str]:
    """Load model + tokenizer, return (runner, device_str)."""
    device = "auto" if torch.cuda.is_available() else "cpu"
    model_source = _resolve_model_source(model_name)

    print(f"[build_runner] model_source: {model_source}")
    print(f"[build_runner] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    kwargs: dict = {"trust_remote_code": True, "dtype": "auto"}
    if device == "auto":
        kwargs["device_map"] = "auto"

    # Match autokv/kvpress: prefer flash_attention_2 for comparable results.
    attn_impl = "default (auto-select)"
    try:
        import flash_attn
        kwargs["attn_implementation"] = "flash_attention_2"
        attn_impl = f"flash_attention_2 (flash_attn {flash_attn.__version__})"
    except ImportError:
        attn_impl = "sdpa/eager (flash_attn not installed)"

    print(f"[build_runner] attn_implementation: {attn_impl}")
    print(f"[build_runner] model kwargs: {kwargs}")

    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()

    # Report actual attention implementation used by the model
    attn_module = None
    lm = model.model if hasattr(model, "model") else model
    if hasattr(lm, "layers") and len(lm.layers) > 0:
        attn_module = getattr(lm.layers[0], "self_attn", None)
    if attn_module is not None:
        actual_impl = getattr(model.config, "_attn_implementation", "unknown")
        print(f"[build_runner] model config._attn_implementation: {actual_impl}")
        print(f"[build_runner] attention module type: {type(attn_module).__name__}")
    print(f"[build_runner] model dtype: {next(model.parameters()).dtype}")
    print(f"[build_runner] model class: {type(model).__name__}")

    return KVPressTextGenerationRunner(model=model, tokenizer=tokenizer), device
