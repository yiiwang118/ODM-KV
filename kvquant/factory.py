"""One method, with reference and native execution backends."""
from __future__ import annotations

import math

from kvquant.allocator import DEFAULT_BIT_LEVELS
from kvquant.scorer import ODMScorer


def make_press(config=None, seed=42):
    """Create ODM-KV; retain backend_per_token for existing evaluation YAMLs."""
    cfg = dict(config or {})
    mode = cfg.get("mode", "reference")
    if mode in {"baseline", "none"}:
        return None
    reference_modes = {"reference", "backend_per_token", "adaptive_backend", "per_token_backend"}
    native_modes = {"native", "odmkv"}
    if mode not in reference_modes | native_modes:
        raise ValueError(f"Unknown ODM-KV execution mode: {mode!r}")
    allowed = {
        "mode", "label", "target_avg_bits", "bits", "epsilon", "n_future_positions",
        "sink_tokens", "buffer_size", "eviction_cost", "n_outlier_channels", "outlier_min_bits",
        "scorer", "normalize_grain", "layerwise", "key_quantizer", "value_quantizer",
        "decode_quant", "allow_decode_eviction", "initial_layers_fp16", "above_target_alpha",
        "score_sink_tokens", "value_group_size",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unsupported ODM-KV settings: {sorted(unknown)}")
    fixed = {
        "scorer": "odm", "normalize_grain": "global", "layerwise": True,
        "key_quantizer": "mse", "value_quantizer": "mse", "decode_quant": True,
        "allow_decode_eviction": False, "initial_layers_fp16": 0, "above_target_alpha": 1.0,
    }
    for name, expected in fixed.items():
        if name in cfg and cfg[name] != expected:
            raise ValueError(f"ODM-KV fixes {name}={expected!r}; received {cfg[name]!r}")
    target = float(cfg.get("target_avg_bits", 2.0))
    if not math.isfinite(target) or not 0 <= target <= 16:
        raise ValueError("target_avg_bits must be finite and in [0, 16]")
    levels = tuple(sorted(set(cfg.get("bits", DEFAULT_BIT_LEVELS))))
    if levels != DEFAULT_BIT_LEVELS:
        raise ValueError(f"ODM-KV uses bits={list(DEFAULT_BIT_LEVELS)}")
    sink = int(cfg.get("sink_tokens", 4))
    buffer = int(cfg.get("buffer_size", 128))
    if sink < 0 or buffer < 4:
        raise ValueError("sink_tokens must be nonnegative and buffer_size must be >= 4")
    if int(cfg.get("score_sink_tokens", sink)) != sink:
        raise ValueError("score_sink_tokens must equal sink_tokens")
    eviction_cost = float(cfg.get("eviction_cost", 0.5))
    if eviction_cost != 0.5:
        raise ValueError("ODM-KV uses eviction_cost=0.5")
    n_outliers = int(cfg.get("n_outlier_channels", 0))
    outlier_bits = int(cfg.get("outlier_min_bits", 3))
    if n_outliers < 0 or outlier_bits not in (2, 3, 4, 8):
        raise ValueError("invalid outlier-channel settings")
    scorer = ODMScorer(
        n_future_positions=int(cfg.get("n_future_positions", 512)),
        n_sink=sink, epsilon=float(cfg.get("epsilon", 1e-2)),
    )
    common = dict(
        scorer=scorer, bit_levels=levels, target_avg_bits=target,
        eviction_cost=eviction_cost, sink_tokens=sink, buffer_size=buffer,
        n_outlier_channels=n_outliers, outlier_min_bits=outlier_bits, seed=seed,
        value_group_size=int(cfg.get("value_group_size", 32)),
    )
    if mode in native_modes:
        if n_outliers:
            raise ValueError("the native backend does not support outlier-channel separation")
        from kvquant.runtime.backend_press import ODMNativeBackend
        return ODMNativeBackend(**common, score_sink_tokens=sink)
    from kvquant.press import ODMPress
    return ODMPress(**common)
