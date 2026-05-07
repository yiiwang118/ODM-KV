"""Press dispatcher driven by experiment YAML config.

``make_press(exp, seed)`` translates a config dict into a ready-to-call
:class:`BasePress` instance.

Supported modes:
    baseline / none                       → no compression (fp16 forward)
    backend_per_token / adaptive_backend  → TurboQuantPerTokenBackendPress (our method)

The following are hard-coded for reproducibility / paper-final settings and
NOT exposed in the YAML:

    SINK_TOKENS = 4              attention sinks always fp16
    EPSILON = 1.0e-2             numerical floor inside the score
    NORMALIZE_GRAIN = "global"   raw scores fed to Lagrangian (no min-max)
    LAYERWISE = True             Lagrangian solved per layer
    KEY_QUANTIZER = "mse"        Lloyd-Max codebook for K
    VALUE_QUANTIZER = "mse"      Lloyd-Max codebook for V
    BUFFER_SIZE = 128            decode tokens before flush + re-quantize
    DECODE_QUANT = True          quantize decode tokens too
    ALLOW_DECODE_EVICTION = False    never evict newly-generated tokens
"""
from __future__ import annotations

from typing import Any, Optional

from benchmark.core.base_press import BasePress
from kvquant import (
    RiskScorer,
    TurboQuantPerTokenBackendPress,
)

# ── Hard-coded paper-final settings ──────────────────────────────────────────
SINK_TOKENS           = 4
EPSILON               = 1.0e-2
NORMALIZE_GRAIN       = "global"
LAYERWISE             = True
KEY_QUANTIZER         = "mse"
VALUE_QUANTIZER       = "mse"
BUFFER_SIZE           = 128
DECODE_QUANT          = True
ALLOW_DECODE_EVICTION = False


def make_press(exp: dict[str, Any], seed: int) -> Optional[BasePress]:
    mode = exp.get("mode", "backend_per_token")

    if mode in {"baseline", "none"}:
        return None

    if mode in {"backend_per_token", "per_token_backend", "adaptive_backend"}:
        bits = tuple(sorted(int(b) for b in exp.get("bits", [0, 2, 3, 4, 8, 16])))
        scorer = RiskScorer(
            n_future_positions=int(exp.get("n_future_positions", 512)),
            n_sink=SINK_TOKENS,
            use_vnorm=bool(exp.get("use_vnorm", True)),
            epsilon=EPSILON,
        )
        scorer.normalize_grain = NORMALIZE_GRAIN

        ratios_raw = exp.get("ratios")
        ratios = tuple(float(r) for r in ratios_raw) if ratios_raw is not None else None
        target_avg_bits_raw = exp.get("target_avg_bits")
        target_avg_bits = float(target_avg_bits_raw) if target_avg_bits_raw is not None else None

        return TurboQuantPerTokenBackendPress(
            scorer=scorer,
            bit_levels=bits,
            ratios=ratios,
            target_avg_bits=target_avg_bits,
            eviction_cost=float(exp.get("eviction_cost", 0.5)),
            above_target_alpha=float(exp.get("above_target_alpha", 1.0)),
            n_outlier_channels=int(exp.get("n_outlier_channels", 0)),
            outlier_min_bits=int(exp.get("outlier_min_bits", 2)),
            seed=seed,
            decode_quant=DECODE_QUANT,
            sink_tokens=SINK_TOKENS,
            layerwise=LAYERWISE,
            key_quantizer=KEY_QUANTIZER,
            value_quantizer=VALUE_QUANTIZER,
            value_group_size=int(exp.get("value_group_size", 32)),
            buffer_size=BUFFER_SIZE,
            allow_decode_eviction=ALLOW_DECODE_EVICTION,
            initial_layers_fp16=int(exp.get("initial_layers_fp16", 0)),
        )

    raise ValueError(f"Unknown mode: {mode!r}")


def auto_label(exp: dict[str, Any]) -> str:
    mode = exp.get("mode", "backend_per_token")
    if mode in {"baseline", "none"}:
        return "baseline"
    target = exp.get("target_avg_bits")
    if target is not None:
        return f"adaptive_t{target}"
    return mode
