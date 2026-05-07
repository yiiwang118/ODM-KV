"""Back-compat shim: BasePress now lives in :mod:`kvquant.base_press`.

Moved to ``kvquant/`` so that ``kvquant.backend_press`` does not import from
``benchmark.*`` (the algorithm layer should not depend on the evaluation harness).

This shim preserves ``from benchmark.core.base_press import BasePress``.
New code should import from :mod:`kvquant.base_press` directly.
"""
from kvquant.base_press import (
    BasePress,
    extract_keys_and_values,
    set_keys_and_values,
    _is_quantized,
    logger,
)

__all__ = [
    "BasePress",
    "extract_keys_and_values",
    "set_keys_and_values",
    "_is_quantized",
    "logger",
]
