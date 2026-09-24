"""Process-local safety guards for request-bound model integrations."""
from __future__ import annotations

import contextlib
import threading
import weakref

from torch import nn


# Attention backend selection and forward hooks are model-global mutations.
# ContextVars isolate request data read *inside* an attention call, but cannot
# make two independently-installed hook sets safe on the same module tree.
_ACTIVE: "weakref.WeakKeyDictionary[nn.Module, tuple[object, str]]" = (
    weakref.WeakKeyDictionary()
)
_LOCK = threading.Lock()


@contextlib.contextmanager
def exclusive_model_context(model: nn.Module, label: str):
    """Admit one backend context per model or fail before model mutation."""
    token = object()
    with _LOCK:
        active = _ACTIVE.get(model)
        if active is not None:
            raise RuntimeError(
                f"{label} does not support overlapping contexts on the same "
                f"model (active={active[1]}); combine requests into one batch "
                "or serialize the contexts. Model-global attention hooks/config "
                "cannot safely bind two request caches."
            )
        _ACTIVE[model] = (token, label)
    try:
        yield
    finally:
        with _LOCK:
            active = _ACTIVE.get(model)
            if active is not None and active[0] is token:
                del _ACTIVE[model]


__all__ = ["exclusive_model_context"]
