"""ODM-KV public API. Native CUDA components are imported on demand."""
from kvquant.scorer import ODMScorer

__all__ = ["ODMScorer", "ODMPress", "make_press"]


def __getattr__(name):
    if name == "ODMPress":
        from kvquant.press import ODMPress
        return ODMPress
    if name == "make_press":
        from kvquant.factory import make_press
        return make_press
    raise AttributeError(name)
