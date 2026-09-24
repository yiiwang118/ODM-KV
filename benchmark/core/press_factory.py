"""Evaluation adapter for the public ODM-KV factory."""
from kvquant.factory import make_press as _make_press


def make_press(exp, seed=42):
    if exp.get("mode") in {"native", "odmkv"}:
        raise ValueError("these evaluators use the reference backend; use graph_generate for native inference")
    return _make_press(exp, seed=seed)


def auto_label(exp):
    if exp.get("mode") in {"baseline", "none"}:
        return "FullKV"
    return f"ODM-KV-{exp.get('target_avg_bits', 2.0):g}bit"


__all__ = ["make_press", "auto_label"]
