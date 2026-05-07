"""Shared infrastructure for benchmarks.

- ``benchmark.core.base_press``     BasePress + cache helpers (forward-hook machinery)
- ``benchmark.core.pipeline``       KVPressTextGenerationRunner (prefill + multi-question decode)
- ``benchmark.core.runner``         Model loading helpers (DEFAULT_MODEL / build_runner)
- ``benchmark.core.press_factory``  make_press dispatchers driven by YAML config
"""
