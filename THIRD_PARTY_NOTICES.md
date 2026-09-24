# Third-party notices

The ODM-KV contributions are licensed under MIT. This does not replace the
licenses of third-party code or dependencies.

## NVIDIA kvpress

The evaluation pipeline, benchmark metric conventions, and attention/RoPE
utilities include adaptations of [NVIDIA kvpress](https://github.com/NVIDIA/kvpress),
which is distributed under the Apache License 2.0. A copy is included at
[licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The adaptations add ODM-KV configuration, quantized-cache integration, and
version-specific Transformers support. Relevant files include `benchmark/`,
`kvquant/base_press.py`, and `kvquant/attention_utils.py`.

## Dependencies

PyTorch, Transformers, Triton, FlashAttention, and the benchmark packages retain
their respective licenses. They are installed as dependencies, not relicensed
by this repository.
