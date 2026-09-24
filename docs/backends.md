# Execution backends

| | Reference | Native |
| --- | --- | --- |
| Entry point | `ODMPress` | `kvquant.runtime.generate.graph_generate` |
| Attention | SDPA with exact head-wise eviction masks | FlashAttention-2 prefill, Triton mixed-bit decode |
| Persistent representation | Packed reference state plus dense reconstructed cache | Packed mixed-bit banks plus exact sink/tail storage |
| Purpose | Readable algorithm and quality evaluation | Physical compressed-cache inference |
| Requirements | PyTorch and Transformers | Linux, NVIDIA CUDA/BF16, Triton, FlashAttention 2 |

Both paths use `ODMScorer`, independent per-request/per-layer budgets, and a
fresh allocation at each decode flush. The public factory accepts only ODM-KV
and uncompressed FullKV. Unsupported scorer or allocation settings raise an
error instead of silently selecting another method.

## Native generation

```python
from kvquant.runtime.generate import graph_generate

# model: BF16, eval mode, entirely on one CUDA device, with FlashAttention 2
# input_ids: [batch, prompt_length] on that device
generated_ids = graph_generate(
    model,
    input_ids,
    max_new_tokens=128,
    press_cfg={"target_avg_bits": 2.0},
    eos_token_id=model.generation_config.eos_token_id,
)
```

The returned tensor contains **new tokens only**. By contrast, Hugging Face
`model.generate` in the reference example returns prompt plus generated tokens.

The native graph path supports head dimensions 64/128, MSE quantization for K
and V, and one CUDA device per model. It requires full precision in the bit
set for protected entries and 2 bits as the lowest nonzero level. It rejects
outlier-channel separation, active sliding-window attention, CPU offloading,
and multi-token/speculative decode. An unsupported configuration is an error;
there is no dense-cache fallback hidden in the native decode dispatcher.

Greedy generation supports a fixed batch, including strict left-padded input
with a two-dimensional attention mask. Sampling is limited to unpadded input.
The reference example uses one unpadded request; its scoring does not account
for padding tokens and padded batches are rejected.

## Memory accounting

The target budget applies to the allocation labels before recent-tail
protection. It excludes vector norms, descriptors, and the extra cost of exact
sinks and the recent window. Native CUDA graphs also reserve capacity so that
buffer flushes preserve storage addresses. Reserved but unused space remains
allocated GPU memory and must be counted in physical-memory measurements.

Use an identical model, prompt protocol, dtype, batch size, and output length
when measuring native memory or latency. Warm up compilation and graph capture
before reporting steady-state decode timing. The reference path is not a
proxy for the native backend's memory use or throughput.
