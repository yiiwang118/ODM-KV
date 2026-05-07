# ODM-KV

**Output-Distortion-Minimizing Mixed-precision KV-cache Quantization.**

ODM-KV assigns a per-token bit budget that minimizes the first-order
expected attention-output distortion under independent K and V quantization
noise, solved with a per-layer Lagrangian over a discrete bit set
`{0, 2, 3, 4, 8, 16}` (where 0 = evict, 16 = fp16).

The score is closed-form, score-only (no extra forward), and supports both
prefill quantization and on-the-fly decode quantization with periodic buffer
re-allocation.

---

## Score

For each cached token *j*:

$$
s_j \;=\; p_j^{\,2} \cdot
\Big[\;\|v_j\|^2 \;+\; \tfrac{\|k_j\|^2}{d}\,\|v_j - \bar o\|^2\;\Big],
\qquad \bar o = \sum_{j} p_j v_j .
$$

This is the leading order of $\mathbb{E}\!\left[\|\Delta o\|^2\right]$
under independent value- and key-quantization noise:

- $p_j^{\,2}\,\|v_j\|^2$ — *V-direct error*. Scales with attention weight
  squared and value norm; protects high-attention high-norm tokens such as
  the BOS / sink tokens.
- $p_j^{\,2}\,\tfrac{\|k_j\|^2}{d}\,\|v_j-\bar o\|^2$ — *K-induced
  attention drift*. When key noise perturbs the softmax, the resulting
  shift in the output is proportional to $v_j-\bar o$; tokens whose values
  point far from the mean output therefore matter most.

The cross-term $\mathbb{E}[\Delta o_V\!\cdot\!\Delta o_K] = 0$ because
V- and K-noise are independent.

A per-layer Lagrangian then minimises
$\sum_j \varepsilon(b_j)\,s_j + \lambda \sum_j b_j$
subject to the average-bit constraint, picking the optimal $b_j \in
\{0, 2, 3, 4, 8, 16\}$ for each token.

---

## What's in this repo

```
ODM-KV/
├── kvquant/                       core scorer + Lagrangian + quantization backend
│   ├── scorer.py                  RiskScorer (the score above) + ExpectedAttentionScorer base
│   ├── allocator.py               scores → bits  (ratio / Lagrangian / fixed-ε)
│   ├── adaptive_backend_press.py  per-token quant press (production path)
│   ├── tq_adaptive_backend.py     adaptive TurboQuant cache state (per-layer)
│   ├── tq_backend.py              TurboQuant Lloyd-Max codebook + dequant
│   ├── attention_patch.py         masks 0-bit tokens during attention
│   └── attention_utils.py         RoPE / GQA / chunked attention helpers
│
├── benchmark/
│   ├── core/                      BasePress, generation pipeline, model loader
│   └── longbench/, ruler/         eval harnesses + metrics
│
├── configs/
│   ├── exp_ruler.yaml             RULER (13 long-context tasks)
│   └── exp_longbench.yaml         LongBench (15 English subtasks)
│
├── codebook.py                    offline Lloyd-Max codebook precomputation
├── eval_ruler.py                  RULER entry point
├── eval_longbench.py              LongBench entry point
└── requirements.txt
```

---

## Install

```bash
pip install -r requirements.txt
```

Tested with PyTorch ≥ 2.5 and `transformers ≥ 4.45`. A CUDA GPU is
required (we test on a single 4090; multi-GPU sharding via
`device_map="auto"` is supported automatically).

---

## Quick start

**RULER**

```bash
python eval_ruler.py --config configs/exp_ruler.yaml
```

Quick sanity check (one task, 5% sample):

```bash
python eval_ruler.py --config configs/exp_ruler.yaml \
    --tasks niah_multikey_3 --fraction 0.05
```

**LongBench**

```bash
python eval_longbench.py --config configs/exp_longbench.yaml
python eval_longbench.py --config configs/exp_longbench.yaml \
    --tasks qasper,hotpotqa --fraction 0.1
```

Both scripts run an fp16 `baseline` and the ODM-KV `target=2.0` config
side-by-side and print a comparison table. Outputs go to `results/`.

---

## Configuration

### YAML-tunable (per experiment)

| field | default | meaning |
|---|---|---|
| `bits` | `[0, 2, 3, 4, 8, 16]` | bit levels Lagrangian can pick from per token |
| `target_avg_bits` | `2.0` | average bit budget |
| `eviction_cost` | `0.5` | multiplier on $\varepsilon(0)$; lower → more eviction |
| `n_outlier_channels` | `0` | OCS channels (Llama: 0; Qwen: 40 helps) |
| `outlier_min_bits` | `2` | floor for OCS channels |

### Hard-coded (paper-final values)

Edit the constants at the top of `benchmark/core/press_factory.py` to
override:

| constant | value | meaning |
|---|---|---|
| `SINK_TOKENS` | `4` | first 4 tokens kept fp16 (attention-sink protection) |
| `EPSILON` | `1.0e-2` | numerical floor inside the score |
| `NORMALIZE_GRAIN` | `"global"` | raw scores fed to Lagrangian (no min-max) |
| `LAYERWISE` | `True` | Lagrangian solved per layer |
| `KEY_QUANTIZER` | `"mse"` | Lloyd-Max codebook for K |
| `VALUE_QUANTIZER` | `"mse"` | Lloyd-Max codebook for V |
| `BUFFER_SIZE` | `128` | decode tokens before flush + re-quantize |
| `DECODE_QUANT` | `True` | quantize decode tokens too |
| `ALLOW_DECODE_EVICTION` | `False` | never evict newly generated tokens |

---

## Reproducing the paper

The shipped configs match the paper-final settings on
**Llama-3.1-8B-Instruct** at `target_avg_bits = 2.0`:

```bash
# RULER (13 tasks, ctx=4096, fraction=1.0) — single GPU, ~4h on a 4090
python eval_ruler.py --config configs/exp_ruler.yaml

# LongBench (15 tasks, fraction=1.0)              — single GPU, ~6h on a 4090
python eval_longbench.py --config configs/exp_longbench.yaml
```

To sweep budgets, change `target_avg_bits` to one of `{1.0, 1.5, 2.0, 3.0}`
in the yaml. To switch model, change `model:` to e.g. `Qwen/Qwen3-8B`
(remember to also set `n_outlier_channels: 40`, `outlier_min_bits: 3` for
Qwen, which has a sharper outlier-channel structure).

---

## Citation

```bibtex
TODO: add bibtex on paper acceptance
```

## License

TBD.
