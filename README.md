# ODM-KV

**Output-Distortion-Minimizing Mixed-precision KV-cache Quantization.**

Per-token bit allocation that minimizes the leading-order expected
attention-output distortion under independent K and V quantization noise.
A per-layer Lagrangian picks each token's bit width from
`{0, 2, 3, 4, 8, 16}` (where 0 = evict, 16 = fp16) under an average-bit
budget. The score is closed-form and adds no extra forward pass.

## Score

For each cached token *j*:

$$
s_j \;=\; p_j^{\,2} \cdot
\Big[\;\|v_j\|^2 \;+\; \tfrac{\|k_j\|^2}{d}\,\|v_j - \bar o\|^2\;\Big],
\qquad \bar o = \sum_{j} p_j v_j .
$$

This is the leading order of $\mathbb{E}\!\left[\|\Delta o\|^2\right]$.
The first term protects high-attention high-norm tokens (e.g. attention
sinks); the second protects content-unique tokens whose values point far
from the mean output.

## Repo layout

```
ODM-KV/
├── kvquant/                       core scorer + Lagrangian + quantization backend
├── benchmark/{core,longbench,ruler}/  generation pipeline + eval harnesses
├── configs/{exp_ruler,exp_longbench}.yaml
├── eval_ruler.py / eval_longbench.py
├── codebook.py                    offline Lloyd-Max codebook precomputation
└── requirements.txt
```

## Install

```bash
pip install -r requirements.txt
```

## Run

**RULER**

```bash
python eval_ruler.py --config configs/exp_ruler.yaml
# Quick check (one task, 5% sample):
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

## Citation

```bibtex
TODO: add bibtex on paper acceptance
```

## License

TBD.
