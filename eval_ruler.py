#!/usr/bin/env python3
"""RULER evaluation for TurboQuant KV quantization.

Usage:
    python eval_ruler.py --config configs/exp_ruler.yaml
    python eval_ruler.py --config configs/exp_ruler.yaml --context_length 8192
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from benchmark.core.base_press import BasePress
from benchmark.longbench.evaluator import DEFAULT_MODEL, build_runner
from benchmark.longbench.press_factory import auto_label as _auto_label
from benchmark.longbench.press_factory import make_press as _make_press
from benchmark.ruler import RulerEvalConfig, evaluate_ruler


def _parse_experiments(cfg: dict[str, Any], seed: int) -> list[tuple[str, Optional[BasePress]]]:
    result = []
    for exp in cfg.get("experiments", []):
        label = exp.get("label") or _auto_label(exp)
        result.append((label, _make_press(exp, seed)))
    return result


def _print_table(results: dict[str, dict[str, float]], labels: list[str]) -> None:
    col = max(len(l) for l in labels) + 2
    header = f"{'task':<26}" + "".join(f"{l:>{col}}" for l in labels)
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for row_key in sorted(results):
        if row_key == "average":
            continue
        vals = "".join(f"{results[row_key].get(l, float('nan')):>{col}.2f}" for l in labels)
        print(f"{row_key:<26}{vals}")
    # Average row at bottom
    if "average" in results:
        vals = "".join(f"{results['average'].get(l, float('nan')):>{col}.2f}" for l in labels)
        print(f"{'-' * len(header)}")
        print(f"{'average':<26}{vals}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to YAML config file")
    parser.add_argument("--model", default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="path to save JSON results")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated RULER task names (e.g. niah_single_1,qa_1)")
    args = parser.parse_args()

    import yaml
    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text()) or {}

    def pick(cli_val: Any, key: str, default: Any) -> Any:
        return cli_val if cli_val is not None else cfg.get(key, default)

    seed = int(pick(args.seed, "seed", 42))
    experiments = _parse_experiments(cfg, seed)
    labels = [l for l, _ in experiments]
    output = pick(args.output, "output", None)

    tasks = None
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    config = RulerEvalConfig(
        model=pick(args.model, "model", DEFAULT_MODEL),
        experiments=experiments,
        context_length=int(pick(args.context_length, "context_length", 4096)),
        fraction=float(pick(args.fraction, "fraction", 1.0)),
        max_new_tokens=pick(args.max_new_tokens, "max_new_tokens", None),
        seed=seed,
        tasks=tasks,
    )
    config.validate()

    print(f"Model:          {config.model}")
    print(f"Context length: {config.context_length}")
    runner, device = build_runner(config.model)
    print(f"Loaded on {device}.")
    print(f"Experiments: {labels}\n")

    results = evaluate_ruler(runner=runner, config=config)
    _print_table(results, labels)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
