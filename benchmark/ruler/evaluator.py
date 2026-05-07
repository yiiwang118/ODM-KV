"""RULER benchmark evaluator.

Loads the RULER dataset from HuggingFace Hub (simonjegou/ruler),
runs inference with optional KV cache compression, and reports
per-task string-match scores.

RULER has 13 tasks across 4 categories:
  - NIAH (8): niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue, niah_multiquery
  - Variable Tracing (1): vt
  - Common Word Extraction (1): cwe
  - Free-form Word Extraction (1): fwe
  - Question Answering (2): qa_1, qa_2
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from benchmark.core.base_press import BasePress
from benchmark.core.pipeline import KVPressTextGenerationRunner
from benchmark.core.runner import DEFAULT_MODEL, build_runner
from benchmark.ruler.metrics import calculate_metrics

RULER_DATASET = "simonjegou/ruler"


@dataclass
class RulerEvalConfig:
    model: str = DEFAULT_MODEL
    experiments: list[tuple[str, Optional[BasePress]]] = field(default_factory=list)
    context_length: int = 4096
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    seed: int = 42
    tasks: Optional[list[str]] = None

    def validate(self) -> None:
        if not self.experiments:
            raise ValueError("`experiments` cannot be empty.")
        if not 0 < self.fraction <= 1.0:
            raise ValueError(f"`fraction` must be in (0, 1], got {self.fraction}.")


def _load_ruler(context_length: int, fraction: float, seed: int,
                tasks: Optional[list[str]] = None):
    """Load RULER dataset for a given context length."""
    try:
        df = load_dataset(
            RULER_DATASET,
            data_dir=str(context_length),
            split="test",
            trust_remote_code=True,
        ).to_pandas()
    except (ValueError, FileNotFoundError):
        print(f"[RULER] data_dir={context_length} not in cache, trying default config...")
        try:
            df = load_dataset(
                RULER_DATASET,
                split="test",
                trust_remote_code=True,
            ).to_pandas()
        except (ValueError, FileNotFoundError):
            # Last-resort: direct arrow read from the HF datasets cache.
            # Handles the offline case where only a hashed config dir exists
            # (e.g. ``default-<hash>/``) and load_dataset can't map "default"
            # to it.
            from pathlib import Path as _P
            import pyarrow.ipc as _ipc
            cache_root = _P.home() / ".cache/huggingface/datasets/simonjegou___ruler"
            hits = list(cache_root.glob("default-*/*/*/ruler-test.arrow"))
            if not hits:
                raise
            arrow_path = hits[0]
            print(f"[RULER] reading arrow directly: {arrow_path}")
            with open(arrow_path, "rb") as f:
                df = _ipc.open_stream(f).read_all().to_pandas()

    if tasks is not None:
        df = df[df["task"].isin(tasks)].copy()
        if df.empty:
            raise ValueError(f"No RULER rows matched tasks={tasks}")

    if fraction < 1.0:
        # Sample within each task so every selected task keeps a representative slice.
        df = (
            df.groupby("task", group_keys=False)
            .apply(lambda g: g.sample(frac=fraction, random_state=seed))
            .reset_index(drop=True)
        )

    return df


def run_ruler(
    runner: KVPressTextGenerationRunner,
    label: str,
    press: Optional[BasePress],
    context_length: int,
    fraction: float,
    max_new_tokens: Optional[int],
    seed: int,
    tasks: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Run RULER evaluation for one experiment, return per-task metrics."""
    df = _load_ruler(context_length, fraction, seed, tasks=tasks)
    df["predicted_answer"] = None

    # RULER groups multiple questions per context (like LongBench)
    grouped = df.groupby("context")

    with tqdm(total=len(df), desc=f"RULER-{context_length} [{label}]", unit="q", leave=False) as pbar:
        for context, group in grouped:
            questions = group["question"].tolist()
            answer_prefix = group["answer_prefix"].iloc[0]
            task_max_tokens = max_new_tokens or int(group["max_new_tokens"].iloc[0])

            out = runner(
                context=context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=task_max_tokens,
                max_context_length=context_length,
            )
            df.loc[group.index, "predicted_answer"] = out["answers"]
            pbar.update(len(group))
            torch.cuda.empty_cache()

    return calculate_metrics(df)


def evaluate_ruler(
    runner: KVPressTextGenerationRunner,
    config: RulerEvalConfig,
) -> dict[str, dict[str, float]]:
    """Run all experiments, return {row_label: {experiment_label: score}}.

    Row labels are per-task scores plus an "average" row.
    """
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Collect all task names from first experiment
    first_label, first_press = config.experiments[0]
    first_metrics = run_ruler(
        runner, first_label, first_press,
        config.context_length, config.fraction,
        config.max_new_tokens, config.seed,
        tasks=config.tasks,
    )
    task_names = sorted(first_metrics.keys())

    # Build results table: {task: {label: score}, ..., "average": {label: avg}}
    results: dict[str, dict[str, float]] = {t: {} for t in task_names}
    results["average"] = {}

    # Fill first experiment
    for task in task_names:
        results[task][first_label] = first_metrics[task]["string_match"]
    results["average"][first_label] = round(
        float(np.mean([first_metrics[t]["string_match"] for t in task_names])), 2
    )

    # Remaining experiments
    for label, press in config.experiments[1:]:
        metrics = run_ruler(
            runner, label, press,
            config.context_length, config.fraction,
            config.max_new_tokens, config.seed,
            tasks=config.tasks,
        )
        for task in task_names:
            score = metrics.get(task, {}).get("string_match", 0.0)
            results[task][label] = score
        results["average"][label] = round(
            float(np.mean([results[t].get(label, 0.0) for t in task_names])), 2
        )

    return results
