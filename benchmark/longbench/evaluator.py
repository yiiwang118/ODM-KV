# LongBench evaluation workflow for TurboQuant
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from benchmark.core.base_press import BasePress
from benchmark.core.pipeline import KVPressTextGenerationRunner
from benchmark.core.runner import DEFAULT_MODEL, _resolve_model_source, build_runner
from benchmark.longbench.metrics import score_predictions
from benchmark.longbench.press_factory import auto_label as _auto_label
from benchmark.longbench.press_factory import make_press as _make_press

LONGBENCH_DATASET = "Xnhyacinth/LongBench"
DEFAULT_TASKS     = ["qasper", "triviaqa", "hotpotqa"]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LongBenchEvalConfig:
    model:              str        = DEFAULT_MODEL
    tasks:              list[str]  = field(default_factory=lambda: list(DEFAULT_TASKS))
    # experiments: list of (label, press) pairs.  press=None means baseline.
    experiments:        list[tuple[str, Optional[BasePress]]] = field(default_factory=list)
    fraction:              float      = 1.0
    max_new_tokens:        Optional[int] = None
    max_context_length:    Optional[int] = None
    seed:                  int        = 42
    mix_samples_per_task:  int        = 5

    def validate(self) -> None:
        if not self.tasks:
            raise ValueError("`tasks` cannot be empty.")
        if not self.experiments:
            raise ValueError("`experiments` cannot be empty.")
        if not 0 < self.fraction <= 1.0:
            raise ValueError(f"`fraction` must be in (0, 1], got {self.fraction}.")


# ── LongBench dataset loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _longbench_cache_index() -> dict[str, Path]:
    """Map LongBench task names to cached Arrow files, if present."""
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "Xnhyacinth___long_bench"
    task_to_arrow: dict[str, Path] = {}

    if not cache_root.exists():
        return task_to_arrow

    for dataset_info in sorted(cache_root.glob("default-*/0.0.0/*/dataset_info.json")):
        try:
            info = json.loads(dataset_info.read_text())
            checksum_key = next(iter(info.get("download_checksums", {})))
            task = checksum_key.split("/")[-2]
        except (StopIteration, IndexError, json.JSONDecodeError, OSError):
            continue

        arrow_path = dataset_info.with_name("long_bench-test.arrow")
        if arrow_path.exists():
            task_to_arrow[task] = arrow_path

    return task_to_arrow


def _load_longbench_task(task: str) -> Dataset:
    """Load a LongBench split from the local cache when available."""
    cached_arrow = _longbench_cache_index().get(task)
    if cached_arrow is not None:
        return Dataset.from_file(str(cached_arrow))

    return load_dataset(
        LONGBENCH_DATASET,
        data_dir=task,
        split="test",
        trust_remote_code=True,
    )


# ── Per-task evaluation ───────────────────────────────────────────────────────

def run_task(
    runner: KVPressTextGenerationRunner,
    task: str,
    label: str,
    press: Optional[BasePress],
    fraction: float,
    max_new_tokens: Optional[int],
    max_context_length: Optional[int],
    seed: int,
) -> float:
    df = _load_longbench_task(task).to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=seed)

    n_contexts = df["context"].nunique()
    n_questions = len(df)
    print(f"\n[run_task] task={task}  label={label}  samples={n_questions}  contexts={n_contexts}")
    print(f"[run_task] press={type(press).__name__ if press else 'None (baseline)'}")
    if max_new_tokens is not None:
        print(f"[run_task] max_new_tokens={max_new_tokens}")
    if max_context_length is not None:
        print(f"[run_task] max_context_length={max_context_length}")

    df["predicted_answer"] = None

    grouped = df.groupby("context")
    assert all(grouped["answer_prefix"].nunique() == 1), \
        "Inconsistent answer_prefix within a context group."

    # Few-shot classification tasks need raw prompt (no chat template)
    # to preserve the Question→Type pattern without chat boundary disruption.
    _CLASSIFICATION_TASKS = {"trec", "lsht"}
    use_chat = task not in _CLASSIFICATION_TASKS

    with tqdm(total=len(df), desc=f"{task} [{label}]", unit="q") as pbar:
        for context, group in grouped:
            out = runner(
                context=context,
                questions=group["question"].tolist(),
                answer_prefix=group["answer_prefix"].iloc[0],
                press=press,
                max_new_tokens=max_new_tokens or int(group["max_new_tokens"].iloc[0]),
                max_context_length=max_context_length,
                use_chat_template=use_chat,
            )
            df.loc[group.index, "predicted_answer"] = out["answers"]
            pbar.update(len(group))

    all_classes  = df["all_classes"].iloc[0] if "all_classes" in df.columns else []
    predictions  = df["predicted_answer"].tolist()
    if any(pred is None for pred in predictions):
        raise RuntimeError("Missing predictions after grouped inference.")
    return score_predictions(task, predictions, df["answers"].tolist(), all_classes)


# ── Mixed mini-benchmark ─────────────────────────────────────────────────────

# English-only subtasks for the mix (skip zh tasks that need Chinese tokeniser)
_MIX_SUBTASKS: list[str] = [
    "narrativeqa",       # single-doc QA
    "qasper",            # single-doc QA
    "multifieldqa_en",   # single-doc QA
    "hotpotqa",          # multi-doc QA
    "2wikimqa",          # multi-doc QA
    "musique",           # multi-doc QA
    "triviaqa",          # multi-doc QA
    "gov_report",        # summarisation
    "qmsum",             # summarisation
    "multi_news",        # summarisation
    "samsum",            # summarisation
    "trec",              # classification
    "passage_retrieval_en",  # retrieval
    "passage_count",     # counting
    "lcc",               # code
    "repobench-p",       # code
]


def run_mixed_task(
    runner: KVPressTextGenerationRunner,
    label: str,
    press: Optional[BasePress],
    samples_per_task: int,
    max_new_tokens: Optional[int],
    max_context_length: Optional[int],
    seed: int,
) -> float:
    """Run a mini-benchmark that samples from every English LongBench subtask.

    Each subtask is scored with its own metric, then scores are macro-averaged
    across subtasks for a single composite number.
    """
    subtask_scores: list[float] = []
    skipped: list[str] = []

    for subtask in _MIX_SUBTASKS:
        try:
            df = _load_longbench_task(subtask).to_pandas()
        except Exception:
            skipped.append(subtask)
            continue

        # Sample a fixed number of rows (by context group to keep questions coherent)
        contexts = df["context"].unique()
        rng = np.random.RandomState(seed)
        n_ctx = min(samples_per_task, len(contexts))
        chosen_ctx = rng.choice(contexts, size=n_ctx, replace=False)
        df = df[df["context"].isin(chosen_ctx)]

        if len(df) == 0:
            skipped.append(subtask)
            continue

        df["predicted_answer"] = None
        grouped = df.groupby("context")

        with tqdm(total=len(df), desc=f"mix/{subtask} [{label}]", unit="q", leave=False) as pbar:
            for context, group in grouped:
                out = runner(
                    context=context,
                    questions=group["question"].tolist(),
                    answer_prefix=group["answer_prefix"].iloc[0],
                    press=press,
                    max_new_tokens=max_new_tokens or int(group["max_new_tokens"].iloc[0]),
                    max_context_length=max_context_length,
                )
                df.loc[group.index, "predicted_answer"] = out["answers"]
                pbar.update(len(group))

        all_classes = df["all_classes"].iloc[0] if "all_classes" in df.columns else []
        predictions = df["predicted_answer"].tolist()
        if any(pred is None for pred in predictions):
            skipped.append(subtask)
            continue

        score = score_predictions(subtask, predictions, df["answers"].tolist(), all_classes)
        subtask_scores.append(score)
        print(f"  mix/{subtask}: {score:.2f}  ({len(df)} samples)")

    if skipped:
        print(f"  [mix] skipped {len(skipped)} subtasks: {skipped}")

    return float(np.mean(subtask_scores)) if subtask_scores else 0.0


# ── Top-level evaluation loop ─────────────────────────────────────────────────

def evaluate_longbench(
    runner: KVPressTextGenerationRunner,
    config: LongBenchEvalConfig,
) -> dict[str, dict[str, float]]:
    """Run all (task, experiment) combinations; return {task: {label: score}}."""
    config.validate()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    results: dict[str, dict[str, float]] = {}
    for task in config.tasks:
        results[task] = {}
        for label, press in config.experiments:
            if task == "longbench_mix":
                results[task][label] = run_mixed_task(
                    runner=runner, label=label, press=press,
                    samples_per_task=config.mix_samples_per_task,
                    max_new_tokens=config.max_new_tokens,
                    max_context_length=config.max_context_length,
                    seed=config.seed,
                )
            else:
                results[task][label] = run_task(
                    runner=runner, task=task, label=label, press=press,
                    fraction=config.fraction,
                    max_new_tokens=config.max_new_tokens,
                    max_context_length=config.max_context_length,
                    seed=config.seed,
                )
    return results
