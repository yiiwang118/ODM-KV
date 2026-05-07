"""RULER benchmark metrics.

Adapted from kvpress/evaluation/benchmarks/ruler/calculate_metrics.py
"""
from __future__ import annotations

import re

import pandas as pd


def string_match_part(preds: list[str], refs: list[list[str]]) -> float:
    """For QA tasks: at least one reference substring appears in prediction."""
    score = (
        sum(max(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) for pred, ref in zip(preds, refs))
        / len(preds)
        * 100
    )
    return round(score, 2)


def string_match_all(preds: list[str], refs: list[list[str]]) -> float:
    """For non-QA tasks: fraction of reference substrings found in prediction."""
    score = (
        sum(
            sum(1.0 if r.lower() in pred.lower() else 0.0 for r in ref) / len(ref)
            for pred, ref in zip(preds, refs)
        )
        / len(preds)
        * 100
    )
    return round(score, 2)


_NP_PATTERN = re.compile(r"[\x00-\x1f]")


def calculate_metrics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Calculate per-task RULER metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: predicted_answer, answer (list[str]), task.

    Returns
    -------
    dict mapping task → {"string_match": score}.
    """
    scores: dict[str, dict[str, float]] = {}

    df = df.copy()
    df["predicted_answer"] = df["predicted_answer"].apply(
        lambda x: _NP_PATTERN.sub("", str(x).strip()).strip()
    )

    for task, df_task in df.groupby("task"):
        task_category = str(task).split("_")[0]
        metric_fn = string_match_part if task_category == "qa" else string_match_all
        preds = df_task["predicted_answer"].tolist()
        refs = df_task["answer"].tolist()
        score = metric_fn(preds, refs)
        scores[str(task)] = {"string_match": score}

    return scores
