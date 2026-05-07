#!/usr/bin/env python3
"""LongBench evaluation for TurboQuant KV quantization.

Usage:
    python eval_longbench.py --config configs/uniform.yaml
    python eval_longbench.py --config configs/mixed_2.5bit.yaml
    python eval_longbench.py --config configs/mixed_3.5bit.yaml

Config file format — see configs/ for examples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from benchmark.core.base_press import BasePress
from benchmark.longbench import (
    DEFAULT_MODEL, DEFAULT_TASKS,
    LongBenchEvalConfig, build_runner, evaluate_longbench,
)
from benchmark.longbench.press_factory import auto_label as _auto_label_impl
from benchmark.longbench.press_factory import make_press as _make_press_impl


# ── Press factory ─────────────────────────────────────────────────────────────

def _make_press(exp: dict[str, Any], seed: int) -> Optional[BasePress]:
    return _make_press_impl(exp, seed)


def _parse_experiments(
    cfg: dict[str, Any],
    seed: int,
) -> list[tuple[str, Optional[BasePress]]]:
    """Build (label, press) pairs from config.

    Supports two formats:

    New format (preferred):
        experiments:
          - label: baseline
            mode: baseline
          - label: b=4
            mode: uniform
            bits: 4
          - label: 2.5bit
            mode: mixed
            b_high: 3
            b_low: 2
            n_outlier: 64

    Legacy format (backward compat):
        bits: [0, 8, 4, 2, 1]   # 0 = baseline
    """
    if "experiments" in cfg:
        result = []
        for exp in cfg["experiments"]:
            label = exp.get("label") or _auto_label(exp)
            result.append((label, _make_press(exp, seed)))
        return result

    # Legacy: bits list
    bits_raw = cfg.get("bits", [0, 8, 4, 2, 1])
    bits = [int(b) for b in bits_raw]
    return [
        ("baseline" if b == 0 else f"b={b}", _make_press({"mode": "baseline" if b == 0 else "uniform", "bits": b}, seed))
        for b in bits
    ]


def _auto_label(exp: dict[str, Any]) -> str:
    return _auto_label_impl(exp)


# ── Table printer ─────────────────────────────────────────────────────────────

def _print_table(results: dict[str, dict[str, float]], tasks: list[str], labels: list[str]) -> None:
    col    = max(len(l) for l in labels) + 2
    header = f"{'task':<22}" + "".join(f"{l:>{col}}" for l in labels)
    sep    = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for task in tasks:
        row = f"{task:<22}" + "".join(f"{results[task].get(l, float('nan')):>{col}.2f}" for l in labels)
        print(row)
    print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

# ── Parallel launch helpers ──────────────────────────────────────────────────

def _detect_gpus() -> list[int]:
    """Return list of available CUDA GPU ids."""
    import os
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [int(x.strip()) for x in visible.split(",") if x.strip()]
    try:
        import torch as _torch
        return list(range(_torch.cuda.device_count()))
    except Exception:
        return []


def _distribute_tasks(tasks: list[str], n_workers: int) -> list[list[str]]:
    """Round-robin distribute tasks into n_workers buckets."""
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        buckets[i % n_workers].append(task)
    return [b for b in buckets if b]  # drop empty


def _launch_worker(
    worker_tasks: list[str],
    config_path: str,
    gpu_id: int,
    cli_args: argparse.Namespace,
    out_path: str,
) -> Any:
    """Launch one worker subprocess that runs multiple tasks on one GPU."""
    import os, subprocess, sys

    cmd = [
        sys.executable, __file__,
        "--config", config_path,
        "--tasks", ",".join(worker_tasks),
        "--output", out_path,
    ]
    # Forward CLI overrides
    if cli_args.model is not None:
        cmd += ["--model", cli_args.model]
    if cli_args.fraction is not None:
        cmd += ["--fraction", str(cli_args.fraction)]
    if cli_args.max_new_tokens is not None:
        cmd += ["--max_new_tokens", str(cli_args.max_new_tokens)]
    if cli_args.max_context_length is not None:
        cmd += ["--max_context_length", str(cli_args.max_context_length)]
    if cli_args.seed is not None:
        cmd += ["--seed", str(cli_args.seed)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Force line-buffered output so the streaming thread sees lines immediately
    env["PYTHONUNBUFFERED"] = "1"

    tasks_str = ",".join(worker_tasks)
    print(f"[parallel] GPU {gpu_id}: launching worker for [{tasks_str}]", flush=True)
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[parallel] GPU {gpu_id}: pid={proc.pid}  tasks=[{tasks_str}]", flush=True)
    return proc


def _stream_worker_output(proc: Any, prefix: str) -> None:
    """Read and print subprocess output line by line in real time."""
    try:
        for line in proc.stdout:
            print(f"  [{prefix}] {line}", end="", flush=True)
    except (ValueError, OSError):
        pass  # pipe closed


def _run_parallel(
    tasks: list[str],
    gpu_ids: list[int],
    config_path: str,
    cli_args: argparse.Namespace,
    output_path: Optional[str],
    labels: list[str],
) -> None:
    """Distribute tasks across GPUs.  Each GPU loads model once, runs its tasks sequentially."""
    import os, tempfile, threading

    tmp_dir = tempfile.mkdtemp(prefix="longbench_parallel_")
    n_workers = len(gpu_ids)
    buckets = _distribute_tasks(tasks, n_workers)

    print(f"[parallel] {len(tasks)} tasks → {len(buckets)} workers on GPUs {gpu_ids}")
    for i, bucket in enumerate(buckets):
        print(f"  GPU {gpu_ids[i]}: {bucket}")

    # Launch workers + real-time output streaming threads
    workers: list[tuple[int, list[str], str, Any, threading.Thread]] = []
    for i, bucket in enumerate(buckets):
        gpu_id = gpu_ids[i]
        out_path = os.path.join(tmp_dir, f"gpu{gpu_id}.json")
        proc = _launch_worker(bucket, config_path, gpu_id, cli_args, out_path)
        thread = threading.Thread(
            target=_stream_worker_output,
            args=(proc, f"GPU {gpu_id}"),
            daemon=True,
        )
        thread.start()
        workers.append((gpu_id, bucket, out_path, proc, thread))

    # Wait for all workers to complete
    merged: dict[str, dict[str, float]] = {}
    failed_tasks: list[str] = []
    for gpu_id, worker_tasks, out_path, proc, thread in workers:
        proc.wait()
        thread.join(timeout=10)
        tasks_str = ",".join(worker_tasks)
        if proc.returncode != 0:
            print(f"[parallel] ✗ GPU {gpu_id} failed (exit={proc.returncode})  tasks=[{tasks_str}]")
            failed_tasks.extend(worker_tasks)
            continue
        try:
            partial = json.loads(Path(out_path).read_text())
            merged.update(partial)
            print(f"[parallel] ✓ GPU {gpu_id} done  tasks=[{tasks_str}]")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[parallel] ✗ GPU {gpu_id} result missing or corrupt: {e}")
            failed_tasks.extend(worker_tasks)

    if failed_tasks:
        print(f"\n[parallel] WARNING: {len(failed_tasks)} task(s) failed: {failed_tasks}")

    # Print combined table
    completed_tasks = [t for t in tasks if t in merged]
    if completed_tasks:
        _print_table(merged, completed_tasks, labels)

    # Save merged results
    if output_path and merged:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"\nSaved → {out}")

    # Cleanup tmp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",             required=True, help="path to YAML config file")
    parser.add_argument("--model",              default=None)
    parser.add_argument("--tasks",              default=None, help="comma-separated task names")
    parser.add_argument("--fraction",           type=float,   default=None)
    parser.add_argument("--max_new_tokens",     type=int,     default=None)
    parser.add_argument("--max_context_length", type=int,     default=None)
    parser.add_argument("--seed",               type=int,     default=None)
    parser.add_argument("--output",             default=None, help="path to save JSON results")
    parser.add_argument("--parallel",           action="store_true",
                        help="run each task as a separate subprocess (1 task per GPU)")
    parser.add_argument("--gpus",               default=None,
                        help="comma-separated GPU ids for parallel mode (default: all visible)")
    args = parser.parse_args()

    import yaml
    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text()) or {}

    def pick(cli_val: Any, key: str, default: Any) -> Any:
        return cli_val if cli_val is not None else cfg.get(key, default)

    def csv_str(raw: Any, default: list[str]) -> list[str]:
        if isinstance(raw, list): return [str(x).strip() for x in raw]
        if isinstance(raw, str):  return [x.strip() for x in raw.split(",") if x.strip()]
        return default

    seed  = int(pick(args.seed, "seed", 42))
    tasks = csv_str(pick(args.tasks, "tasks", DEFAULT_TASKS), DEFAULT_TASKS)
    experiments = _parse_experiments(cfg, seed)
    labels = [label for label, _ in experiments]
    output = pick(args.output, "output", None)

    # ── Parallel mode: one subprocess per task ───────────────────────────
    if args.parallel:
        gpu_ids = (
            [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
            if args.gpus else _detect_gpus()
        )
        if not gpu_ids:
            gpu_ids = [0]
            print("[parallel] no GPUs detected, defaulting to GPU 0 (all tasks sequential on one device)")
        print(f"[parallel] tasks={tasks}  gpus={gpu_ids}  experiments={labels}")
        _run_parallel(tasks, gpu_ids, args.config, args, output, labels)
        return

    # ── Sequential mode (original) ──────────────────────────────────────
    config = LongBenchEvalConfig(
        model              = pick(args.model,              "model",              DEFAULT_MODEL),
        tasks              = tasks,
        experiments        = experiments,
        fraction           = float(pick(args.fraction,     "fraction",           1.0)),
        max_new_tokens     = pick(args.max_new_tokens,     "max_new_tokens",     None),
        max_context_length = pick(args.max_context_length, "max_context_length", None),
        seed               = seed,
        mix_samples_per_task = int(cfg.get("mix_samples_per_task", 5)),
    )
    config.validate()

    print(f"Model: {config.model}")
    runner, device = build_runner(config.model)
    print(f"Loaded on {device}.")
    print(f"Experiments: {labels}\n")

    results = evaluate_longbench(runner=runner, config=config)
    _print_table(results, config.tasks, labels)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
