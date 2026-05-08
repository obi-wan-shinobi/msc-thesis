#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

EXPERIMENT_MODULE = "experiments.ntk_frozen_vs_evolving_mode_dynamics"

SMOKE_CONFIGS = [
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_w128.yaml",
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_78_w128.yaml",
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_1112_w128.yaml",
]

FULL_CONFIGS = [
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_equalamp_w128.yaml",
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_equalamp_w256.yaml",
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_equalamp_w512.yaml",
    "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_equalamp_w1024.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_78_w128.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_78_w256.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_78_w512.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_78_w1024.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_1112_w128.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_1112_w256.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_1112_w512.yaml",
    # "configs/ntk_frozen_vs_evolving_mode_dynamics_k0256_1112_w1024.yaml",
]


@dataclass(frozen=True)
class RunResult:
    config: str
    returncode: int
    log_path: Path


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_one(config: str, log_dir: Path) -> RunResult:
    tag = Path(config).stem
    log_path = log_dir / f"{tag}.log"
    print(f"[{now()}] START {config}", flush=True)

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, "-m", EXPERIMENT_MODULE, config],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if proc.returncode == 0:
        print(f"[{now()}] DONE  {config} (log: {log_path})", flush=True)
    else:
        print(f"[{now()}] FAIL  {config} (log: {log_path})", flush=True)

    return RunResult(config=config, returncode=proc.returncode, log_path=log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-vs-evolving NTK experiments in parallel.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="smoke",
        choices=["smoke", "full"],
        help="Which config set to run.",
    )
    parser.add_argument(
        "jobs",
        nargs="?",
        default=2,
        type=int,
        help="Max number of parallel jobs.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for per-config logs.",
    )
    parser.add_argument(
        "--no-xla-prealloc",
        action="store_true",
        help="Do not set XLA_PYTHON_CLIENT_PREALLOCATE=false.",
    )
    return parser.parse_args()


def select_configs(mode: str) -> list[str]:
    if mode == "smoke":
        return list(SMOKE_CONFIGS)
    return list(FULL_CONFIGS)


def main() -> int:
    args = parse_args()
    if args.jobs <= 0:
        print(f"jobs must be positive, got {args.jobs}")
        return 2

    if not args.no_xla_prealloc:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    configs = select_configs(args.mode)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Parallel jobs: {args.jobs}")
    print(f"Total configs: {len(configs)}")

    results: list[RunResult] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = [ex.submit(run_one, cfg, log_dir) for cfg in configs]
        for fut in as_completed(futures):
            results.append(fut.result())

    failures = [r for r in results if r.returncode != 0]
    if failures:
        print("\nCompleted with failures:")
        for res in failures:
            print(f"- {res.config} (log: {res.log_path})")
        return 1

    print("\nAll requested runs finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
