#!/usr/bin/env python3
"""General L3P training launcher for Kaggle notebooks.

The launcher keeps generated artifacts in a single run directory and adds
optional W&B integration without changing the normal local training scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3p.config import get_config, list_envs, resolve_env


def default_output_dir() -> Path:
    kaggle_root = Path("/kaggle/working")
    return kaggle_root / "l3p_runs" if kaggle_root.is_dir() else Path("l3p_runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train L3P or PN-LMCGS with Kaggle-friendly artifacts.",
    )
    parser.add_argument("--env", default="FetchPickAndPlace", choices=list_envs())
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--pn-lmcgs", action="store_true")
    parser.add_argument("--hazardous-pointmaze", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-project", default="latent-landmarks")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="offline",
    )
    return parser


def _validate_positive(value: int | None, name: str, *, allow_zero: bool = False) -> None:
    if value is None:
        return
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def build_config(args: argparse.Namespace):
    if args.hazardous_pointmaze and resolve_env(args.env) != "PointMaze":
        raise ValueError("--hazardous-pointmaze is only valid for the PointMaze environment")
    _validate_positive(args.steps, "--steps")
    _validate_positive(args.workers, "--workers")
    _validate_positive(args.eval_episodes, "--eval-episodes")
    _validate_positive(args.save_every, "--save-every", allow_zero=True)

    overrides: dict[str, Any] = {"seed": args.seed, "total_steps": args.steps}
    if args.workers is not None:
        overrides["n_workers"] = args.workers
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.pn_lmcgs or args.hazardous_pointmaze:
        overrides["pn_lmcgs_enabled"] = True
    if args.hazardous_pointmaze:
        overrides["pn_pointmaze_hazard_enabled"] = True
    return get_config(args.env, **overrides)


def default_run_name(env_name: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{env_name.lower()}-seed{seed}-{timestamp}"


def prepare_run_directory(
        output_dir: Path, run_name: str, config: Mapping[str, Any],
        *, overwrite: bool = False) -> Path:
    run_dir = Path(output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"run directory {run_dir} exists; pass --overwrite to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return run_dir


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_config(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
