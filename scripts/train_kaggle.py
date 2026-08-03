#!/usr/bin/env python3
"""General L3P training launcher for Kaggle notebooks.

The launcher keeps generated artifacts in a single run directory and adds
optional W&B integration without changing the normal local training scripts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3p.config import get_config, list_envs, resolve_env
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


class WandbSink:
    """Translate scalar trainer events into a single W&B run."""

    def __init__(self, wandb, run, run_name: str):
        self.wandb = wandb
        self.run = run
        self.run_name = run_name

    def __call__(self, event: str, step: int, metrics: Mapping[str, float]) -> None:
        values = {f"{event}/{name}": float(value) for name, value in metrics.items()}
        self.wandb.log(values, step=int(step))

    def finish(self) -> None:
        self.wandb.finish()

    def log_model(self, model_path: Path) -> None:
        artifact = self.wandb.Artifact(f"{self.run_name}-model", type="model")
        artifact.add_file(str(model_path))
        self.run.log_artifact(artifact)


def create_wandb_sink(mode: str, project: str, entity: str | None,
                      run_name: str, config: Mapping[str, Any]) -> WandbSink | None:
    if mode == "disabled":
        return None
    try:
        wandb = importlib.import_module("wandb")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "W&B is unavailable; run 'pip install -r requirements-kaggle.txt' "
            "or pass --wandb-mode disabled",
        ) from error
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=dict(config),
            mode=mode,
        )
    except Exception as error:
        raise RuntimeError(
            "W&B initialization failed; use --wandb-mode offline or disabled "
            f"to continue without an online run ({error})",
        ) from error
    return WandbSink(wandb, run, run_name)


def default_output_dir() -> Path:
    kaggle_root = Path("/kaggle/working")
    return kaggle_root / "l3p_runs" if kaggle_root.is_dir() else Path("l3p_runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train L3P or PN-LMCGS with Kaggle-friendly artifacts.",
    )
    parser.add_argument("--env", default="PointMaze", choices=list_envs())
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
    run_name_path = Path(run_name)
    if (not run_name or run_name_path.is_absolute() or len(run_name_path.parts) != 1
            or run_name_path.name in ("", ".", "..")):
        raise ValueError("run name must be a single relative directory name")
    run_dir = Path(output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"run directory {run_dir} exists; pass --overwrite to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as handle:
        json.dump(dict(config), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return run_dir


class Tee:
    """Write launcher output to both the terminal and a line-buffered file."""

    def __init__(self, path: Path):
        self.path = path
        self.terminal = None
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self.file = self.path.open("a", buffering=1)
        sys.stdout = self
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        sys.stdout = self.terminal
        self.file.close()

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.file.flush()


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = build_config(args)
    except ValueError as error:
        parser.error(str(error))

    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        parser.error("--wandb-mode online requires WANDB_API_KEY (set it in Kaggle Secrets)")

    run_name = args.run_name or default_run_name(cfg.env_name, cfg.seed)
    run_config = {
        "config": vars(cfg),
        "launcher": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
    }
    try:
        run_dir = prepare_run_directory(
            args.output_dir, run_name, run_config, overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))

    sink = create_wandb_sink(
        args.wandb_mode,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_run_name or run_name,
        run_config,
    )
    active_error = None
    try:
        with Tee(run_dir / "train.log"):
            print(f"===== L3P on {cfg.env_name} | steps={cfg.total_steps} seed={cfg.seed} =====")
            print(f"(artifacts -> {run_dir})")
            env = make_vec_env(cfg, cfg.n_workers, cfg.seed)
            trainer = L3PTrainer(env, cfg, metrics_callback=sink)
            checkpoint_path = run_dir / "checkpoint.pt"
            trainer.train(
                checkpoint_path=str(checkpoint_path) if args.save_every else None,
                checkpoint_every=args.save_every,
            )
            model_path = run_dir / "model.pt"
            trainer.save(str(model_path))
            print(f"Saved model to {model_path}")
            success_rate = trainer.evaluate(cfg.eval_episodes)
            final_metrics = {"success_rate": float(success_rate)}
            if cfg.pn_lmcgs_enabled:
                final_metrics.update(trainer.last_eval_metrics)
            if sink is not None:
                sink("final", trainer.total_env_steps, final_metrics)
                try:
                    sink.log_model(model_path)
                except Exception as error:
                    print(f"W&B artifact upload failed; local model preserved: {error}")
            print(f"Final test success rate: {success_rate:.2f}")
    except BaseException as error:
        active_error = error
        raise
    finally:
        if sink is not None:
            try:
                sink.finish()
            except Exception as error:
                if active_error is None:
                    raise
                print(f"W&B cleanup failed after training error: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
