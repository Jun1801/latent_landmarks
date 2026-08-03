#!/usr/bin/env python3
"""Train L3P on PointMaze-Hard.

Runs end-to-end with only torch + numpy installed. Use --short for a quick
smoke run that exercises the whole collect -> train -> plan -> eval pipeline.

Examples:
    python scripts/train_pointmaze.py --short
    python scripts/train_pointmaze.py --steps 500000 --n-landmarks 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


class Tee:
    """Duplicate stdout to a log file (line-buffered), so all training output is
    both printed to the terminal and persisted to a single file."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.terminal = sys.stdout
        self.file = open(path, "a", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None, help="total env steps (default: 500000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-landmarks", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--maze-size", type=int, default=None,
                   help="maze grid size (larger -> longer horizon, planning matters more)")
    p.add_argument("--hindsight", type=int, default=None,
                   help="HER hindsight range (shorter -> low-level only learns short hops)")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="episodes per evaluation (more -> smoother success-rate curve)")
    p.add_argument("--save", type=str, default="l3p_pointmaze.pt")
    p.add_argument("--save-every", type=int, default=0,
                   help="also save a checkpoint every N env steps (0 = only at end)")
    p.add_argument("--log-file", type=str, default="logs/pointmaze.log",
                   help="file to append all training logs to (set '' to disable)")
    p.add_argument("--short", action="store_true",
                   help="tiny config for a fast smoke test")
    p.add_argument("--pn-lmcgs", action="store_true",
                   help="enable the optional positive-negative landmark MCGS workflow")
    p.add_argument("--hazardous-pointmaze", action="store_true",
                   help="use PointMaze's two-route hazardous geometry (enables PN-LMCGS)")
    return p


def build_config(args):
    total_steps = args.steps if args.steps is not None else 500_000
    overrides = dict(seed=args.seed, total_steps=total_steps)

    if args.n_landmarks is not None:
        overrides["n_landmarks"] = args.n_landmarks
    if args.n_workers is not None:
        overrides["n_workers"] = args.n_workers
    if args.maze_size is not None:
        overrides["maze_size"] = args.maze_size
    if args.hindsight is not None:
        overrides["hindsight_range"] = args.hindsight
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.pn_lmcgs or args.hazardous_pointmaze:
        overrides["pn_lmcgs_enabled"] = True
    if args.hazardous_pointmaze:
        overrides["pn_pointmaze_hazard_enabled"] = True

    if args.short:
        overrides.update(
            total_steps=total_steps if args.steps is not None else 30_000,
            max_episode_steps=100, n_workers=1,
            n_landmarks=15, n_warmup_trajs=20, initial_random_trajs=10,
            random_landmarks_train=10, batch_size=128, gls_batch_size=128,
            landmark_batch_size=64, train_after=1, n_grad_steps=1,
            env_steps_per_opt=100, eval_interval=5000, eval_episodes=10,
            log_interval=2000,
        )
    return get_config("PointMaze", **overrides)


def _print_pn_mode(cfg) -> None:
    print(f"(PN-LMCGS enabled={cfg.pn_lmcgs_enabled} "
          f"hazardous_pointmaze={cfg.pn_pointmaze_hazard_enabled} "
          "safety_cost_precedence=Safety-Gym cost > info[safety_cost] > "
          "info[cost] > explicit collision adapter > zero)")
    print(f"(PN phase gates: WARMUP -> LANDMARKS -> MACRO_BOOTSTRAP -> JOINT | "
          f"collision_adapter=enabled:{cfg.pn_collision_cost_enabled} "
          f"key:{cfg.pn_collision_cost_info_key or '<none>'} | "
          f"pn_training_unsafe_fallback={cfg.pn_training_unsafe_fallback})")


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = build_config(args)

    if args.log_file:
        sys.stdout = Tee(args.log_file)
        print(f"===== new run: steps={cfg.total_steps} seed={args.seed} =====")
        print(f"(logging to {args.log_file})")

    env = make_vec_env(cfg, cfg.n_workers, cfg.seed)
    trainer = L3PTrainer(env, cfg)

    print(f"Training L3P on PointMaze-Hard | steps={cfg.total_steps} "
          f"workers={cfg.n_workers} landmarks={cfg.n_landmarks}")
    _print_pn_mode(cfg)
    trainer.train(checkpoint_path=args.save if args.save_every else None,
                  checkpoint_every=args.save_every)
    trainer.save(args.save)
    print(f"Saved model to {args.save}")

    sr = trainer.evaluate(cfg.eval_episodes)
    if cfg.pn_lmcgs_enabled:
        print(f"Final safe_success_rate (primary safety-aware metric): "
              f"{trainer.last_eval_metrics['safe_success_rate']:.2f}")
    print(f"Final long-horizon test success rate: {sr:.2f}")


if __name__ == "__main__":
    main()
