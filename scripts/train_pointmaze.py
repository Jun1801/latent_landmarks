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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000, help="total env steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-landmarks", type=int, default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--maze-size", type=int, default=None,
                   help="maze grid size (larger -> longer horizon, planning matters more)")
    p.add_argument("--hindsight", type=int, default=None,
                   help="HER hindsight range (shorter -> low-level only learns short hops)")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="episodes per evaluation (more -> smoother success-rate curve)")
    p.add_argument("--device", type=str, default=None,
                   help="torch device override, e.g. cpu or cuda")
    p.add_argument("--save", type=str, default="l3p_pointmaze.pt")
    p.add_argument("--load", type=str, default=None,
                   help="resume from a saved checkpoint before training")
    p.add_argument("--save-training-state", action="store_true",
                   help="include replay buffer, optimizer state, counters, and RNG in checkpoints")
    p.add_argument("--save-every", type=int, default=0,
                   help="also save a checkpoint every N env steps (0 = only at end)")
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="stop training gracefully after this many hours")
    p.add_argument("--log-file", type=str, default="logs/pointmaze.log",
                   help="file to append all training logs to (set '' to disable)")
    p.add_argument("--value-contrastive", action="store_true",
                   help="enable InfoNCE negative-sampling loss for V(g1,g2)")
    p.add_argument("--value-contrastive-lambda", type=float, default=None)
    p.add_argument("--value-contrastive-temperature", type=float, default=None)
    p.add_argument("--n-value-negatives", type=int, default=None)
    p.add_argument("--negative-sampling-strategy", choices=["random", "cross_episode"],
                   default=None)
    p.add_argument("--short", action="store_true",
                   help="tiny config for a fast smoke test")
    args = p.parse_args()

    if args.log_file:
        sys.stdout = Tee(args.log_file)
        print(f"===== new run: steps={args.steps} seed={args.seed} =====")
        print(f"(logging to {args.log_file})")

    overrides = dict(seed=args.seed, total_steps=args.steps)
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
    if args.device is not None:
        overrides["device"] = args.device
    if args.value_contrastive:
        overrides["use_value_contrastive"] = True
    if args.value_contrastive_lambda is not None:
        overrides["value_contrastive_lambda"] = args.value_contrastive_lambda
    if args.value_contrastive_temperature is not None:
        overrides["value_contrastive_temperature"] = args.value_contrastive_temperature
    if args.n_value_negatives is not None:
        overrides["n_value_negatives"] = args.n_value_negatives
    if args.negative_sampling_strategy is not None:
        overrides["negative_sampling_strategy"] = args.negative_sampling_strategy

    if args.short:
        overrides.update(
            total_steps=30_000, max_episode_steps=100, n_workers=1,
            n_landmarks=15, n_warmup_trajs=20, initial_random_trajs=10,
            random_landmarks_train=10, batch_size=128, gls_batch_size=128,
            landmark_batch_size=64, train_after=1000, eval_interval=5000,
            eval_episodes=10, log_interval=2000,
        )

    cfg = get_config("PointMaze", **overrides)
    env = make_vec_env(cfg, cfg.n_workers, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    if args.load:
        trainer.load(args.load, restore_training_state=True)
        print(f"Loaded checkpoint from {args.load} "
              f"(env_steps={trainer.total_env_steps}, episodes={trainer.episodes_collected})")

    print(f"Training L3P on PointMaze-Hard | steps={cfg.total_steps} "
          f"workers={cfg.n_workers} landmarks={cfg.n_landmarks}")
    print(f"Value contrastive={'on' if cfg.use_value_contrastive else 'off'} "
          f"lambda={cfg.value_contrastive_lambda} temp={cfg.value_contrastive_temperature} "
          f"K={cfg.n_value_negatives} negatives={cfg.negative_sampling_strategy}")
    trainer.train(checkpoint_path=args.save if args.save_every else None,
                  checkpoint_every=args.save_every,
                  save_training_state=args.save_training_state,
                  time_limit_seconds=(args.time_limit_hours * 3600
                                      if args.time_limit_hours is not None else None))
    trainer.save(args.save, include_training_state=args.save_training_state)
    print(f"Saved model to {args.save}")

    sr = trainer.evaluate(cfg.eval_episodes)
    print(f"Final long-horizon test success rate: {sr:.2f}")


if __name__ == "__main__":
    main()
