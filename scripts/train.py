#!/usr/bin/env python3
"""General L3P training launcher for any of the paper's environments.

Uses the faithful Appendix-E hyper-parameters from l3p/config.py by default;
any of them can be overridden on the command line.

Examples:
    python scripts/train.py --env FetchPickAndPlace --steps 1000000
    python scripts/train.py --env PointMaze --steps 500000
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # silence gymnasium/robotics deprecation spam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3p.config import get_config, list_envs
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


class Tee:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.terminal = sys.stdout
        self.file = open(path, "a", buffering=1)

    def write(self, m):
        self.terminal.write(m); self.file.write(m)

    def flush(self):
        self.terminal.flush(); self.file.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="FetchPickAndPlace", choices=list_envs())
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", type=str, default=None,
                   help="torch device override, e.g. cpu or cuda")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--load", type=str, default=None,
                   help="resume from a saved checkpoint before training")
    p.add_argument("--save-training-state", action="store_true",
                   help="include replay buffer, optimizer state, counters, and RNG in checkpoints")
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--time-limit-hours", type=float, default=None,
                   help="stop training gracefully after this many hours")
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--value-contrastive", action="store_true",
                   help="enable InfoNCE negative-sampling loss for V(g1,g2)")
    p.add_argument("--value-contrastive-lambda", type=float, default=None)
    p.add_argument("--value-contrastive-temperature", type=float, default=None)
    p.add_argument("--n-value-negatives", type=int, default=None)
    p.add_argument("--negative-sampling-strategy", choices=["random", "cross_episode"],
                   default=None)
    p.add_argument("--ae-contrastive-lambda", type=float, default=None,
                   help="weight for the auxiliary AE latent triplet loss; use 0.0 for baseline")
    p.add_argument("--ae-contrastive-margin", type=float, default=None)
    p.add_argument("--ae-negatives-per-anchor", type=int, default=None)
    p.add_argument("--ae-negative-mode", choices=["random", "hard"], default=None)
    args = p.parse_args()

    overrides = dict(seed=args.seed, total_steps=args.steps)
    if args.workers is not None:
        overrides["n_workers"] = args.workers
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
    if args.ae_contrastive_lambda is not None:
        overrides["ae_contrastive_lambda"] = args.ae_contrastive_lambda
    if args.ae_contrastive_margin is not None:
        overrides["ae_contrastive_margin"] = args.ae_contrastive_margin
    if args.ae_negatives_per_anchor is not None:
        overrides["ae_negatives_per_anchor"] = args.ae_negatives_per_anchor
    if args.ae_negative_mode is not None:
        overrides["ae_negative_mode"] = args.ae_negative_mode

    cfg = get_config(args.env, **overrides)
    save = args.save or f"l3p_{cfg.env_name}.pt"
    log_file = args.log_file or f"logs/{cfg.env_name}.log"
    sys.stdout = Tee(log_file)

    print(f"===== L3P on {cfg.env_name} | steps={cfg.total_steps} seed={cfg.seed} =====")
    print(f"(Appendix-E: workers={cfg.n_workers} batch={cfg.batch_size} gamma={cfg.gamma} "
          f"hindsight={cfg.hindsight_range} N={cfg.n_landmarks} d_max={cfg.d_max} "
          f"warmup_trajs={cfg.n_warmup_trajs})")
    print(f"(value contrastive={'on' if cfg.use_value_contrastive else 'off'} "
          f"lambda={cfg.value_contrastive_lambda} temp={cfg.value_contrastive_temperature} "
          f"K={cfg.n_value_negatives} negatives={cfg.negative_sampling_strategy})")
    print(f"(AE contrastive={'on' if cfg.ae_contrastive_lambda > 0 else 'off'} "
          f"lambda={cfg.ae_contrastive_lambda} margin={cfg.ae_contrastive_margin} "
          f"K={cfg.ae_negatives_per_anchor} negatives={cfg.ae_negative_mode})")
    print(f"(logging to {log_file}, model -> {save})")

    venv = make_vec_env(cfg, cfg.n_workers, cfg.seed)
    print(f"env ready: obs={venv.obs_dim} goal={venv.goal_dim} act={venv.act_dim} "
          f"workers={venv.n} horizon={venv.max_episode_steps}")
    trainer = L3PTrainer(venv, cfg)
    if args.load:
        trainer.load(args.load, restore_training_state=True)
        print(f"Loaded checkpoint from {args.load} "
              f"(env_steps={trainer.total_env_steps}, episodes={trainer.episodes_collected})")
    trainer.train(checkpoint_path=save if args.save_every else None,
                  checkpoint_every=args.save_every,
                  save_training_state=args.save_training_state,
                  time_limit_seconds=(args.time_limit_hours * 3600
                                      if args.time_limit_hours is not None else None))
    trainer.save(save, include_training_state=args.save_training_state)
    print(f"Saved model to {save}")
    sr = trainer.evaluate(cfg.eval_episodes)
    print(f"Final test success rate: {sr:.2f}")


if __name__ == "__main__":
    main()
