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

from l3p.config import get_config, list_envs, resolve_env
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="FetchPickAndPlace", choices=list_envs())
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--log-file", type=str, default=None)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--pn-lmcgs", action="store_true",
                   help="enable the optional positive-negative landmark MCGS workflow")
    p.add_argument("--hazardous-pointmaze", action="store_true",
                   help="use PointMaze's two-route hazardous geometry (enables PN-LMCGS)")
    return p


def build_config(args):
    if args.hazardous_pointmaze and resolve_env(args.env) != "PointMaze":
        raise ValueError("--hazardous-pointmaze is only valid for the PointMaze environment")

    overrides = dict(seed=args.seed, total_steps=args.steps)
    if args.workers is not None:
        overrides["n_workers"] = args.workers
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.pn_lmcgs or args.hazardous_pointmaze:
        overrides["pn_lmcgs_enabled"] = True
    if args.hazardous_pointmaze:
        overrides["pn_pointmaze_hazard_enabled"] = True
    return get_config(args.env, **overrides)


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
    save = args.save or f"l3p_{cfg.env_name}.pt"
    log_file = args.log_file or f"logs/{cfg.env_name}.log"
    sys.stdout = Tee(log_file)

    print(f"===== L3P on {cfg.env_name} | steps={cfg.total_steps} seed={cfg.seed} =====")
    print(f"(Appendix-E: workers={cfg.n_workers} batch={cfg.batch_size} gamma={cfg.gamma} "
          f"hindsight={cfg.hindsight_range} N={cfg.n_landmarks} d_max={cfg.d_max} "
          f"warmup_trajs={cfg.n_warmup_trajs})")
    _print_pn_mode(cfg)
    print(f"(logging to {log_file}, model -> {save})")

    venv = make_vec_env(cfg, cfg.n_workers, cfg.seed)
    print(f"env ready: obs={venv.obs_dim} goal={venv.goal_dim} act={venv.act_dim} "
          f"workers={venv.n} horizon={venv.max_episode_steps}")
    trainer = L3PTrainer(venv, cfg)
    trainer.train(checkpoint_path=save if args.save_every else None,
                  checkpoint_every=args.save_every)
    trainer.save(save)
    print(f"Saved model to {save}")
    sr = trainer.evaluate(cfg.eval_episodes)
    if cfg.pn_lmcgs_enabled:
        print(f"Final safe_success_rate (primary safety-aware metric): "
              f"{trainer.last_eval_metrics['safe_success_rate']:.2f}")
    print(f"Final test success rate: {sr:.2f}")


if __name__ == "__main__":
    main()
