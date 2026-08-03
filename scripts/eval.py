#!/usr/bin/env python3
"""Evaluate a trained L3P checkpoint on the long-horizon PointMaze-Hard test.

Reports the test-time success rate and, optionally, the decoded latent landmark
coordinates (to check they scatter across the free space, as in the paper).

Example:
    python scripts/eval.py --load l3p_pointmaze.pt --episodes 50 --show-landmarks
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=str, default="l3p_pointmaze.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--no-planning", action="store_true",
                   help="evaluate the flat goal-conditioned policy without planning")
    p.add_argument("--show-landmarks", action="store_true")
    p.add_argument("--pn-lmcgs", action="store_true",
                   help="construct the optional PN-LMCGS checkpoint architecture")
    p.add_argument("--hazardous-pointmaze", action="store_true",
                   help="use matching hazardous PointMaze geometry (enables PN-LMCGS)")
    return p


def build_config(args):
    overrides = dict(seed=args.seed)
    if args.pn_lmcgs or args.hazardous_pointmaze:
        overrides["pn_lmcgs_enabled"] = True
    if args.hazardous_pointmaze:
        overrides["pn_pointmaze_hazard_enabled"] = True
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
    _print_pn_mode(cfg)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)

    sr = trainer.evaluate(args.episodes, use_planning=not args.no_planning)
    mode = "flat policy" if args.no_planning else "L3P planner"
    if cfg.pn_lmcgs_enabled:
        print(f"[{mode}] safe_success_rate (primary safety-aware metric) over "
              f"{args.episodes} eps: {trainer.last_eval_metrics['safe_success_rate']:.2f}")
    print(f"[{mode}] long-horizon test success rate over {args.episodes} eps: {sr:.2f}")

    if args.show_landmarks and trainer.centroids_initialized:
        with torch.no_grad():
            coords = trainer.ae.decode(trainer.landmarks.centroids.detach()).cpu().numpy()
        print(f"\n{len(coords)} decoded latent landmarks (x, y):")
        for i, c in enumerate(coords):
            print(f"  L{i:02d}: ({c[0]:5.2f}, {c[1]:5.2f})")
        print(f"  spread: x in [{coords[:,0].min():.2f}, {coords[:,0].max():.2f}], "
              f"y in [{coords[:,1].min():.2f}, {coords[:,1].max():.2f}]")


if __name__ == "__main__":
    main()
