#!/usr/bin/env python3
"""Preflight a checkpoint before spending hours on E1 experiments.

The E1 MCTS runs are only meaningful if the clean, no-noise baseline can solve
the environment at a non-trivial rate. This script checks that condition by
evaluating:
  * the flat goal-conditioned policy, no landmark planning;
  * Soft Floyd with the checkpoint/config default d_max;
  * Soft Floyd after a small fair d_max calibration sweep.

Example:
    python scripts/check_checkpoint_readiness.py --env AntMaze \
        --load checkpoint/l3p_AntMaze.pt --episodes 20 --calibrate-episodes 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config, list_envs
from l3p.envs import make_vec_env
from l3p.planning.noise import dmax_candidates
from l3p.planning.planner import LatentPlanner
from l3p.trainer import L3PTrainer


def reseed_eval_env(trainer, seed: int) -> None:
    """Best-effort deterministic episode reset for both NumPy and Gym envs."""
    env = trainer.env.envs[0]
    env.rng = np.random.default_rng(seed)


def eval_direct(trainer, episodes: int, base_seed: int) -> list[int]:
    outcomes = []
    for ep in range(episodes):
        reseed_eval_env(trainer, base_seed * 1_000_003 + ep)
        outcomes.append(int(trainer.evaluate(1, use_planning=False) > 0))
    return outcomes


def eval_soft_floyd(trainer, episodes: int, base_seed: int) -> list[int]:
    outcomes = []
    for ep in range(episodes):
        reseed_eval_env(trainer, base_seed * 1_000_003 + ep)
        planner = LatentPlanner(trainer.agent, trainer.landmarks, trainer.ae,
                                trainer.graph_search, trainer.cfg)
        outcomes.append(int(trainer.evaluate(1, planner=planner) > 0))
    return outcomes


def value_dmax_candidates(trainer, percentiles: list[float]) -> list[float]:
    with torch.no_grad():
        goals = trainer.ae.decode(trainer.landmarks.centroids.detach())
        m = goals.shape[0]
        gi = goals[:, None, :].expand(m, m, goals.shape[1]).reshape(m * m, -1)
        gj = goals[None, :, :].expand(m, m, goals.shape[1]).reshape(m * m, -1)
        V = trainer.agent.value(gi, gj).view(m, m).cpu().numpy()
    return dmax_candidates(V, percentiles=percentiles, fallback=trainer.cfg.d_max)


def mean(xs: list[int]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="AntMaze", choices=list_envs())
    p.add_argument("--load", type=str, default="checkpoint/l3p_AntMaze.pt")
    p.add_argument("--episodes", type=int, default=10,
                   help="final eval episodes for each reported baseline")
    p.add_argument("--calibrate-episodes", type=int, default=5,
                   help="episodes per d_max candidate during calibration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-success", type=float, default=0.30,
                   help="minimum calibrated Soft Floyd success required for READY")
    p.add_argument("--percentiles", type=float, nargs="+",
                   default=[5, 8, 11, 15, 20, 25, 30])
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    cfg = get_config(args.env, seed=args.seed)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)
    if not trainer.centroids_initialized:
        raise SystemExit("NOT READY: checkpoint has no initialized landmark centroids.")

    print(f"Checkpoint readiness: env={args.env} load={args.load}")
    print(f"episodes={args.episodes} calibrate_episodes={args.calibrate_episodes} "
          f"min_success={args.min_success:.2f}")

    direct = eval_direct(trainer, args.episodes, args.seed)
    print(f"flat policy success: {mean(direct):.2f} ({sum(direct)}/{len(direct)})")

    default_dmax = trainer.cfg.d_max
    default_soft = eval_soft_floyd(trainer, args.episodes, args.seed)
    print(f"Soft Floyd default d_max={default_dmax:.6g}: "
          f"{mean(default_soft):.2f} ({sum(default_soft)}/{len(default_soft)})")

    sweep = []
    best_dmax, best_sr = default_dmax, -1.0
    for dmax in value_dmax_candidates(trainer, args.percentiles):
        trainer.cfg.d_max = dmax
        out = eval_soft_floyd(trainer, args.calibrate_episodes, args.seed)
        sr = mean(out)
        sweep.append(dict(d_max=dmax, success=sr, successes=sum(out),
                          episodes=len(out)))
        if sr > best_sr:
            best_dmax, best_sr = dmax, sr
    trainer.cfg.d_max = best_dmax
    tuned = eval_soft_floyd(trainer, args.episodes, args.seed)
    tuned_sr = mean(tuned)

    print("d_max calibration sweep:")
    for row in sweep:
        mark = " <- chosen" if row["d_max"] == best_dmax else ""
        print(f"  d_max={row['d_max']:.6g}: {row['success']:.2f} "
              f"({row['successes']}/{row['episodes']}){mark}")
    print(f"Soft Floyd calibrated d_max={best_dmax:.6g}: "
          f"{tuned_sr:.2f} ({sum(tuned)}/{len(tuned)})")

    ready = tuned_sr >= args.min_success
    status = "READY" if ready else "NOT_READY"
    print(f"{status}: calibrated Soft Floyd {tuned_sr:.2f} "
          f"{'>=' if ready else '<'} threshold {args.min_success:.2f}")

    result = dict(
        env=args.env,
        load=args.load,
        episodes=args.episodes,
        calibrate_episodes=args.calibrate_episodes,
        min_success=args.min_success,
        flat_policy=dict(mean=mean(direct), successes=sum(direct), n=len(direct)),
        soft_floyd_default=dict(d_max=default_dmax, mean=mean(default_soft),
                                successes=sum(default_soft), n=len(default_soft)),
        d_max_sweep=sweep,
        soft_floyd_calibrated=dict(d_max=best_dmax, mean=tuned_sr,
                                   successes=sum(tuned), n=len(tuned)),
        status=status,
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {args.out}")

    if not ready:
        sys.exit(2)


if __name__ == "__main__":
    main()
