#!/usr/bin/env python3
"""Visualize an L3P planned episode on the PointMaze map (cf. paper Figure 6).

Loads a trained checkpoint, runs a long-horizon test episode with the L3P
planner, and draws:
  * the maze walls
  * all decoded latent landmarks (blue dots)
  * the landmarks actually chosen as sub-goals by the planner (blue stars)
  * the agent's trajectory (orange)
  * start (orange dot) and goal (red star)

Example:
    python scripts/plot_trajectory.py --load l3p_pointmaze_full.pt --out logs/trajectory.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


def run_episode(trainer, env):
    """Run one planned long-horizon episode; return trajectory, chosen sub-goal
    indices, goal, and success."""
    env.set_eval(True)
    obs = env.reset()
    goal = obs["desired_goal"].astype(np.float32)
    trainer.planner.reset(goal)
    traj = [obs["observation"].copy()]
    chosen = []
    success = False
    for _ in range(env.max_episode_steps):
        a = trainer.planner.act(obs["observation"])
        idx = trainer.planner.subg_idx
        if idx is not None and idx < trainer.planner.n_landmarks:
            chosen.append(idx)
        obs, _, done, info = env.step(a)
        traj.append(obs["observation"].copy())
        if info["is_success"] > 0:
            success = True
            break
    return np.array(traj), sorted(set(chosen)), goal, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", default="l3p_pointmaze_full.pt")
    p.add_argument("--out", default="logs/trajectory.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tries", type=int, default=8, help="episodes to try for a successful one")
    args = p.parse_args()

    cfg = get_config("PointMaze", seed=args.seed)
    venv = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(venv, cfg)
    trainer.load(args.load)
    env = venv.envs[0]

    # pick a successful episode if possible, else the longest one
    best = None
    for _ in range(args.tries):
        traj, chosen, goal, success = run_episode(trainer, env)
        if success:
            best = (traj, chosen, goal, success)
            break
        if best is None or len(traj) > len(best[0]):
            best = (traj, chosen, goal, success)
    traj, chosen, goal, success = best

    # decoded landmarks
    with torch.no_grad():
        landmarks = trainer.ae.decode(trainer.landmarks.centroids.detach()).cpu().numpy()

    maze = env.maze
    H, W = maze.shape

    fig, ax = plt.subplots(figsize=(7, 7))
    # walls
    for r in range(H):
        for c in range(W):
            if maze[r, c] == 1:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="0.6", ec="none"))

    # all landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="#1f77b4", s=45, alpha=0.5,
               edgecolors="white", linewidths=0.5, label="latent landmarks (all)", zorder=3)
    # chosen sub-goal landmarks
    if chosen:
        cl = landmarks[chosen]
        ax.scatter(cl[:, 0], cl[:, 1], marker="*", c="#1f77b4", s=320,
                   edgecolors="black", linewidths=0.8, label="chosen as sub-goals", zorder=5)

    # trajectory
    ax.plot(traj[:, 0], traj[:, 1], "-", color="#ff7f0e", lw=2.2, alpha=0.9,
            label="agent trajectory", zorder=4)
    ax.scatter([traj[0, 0]], [traj[0, 1]], c="#ff7f0e", s=140, edgecolors="black",
               linewidths=1, label="start", zorder=6)
    ax.scatter([goal[0]], [goal[1]], marker="*", c="#d62728", s=360, edgecolors="black",
               linewidths=1, label="goal", zorder=6)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.invert_yaxis()               # row 0 at top (natural maze view)
    ax.set_xticks([]); ax.set_yticks([])
    status = "SUCCESS" if success else f"not reached ({len(traj)} steps)"
    ax.set_title(f"$L^3P$ planned path on PointMaze-Hard — {status}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9,
              frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out}")
    print(f"episode: {len(traj)} steps, success={success}, "
          f"sub-goals used: {len(chosen)}")


if __name__ == "__main__":
    main()
