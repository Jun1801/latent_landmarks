#!/usr/bin/env python3
"""Visualize an L3P planned episode on a real MuJoCo environment (cf. Figure 6).

Works for the gymnasium-robotics maze envs (draws the maze walls) and, in a
top-down mode, for Fetch (draws the object's path on the table). Plots:
  * maze walls (maze envs) / table region (Fetch)
  * all decoded latent landmarks (blue)
  * the agent/object trajectory (orange), start (dot) and goal (red star)

Example:
    python scripts/plot_trajectory_mujoco.py --env PointMazeMuJoCo \
        --load logs/point_maze_mujoco/model.pt --out logs/point_maze_mujoco/trajectory.png
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer


def run_episode(trainer, env, T):
    obs = env.reset()
    goal = obs["desired_goal"].astype(np.float32)
    trainer.planner.reset(goal)
    traj = [obs["achieved_goal"].copy()]
    chosen = []
    success = 0.0
    for _ in range(T):
        a = trainer.planner.act(obs["observation"])
        idx = trainer.planner.subg_idx
        if idx is not None and idx < trainer.planner.n_landmarks:
            chosen.append(idx)
        obs, r, done, info = env.step(a)
        traj.append(obs["achieved_goal"].copy())
        if float(info.get("success", info.get("is_success", 0.0))) > 0:
            success = 1.0
            break
    return np.array(traj), sorted(set(chosen)), goal, success


def draw_maze_walls(ax, maze):
    """Draw wall cells of a gymnasium-robotics maze using its cell->xy mapping."""
    scaling = getattr(maze, "maze_size_scaling", 1.0)
    mp = maze.maze_map
    for r in range(len(mp)):
        for c in range(len(mp[0])):
            if mp[r][c] == 1:
                x, y = maze.cell_rowcol_to_xy((r, c))
                ax.add_patch(plt.Rectangle((x - scaling / 2, y - scaling / 2),
                                           scaling, scaling, color="0.6", ec="none"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tries", type=int, default=10)
    args = p.parse_args()

    cfg = get_config(args.env, seed=1)
    venv = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(venv, cfg)
    trainer.load(args.load)
    env = venv.envs[0]
    T = cfg.test_episode_steps

    # Prefer a SUCCESSFUL episode whose start<->goal are the farthest apart in
    # space (a genuine long-horizon traversal, cf. Figure 6), rather than one
    # that merely took many steps oscillating over a short distance.
    def span(r):
        traj, _, goal, _ = r
        return float(np.linalg.norm(traj[0][:2] - goal[:2]))
    succ_runs, all_runs = [], []
    for _ in range(args.tries):
        traj, chosen, goal, success = run_episode(trainer, env, T)
        all_runs.append((traj, chosen, goal, success))
        if success:
            succ_runs.append((traj, chosen, goal, success))
    pool = succ_runs if succ_runs else all_runs
    traj, chosen, goal, success = max(pool, key=span)

    with torch.no_grad():
        landmarks = trainer.ae.decode(trainer.landmarks.centroids.detach()).cpu().numpy()

    # take the first 2 dims (xy) — 3D Fetch goals project to the table plane
    traj2, lm2, goal2 = traj[:, :2], landmarks[:, :2], goal[:2]

    fig, ax = plt.subplots(figsize=(7, 7))
    base = env.env.unwrapped if hasattr(env, "env") else env
    maze = getattr(base, "maze", None)
    if maze is not None and hasattr(maze, "maze_map"):
        draw_maze_walls(ax, maze)
    else:  # Fetch: draw the table footprint + any obstacle / container
        xy = np.vstack([traj2, lm2, goal2[None, :]])
        lo, hi = xy.min(0) - 0.05, xy.max(0) + 0.05
        ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0]-lo[0], hi[1]-lo[1],
                                   color="0.92", ec="0.6"))
        from l3p.config import resolve_env
        from l3p.envs import fetch_variants as fv
        canonical = resolve_env(args.env)
        if canonical == "BoxDistractorPickAndPlace":
            x, y = fv.DISTRACTOR_XY; h = fv.DISTRACTOR_HALF
            ax.add_patch(plt.Rectangle((x-h, y-h), 2*h, 2*h, color="0.45",
                                       ec="black", zorder=2, label="box distractor"))
        elif canonical == "PlaceInsideBox":
            x, y = fv.CONTAINER_XY; h = fv.CONTAINER_HALF
            ax.add_patch(plt.Rectangle((x-h, y-h), 2*h, 2*h, fill=False,
                                       ec="#8c564b", lw=3, zorder=2, label="target box"))

    ax.scatter(lm2[:, 0], lm2[:, 1], c="#1f77b4", s=45, alpha=0.5, edgecolors="white",
               linewidths=0.5, label="latent landmarks", zorder=3)
    if chosen:
        cl = lm2[chosen]
        ax.scatter(cl[:, 0], cl[:, 1], marker="*", c="#1f77b4", s=300,
                   edgecolors="black", linewidths=0.8, label="chosen sub-goals", zorder=5)
    ax.plot(traj2[:, 0], traj2[:, 1], "-", color="#ff7f0e", lw=2.2, alpha=0.9,
            label="trajectory", zorder=4)
    ax.scatter([traj2[0, 0]], [traj2[0, 1]], c="#ff7f0e", s=140, edgecolors="black",
               linewidths=1, label="start", zorder=6)
    ax.scatter([goal2[0]], [goal2[1]], marker="*", c="#d62728", s=340, edgecolors="black",
               linewidths=1, label="goal", zorder=6)

    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    status = "SUCCESS" if success else f"{len(traj)} steps"
    ax.set_title(f"$L^3P$ planned path — {args.env} ({status})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Saved {args.out} | success={success} sub-goals={len(chosen)} steps={len(traj)}")


if __name__ == "__main__":
    main()
