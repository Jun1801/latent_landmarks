#!/usr/bin/env python3
"""Visualize the landmark world-model graph BEFORE vs AFTER injecting noise
(Track V). Two panels side by side: clean edges vs noisy edges, with "wormhole"
edges (noise made them look much SHORTER than they are -- the traps that fool a
static planner) highlighted in red. Optional --traj overlays one soft-Floyd
episode's achieved trajectory + chosen sub-goals on the noisy graph.

PointMaze (2D goal) runs fully locally:
    python scripts/viz_graph_noise.py --load checkpoint/l3p_pointmaze_full.pt \
        --regime e1c --sigma 0.3 --traj --out logs/exp_suite/viz_pointmaze_e1c.png
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
from l3p.planning.planner import LatentPlanner
from l3p.planning.mcts_planner import SoftFloydE1c
from l3p.planning.noise import CriticEdgeFn


def pairwise_D(agent, goals):
    """Critic-D between every pair of goal-space landmark points -> [N, N]."""
    N = goals.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = agent.distance_after_action(goals[i], goals)
    return D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--env", default="PointMaze")
    p.add_argument("--regime", choices=["e1a", "e1c"], default="e1c")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--pct", type=float, default=25, help="admissibility: keep edges below this D percentile")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj", action="store_true", help="overlay one soft-Floyd episode")
    p.add_argument("--out", default="logs/exp_suite/viz_graph_noise.png")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = get_config(a.env, seed=a.seed)
    env = make_vec_env(cfg, 1, cfg.seed)
    tr = L3PTrainer(env, cfg)
    tr.load(a.load)
    with torch.no_grad():
        xy = tr.ae.decode(tr.landmarks.centroids.detach()).cpu().numpy()   # [N, 2]
    N = xy.shape[0]
    D = pairwise_D(tr.agent, xy.astype(np.float32))
    np.fill_diagonal(D, np.inf)
    dmax = np.percentile(D[np.isfinite(D)], a.pct)
    adm = D <= dmax

    rng = np.random.default_rng(a.seed)
    if a.regime == "e1c":                        # fixed multiplicative bias
        Dn = D * (1.0 + rng.normal(0.0, a.sigma, D.shape))
    else:                                        # additive
        Dn = D + rng.normal(0.0, a.sigma * np.nanmedian(D[np.isfinite(D)]), D.shape)
    Dn = np.clip(Dn, 0.0, None)
    worm = adm & (Dn < 0.5 * D)                  # looks >=2x shorter than reality

    tr.env.set_eval(True)
    obs = tr.env.envs[0].reset()
    goal = obs["desired_goal"].astype(np.float32)
    start = obs["achieved_goal"].astype(np.float32)

    traj = subg = None
    if a.traj:
        pl = (SoftFloydE1c(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg,
                           sigma=a.sigma, noise_seed=a.seed, edge_value_fn=CriticEdgeFn(tr.agent))
              if a.regime == "e1c" else
              LatentPlanner(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg))
        pl.reset(goal)
        path, subs = [start.copy()], []
        for _ in range(cfg.test_episode_steps):
            act = pl.act(obs["observation"], achieved_goal=obs["achieved_goal"])
            obs, r, d, _ = tr.env.envs[0].step(act)
            path.append(obs["achieved_goal"].copy())
            si = getattr(pl, "subg_idx", None)
            if si is not None and si < N:
                subs.append(si)
            if r > 0 or d:
                break
        traj = np.array(path)
        subg = sorted(set(subs))

    def panel(ax, mat, title, hi=False):
        finite = mat[np.isfinite(mat)]
        vmax = np.percentile(finite, 90) if finite.size else 1.0
        for i in range(N):
            for j in range(i + 1, N):
                if not adm[i, j]:
                    continue
                red = hi and (worm[i, j] or worm[j, i])
                ax.plot(xy[[i, j], 0], xy[[i, j], 1],
                        color=("red" if red else "0.7"),
                        lw=(2.0 if red else 0.5), alpha=(0.9 if red else 0.35), zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c="steelblue", zorder=3)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=260, c="gold", edgecolors="k", zorder=5)
        ax.scatter([start[0]], [start[1]], marker="s", s=80, c="green", edgecolors="k", zorder=5)
        if hi and traj is not None:
            ax.plot(traj[:, 0], traj[:, 1], "-", color="navy", lw=2.2, alpha=0.95, zorder=4)
            if subg:
                ax.scatter(xy[subg, 0], xy[subg, 1], s=110, facecolors="none",
                           edgecolors="navy", linewidths=2.2, zorder=6)
        ax.set_title(title, fontsize=11); ax.set_aspect("equal"); ax.axis("off")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    panel(a1, D, f"CLEAN world model  (N={N})")
    panel(a2, Dn, f"NOISY ({a.regime}, sigma={a.sigma})  red=wormhole"
                  + ("  + soft-Floyd path" if a.traj else ""), hi=True)
    fig.suptitle("Landmark graph before vs after world-model noise", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}  |  {int(worm.sum())} wormhole edges, {int(adm.sum()//2)} admissible")


if __name__ == "__main__":
    main()
