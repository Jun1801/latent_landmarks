#!/usr/bin/env python3
"""Compare the PLAN (route) each planner takes on the SAME episode + SAME noisy
world model (E1c bias). One panel per planner: landmark graph (wormholes red),
the committed sub-goal sequence (dashed = the intended route), and the achieved
trajectory (solid navy), with success in the title. Shows concretely that
soft-Floyd routes INTO a wormhole and fails while mcts_fb detects+routes around it.

    python scripts/viz_plans.py --load checkpoint/l3p_pointmaze_full.pt \
        --sigma 0.3 --seed 3 --out logs/exp_suite/plans_pointmaze_e1c.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import numpy as np
import torch

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer
from l3p.planning.mcts_planner import SoftFloydE1c, FeedbackMCTSPlanner
from l3p.planning.noise import CriticEdgeFn
from run_e1c import calibrate_d_max


def run_episode(tr, planner, es, max_steps):
    """Record achieved trajectory + ordered distinct sub-goals + success + goal/start."""
    tr.env.set_eval(True)
    e = tr.env.envs[0]
    e.rng = np.random.default_rng(es); obs = e.reset()   # seed BEFORE reset (start/goal)
    goal = obs["desired_goal"].astype(np.float32)
    planner.reset(goal)
    traj, subs, succ = [obs["achieved_goal"].copy()], [], 0
    for _ in range(max_steps):
        a = planner.act(obs["observation"], achieved_goal=obs["achieved_goal"])
        obs, r, d, info = e.step(a)
        traj.append(obs["achieved_goal"].copy())
        si = getattr(planner, "subg_idx", None)
        if si is not None and (not subs or subs[-1] != si):
            subs.append(int(si))
        if float(info.get("is_success", r == 0.0)) > 0:   # PointMaze: reward 0 = reached
            succ = 1; break
        if d:
            break
    return np.array(traj), subs, succ, goal, np.asarray(traj[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--env", default="PointMaze")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=3, help="episode seed (pick one where soft-Floyd fails)")
    p.add_argument("--pct", type=float, default=25)
    p.add_argument("--dmax", type=float, default=None, help="fix d_max, skip calibration (faster seed search)")
    p.add_argument("--out", default="logs/exp_suite/plans_pointmaze_e1c.png")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = get_config(a.env, seed=0, mcts_n_simulations=100, mcts_rollout_horizon=10)
    tr = L3PTrainer(make_vec_env(cfg, 1, cfg.seed), cfg); tr.load(a.load)
    if a.dmax is not None:
        tr.cfg.d_max = a.dmax
    else:
        calibrate_d_max(tr, tr.cfg, 15, 987654)
    dmax = tr.cfg.d_max
    N = tr.landmarks.centroids.shape[0]
    with torch.no_grad():
        xy = tr.ae.decode(tr.landmarks.centroids.detach()).cpu().numpy()
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = tr.agent.distance_after_action(xy[i].astype(np.float32), xy.astype(np.float32))
    np.fill_diagonal(D, np.inf)
    adm = D <= max(dmax, np.percentile(D[np.isfinite(D)], a.pct))

    def mk(kind, s):
        if kind == "soft_floyd":
            return SoftFloydE1c(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg,
                                sigma=a.sigma, noise_seed=s, edge_value_fn=CriticEdgeFn(tr.agent))
        fb = (kind == "mcts_fb")
        return FeedbackMCTSPlanner(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg,
                                   sigma=a.sigma, noise_seed=s, feedback=fb,
                                   rho=0.5, tau_reach=0.25 * dmax, tau_progress=0.5 * dmax,
                                   tau_snap=dmax, r_max=2, edge_value_fn=CriticEdgeFn(tr.agent),
                                   rng=np.random.default_rng(s + 1))

    kinds = ["soft_floyd", "mcts_nofb", "mcts_fb"]
    # pick a bias/episode where soft_floyd FAILS but mcts_fb REACHES (illustrative)
    chosen, best = a.seed, None
    for s in [a.seed] + [x for x in range(12) if x != a.seed]:
        res = {k: run_episode(tr, mk(k, s), s, cfg.test_episode_steps) for k in kinds}
        if best is None:
            best = (s, res)
        if res["soft_floyd"][2] == 0 and res["mcts_fb"][2] == 1:
            chosen, best = s, (s, res); break
    chosen, runs = best[0], best[1]
    goal, start = runs["soft_floyd"][3], runs["soft_floyd"][4]
    Dn = D * (1.0 + np.random.default_rng(chosen).normal(0.0, a.sigma, D.shape))
    worm = adm & (Dn < 0.5 * D)
    print(f"[viz_plans] chosen seed={chosen}  success: "
          + ", ".join(f"{k}={runs[k][2]}" for k in kinds), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, k in zip(axes, kinds):
        traj, subs, succ = runs[k][0], runs[k][1], runs[k][2]
        for i in range(N):
            for j in range(i + 1, N):
                if adm[i, j]:
                    red = worm[i, j] or worm[j, i]
                    ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=("red" if red else "0.8"),
                            lw=(1.5 if red else 0.4), alpha=(0.7 if red else 0.3), zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=14, c="steelblue", zorder=3)
        # planned route: start -> chosen subgoals -> goal (dashed)
        route = [start] + [xy[i] for i in subs if i < N] + [goal]
        route = np.array(route)
        ax.plot(route[:, 0], route[:, 1], "--", color="orange", lw=1.6, alpha=0.85, zorder=4,
                label="planned route")
        ax.scatter([xy[i, 0] for i in subs if i < N], [xy[i, 1] for i in subs if i < N],
                   s=120, facecolors="none", edgecolors="orange", linewidths=2, zorder=6)
        ax.plot(traj[:, 0], traj[:, 1], "-", color="navy", lw=2.2, alpha=0.95, zorder=5,
                label="achieved")
        ax.scatter([start[0]], [start[1]], marker="s", s=90, c="green", edgecolors="k", zorder=7)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=300, c="gold", edgecolors="k", zorder=7)
        ax.set_title(f"{k}   {'REACHED' if succ else 'FAILED'}   ({len(subs)} subgoals)",
                     fontsize=11, color=("green" if succ else "crimson"))
        ax.set_aspect("equal"); ax.axis("off")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Plan comparison on the SAME noisy world model (E1c bias, sigma={a.sigma}) "
                 f"— red=wormhole", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}  |  success: " + ", ".join(f"{k}={runs[k][2]}" for k in kinds))


if __name__ == "__main__":
    main()
