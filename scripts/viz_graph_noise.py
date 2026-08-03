#!/usr/bin/env python3
"""Visualize the landmark world-model graph BEFORE vs AFTER injecting noise
(Track V): two panels (clean vs noisy edges) with "wormhole" edges (noise made
them look >=2x SHORTER -- the traps that fool a static planner) in red, plus an
optional soft-Floyd trajectory + chosen sub-goals on the noisy panel.

PointMaze (2D goal) runs fully locally from a checkpoint:
    python scripts/viz_graph_noise.py --load checkpoint/l3p_pointmaze_full.pt \
        --regime e1c --sigma 0.3 --traj --out logs/exp_suite/viz_pointmaze_e1c.png

Paper envs (AntMaze/Fetch, MuJoCo) can't run on this box -- plot from a JSON dumped
on Kaggle by eval_ablation.py --dump:
    python scripts/viz_graph_noise.py --from-dump antmaze_dump.json \
        --out logs/exp_suite/viz_antmaze_e1c.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def render(xy, D, Dn, adm, worm, goal, start, traj, subg, out, subtitle):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    N = xy.shape[0]

    def panel(ax, title, hi=False):
        for i in range(N):
            for j in range(i + 1, N):
                if not adm[i, j]:
                    continue
                red = hi and (worm[i, j] or worm[j, i])
                ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=("red" if red else "0.7"),
                        lw=(2.0 if red else 0.5), alpha=(0.9 if red else 0.35), zorder=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c="steelblue", zorder=3)
        ax.scatter([goal[0]], [goal[1]], marker="*", s=260, c="gold", edgecolors="k", zorder=5)
        ax.scatter([start[0]], [start[1]], marker="s", s=80, c="green", edgecolors="k", zorder=5)
        if hi and traj is not None and len(traj):
            t = np.asarray(traj)
            ax.plot(t[:, 0], t[:, 1], "-", color="navy", lw=2.2, alpha=0.95, zorder=4)
            if subg:
                ax.scatter(xy[subg, 0], xy[subg, 1], s=110, facecolors="none",
                           edgecolors="navy", linewidths=2.2, zorder=6)
        ax.set_title(title, fontsize=11); ax.set_aspect("equal"); ax.axis("off")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
    panel(a1, f"CLEAN world model  (N={N})")
    panel(a2, "NOISY  " + subtitle + "  red=wormhole"
              + ("  + soft-Floyd path" if (traj is not None and len(traj)) else ""), hi=True)
    fig.suptitle("Landmark graph before vs after world-model noise", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}  |  {int(worm.sum())} wormhole edges, {int(adm.sum() // 2)} admissible")


def from_dump(path, pct, worm_frac=0.5):
    d = json.load(open(path))
    n = d["n"]
    xy = np.asarray(d["landmark_xy"])
    D = -np.asarray(d["edge_clean"])[:n, :n]     # paper V<=0 -> distance = -V
    Dn = -np.asarray(d["edge_noisy"])[:n, :n]
    np.fill_diagonal(D, np.inf)
    adm = D <= np.percentile(D[np.isfinite(D)], pct)
    worm = adm & (Dn < worm_frac * D)            # edge looks >= 1/worm_frac x shorter
    if "planners" in d:                          # --dump-plans format: reuse graph + one route
        pk = "soft_floyd" if "soft_floyd" in d["planners"] else list(d["planners"])[0]
        p = d["planners"][pk]; traj = p.get("traj"); raw_sub = p.get("subgoals")
        sub = f"({d['regime']}, sigma={d['sigma']}, {pk})"
    else:                                        # single --dump format
        traj = d.get("traj"); raw_sub = d.get("subgoals")
        sub = f"({d['regime']}, sigma={d['sigma']})"
    subg = [s for s in (raw_sub or []) if s < n]   # drop the goal node (idx n)
    return dict(xy=xy, D=D, Dn=Dn, adm=adm, worm=worm, goal=np.asarray(d["goal"]),
                start=np.asarray(d["start"]), traj=traj, subg=subg, subtitle=sub)


def from_checkpoint(a):
    import torch
    from l3p.config import get_config
    from l3p.envs import make_vec_env
    from l3p.trainer import L3PTrainer
    from l3p.planning.planner import LatentPlanner
    from l3p.planning.mcts_planner import SoftFloydE1c
    from l3p.planning.noise import CriticEdgeFn

    cfg = get_config(a.env, seed=a.seed)
    tr = L3PTrainer(make_vec_env(cfg, 1, cfg.seed), cfg); tr.load(a.load)
    with torch.no_grad():
        xy = tr.ae.decode(tr.landmarks.centroids.detach()).cpu().numpy()
    N = xy.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i] = tr.agent.distance_after_action(xy[i].astype(np.float32), xy.astype(np.float32))
    np.fill_diagonal(D, np.inf)
    adm = D <= np.percentile(D[np.isfinite(D)], a.pct)
    rng = np.random.default_rng(a.seed)
    if a.regime == "e1c":
        Dn = D * (1.0 + rng.normal(0.0, a.sigma, D.shape))
    else:
        Dn = D + rng.normal(0.0, a.sigma * np.nanmedian(D[np.isfinite(D)]), D.shape)
    Dn = np.clip(Dn, 0.0, None)
    worm = adm & (Dn < 0.5 * D)
    tr.env.set_eval(True)
    obs = tr.env.envs[0].reset()
    goal = obs["desired_goal"].astype(np.float32); start = obs["achieved_goal"].astype(np.float32)
    traj = subg = None
    if a.traj:
        pl = (SoftFloydE1c(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg,
                           sigma=a.sigma, noise_seed=a.seed, edge_value_fn=CriticEdgeFn(tr.agent))
              if a.regime == "e1c" else
              LatentPlanner(tr.agent, tr.landmarks, tr.ae, tr.graph_search, tr.cfg))
        pl.reset(goal); path, subs = [start.copy()], []
        for _ in range(cfg.test_episode_steps):
            obs, r, d, _ = tr.env.envs[0].step(pl.act(obs["observation"], achieved_goal=obs["achieved_goal"]))
            path.append(obs["achieved_goal"].copy())
            si = getattr(pl, "subg_idx", None)
            if si is not None and si < N:
                subs.append(si)
            if r > 0 or d:
                break
        traj, subg = np.array(path), sorted(set(subs))
    return dict(xy=xy, D=D, Dn=Dn, adm=adm, worm=worm, goal=goal, start=start,
                traj=traj, subg=subg, subtitle=f"({a.regime}, sigma={a.sigma})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-dump", default=None, help="plot from eval_ablation --dump JSON (paper envs)")
    p.add_argument("--load", default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--env", default="PointMaze")
    p.add_argument("--regime", choices=["e1a", "e1c"], default="e1c")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--pct", type=float, default=25)
    p.add_argument("--worm-frac", type=float, default=0.5,
                   help="flag an edge as wormhole if noisy dist < worm_frac*clean (raise for mild bias)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj", action="store_true")
    p.add_argument("--out", default="logs/exp_suite/viz_graph_noise.png")
    a = p.parse_args()
    data = from_dump(a.from_dump, a.pct, a.worm_frac) if a.from_dump else from_checkpoint(a)
    render(out=a.out, **data)


if __name__ == "__main__":
    main()
