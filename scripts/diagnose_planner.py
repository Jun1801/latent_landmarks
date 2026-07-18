#!/usr/bin/env python3
"""Diagnose whether the L3P planner routes through landmarks or just goes direct.

Loads a checkpoint and, over several long-horizon eval episodes, reports:
  * D(s, pi(s,g), g): the critic's DIRECT state->goal distance estimate at start
    (if this is underestimated << the true path length, the direct option wins).
  * graph d_{c->g}: how many landmark->goal graph distances are finite (reachable).
  * planner behavior: how many sub-goal choices were landmarks vs the direct goal.
  * per-episode success.

Hypothesis under test: with a short hindsight range the critic underestimates
far-goal distance, so the un-clipped direct-to-goal option in Eq. 7 wins and the
planner never routes through landmarks -> fails when the flat policy can't reach.
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


def diagnose_episode(trainer, env):
    env.set_eval(True)
    obs = env.reset()
    goal = obs["desired_goal"].astype(np.float32)
    trainer.planner.reset(goal)

    # direct state->goal distance estimate at start
    d_direct = float(trainer.agent.distance_after_action(obs["observation"], goal[None, :])[0])
    dcg = trainer.planner.d_c2g
    n_land = trainer.planner.n_landmarks
    finite_edges = int(np.sum(dcg[:n_land] > -1e5)) if dcg is not None else 0

    chosen_landmark, chosen_direct = 0, 0
    success = 0.0
    for _ in range(env.max_episode_steps):
        a = trainer.planner.act(obs["observation"])
        idx = trainer.planner.subg_idx
        if idx is not None:
            if idx < n_land:
                chosen_landmark += 1
            else:
                chosen_direct += 1
        obs, _, done, info = env.step(a)
        success = max(success, info["is_success"])
        if success > 0:
            break
    return dict(d_direct=d_direct, finite_edges=finite_edges, n_land=n_land,
                chosen_landmark=chosen_landmark, chosen_direct=chosen_direct,
                success=success)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", required=True)
    p.add_argument("--maze-size", type=int, default=13)
    p.add_argument("--hindsight", type=int, default=25)
    p.add_argument("--episodes", type=int, default=15)
    args = p.parse_args()

    cfg = get_config("PointMaze", maze_size=args.maze_size, hindsight_range=args.hindsight)
    venv = make_vec_env(cfg, 1, 0)
    trainer = L3PTrainer(venv, cfg)
    trainer.load(args.load)
    env = venv.envs[0]

    # true path length (BFS) for reference
    from collections import deque
    m = env.maze; H, W = m.shape
    def bfs(a, b):
        q = deque([(a, 0)]); seen = {a}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == b: return d
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and m[nr, nc] == 0 and (nr, nc) not in seen:
                    seen.add((nr, nc)); q.append(((nr, nc), d + 1))
        return -1
    true_path = bfs(env._start_cell, env._goal_cell)

    rows = [diagnose_episode(trainer, env) for _ in range(args.episodes)]
    d_direct = np.mean([r["d_direct"] for r in rows])
    land = np.mean([r["chosen_landmark"] for r in rows])
    direct = np.mean([r["chosen_direct"] for r in rows])
    fin = np.mean([r["finite_edges"] for r in rows])
    sr = np.mean([r["success"] > 0 for r in rows])

    print(f"\n=== PLANNER DIAGNOSIS ({args.episodes} eval episodes) ===")
    print(f"true path length (corner-to-corner): ~{true_path} steps")
    print(f"critic D(s, pi, goal) at start      : {d_direct:5.1f}  "
          f"(if << {true_path}, it UNDERESTIMATES the far goal)")
    print(f"landmarks (N)                        : {rows[0]['n_land']}")
    print(f"graph edges landmark->goal that are FINITE (reachable): {fin:.1f}/{rows[0]['n_land']}")
    print(f"planner steps choosing a LANDMARK sub-goal : {land:5.1f}")
    print(f"planner steps choosing DIRECT-to-goal      : {direct:5.1f}")
    print(f"eval success rate                    : {sr:.2f}")
    print("\nInterpretation:")
    if direct > land and sr < 0.5:
        print("  -> planner mostly goes DIRECT and fails: confirms the un-clipped")
        print("     direct-to-goal option wins. Fix: apply d_max to the direct option.")
    elif land > direct and sr < 0.5:
        print("  -> planner DOES route through landmarks but still fails: the issue is")
        print("     graph accuracy / low-level execution, not the direct-option bug.")
    else:
        print("  -> planner routes and succeeds: mechanism healthy at this checkpoint.")


if __name__ == "__main__":
    main()
