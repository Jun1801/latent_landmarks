# Environments & Infrastructure Guide

L³P (this repo) is fully **environment-agnostic**: the algorithm, planner and
training loop never change between tasks. Only the environment behind the
GoalEnv interface changes. This document lists the five environments from the
paper (Section 5), their settings, and **exactly what infrastructure is needed
to run each one**.

## The five environments

| Env (canonical name) | Family (Appendix-E column) | Backend | Runnable here? |
|---|---|---|---|
| `PointMaze` | Point-Maze | pure NumPy | ✅ yes, out of the box |
| `AntMaze` | Ant-Maze | MuJoCo (custom maze env) | ⚠️ needs MuJoCo + custom env |
| `FetchPickAndPlace` | Fetch | MuJoCo (gym-robotics) | ⚠️ needs MuJoCo |
| `BoxDistractorPickAndPlace` | Fetch | MuJoCo (custom env) | ⚠️ needs MuJoCo + custom env |
| `PlaceInsideBox` | Fetch | MuJoCo (custom env) | ⚠️ needs MuJoCo + custom env |

All settings live in `l3p/config.py`:
- `_PER_ENV` — the Appendix-E hyper-parameters, one column per family.
- `ENV_SPECS` — per-env structural settings: gym id, train/test horizon,
  curriculum ratio (Place-Inside-Box), distractor flag (Box-Distractor).

Inspect them:
```python
from l3p.config import get_config, list_envs, env_spec
for name in list_envs():
    print(name, get_config(name), env_spec(name))
```

## Why PointMaze runs but the others don't (yet)

`PointMaze` is a dependency-free kinematic maze (`l3p/envs/point_maze.py`). The
other four need a **physics simulator (MuJoCo)** to step the Ant/Fetch dynamics.
Two of the pieces are needed:

1. **A physics backend** — MuJoCo, via one of two Python stacks (below).
2. **The environment code** — only `FetchPickAndPlace-v1` ships with
   gym/gymnasium-robotics. `AntMaze`, `BoxDistractorPickAndPlace` and
   `PlaceInsideBox` are **custom environments from the paper** (in the original
   repo under `goal_env/mujoco/`: `ant_maze_env.py`, `maze_env.py`,
   `create_maze_env.py`, plus modified Fetch XMLs). They must be dropped in and
   registered; they are not on PyPI.

`l3p/envs/mujoco.py` is the adapter: once a backend + env id exist, it wraps the
gym/gymnasium GoalEnv onto L³P's interface (it already handles both the old gym
4-tuple `step` and the new gymnasium 5-tuple, and the longer test horizon).

## Install path A — paper-faithful (Linux x86_64 recommended)

Matches the versions in the paper's repo. **Easiest on Ubuntu x86_64; does not
build cleanly on macOS ARM.**

```bash
# system: MuJoCo 2.0 binaries in ~/.mujoco/mujoco200, plus a license key (now free)
sudo apt-get install -y libgl1-mesa-dev libosmesa6-dev patchelf gcc
pip install gym==0.13.1 "mujoco-py<2.1,>=2.0" numpy torch
# then copy the paper's goal_env/mujoco/* into l3p/envs/ and register the ids
```

`mujoco-py` compiles a C extension on first import — needs a working compiler
and the MuJoCo 2.0 shared library on the linker path.

## Install path B — modern (works on macOS ARM and Linux)

Uses DeepMind's official `mujoco` (pip wheels for ARM/Linux/x86) plus
`gymnasium-robotics` (ships the Fetch envs). This gets **FetchPickAndPlace**
running quickly; AntMaze / the two custom Fetch variants still need their env
code ported to the gymnasium API.

```bash
pip install "mujoco>=3" gymnasium-robotics torch numpy
```

`l3p/envs/mujoco.py` auto-detects this stack (tries `gym`, then
`gymnasium` + `gymnasium_robotics`).

## Hardware / compute expectations

| | Requirement |
|---|---|
| CPU | Multi-core helps; MuJoCo stepping and graph search are CPU-bound. |
| GPU | Optional. Speeds up the (small) neural nets, but the bottleneck is env sim + the sequential planner, so gains are modest. |
| RAM | A few GB; the centralized replay stores whole episodes. |
| Workers | Paper uses 1 / 3 / 12 parallel workers for Point / Ant / Fetch (Appendix E). More workers → more throughput into the shared replay. |
| Wall-clock | Paper trains to ~1.6M (Point), ~3M (Ant), ~1M (Fetch) steps. On a good multi-core box: hours; on a single CPU: substantially longer. |

## Env-specific settings already encoded

- **AntMaze** — `grad_norm_clip=15.0` (Appendix D), γ=0.98, train horizon 200 /
  test horizon 500 (generalize to the longest unseen path).
- **FetchPickAndPlace** — inputs normalized by running mean/std (Appendix D),
  γ=0.99, horizon 50, N=80 landmarks, 6000 warm-up trajectories.
- **BoxDistractorPickAndPlace** — `has_distractor=True`: a box distractor sits on
  the table; the arm must pick/place while avoiding it.
- **PlaceInsideBox** — `place_inside_box_ratio=0.2`: curriculum with 80% regular
  pick-and-place goals and 20% inside-the-box goals (Section 5.3).

These flags are read by the env layer; the RL side needs no changes.

## Launching (once a backend + env code are in place)

```bash
# any of the canonical names or aliases (ant / fetch / distractor / inside_box)
bash scripts/train_mujoco.sh antmaze
# or directly:
python -c "from l3p.config import get_config; from l3p.envs import make_vec_env; \
from l3p.trainer import L3PTrainer; \
cfg=get_config('AntMaze'); tr=L3PTrainer(make_vec_env(cfg,cfg.n_workers,cfg.seed),cfg); tr.train()"
```

If MuJoCo is absent you get a clear, actionable ImportError pointing back here —
never a crash.

## Summary

- **Runs today, no setup:** `PointMaze` (pure NumPy).
- **Runs after `pip install "mujoco>=3" gymnasium-robotics`:** `FetchPickAndPlace`.
- **Needs backend + porting the paper's custom env code:** `AntMaze`,
  `BoxDistractorPickAndPlace`, `PlaceInsideBox`.

All five already have faithful hyper-parameters and structural settings wired in;
the remaining work for the MuJoCo four is purely providing the simulator and the
custom env definitions, not touching the L³P algorithm.
