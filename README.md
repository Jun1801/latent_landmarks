# L³P — World Model as a Graph: Learning Latent Landmarks for Planning

A faithful, from-scratch PyTorch reimplementation of

> **World Model as a Graph: Learning Latent Landmarks for Planning**
> Lunjun Zhang, Ge Yang, Bradly Stadie — ICML 2021 ([arXiv:2011.12491](https://arxiv.org/abs/2011.12491))

L³P learns a **graph-structured world model**: the nodes are a small set of
*latent landmarks* scattered across goal space by reachability, and the edges
are reachability (distance) estimates distilled from Q-functions. Planning is a
graph search over this graph plus an online planner that exploits temporal
abstraction. Unlike step-by-step model-based RL (which diverges over long
horizons), this graph is a sparse multi-step transition model that supports
temporally extended reasoning.

This repo runs **end-to-end out of the box** on a self-contained pure-NumPy
`PointMaze-Hard` environment (only `torch` + `numpy` required), and includes a
documented scaffold for the paper's MuJoCo `AntMaze`/`Fetch` benchmarks.

---

## Quick start

```bash
pip install -r requirements.txt          # torch + numpy

# unit tests (fast, synthetic data)
python tests/test_modules.py

# short smoke run: full collect -> train -> plan -> eval pipeline (~1 min CPU)
python scripts/train_pointmaze.py --short

# full run
python scripts/train_pointmaze.py --steps 500000

# evaluate a checkpoint on the long-horizon test + inspect landmarks
python scripts/eval.py --load l3p_pointmaze.pt --episodes 50 --show-landmarks
```

---

## PN-LMCGS safety workflow (optional)

The baseline remains unchanged: PN-LMCGS is disabled unless `--pn-lmcgs` is
selected. Train and evaluate a PN checkpoint with:

```bash
python scripts/train_pointmaze.py --pn-lmcgs --steps 500000 --save pn_pointmaze.pt
python scripts/eval.py --pn-lmcgs --load pn_pointmaze.pt --episodes 50
```

PointMaze also has an opt-in two-route task with a short hazardous route and a
longer safe detour. It is PointMaze-only; `--hazardous-pointmaze` automatically
enables PN-LMCGS, and evaluation must use the same structural geometry:

```bash
python scripts/train_pointmaze.py --pn-lmcgs --hazardous-pointmaze --steps 500000 --save pn_hazard.pt
python scripts/eval.py --pn-lmcgs --hazardous-pointmaze --load pn_hazard.pt --episodes 50
```

Safety input precedence is: separate Safety-Gym cost, then `info["safety_cost"]`,
then `info["cost"]`, then an explicitly enabled collision adapter, otherwise
zero. Collision inference is disabled by default: ordinary contact is not
treated as unsafe. For an environment that needs an adapter, explicitly set
`pn_collision_cost_enabled=True`, `pn_collision_cost_info_key`, and optionally
`pn_collision_cost_unsafe_value` in its `Config`.

PN checkpoints use v3 state and retain legacy checkpoint compatibility. The
trainer progresses through `WARMUP`, `LANDMARKS`, `MACRO_BOOTSTRAP`, and `JOINT`
gates; PN runs report `safe_success_rate` as the primary safety-aware metric
while preserving the float goal-success output for existing scripts.

---

## How the code maps to the paper

Every component maps to a specific equation / algorithm from the paper.

| Paper | Where | What |
|-------|-------|------|
| **Eq. 3** — `Q = -(1-γ^D)/(1-γ)` | `l3p/models/networks.py` (`Critic`) | Distance-parameterized critic; `D(s,a,g) ≥ 0` via softplus, `Q` recovered analytically. |
| **Eq. 1** — TD loss | `l3p/agent/ddpg.py` (`update_critic`) | Goal-conditioned Q-learning with target network + binary reward. |
| **Eq. 4** — value regression | `l3p/agent/ddpg.py` (`update_value`) | `V(g1,g2)` regressed toward the distance `D` (note the s_t vs s_{t+1} asymmetry). |
| actor loss | `l3p/agent/ddpg.py` (`update_actor`) | Maximize `Q(s,π(s,g),g)` + action-L2 penalty. |
| **HER** | `l3p/replay/her_buffer.py` | Future-goal relabelling with a shortened *hindsight range* (Appendix D/E). |
| **Eq. 2** — reachability AE | `l3p/models/autoencoder.py`, `l3p/losses.py` | `L_rec + λ·L_latent`; latent L2 distance ≈ reachability. |
| **Eq. 5** — latent landmarks | `l3p/models/landmarks.py` (`LatentLandmarks`) | Mixture-of-Gaussians centroids + ELBO (uniform prior). |
| **Algorithm 2** — GLS | `l3p/models/landmarks.py` (`greedy_latent_sparsification`) | Farthest-point sub-sampling for diverse clustering batches / random landmarks. |
| **Eq. 6, Eq. 8** — soft Floyd | `l3p/planning/graph_search.py` | Weight matrix `W`, `d_max` masking (Appendix B), soft relaxation with temperature β. |
| **Algorithm 1, Eq. 7** — online planner | `l3p/planning/planner.py` | Sub-goal selection, K-step commitment, previous-landmark removal. |
| **Algorithm 3** — training loop | `l3p/trainer.py` | Collect (with planner) → gradient steps on every module → repeat. |
| **Appendix E** — hyper-parameters | `l3p/config.py` | Common + per-env (Point/Ant/Fetch) tables verbatim. |

### The algorithm in one paragraph

The low-level agent is DDPG+HER, but its critic is parameterized through a
*distance* `D(s,a,g)` (expected steps-to-goal), from which `Q` follows by Eq. 3.
A separate value function `V(g1,g2)` estimates goal-to-goal distances and is
regressed toward `D`. An auto-encoder embeds goals into a latent space
*constrained* so that latent L2 distance matches reachability (Eq. 2). We then
fit a mixture of Gaussians in that latent space (Eq. 5); the decoded centroids
are the **latent landmarks**. At planning time we build a graph whose nodes are
the landmarks plus the current goal and whose edge weights are negated `V`
distances, run a **soft Floyd** relaxation to get each landmark's distance to
the goal, and use an **online planner** (Algorithm 1) that picks the sub-goal
maximizing `d_{s→c} + d_{c→g}`, commits to it for `K` steps (temporal
abstraction, no re-planning every step), and drops the previous landmark on
re-plan to avoid getting stuck.

---

## Repository layout

```
l3p/
  config.py                 Appendix-E hyper-parameters (common + per-env)
  models/
    networks.py             Actor π(s,g); Critic→D(s,a,g)→Q; Value V(g1,g2)
    autoencoder.py          reachability-constrained auto-encoder f_E / f_D
    landmarks.py            MoG centroids + ELBO (Eq. 5) + GLS (Algorithm 2)
  agent/
    ddpg.py                 distance-parameterized DDPG; Eq. 1 / Eq. 4 / actor
    normalizer.py           running mean/std input normalization
  replay/her_buffer.py      episodic HER buffer (future relabelling, hindsight range)
  planning/
    graph_search.py         W (Eq. 6), d_max masking, soft Floyd (Eq. 8)
    planner.py              online planner (Algorithm 1, Eq. 7)
  losses.py                 auto-encoder losses (Eq. 2)
  trainer.py                overall training loop (Algorithm 3)
  envs/
    point_maze.py           pure-NumPy PointMaze-Hard GoalEnv (runs everywhere)
    vec_env.py              in-process vectorized workers → centralized replay
    mujoco.py               guarded AntMaze/Fetch wrappers (optional)
scripts/
  train_pointmaze.py        train on PointMaze-Hard  (--short for a smoke run)
  eval.py                   evaluate + visualize landmark coordinates
  train_mujoco.sh           launchers for the MuJoCo benchmarks (optional)
tests/test_modules.py       unit tests for every component (synthetic data)
```

---

## The PointMaze-Hard environment

`l3p/envs/point_maze.py` is a dependency-free 2-D point-mass maze with a width-1
serpentine corridor. It follows the gym GoalEnv convention (dict observations
with `observation` / `achieved_goal` / `desired_goal`, sparse `compute_reward`).
During **training** the start and goal are uniform over free space; during
**evaluation** (`set_eval(True)`) the agent always starts at one end and must
traverse the entire maze to a goal at the far end — the long-horizon
generalization test of the paper (Figure 5), with no prior knowledge of the map.

A ~30k-step smoke run already shows the intended behavior: the low-level policy
learns local goal-reaching, and the decoded landmarks scatter across the free
corridors (their x-coordinates cluster at the corridor columns). Reproducing the
paper's asymptotic success rate needs the full multi-million-step budget.

---

## Running the MuJoCo environments (optional, not verified here)

The paper's harder benchmarks — `AntMaze`, `FetchPickAndPlace`,
`Box-Distractor-PickAndPlace`, `Place-Inside-Box` — require MuJoCo, which is
heavy/fragile to install (especially on macOS ARM) and is **not** part of the
default setup. The algorithm code is fully env-agnostic; only a wrapper is
needed. To try them:

```bash
pip install gym==0.13.1 "mujoco-py<2.1,>=2.0"   # + MuJoCo 2.0 system setup
bash scripts/train_mujoco.sh antmaze            # or: fetch | distractor | inside_box
```

`l3p/envs/mujoco.py` guards these imports: without MuJoCo installed, requesting
one of these envs raises a clear install message rather than crashing, and
PointMaze continues to work.

All five paper environments (`PointMaze`, `AntMaze`, `FetchPickAndPlace`,
`BoxDistractorPickAndPlace`, `PlaceInsideBox`) have faithful hyper-parameters
(Appendix E) and structural settings wired into `l3p/config.py` (`ENV_SPECS`).
**See [`docs/ENVIRONMENTS.md`](docs/ENVIRONMENTS.md)** for the full per-env
settings and the exact infrastructure needed to run each — including the modern
`mujoco>=3` + `gymnasium-robotics` path that also works on macOS ARM.

---

## Notes on fidelity

- **Multi-worker / centralized replay** (Appendix D) is implemented as
  in-process vectorized workers sharing one HER buffer. In a single process the
  network parameters are shared, so "averaging gradients across workers" is
  automatic; true MPI parallelism is an optional extension.
- **Soft vs. hard Floyd**: we use the soft relaxation (Eq. 8) the paper found
  more robust; setting `beta → 0` recovers hard Floyd.
- Full-scale reproduction of the training curves is out of scope for a
  CPU/no-MuJoCo setup; the goal here is a faithful, correct, runnable
  implementation of the method.
