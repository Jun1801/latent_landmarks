"""MuJoCo environment scaffold (AntMaze / Fetch tasks) — paper Section 5.

These environments reproduce the paper's harder benchmarks but require MuJoCo,
which is NOT installed by default in this repo (heavy/fragile, macOS ARM in
particular). Imports are guarded so the package works without them; a clear,
actionable message is raised only if a MuJoCo env is actually requested.

The per-environment settings (gym id, horizons, curriculum, distractor) live in
`l3p.config.ENV_SPECS`; the Appendix-E hyper-parameters live in `l3p.config`.
This file only holds the thin adapter that maps a gym GoalEnv (dict obs with
observation/achieved_goal/desired_goal, sparse compute_reward) onto the
attribute/method surface L3P expects (obs_dim, goal_dim, act_dim, max_action,
max_episode_steps; reset, step, compute_reward, set_eval). Everything else in
L3P is env-agnostic, so no other module changes between environments.

Two install paths (see docs/ENVIRONMENTS.md for the full infrastructure guide):
  * paper-faithful (Linux x86_64):  gym==0.13.1 + mujoco-py<2.1 + MuJoCo 2.0
  * modern (also macOS ARM):        mujoco>=3 + gymnasium-robotics
The loader below prefers gymnasium-robotics, then falls back to legacy gym, so
either stack works once the corresponding env ids are registered.
"""

from __future__ import annotations

import numpy as np

from l3p.config import env_spec, resolve_env, list_envs

_INSTALL_MSG = (
    "The MuJoCo environments ({env}) require a physics backend that is not "
    "installed.\n"
    "  Paper-faithful (Linux x86_64):\n"
    "      pip install gym==0.13.1 \"mujoco-py<2.1,>=2.0\"   # + MuJoCo 2.0 binaries\n"
    "  Modern (works on macOS ARM / Linux):\n"
    "      pip install \"mujoco>=3\" gymnasium-robotics\n"
    "See docs/ENVIRONMENTS.md for the full setup + which env ids to register.\n"
    "PointMaze runs with no MuJoCo."
)


# Modern (gymnasium-robotics) env ids for the paper environments that ship with
# the library. The custom paper envs (AntMaze / Box-Distractor / Place-Inside-Box)
# are not on PyPI and need porting — they map to None.
_GYMNASIUM_IDS = {
    "FetchPickAndPlace": "FetchPickAndPlace-v4",
    "PointMazeMuJoCo": "PointMaze_Large-v3",  # real point-in-maze (modern MuJoCo)
    "AntMaze": "AntMaze_Medium-v5",     # goal-conditioned Ant in a medium maze (tractable on CPU)
    "AntMazeUMaze": "AntMaze_UMaze-v5",
    "AntMazeMedium": "AntMaze_Medium-v5",
    "AntMazeLarge": "AntMaze_Large-v5",
    "AntMazeLargeDiverseG": "AntMaze_Large_Diverse_G-v5",
    "AntMazeLargeDiverseGR": "AntMaze_Large_Diverse_GR-v5",
    # Box-Distractor / Place-Inside-Box are custom Fetch variants, built locally
    # (see l3p/envs/fetch_variants.py) rather than via a gym id.
    "BoxDistractorPickAndPlace": "__local_box_distractor__",
    "PlaceInsideBox": "__local_place_inside_box__",
}

# Maze envs need continuing_task=True + reset_target=False so each episode runs a
# fixed horizon with a fixed goal (matches the fixed-length HER replay buffer).
_MAZE_KWARGS = dict(continuing_task=True, reset_target=False)
_MAZE_FAMILIES = {"AntMaze", "PointMaze"}


def _load_backend():
    """Return (module, kind) where kind is 'gym' or 'gymnasium'; None if absent."""
    # Prefer the maintained gymnasium-robotics stack. Kaggle often has the old
    # unmaintained `gym` package preinstalled; if we import gym first, modern ids
    # such as PointMaze_Large-v3 are not registered and env construction fails.
    try:
        import gymnasium
        import gymnasium_robotics
        gymnasium.register_envs(gymnasium_robotics)   # register Fetch/etc. ids
        return gymnasium, "gymnasium"
    except ImportError:
        pass
    try:
        import gym  # noqa: F401
        return gym, "gym"
    except ImportError:
        return None, None


def make_mujoco_env(cfg, seed: int):  # pragma: no cover - requires MuJoCo
    canonical = resolve_env(cfg.env_name)
    spec = env_spec(cfg.env_name)
    if not spec.get("needs_mujoco", False):
        raise ValueError(f"{canonical} is not a MuJoCo env; use make_env instead.")

    backend, kind = _load_backend()
    if backend is None:
        raise ImportError(_INSTALL_MSG.format(env=canonical))

    env_id = _GYMNASIUM_IDS.get(canonical) if kind == "gymnasium" else spec["gym_id"]
    if env_id is None:
        raise NotImplementedError(
            f"{canonical} needs a custom env implementation (see its ENV_SPECS "
            f"note) that is not bundled here.\n"
            "docs/ENVIRONMENTS.md explains how to port the custom paper envs.")

    # Locally-built custom Fetch variants (box distractor / place-inside-box).
    if isinstance(env_id, str) and env_id.startswith("__local_"):
        from l3p.envs.fetch_variants import make_fetch_variant
        env = make_fetch_variant(canonical, cfg, seed)
        return GymGoalEnvWrapper(env, cfg, "gymnasium")

    # TimeLimit set to the LARGER of train/test horizon so evaluation can run
    # the longer test horizon; the collection loop caps training episodes at the
    # (shorter) train horizon itself.
    kwargs = dict(max_episode_steps=max(cfg.max_episode_steps, cfg.test_episode_steps))
    if spec.get("family") in _MAZE_FAMILIES and kind == "gymnasium":
        kwargs.update(_MAZE_KWARGS)
    env = backend.make(env_id, **kwargs)
    try:
        env.reset(seed=seed)           # gymnasium style
    except TypeError:
        env.seed(seed)                 # gym<=0.13 style
    return GymGoalEnvWrapper(env, cfg, kind)


class GymGoalEnvWrapper:  # pragma: no cover - requires MuJoCo
    """Adapts a gym / gymnasium GoalEnv to the surface L3P uses."""

    def __init__(self, env, cfg, kind="gym"):
        self.env = env
        self.cfg = cfg
        self.kind = kind
        obs = self._unpack(env.reset())
        self.obs_dim = obs["observation"].shape[0]
        self.goal_dim = obs["desired_goal"].shape[0]
        self.act_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])
        self.max_episode_steps = cfg.max_episode_steps
        self.eval_mode = False
        # Different envs use different sparse-reward conventions: Fetch gives
        # {-1 (fail), 0 (success)} while the gymnasium maze envs give
        # {0 (fail), 1 (success)}. L3P's distance parameterization assumes the
        # paper convention R = -1[goal not reached]. Probe success/fail values
        # once and normalize compute_reward to {-1, 0}.
        dg = obs["desired_goal"]
        succ_r = float(np.asarray(env.unwrapped.compute_reward(dg, dg, {})).reshape(-1)[0])
        fail_r = float(np.asarray(env.unwrapped.compute_reward(dg + 1e3, dg, {})).reshape(-1)[0])
        self._reward_mid = 0.5 * (succ_r + fail_r)

    @staticmethod
    def _unpack(ret):
        # gymnasium reset returns (obs, info); gym returns obs
        return ret[0] if isinstance(ret, tuple) else ret

    def set_eval(self, flag: bool) -> None:
        self.eval_mode = flag
        # At test time the paper uses a longer horizon (ENV_SPECS test_horizon).
        self.max_episode_steps = (self.cfg.test_episode_steps if flag
                                  else self.cfg.max_episode_steps)

    def reset(self):
        rng = getattr(self, "rng", None)
        if rng is not None:
            seed = int(rng.integers(0, 2**31 - 1))
            try:
                return self._unpack(self.env.reset(seed=seed))
            except TypeError:
                self.env.seed(seed)
        return self._unpack(self.env.reset())

    def step(self, action):
        ret = self.env.step(action)
        if len(ret) == 5:                       # gymnasium: obs, rew, term, trunc, info
            obs, reward, term, trunc, info = ret
            return obs, reward, term or trunc, info
        return ret                               # gym: obs, rew, done, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        # the sparse GoalEnv reward lives on the unwrapped env; normalize to the
        # paper convention R = -1 (goal not reached) / 0 (reached).
        r = np.asarray(self.env.unwrapped.compute_reward(achieved_goal, desired_goal, info))
        return np.where(r > self._reward_mid, 0.0, -1.0).astype(np.float32)
