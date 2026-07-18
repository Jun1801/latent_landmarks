"""Lightweight in-process vectorized environments.

The paper uses multiple parallel workers that feed a *centralized* replay buffer
(Appendix D: "a centralized replay for all parallel workers is significantly
more sample efficient than separate replays"). Because L3P's planner is stateful
per episode, we collect episodes worker-by-worker into a single shared buffer
rather than stepping all workers in lockstep. In a single process the network
parameters are shared, so "averaging gradients across workers" is automatic;
true MPI parallelism is left as an optional extension (see README).
"""

from __future__ import annotations

from typing import Callable, List

from l3p.envs.point_maze import PointMazeEnv


def make_env(cfg, seed: int) -> object:
    """Factory that maps a config's env_name to an environment instance.

    MuJoCo envs (needs_mujoco=True in ENV_SPECS) go through the guarded MuJoCo
    wrapper; everything else uses the pure-NumPy PointMaze.
    """
    from l3p.config import env_spec
    if env_spec(cfg.env_name).get("needs_mujoco", False):
        from l3p.envs.mujoco import make_mujoco_env
        return make_mujoco_env(cfg, seed)
    return PointMazeEnv(cfg=cfg, seed=seed)


def make_vec_env(cfg, n_workers: int, seed: int = 0) -> "VecEnv":
    envs = [make_env(cfg, seed + i) for i in range(n_workers)]
    return VecEnv(envs)


class VecEnv:
    def __init__(self, envs: List[object]):
        self.envs = envs
        self.n = len(envs)
        e0 = envs[0]
        self.obs_dim = e0.obs_dim
        self.goal_dim = e0.goal_dim
        self.act_dim = e0.act_dim
        self.max_action = e0.max_action
        self.max_episode_steps = e0.max_episode_steps

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return self.envs[i]

    def set_eval(self, flag: bool) -> None:
        for e in self.envs:
            e.set_eval(flag)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.envs[0].compute_reward(achieved_goal, desired_goal, info)
