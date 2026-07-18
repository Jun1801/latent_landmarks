"""Episodic replay buffer with Hindsight Experience Replay (Andrychowicz et al. 2017).

Stores whole episodes so that goals can be relabelled with *future* achieved
goals at sample time. Two paper-specific details:

  * ``her_ratio`` fraction of sampled transitions get their desired goal
    relabelled (paper uses 0.85).
  * relabelling is restricted to a shortened *hindsight range* (Appendix D/E):
    since L3P decomposes long horizons into short sub-goals, the low-level agent
    only needs to learn to reach nearby goals, so future goals are drawn from a
    limited window rather than the whole remaining episode.

Every sampled transition also carries a ``future_ag`` (a future achieved goal),
used as the target goal g2 for the value-function regression (Eq. 4).
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


class HERReplayBuffer:
    def __init__(self, size_episodes: int, horizon: int, obs_dim: int, goal_dim: int,
                 act_dim: int, compute_reward: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 her_ratio: float = 0.85, hindsight_range: int = 80):
        self.size = size_episodes
        self.T = horizon
        self.compute_reward = compute_reward
        self.her_ratio = her_ratio
        self.hindsight_range = hindsight_range

        self.obs = np.zeros((size_episodes, horizon + 1, obs_dim), dtype=np.float32)
        self.ag = np.zeros((size_episodes, horizon + 1, goal_dim), dtype=np.float32)
        self.g = np.zeros((size_episodes, horizon, goal_dim), dtype=np.float32)
        self.act = np.zeros((size_episodes, horizon, act_dim), dtype=np.float32)

        self.ptr = 0
        self.n_episodes = 0

    def __len__(self) -> int:
        return self.n_episodes

    @property
    def n_transitions(self) -> int:
        return self.n_episodes * self.T

    def store_episode(self, ep: Dict[str, np.ndarray]) -> None:
        i = self.ptr
        self.obs[i] = ep["obs"]
        self.ag[i] = ep["ag"]
        self.g[i] = ep["g"]
        self.act[i] = ep["act"]
        self.ptr = (self.ptr + 1) % self.size
        self.n_episodes = min(self.n_episodes + 1, self.size)

    # ------------------------------------------------------------------ sampling
    def _future_index(self, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample a future timestep in (t, min(t+range, T)] for each t."""
        high = np.minimum(t + self.hindsight_range, self.T)
        # uniform integer in (t, high]; +1 so we never pick t itself
        offset = (rng.random(t.shape) * (high - t)).astype(np.int64) + 1
        return np.minimum(t + offset, self.T)

    def sample(self, batch_size: int, rng: np.random.Generator = None) -> Dict[str, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        ep_idx = rng.integers(0, self.n_episodes, size=batch_size)
        t = rng.integers(0, self.T, size=batch_size)

        obs = self.obs[ep_idx, t]
        next_obs = self.obs[ep_idx, t + 1]
        ag = self.ag[ep_idx, t]
        next_ag = self.ag[ep_idx, t + 1]
        act = self.act[ep_idx, t]
        g = self.g[ep_idx, t].copy()

        # HER relabelling: replace desired goal with a future achieved goal.
        her_mask = rng.random(batch_size) < self.her_ratio
        fut_t = self._future_index(t, rng)
        future_ag = self.ag[ep_idx, fut_t]
        g[her_mask] = future_ag[her_mask]

        # Independent future achieved goal for the value regression target (Eq. 4).
        fut_t2 = self._future_index(t, rng)
        future_ag_v = self.ag[ep_idx, fut_t2]

        reward = self.compute_reward(next_ag, g).astype(np.float32)

        return dict(obs=obs, next_obs=next_obs, ag=ag, next_ag=next_ag, act=act,
                    g=g, future_ag=future_ag_v, reward=reward)

    def sample_achieved_goals(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Sample a flat batch of achieved goals (used by the AE and GLS)."""
        if rng is None:
            rng = np.random.default_rng()
        ep_idx = rng.integers(0, self.n_episodes, size=n)
        t = rng.integers(0, self.T + 1, size=n)
        return self.ag[ep_idx, t]
