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

from typing import Callable, Dict, Optional

import numpy as np


class HERReplayBuffer:
    """Fixed-capacity episodic HER storage with per-episode valid lengths."""

    REASON_OTHER = 0
    REASON_GOAL = 1
    REASON_TIMEOUT = 2
    REASON_VIOLATION = 3
    _REASON_CODES = {
        "other": REASON_OTHER,
        "goal": REASON_GOAL,
        "timeout": REASON_TIMEOUT,
        "violation": REASON_VIOLATION,
    }
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
        self.cmd_g = np.zeros((size_episodes, horizon, goal_dim), dtype=np.float32)
        self.act = np.zeros((size_episodes, horizon, act_dim), dtype=np.float32)
        self.cost = np.zeros((size_episodes, horizon), dtype=np.float32)
        self.violation = np.zeros((size_episodes, horizon), dtype=bool)
        self.goal_reached = np.zeros((size_episodes, horizon), dtype=bool)
        self.terminated = np.zeros((size_episodes, horizon), dtype=bool)
        self.truncated = np.zeros((size_episodes, horizon), dtype=bool)
        self.termination_reason = np.full(
            (size_episodes, horizon), self.REASON_OTHER, dtype=np.int8
        )
        self.valid = np.zeros((size_episodes, horizon), dtype=bool)
        self.episode_lengths = np.zeros(size_episodes, dtype=np.int64)

        self.ptr = 0
        self.n_episodes = 0

    def __len__(self) -> int:
        return self.n_episodes

    @property
    def n_transitions(self) -> int:
        return int(self.episode_lengths.sum())

    def store_episode(self, ep: Dict[str, np.ndarray]) -> None:
        length = self._episode_length(ep)
        i = self.ptr
        self._clear_slot(i)
        self.obs[i, :length + 1] = self._field(
            ep, "obs", length + 1, self.obs.shape[2:], allow_scalar=False, exact=True
        )
        self.ag[i, :length + 1] = self._field(
            ep, "ag", length + 1, self.ag.shape[2:], allow_scalar=False, exact=True
        )
        self.g[i, :length] = self._field(
            ep, "g", length, self.g.shape[2:], allow_scalar=False, exact=True
        )
        self.act[i, :length] = self._field(
            ep, "act", length, self.act.shape[2:], allow_scalar=False, exact=True
        )
        cmd_g_source = ep if "cmd_g" in ep else {"cmd_g": ep["g"]}
        self.cmd_g[i, :length] = self._field(
            cmd_g_source, "cmd_g", length, self.cmd_g.shape[2:], allow_scalar=False, exact=True
        )
        self.cost[i, :length] = self._field(ep, "cost", length, (), default=0.0)
        self.violation[i, :length] = self._field(
            ep, "violation", length, (), default=self.cost[i, :length] > 0
        ).astype(bool)
        self.goal_reached[i, :length] = self._field(ep, "goal_reached", length, (), default=False).astype(bool)
        self.terminated[i, :length] = self._field(ep, "terminated", length, (), default=False).astype(bool)
        self.truncated[i, :length] = self._field(ep, "truncated", length, (), default=False).astype(bool)
        self.termination_reason[i, :length] = self._reason_field(ep, length)
        self.valid[i, :length] = True
        self.episode_lengths[i] = length
        self.ptr = (self.ptr + 1) % self.size
        self.n_episodes = min(self.n_episodes + 1, self.size)

    def _clear_slot(self, i: int) -> None:
        for field in (self.obs, self.ag, self.g, self.cmd_g, self.act, self.cost,
                      self.violation, self.goal_reached, self.terminated,
                      self.truncated, self.valid):
            field[i].fill(0)
        self.termination_reason[i].fill(self.REASON_OTHER)
        self.episode_lengths[i] = 0

    def _episode_length(self, ep: Dict[str, np.ndarray]) -> int:
        if not isinstance(ep, dict):
            raise TypeError("episode must be a dictionary")
        for name in ("obs", "ag", "g", "act"):
            if name not in ep:
                raise ValueError(f"episode missing required field {name!r}")
        explicit = ep.get("length", ep.get("episode_length"))
        if explicit is None:
            lengths = tuple(
                array.shape[0] - offset if array.ndim else -1
                for array, offset in (
                    (np.asarray(ep["g"]), 0),
                    (np.asarray(ep["act"]), 0),
                    (np.asarray(ep["obs"]), 1),
                    (np.asarray(ep["ag"]), 1),
                )
            )
            if len(set(lengths)) != 1:
                raise ValueError("episode shapes imply inconsistent transition lengths; provide length")
            length = lengths[0]
        else:
            if isinstance(explicit, (bool, np.bool_)) or int(explicit) != explicit:
                raise ValueError("episode length must be an integer")
            length = int(explicit)
        if not 0 < length <= self.T:
            raise ValueError(f"episode length must be in [1, {self.T}], got {length}")
        return length

    @staticmethod
    def _field(ep: Dict[str, np.ndarray], name: str, length: int, tail: tuple,
               default=None, allow_scalar: bool = True, exact: bool = False) -> np.ndarray:
        if name not in ep:
            value = default
            if value is None:
                raise ValueError(f"episode missing required field {name!r}")
        else:
            value = ep[name]
        array = np.asarray(value)
        if array.ndim == 0:
            if not allow_scalar:
                raise ValueError(f"episode field {name!r} must have shape ({length}, {tail}), got scalar")
            array = np.full((length,) + tail, array, dtype=array.dtype)
        expected = (length,) + tail
        if (exact and array.shape != expected) or (
            not exact and (array.shape[0] < length or array.shape[1:] != tail)
        ):
            raise ValueError(f"episode field {name!r} must have shape ({length}, {tail}), got {array.shape}")
        return array[:length]

    def _reason_field(self, ep: Dict[str, np.ndarray], length: int) -> np.ndarray:
        raw = ep.get("termination_reason", self.REASON_OTHER)
        values = self._field({"value": raw}, "value", length, ())
        if np.issubdtype(values.dtype, np.bool_):
            raise ValueError("termination_reason must contain integer codes, not booleans")
        if np.issubdtype(values.dtype, np.integer):
            result = values
        elif np.issubdtype(values.dtype, np.floating):
            if not np.all(np.isfinite(values)):
                raise ValueError("termination_reason must contain finite integer codes")
            if not np.all(values == np.trunc(values)):
                raise ValueError("termination_reason must contain integer codes")
            result = values
        elif np.issubdtype(values.dtype, np.number):
            raise ValueError("termination_reason must contain real integer codes")
        else:
            try:
                return np.asarray([self._REASON_CODES[str(x)] for x in values], dtype=np.int8)
            except KeyError as exc:
                raise ValueError(f"unknown termination_reason {exc.args[0]!r}") from exc
        if np.any((result < self.REASON_OTHER) | (result > self.REASON_VIOLATION)):
            raise ValueError("termination_reason contains an unknown numeric code")
        return result.astype(np.int8)

    # ------------------------------------------------------------------ sampling
    def _future_index(self, t: np.ndarray, rng: np.random.Generator,
                      episode_lengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample a future timestep in (t, min(t+range, episode length)]."""
        horizon = self.T if episode_lengths is None else episode_lengths
        high = np.minimum(t + self.hindsight_range, horizon)
        # uniform integer in (t, high]; +1 so we never pick t itself
        offset = (rng.random(t.shape) * (high - t)).astype(np.int64) + 1
        return np.minimum(t + offset, horizon)

    def _sample_indices(self, batch_size: int, rng: np.random.Generator) -> tuple:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        candidates = np.flatnonzero(self.episode_lengths > 0)
        if not len(candidates):
            raise ValueError("cannot sample from an empty replay buffer")
        ep_idx = candidates[rng.integers(0, len(candidates), size=batch_size)]
        lengths = self.episode_lengths[ep_idx]
        t = (rng.random(batch_size) * lengths).astype(np.int64)
        return ep_idx, t, lengths

    def sample(self, batch_size: int, rng: np.random.Generator = None) -> Dict[str, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        ep_idx, t, lengths = self._sample_indices(batch_size, rng)

        obs = self.obs[ep_idx, t]
        next_obs = self.obs[ep_idx, t + 1]
        ag = self.ag[ep_idx, t]
        next_ag = self.ag[ep_idx, t + 1]
        act = self.act[ep_idx, t]
        g = self.g[ep_idx, t].copy()

        # HER relabelling: replace desired goal with a future achieved goal.
        her_mask = rng.random(batch_size) < self.her_ratio
        fut_t = self._future_index(t, rng, lengths)
        future_ag = self.ag[ep_idx, fut_t]
        g[her_mask] = future_ag[her_mask]

        # Independent future achieved goal for the value regression target (Eq. 4).
        fut_t2 = self._future_index(t, rng, lengths)
        future_ag_v = self.ag[ep_idx, fut_t2]

        reward = np.asarray(self.compute_reward(next_ag, g), dtype=np.float32)
        reached = reward == 0
        terminated = self.terminated[ep_idx, t]
        truncated = self.truncated[ep_idx, t]

        return dict(obs=obs, next_obs=next_obs, ag=ag, next_ag=next_ag, act=act,
                    g=g, future_ag=future_ag_v, reward=reward,
                    cmd_g=self.cmd_g[ep_idx, t], cost=self.cost[ep_idx, t],
                    violation=self.violation[ep_idx, t], goal_reached=reached,
                    terminated=terminated, truncated=truncated,
                    termination_reason=self.termination_reason[ep_idx, t],
                    stop=terminated | truncated | reached)

    def sample_achieved_goals(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Sample a flat batch of achieved goals (used by the AE and GLS)."""
        if rng is None:
            rng = np.random.default_rng()
        if n < 1:
            raise ValueError("n must be positive")
        candidates = np.flatnonzero(self.episode_lengths > 0)
        if not len(candidates):
            raise ValueError("cannot sample achieved goals from an empty replay buffer")
        ep_idx = candidates[rng.integers(0, len(candidates), size=n)]
        t = (rng.random(n) * (self.episode_lengths[ep_idx] + 1)).astype(np.int64)
        return self.ag[ep_idx, t]
