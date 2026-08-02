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
                 her_ratio: float = 0.85, hindsight_range: int = 80,
                 n_value_negatives: int = 0, negative_sampling_strategy: str = "random",
                 ae_negatives_per_anchor: int = 1,
                 ae_negative_mode: str = "random",
                 goal_threshold: float = 0.0):
        self.size = size_episodes
        self.T = horizon
        self.compute_reward = compute_reward
        self.her_ratio = her_ratio
        self.hindsight_range = hindsight_range
        self.n_value_negatives = n_value_negatives
        self.negative_sampling_strategy = negative_sampling_strategy
        self.ae_negatives_per_anchor = ae_negatives_per_anchor
        self.ae_negative_mode = ae_negative_mode
        self.goal_threshold = goal_threshold
        if n_value_negatives < 0:
            raise ValueError("n_value_negatives must be >= 0")
        if negative_sampling_strategy not in {"random", "cross_episode"}:
            raise ValueError(
                "negative_sampling_strategy must be 'random' or 'cross_episode'"
            )
        if ae_negatives_per_anchor < 1:
            raise ValueError("ae_negatives_per_anchor must be >= 1")
        if ae_negative_mode not in {"random", "hard"}:
            raise ValueError("ae_negative_mode must be 'random' or 'hard'")

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

    def _negative_achieved_goals(self, ep_idx: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
        """Sample unrelated achieved goals for contrastive value learning."""
        batch_size = ep_idx.shape[0]
        K = self.n_value_negatives
        neg_ep_idx = rng.integers(0, self.n_episodes, size=(batch_size, K))
        if self.negative_sampling_strategy == "cross_episode" and self.n_episodes > 1:
            same_episode = neg_ep_idx == ep_idx[:, None]
            while same_episode.any():
                neg_ep_idx[same_episode] = rng.integers(
                    0, self.n_episodes, size=int(same_episode.sum())
                )
                same_episode = neg_ep_idx == ep_idx[:, None]

        neg_t = rng.integers(0, self.T + 1, size=(batch_size, K))
        return self.ag[neg_ep_idx, neg_t]

    def _ae_negative_indices(self, ep_idx: np.ndarray, t: np.ndarray, n_negatives: int,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample random negative achieved-goal indices for AE contrastive loss.

        Prefer goals from different episodes. If only one episode is available,
        sample outside the anchor's positive future window.
        """
        batch_size = ep_idx.shape[0]
        neg_ep_idx = rng.integers(0, self.n_episodes, size=(batch_size, n_negatives))
        neg_t = rng.integers(0, self.T + 1, size=(batch_size, n_negatives))

        if self.n_episodes > 1:
            same_episode = neg_ep_idx == ep_idx[:, None]
            while same_episode.any():
                neg_ep_idx[same_episode] = rng.integers(
                    0, self.n_episodes, size=int(same_episode.sum())
                )
                same_episode = neg_ep_idx == ep_idx[:, None]
        else:
            high = np.minimum(t + self.hindsight_range, self.T)
            in_future_window = (neg_t > t[:, None]) & (neg_t <= high[:, None])
            while in_future_window.any():
                neg_t[in_future_window] = rng.integers(
                    0, self.T + 1, size=int(in_future_window.sum())
                )
                in_future_window = (neg_t > t[:, None]) & (neg_t <= high[:, None])

        return neg_ep_idx, neg_t

    def sample_ae_triples(self, batch_size: int, rng: np.random.Generator = None,
                          n_negatives: int = None, negative_mode: str = None,
                          goal_threshold: float = None) -> Dict[str, np.ndarray]:
        """Sample (anchor, positive, negative) achieved-goal triples for AE loss.

        anchor: achieved goal at timestep t
        positive: future achieved goal from the same episode within hindsight_range
        negative: random achieved goal from another episode, or outside the
            positive future window if the buffer only contains one episode

        Random negatives that are closer than 2 * goal_threshold to the anchor
        are re-sampled a few times; if no valid replacement is found, the method
        falls back to the original random draw.
        """
        if rng is None:
            rng = np.random.default_rng()
        if self.n_episodes < 1:
            raise ValueError("cannot sample AE triples from an empty replay buffer")

        n_negatives = self.ae_negatives_per_anchor if n_negatives is None else n_negatives
        negative_mode = self.ae_negative_mode if negative_mode is None else negative_mode
        goal_threshold = self.goal_threshold if goal_threshold is None else goal_threshold
        if n_negatives < 1:
            raise ValueError("n_negatives must be >= 1")
        if negative_mode != "random":
            raise NotImplementedError("AE hard negatives are not implemented yet; use random")

        ep_idx = rng.integers(0, self.n_episodes, size=batch_size)
        t = rng.integers(0, self.T, size=batch_size)
        fut_t = self._future_index(t, rng)

        anchor = self.ag[ep_idx, t]
        positive = self.ag[ep_idx, fut_t]

        neg_ep_idx, neg_t = self._ae_negative_indices(ep_idx, t, n_negatives, rng)
        negative = self.ag[neg_ep_idx, neg_t].copy()
        fallback = negative.copy()

        min_dist = 2.0 * float(goal_threshold)
        if min_dist > 0.0:
            anchor_exp = anchor[:, None, :]
            valid = np.linalg.norm(negative - anchor_exp, axis=-1) > min_dist
            for _ in range(16):
                if valid.all():
                    break
                cand_ep_idx, cand_t = self._ae_negative_indices(ep_idx, t, n_negatives, rng)
                cand = self.ag[cand_ep_idx, cand_t]
                cand_valid = np.linalg.norm(cand - anchor_exp, axis=-1) > min_dist
                replace = (~valid) & cand_valid
                negative[replace] = cand[replace]
                valid[replace] = True
            negative[~valid] = fallback[~valid]

        return dict(anchor=anchor, positive=positive, negative=negative)

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

        sample = dict(obs=obs, next_obs=next_obs, ag=ag, next_ag=next_ag, act=act,
                      g=g, future_ag=future_ag_v, reward=reward)
        if self.n_value_negatives > 0:
            sample["neg_ag"] = self._negative_achieved_goals(ep_idx, rng)
        return sample

    def sample_achieved_goals(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        """Sample a flat batch of achieved goals (used by the AE and GLS)."""
        if rng is None:
            rng = np.random.default_rng()
        ep_idx = rng.integers(0, self.n_episodes, size=n)
        t = rng.integers(0, self.T + 1, size=n)
        return self.ag[ep_idx, t]

    def state_dict(self) -> dict:
        """Serialize only the filled part of the buffer for compact checkpoints."""
        n = self.n_episodes
        return dict(
            ptr=self.ptr,
            n_episodes=n,
            obs=self.obs[:n].copy(),
            ag=self.ag[:n].copy(),
            g=self.g[:n].copy(),
            act=self.act[:n].copy(),
        )

    def load_state_dict(self, d: dict) -> None:
        n = int(d["n_episodes"])
        if n > self.size:
            raise ValueError(f"checkpoint has {n} episodes, buffer capacity is {self.size}")
        self.obs[:n] = d["obs"]
        self.ag[:n] = d["ag"]
        self.g[:n] = d["g"]
        self.act[:n] = d["act"]
        self.ptr = int(d["ptr"]) % self.size
        self.n_episodes = n
