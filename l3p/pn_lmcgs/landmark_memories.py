"""Bounded raw-goal memories used to train PN-LMCGS landmarks."""

from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Sequence, Tuple

import numpy as np


class LandmarkMemoryManager:
    """FIFO positive and violation-endpoint memories with copy-on-ingest semantics."""

    def __init__(self, goal_dim: int, positive_capacity: int = 10000,
                 negative_capacity: int = 10000):
        if not isinstance(goal_dim, (int, np.integer)) or goal_dim < 1:
            raise ValueError("goal_dim must be a positive integer")
        if positive_capacity < 1 or negative_capacity < 1:
            raise ValueError("memory capacities must be positive")
        self.goal_dim = int(goal_dim)
        self.positive_capacity = int(positive_capacity)
        self.negative_capacity = int(negative_capacity)
        self._positive: Deque[np.ndarray] = deque()
        self._negative: Deque[Tuple[np.ndarray, int, bool]] = deque()
        self._negative_episode_counts: Counter = Counter()

    @property
    def positive_size(self) -> int:
        return len(self._positive)

    @property
    def negative_size(self) -> int:
        return len(self._negative)

    @property
    def positive_goals(self) -> np.ndarray:
        return self._goals(self._positive)

    @property
    def negative_goals(self) -> np.ndarray:
        return self._goals([entry[0] for entry in self._negative])

    @property
    def distinct_negative_episodes(self) -> int:
        return len(self._negative_episode_counts)

    def negative_landmarks_active(self, min_samples: int, min_episodes: int) -> bool:
        return self.negative_size >= min_samples and self.distinct_negative_episodes >= min_episodes

    def ingest_episode(self, ep: Dict[str, np.ndarray], episode_id: int) -> None:
        length = self._episode_length(ep)
        ag = np.asarray(self._field(ep, "ag", length + 1), dtype=np.float32)
        cost = self._transition_field(ep, "cost", length, default=0.0).astype(np.float32)
        post_available = self._transition_field(ep, "post_state_available", length, default=True).astype(bool)
        episode_id = int(episode_id)
        negative_entries = [
            (ag[t] if not post_available[t] else ag[t + 1], not post_available[t])
            for t in np.flatnonzero(cost > 0)
        ]

        # Positives require an explicitly recorded low-level command; task g is not a substitute.
        if "cmd_g" not in ep:
            for goal, is_preimpact in negative_entries:
                self._append_negative(goal, episode_id, is_preimpact)
            return
        cmd_g = self._field(ep, "cmd_g", length)
        reached = self._transition_field(ep, "goal_reached", length, default=False).astype(bool)
        positive_goals = []
        start = 0
        while start < length:
            end = start + 1
            while end < length and np.array_equal(cmd_g[end], cmd_g[start]):
                end += 1
            successful = np.flatnonzero(reached[start:end]) + start
            for t in range(start, end):
                later = successful[successful >= t]
                if len(later):
                    goal_time = int(later[0])
                    if cost[t:goal_time + 1].sum() == 0:
                        positive_goals.append(ag[t])
            start = end
        for goal, is_preimpact in negative_entries:
            self._append_negative(goal, episode_id, is_preimpact)
        for goal in positive_goals:
            self._append_positive(goal)

    def sample_positive(self, n: int, rng: np.random.Generator = None) -> np.ndarray:
        if not self._positive:
            raise ValueError("cannot sample positive landmarks from an empty memory")
        indices = self._sample_indices(n, len(self._positive), rng)
        return np.asarray([self._positive[i] for i in indices], dtype=np.float32).copy()

    def sample_negative(self, n: int, rng: np.random.Generator = None) -> Dict[str, np.ndarray]:
        if not self._negative:
            raise ValueError("cannot sample negative landmarks from an empty memory")
        indices = self._sample_indices(n, len(self._negative), rng)
        selected = [self._negative[i] for i in indices]
        return {
            "goals": np.asarray([entry[0] for entry in selected], dtype=np.float32).copy(),
            "episode_id": np.asarray([entry[1] for entry in selected], dtype=np.int64),
            "is_preimpact": np.asarray([entry[2] for entry in selected], dtype=bool),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "goal_dim": self.goal_dim,
            "positive_capacity": self.positive_capacity,
            "negative_capacity": self.negative_capacity,
            "positive_goals": self.positive_goals,
            "negative_goals": self.negative_goals,
            "negative_episode_ids": np.asarray([x[1] for x in self._negative], dtype=np.int64),
            "negative_is_preimpact": np.asarray([x[2] for x in self._negative], dtype=bool),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        required = {"goal_dim", "positive_capacity", "negative_capacity", "positive_goals",
                    "negative_goals", "negative_episode_ids", "negative_is_preimpact"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"memory state missing keys: {sorted(missing)}")
        if state["goal_dim"] != self.goal_dim:
            raise ValueError("memory state goal_dim does not match manager")
        if state["positive_capacity"] != self.positive_capacity or state["negative_capacity"] != self.negative_capacity:
            raise ValueError("memory state capacities do not match manager")
        positive = self._goals_from_state(state["positive_goals"], "positive_goals", self.positive_capacity)
        negative = self._goals_from_state(state["negative_goals"], "negative_goals", self.negative_capacity)
        ids = np.asarray(state["negative_episode_ids"])
        preimpact = np.asarray(state["negative_is_preimpact"])
        if ids.shape != (len(negative),) or preimpact.shape != (len(negative),):
            raise ValueError("negative memory metadata has invalid shape")
        self._positive = deque(goal.copy() for goal in positive)
        self._negative = deque(
            (goal.copy(), int(ident), bool(pre))
            for goal, ident, pre in zip(negative, ids, preimpact)
        )
        self._negative_episode_counts = Counter(int(ident) for ident in ids)

    def _append_positive(self, goal: np.ndarray) -> None:
        if len(self._positive) == self.positive_capacity:
            self._positive.popleft()
        self._positive.append(np.asarray(goal, dtype=np.float32).copy())

    def _append_negative(self, goal: np.ndarray, episode_id: int, is_preimpact: bool) -> None:
        if len(self._negative) == self.negative_capacity:
            _, old_episode, _ = self._negative.popleft()
            self._negative_episode_counts[old_episode] -= 1
            if not self._negative_episode_counts[old_episode]:
                del self._negative_episode_counts[old_episode]
        self._negative.append((np.asarray(goal, dtype=np.float32).copy(), int(episode_id), bool(is_preimpact)))
        self._negative_episode_counts[int(episode_id)] += 1

    def _episode_length(self, ep: Dict[str, np.ndarray]) -> int:
        if "ag" not in ep:
            raise ValueError("episode missing required field 'ag'")
        explicit = ep.get("length", ep.get("episode_length"))
        if explicit is None:
            if "cost" in ep:
                length = len(np.asarray(ep["cost"]))
            elif "cmd_g" in ep:
                length = len(np.asarray(ep["cmd_g"]))
            else:
                length = len(np.asarray(ep["ag"])) - 1
        else:
            if isinstance(explicit, (bool, np.bool_)) or int(explicit) != explicit:
                raise ValueError("episode length must be an integer")
            length = int(explicit)
        if length < 1:
            raise ValueError("episode length must be non-zero")
        if len(np.asarray(ep["ag"])) < length + 1:
            raise ValueError("ag must include every post-transition achieved goal")
        return length

    def _field(self, ep: Dict[str, np.ndarray], name: str, length: int) -> np.ndarray:
        array = np.asarray(ep[name])
        if array.shape[0] < length or array.shape[1:] != (self.goal_dim,):
            raise ValueError(f"episode field {name!r} must have shape (N, {self.goal_dim})")
        return array[:length]

    def _transition_field(self, ep: Dict[str, np.ndarray], name: str, length: int,
                          default=None) -> np.ndarray:
        value = ep.get(name, default)
        array = np.asarray(value)
        if array.ndim == 0:
            array = np.full(length, array, dtype=array.dtype)
        if array.shape[0] < length:
            raise ValueError(f"episode field {name!r} is shorter than the episode")
        return array[:length]

    def _goals(self, goals: Sequence[np.ndarray]) -> np.ndarray:
        if not goals:
            return np.empty((0, self.goal_dim), dtype=np.float32)
        return np.asarray(goals, dtype=np.float32).copy()

    def _goals_from_state(self, values: object, name: str, capacity: int) -> np.ndarray:
        goals = np.asarray(values, dtype=np.float32)
        if goals.ndim != 2 or goals.shape[1:] != (self.goal_dim,) or len(goals) > capacity:
            raise ValueError(f"memory state {name} has invalid shape or exceeds capacity")
        return goals

    @staticmethod
    def _sample_indices(n: int, available: int, rng: np.random.Generator = None) -> np.ndarray:
        if n < 1:
            raise ValueError("sample size must be positive")
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, available, size=n)
