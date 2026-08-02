"""Replay storage and current-landmark supervision for macro attempts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch


class Outcome(IntEnum):
    TARGET = 0
    DRIFT = 1
    STUCK = 2
    VIOLATION = 3


def _vector(value, name: str, *, length: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or (length is not None and array.shape[0] != length):
        expected = "a vector" if length is None else f"a vector of length {length}"
        raise ValueError(f"{name} must be {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _boolean(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


@dataclass
class MacroAttempt:
    start_goal: np.ndarray
    command_goal: np.ndarray
    end_goal: np.ndarray
    violation_goal: Optional[np.ndarray]
    duration: int
    context: np.ndarray
    target_reached: bool
    violation: bool
    episode_id: int
    start_t: int
    end_t: int

    def __post_init__(self) -> None:
        self.start_goal = _vector(self.start_goal, "start_goal")
        goal_dim = len(self.start_goal)
        if goal_dim == 0:
            raise ValueError("start_goal must not be empty")
        self.command_goal = _vector(self.command_goal, "command_goal", length=goal_dim)
        self.end_goal = _vector(self.end_goal, "end_goal", length=goal_dim)
        self.violation_goal = (
            None if self.violation_goal is None else _vector(self.violation_goal, "violation_goal", length=goal_dim)
        )
        self.context = _vector(self.context, "context")
        self.duration = _integer(self.duration, "duration", minimum=1)
        self.target_reached = _boolean(self.target_reached, "target_reached")
        self.violation = _boolean(self.violation, "violation")
        self.episode_id = _integer(self.episode_id, "episode_id")
        self.start_t = _integer(self.start_t, "start_t")
        self.end_t = _integer(self.end_t, "end_t")
        if self.end_t < self.start_t:
            raise ValueError("end_t must be at least start_t")

    def copy(self) -> "MacroAttempt":
        return MacroAttempt(
            self.start_goal, self.command_goal, self.end_goal, self.violation_goal,
            self.duration, self.context, self.target_reached, self.violation,
            self.episode_id, self.start_t, self.end_t,
        )


class MacroAttemptBuffer:
    """Bounded FIFO storage whose validation partition is a fixed random split."""

    def __init__(self, capacity: int, validation_fraction: float = 0.1, split_seed: int = 0):
        self.capacity = _integer(capacity, "capacity", minimum=1)
        if not np.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be finite and in [0, 1)")
        self.validation_fraction = float(validation_fraction)
        self.split_seed = _integer(split_seed, "split_seed")
        self._slots: list[Optional[MacroAttempt]] = [None] * self.capacity
        self._slot_validation: list[Optional[bool]] = [None] * self.capacity
        self._head = 0
        self._size = 0
        self._train_slots: list[int] = []
        self._validation_slots: list[int] = []
        self._slot_partition_positions = np.full(self.capacity, -1, dtype=np.int64)
        self._goal_dim: Optional[int] = None
        self._context_dim: Optional[int] = None

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return len(self)

    @property
    def attempts(self) -> list[MacroAttempt]:
        return [self._attempt_at_slot(slot).copy() for slot in self._chronological_slots()]

    @property
    def train_indices(self) -> np.ndarray:
        return self._partition_indices(self._train_slots)

    @property
    def validation_indices(self) -> np.ndarray:
        return self._partition_indices(self._validation_slots)

    def add(self, attempt: MacroAttempt) -> None:
        if not isinstance(attempt, MacroAttempt):
            raise ValueError("attempt must be a MacroAttempt")
        candidate = attempt.copy()
        goal_dim, context_dim = len(candidate.start_goal), len(candidate.context)
        if self._goal_dim is not None and goal_dim != self._goal_dim:
            raise ValueError("attempt goal dimension does not match buffer")
        if self._context_dim is not None and context_dim != self._context_dim:
            raise ValueError("attempt context dimension does not match buffer")
        if self._size == self.capacity:
            slot = self._head
            self._remove_slot_from_partition(slot)
            self._slots[slot] = None
            self._slot_validation[slot] = None
            self._head = (self._head + 1) % self.capacity
            self._size -= 1
        else:
            slot = (self._head + self._size) % self.capacity
        validation = self._validation_size(self._size + 1) > len(self._validation_slots)
        self._slots[slot] = candidate
        self._slot_validation[slot] = validation
        self._add_slot_to_partition(slot, validation)
        self._size += 1
        self._goal_dim, self._context_dim = goal_dim, context_dim

    def extend(self, attempts: Iterable[MacroAttempt]) -> None:
        candidates = list(attempts)
        for attempt in candidates:
            if not isinstance(attempt, MacroAttempt):
                raise ValueError("attempt must be a MacroAttempt")
        for attempt in candidates:
            self.add(attempt)

    def sample_train(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> dict[str, np.ndarray]:
        return self._sample_from_slots(self._train_slots, batch_size, rng)

    def sample_validation(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> dict[str, np.ndarray]:
        return self._sample_from_slots(self._validation_slots, batch_size, rng)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "validation_fraction": self.validation_fraction,
            "split_seed": self.split_seed,
            "attempts": [self._attempt_state(self._attempt_at_slot(slot)) for slot in self._chronological_slots()],
            "validation_membership": [bool(self._slot_validation[slot]) for slot in self._chronological_slots()],
            "partition_order": {
                "train": self._partition_indices(self._train_slots).tolist(),
                "validation": self._partition_indices(self._validation_slots).tolist(),
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if not isinstance(state, dict):
            raise ValueError("buffer state must be a dictionary")
        required = {"capacity", "validation_fraction", "split_seed", "attempts", "validation_membership", "partition_order"}
        missing = required.difference(state)
        if missing:
            raise ValueError(f"buffer state missing keys: {sorted(missing)}")
        if state["capacity"] != self.capacity:
            raise ValueError("buffer state capacity does not match buffer")
        if state["validation_fraction"] != self.validation_fraction or state["split_seed"] != self.split_seed:
            raise ValueError("buffer state split configuration does not match buffer")
        raw_attempts = state["attempts"]
        if not isinstance(raw_attempts, Sequence) or len(raw_attempts) > self.capacity:
            raise ValueError("buffer state attempts is invalid or exceeds capacity")
        candidates = [self._attempt_from_state(item) for item in raw_attempts]
        raw_membership = state["validation_membership"]
        if isinstance(raw_membership, (str, bytes)) or not isinstance(raw_membership, Sequence):
            raise ValueError("buffer state membership is invalid")
        if len(raw_membership) != len(candidates):
            raise ValueError("buffer state membership length does not match attempts")
        membership = [_boolean(value, "buffer state membership") for value in raw_membership]
        if sum(membership) != self._validation_size(len(candidates)):
            raise ValueError("buffer state membership has invalid partition size")
        train_order, validation_order = self._partition_order_from_state(
            state["partition_order"], membership, len(candidates)
        )
        if candidates:
            goal_dim, context_dim = len(candidates[0].start_goal), len(candidates[0].context)
            if any(len(x.start_goal) != goal_dim or len(x.context) != context_dim for x in candidates):
                raise ValueError("buffer state attempt dimensions are inconsistent")
        else:
            goal_dim = context_dim = None
        slots: list[Optional[MacroAttempt]] = [None] * self.capacity
        slot_validation: list[Optional[bool]] = [None] * self.capacity
        for slot, (candidate, validation) in enumerate(zip(candidates, membership)):
            slots[slot] = candidate.copy()
            slot_validation[slot] = validation
        positions = np.full(self.capacity, -1, dtype=np.int64)
        for position, slot in enumerate(train_order):
            positions[slot] = position
        for position, slot in enumerate(validation_order):
            positions[slot] = position
        self._slots, self._slot_validation = slots, slot_validation
        self._head, self._size = 0, len(candidates)
        self._train_slots, self._validation_slots = train_order, validation_order
        self._slot_partition_positions = positions
        self._goal_dim, self._context_dim = goal_dim, context_dim

    def _sample_from_slots(self, slots: Sequence[int], batch_size: int,
                             rng: Optional[np.random.Generator]) -> dict[str, np.ndarray]:
        batch_size = _integer(batch_size, "batch_size", minimum=1)
        if len(slots) == 0:
            raise ValueError("requested replay split is empty")
        rng = np.random.default_rng() if rng is None else rng
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy Generator")
        selected_positions = rng.integers(len(slots), size=batch_size)
        picked = [self._attempt_at_slot(slots[int(position)]) for position in selected_positions]
        violation_goals = np.full((batch_size, self._goal_dim), np.nan, dtype=np.float32)
        violation_goal_present = np.zeros(batch_size, dtype=bool)
        for index, item in enumerate(picked):
            if item.violation_goal is not None:
                violation_goals[index] = item.violation_goal
                violation_goal_present[index] = True
        return {
            "start_goal": np.asarray([x.start_goal for x in picked], dtype=np.float32).copy(),
            "command_goal": np.asarray([x.command_goal for x in picked], dtype=np.float32).copy(),
            "end_goal": np.asarray([x.end_goal for x in picked], dtype=np.float32).copy(),
            "violation_goal": violation_goals,
            "violation_goal_present": violation_goal_present,
            "duration": np.asarray([x.duration for x in picked], dtype=np.int64),
            "context": np.asarray([x.context for x in picked], dtype=np.float32).copy(),
            "target_reached": np.asarray([x.target_reached for x in picked], dtype=bool),
            "violation": np.asarray([x.violation for x in picked], dtype=bool),
            "episode_id": np.asarray([x.episode_id for x in picked], dtype=np.int64),
            "start_t": np.asarray([x.start_t for x in picked], dtype=np.int64),
            "end_t": np.asarray([x.end_t for x in picked], dtype=np.int64),
        }

    def _validation_size(self, size: int) -> int:
        validation_size = int(np.floor(size * self.validation_fraction))
        if self.validation_fraction > 0 and size > 1:
            validation_size = max(1, validation_size)
        return min(validation_size, max(0, size - 1))

    def _chronological_slots(self) -> list[int]:
        return [(self._head + index) % self.capacity for index in range(self._size)]

    def _partition_indices(self, slots: Sequence[int]) -> np.ndarray:
        return np.asarray([(slot - self._head) % self.capacity for slot in slots], dtype=np.int64)

    def _attempt_at_slot(self, slot: int) -> MacroAttempt:
        attempt = self._slots[slot]
        if attempt is None:
            raise RuntimeError("replay partition references an empty slot")
        return attempt

    def _add_slot_to_partition(self, slot: int, validation: bool) -> None:
        partition = self._validation_slots if validation else self._train_slots
        self._slot_partition_positions[slot] = len(partition)
        partition.append(slot)

    def _remove_slot_from_partition(self, slot: int) -> None:
        validation = self._slot_validation[slot]
        if validation is None:
            raise RuntimeError("cannot remove an empty replay slot")
        partition = self._validation_slots if validation else self._train_slots
        position = int(self._slot_partition_positions[slot])
        last_slot = partition.pop()
        if position < len(partition):
            partition[position] = last_slot
            self._slot_partition_positions[last_slot] = position
        self._slot_partition_positions[slot] = -1

    @staticmethod
    def _partition_order_from_state(raw_order: object, membership: Sequence[bool], size: int) -> tuple[list[int], list[int]]:
        if not isinstance(raw_order, dict) or set(raw_order) != {"train", "validation"}:
            raise ValueError("buffer state partition order is invalid")
        orders: list[list[int]] = []
        for name, is_validation in (("train", False), ("validation", True)):
            raw_partition = raw_order[name]
            if isinstance(raw_partition, (str, bytes)) or not isinstance(raw_partition, Sequence):
                raise ValueError("buffer state partition order is invalid")
            partition = [_integer(value, "buffer state partition index") for value in raw_partition]
            expected = [index for index, value in enumerate(membership) if value == is_validation]
            if sorted(partition) != expected:
                raise ValueError("buffer state partition order does not match membership")
            orders.append(partition)
        return orders[0], orders[1]

    @staticmethod
    def _attempt_state(attempt: MacroAttempt) -> dict[str, object]:
        return {
            "start_goal": attempt.start_goal.copy(), "command_goal": attempt.command_goal.copy(),
            "end_goal": attempt.end_goal.copy(),
            "violation_goal": None if attempt.violation_goal is None else attempt.violation_goal.copy(),
            "duration": attempt.duration, "context": attempt.context.copy(),
            "target_reached": attempt.target_reached, "violation": attempt.violation,
            "episode_id": attempt.episode_id, "start_t": attempt.start_t, "end_t": attempt.end_t,
        }

    @staticmethod
    def _attempt_from_state(state: object) -> MacroAttempt:
        if not isinstance(state, dict):
            raise ValueError("buffer state attempt must be a dictionary")
        required = set(MacroAttempt.__dataclass_fields__)
        missing = required.difference(state)
        if missing:
            raise ValueError(f"buffer state attempt missing keys: {sorted(missing)}")
        return MacroAttempt(**{name: state[name] for name in required})


def stratified_macro_indices(outcomes, batch_size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return balanced-training indices; this deliberately is not a natural-frequency sample."""
    batch_size = _integer(batch_size, "batch_size", minimum=1)
    values = np.asarray(outcomes)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("outcomes must be a non-empty vector")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 3) | (values != np.floor(values))):
        raise ValueError("outcomes must contain valid outcome labels")
    rng = np.random.default_rng() if rng is None else rng
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    strata = [np.flatnonzero(values == Outcome.TARGET), np.flatnonzero((values == Outcome.DRIFT) | (values == Outcome.STUCK)), np.flatnonzero(values == Outcome.VIOLATION)]
    requested = np.asarray([int(round(batch_size * 0.50)), int(round(batch_size * 0.25))], dtype=np.int64)
    requested = np.append(requested, batch_size - requested.sum())
    available = [x for x in strata if len(x)]
    selected: list[np.ndarray] = []
    for count, stratum in zip(requested, strata):
        source = stratum if len(stratum) else np.concatenate(available)
        selected.append(source[rng.integers(len(source), size=count)])
    result = np.concatenate(selected)
    return result[rng.permutation(len(result))].astype(np.int64)


def label_macro_attempts(attempts: Sequence[MacroAttempt], encoder: Callable, positive_centroids,
                         negative_centroids, assignment_radius: float, negative_active: bool,
                         device: Optional[torch.device | str] = None):
    """Label raw replay entries against the *current* latent centroids."""
    from l3p.pn_lmcgs.macro_transition_model import MacroLabels

    if not isinstance(negative_active, (bool, np.bool_)):
        raise ValueError("negative_active must be boolean")
    if not np.isfinite(assignment_radius) or assignment_radius < 0:
        raise ValueError("assignment_radius must be finite and non-negative")
    entries = list(attempts)
    if any(not isinstance(x, MacroAttempt) for x in entries):
        raise ValueError("attempts must contain MacroAttempt values")
    if device is None:
        device = _module_device(encoder)
    device = torch.device(device)
    positive = _centroids(positive_centroids, "positive_centroids", device)
    negative = _centroids(negative_centroids, "negative_centroids", device, expected_dim=positive.shape[1] if positive.ndim == 2 and positive.shape[1] else None)
    if not entries:
        return MacroLabels(torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.float32, device=device))
    goal_dim = len(entries[0].end_goal)
    if any(len(x.end_goal) != goal_dim for x in entries):
        raise ValueError("attempt goal dimensions are inconsistent")
    with torch.no_grad():
        end_latent = _encode(encoder, np.asarray([x.end_goal for x in entries], dtype=np.float32), device)
        if positive_centroids is not None and positive.shape[1] != end_latent.shape[1]:
            raise ValueError("positive_centroids dimension does not match encoder output")
        if negative_centroids is not None and negative.shape[1] != end_latent.shape[1]:
            raise ValueError("negative_centroids dimension does not match encoder output")
        outcome = torch.full((len(entries),), int(Outcome.STUCK), dtype=torch.long, device=device)
        positive_id = torch.full_like(outcome, -1)
        negative_id = torch.full_like(outcome, -1)
        for index, entry in enumerate(entries):
            if entry.violation:
                outcome[index] = int(Outcome.VIOLATION)
            elif entry.target_reached:
                outcome[index] = int(Outcome.TARGET)
            elif positive.shape[0]:
                distances = torch.linalg.vector_norm(positive - end_latent[index], dim=1)
                nearest = int(torch.argmin(distances))
                if distances[nearest] <= assignment_radius:
                    outcome[index] = int(Outcome.DRIFT)
                    positive_id[index] = nearest
        violation_indices = [i for i, entry in enumerate(entries) if entry.violation]
        if negative_active and negative.shape[0] and violation_indices:
            endpoints = np.asarray([
                entries[i].end_goal if entries[i].violation_goal is None else entries[i].violation_goal
                for i in violation_indices
            ], dtype=np.float32)
            endpoint_latent = _encode(encoder, endpoints, device)
            distances = torch.cdist(endpoint_latent, negative)
            negative_id[torch.as_tensor(violation_indices, device=device)] = torch.argmin(distances, dim=1)
    return MacroLabels(outcome, positive_id, negative_id,
                       torch.tensor([x.duration for x in entries], dtype=torch.float32, device=device))


def _module_device(encoder: Callable) -> torch.device:
    if isinstance(encoder, torch.nn.Module):
        try:
            return next(encoder.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _centroids(value, name: str, device: torch.device, expected_dim: Optional[int] = None) -> torch.Tensor:
    if value is None:
        if expected_dim is None:
            return torch.empty((0, 0), dtype=torch.float32, device=device)
        return torch.empty((0, expected_dim), dtype=torch.float32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim != 2 or (expected_dim is not None and tensor.shape[1] != expected_dim):
        raise ValueError(f"{name} must have shape [N, latent_dim]")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    return tensor.detach()


def _encode(encoder: Callable, goals: np.ndarray, device: torch.device) -> torch.Tensor:
    result = encoder(torch.as_tensor(goals, dtype=torch.float32, device=device))
    result = torch.as_tensor(result, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.shape[0] != goals.shape[0] or not torch.isfinite(result).all():
        raise ValueError("encoder must return finite [B, latent_dim] embeddings")
    return result.detach()
