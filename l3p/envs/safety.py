"""Normalize environment step results into the PN-LMCGS safety contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class NormalizedStep:
    """A backend-neutral primitive transition with a binary safety cost."""

    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    safety_cost: float

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True)
class CollisionCostAdapter:
    """Optional, explicitly enabled conversion of an info field into a cost."""

    enabled: bool = False
    info_key: str = ""
    unsafe_value: Optional[Any] = None

    def cost_from_info(self, info: Mapping[str, Any]) -> Optional[float]:
        if not self.enabled or not self.info_key or self.info_key not in info:
            return None
        value = info[self.info_key]
        if self.unsafe_value is not None:
            return float(value == self.unsafe_value)
        return _binarize_cost(value)


def _binarize_cost(value: Any) -> float:
    """Convert a scalar safety signal to the v1 binary cost convention."""
    try:
        return float(float(value) > 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"safety cost must be scalar and numeric, got {value!r}") from exc


def _termination_reason(info: Mapping[str, Any], *, terminated: bool,
                        truncated: bool, goal_reached: bool,
                        safety_cost: float) -> str:
    if "termination_reason" in info:
        return str(info["termination_reason"])
    if truncated:
        return "timeout"
    if goal_reached:
        return "goal"
    if terminated and safety_cost > 0.0:
        return "violation"
    return "other"


def normalize_step_result(
    result: Tuple[Any, ...],
    collision_adapter: Optional[CollisionCostAdapter] = None,
) -> NormalizedStep:
    """Normalize legacy Gym, Gymnasium, and Safety-Gymnasium step returns.

    Cost precedence is: Safety-Gymnasium's separate cost, ``safety_cost`` in
    info, generic ``cost`` in info, an explicitly enabled collision adapter,
    then zero. The source info object is never mutated.
    """
    if not isinstance(result, tuple) or len(result) not in (4, 5, 6):
        length = len(result) if isinstance(result, tuple) else "non-tuple"
        raise ValueError(
            "step result must be a tuple of length 4, 5, or 6; "
            f"got {length}"
        )

    separate_cost = None
    if len(result) == 4:
        observation, reward, done, source_info = result
        source_info = dict(source_info) if isinstance(source_info, Mapping) else source_info
        if not isinstance(source_info, Mapping):
            raise TypeError("step result info must be a mapping")
        truncated = bool(source_info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    elif len(result) == 5:
        observation, reward, terminated, truncated, source_info = result
    else:
        observation, reward, separate_cost, terminated, truncated, source_info = result

    if not isinstance(source_info, Mapping):
        raise TypeError("step result info must be a mapping")
    info = dict(source_info)
    terminated = bool(terminated)
    truncated = bool(truncated)

    if separate_cost is not None:
        safety_cost = _binarize_cost(separate_cost)
    elif "safety_cost" in info:
        safety_cost = _binarize_cost(info["safety_cost"])
    elif "cost" in info:
        safety_cost = _binarize_cost(info["cost"])
    else:
        adapter_cost = collision_adapter.cost_from_info(info) if collision_adapter else None
        safety_cost = 0.0 if adapter_cost is None else _binarize_cost(adapter_cost)

    goal_reached = bool(
        info.get("goal_reached", info.get("is_success", info.get("success", False)))
    )
    info["safety_cost"] = safety_cost
    info["goal_reached"] = goal_reached
    info["termination_reason"] = _termination_reason(
        info,
        terminated=terminated,
        truncated=truncated,
        goal_reached=goal_reached,
        safety_cost=safety_cost,
    )
    return NormalizedStep(
        observation=observation,
        reward=float(reward),
        terminated=terminated,
        truncated=truncated,
        info=info,
        safety_cost=safety_cost,
    )
