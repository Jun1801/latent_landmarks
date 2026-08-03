"""Typed bridge between L3P models and risk-constrained macro graph search."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Literal, Optional

import numpy as np
import torch

from l3p.pn_lmcgs.macro_transition_model import MacroTransitionModel
from l3p.pn_lmcgs.mcgs import EdgePrediction, PlanResult, PlanStatus, RiskConstrainedMCGS


def _vector(value: Any, name: str, width: Optional[int] = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 1 or result.size == 0 or (width is not None and result.size != width):
        raise ValueError(f"{name} must be a non-empty vector" if width is None else f"{name} has invalid shape")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result.copy()


@dataclass(frozen=True)
class CommandResult:
    goal: Optional[np.ndarray]
    horizon: int
    source: Literal["mcgs", "direct", "original"]
    plan_result: Optional[PlanResult] = None
    diagnostics: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.goal is not None:
            object.__setattr__(self, "goal", _vector(self.goal, "goal"))
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, (int, np.integer)) or self.horizon < 0:
            raise ValueError("horizon must be a non-negative integer")
        if self.source not in {"mcgs", "direct", "original"}:
            raise ValueError("source is invalid")
        if self.diagnostics is not None:
            object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class PNPlannerAdapter:
    """Bind current L3P modules to :class:`RiskConstrainedMCGS` callbacks.

    Landmark parameters remain live between plans, but each plan captures a
    detached centroid snapshot so every batched callback observes one coherent
    model state.
    """

    def __init__(self, agent: Any, autoencoder: Any, positive_landmarks: Any,
                 negative_landmarks: Optional[Any], macro_model: MacroTransitionModel,
                 graph_search: Any, cfg: Any, rng: Optional[np.random.Generator] = None,
                 original_planner: Optional[Any] = None):
        if not isinstance(macro_model, MacroTransitionModel):
            raise ValueError("macro_model must be a MacroTransitionModel")
        self.agent = agent
        self.ae = autoencoder
        self.positive_landmarks = positive_landmarks
        self.negative_landmarks = negative_landmarks
        self.macro_model = macro_model
        self.graph_search = graph_search
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(getattr(cfg, "seed", None))
        self.original_planner = original_planner
        self._positive_centroids: Optional[torch.Tensor] = None
        self._negative_centroids: Optional[torch.Tensor] = None
        self._goal_dim: Optional[int] = None
        self._context: Optional[np.ndarray] = None
        self.search = RiskConstrainedMCGS(cfg, graph_search, self._root_distance,
                                          self._value_distance, self._edge_prediction, self.rng)

    @property
    def device(self) -> torch.device:
        return self.macro_model.temperature.device

    def _centroids(self, landmarks: Optional[Any], name: str) -> torch.Tensor:
        if landmarks is None:
            return torch.empty((0, self.macro_model.embedding_dim), dtype=torch.float32, device=self.device)
        centroids = getattr(landmarks, "centroids", None)
        if not isinstance(centroids, torch.Tensor) or centroids.ndim != 2:
            raise ValueError(f"{name} must expose a [N, embedding_dim] centroid tensor")
        if centroids.shape[1] != self.macro_model.embedding_dim or not torch.isfinite(centroids).all():
            raise ValueError(f"{name} centroids have invalid shape or values")
        return centroids.detach().to(device=self.device, dtype=torch.float32).clone()

    def _prepare_plan(self, final_goal: np.ndarray, context: Any) -> np.ndarray:
        self._goal_dim = len(final_goal)
        self._positive_centroids = self._centroids(self.positive_landmarks, "positive_landmarks")
        self._negative_centroids = self._centroids(self.negative_landmarks, "negative_landmarks")
        if context is None:
            self._context = np.zeros(self.macro_model.context_dim, dtype=np.float32)
        else:
            self._context = _vector(context, "context", self.macro_model.context_dim)
        with torch.no_grad():
            decoded = self.ae.decode(self._positive_centroids)
        if not isinstance(decoded, torch.Tensor) or decoded.device != self.device:
            raise ValueError("autoencoder decode must return a tensor on the model device")
        if decoded.ndim != 2 or decoded.shape != (len(self._positive_centroids), self._goal_dim):
            raise ValueError("decoded positive centroids have invalid shape")
        if not torch.isfinite(decoded).all():
            raise ValueError("decoded positive centroids must be finite")
        return decoded.detach().cpu().numpy().astype(np.float32, copy=True)

    def plan(self, state: Any, achieved_goal: Any, final_goal: Any, context: Any = None,
             training: bool = False) -> PlanResult:
        state_array = _vector(state, "state")
        achieved = _vector(achieved_goal, "achieved_goal")
        final = _vector(final_goal, "final_goal", len(achieved))
        positives = self._prepare_plan(final, context)
        search_context = None if self._context is not None and self._context.size == 0 \
            else self._context
        return self.search.plan(
            state_array, achieved, final, positives, search_context,
            training=training,
        )

    def _root_distance(self, state: np.ndarray, targets: np.ndarray) -> np.ndarray:
        result = np.asarray(self.agent.distance_after_action(state, targets), dtype=np.float32)
        if result.shape != (len(targets),) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("distance_after_action must return finite non-negative distances")
        return result.copy()

    def _value_distance(self, starts: np.ndarray, commands: np.ndarray) -> np.ndarray:
        if starts.shape != commands.shape or starts.ndim != 2:
            raise ValueError("value callback requires paired goal rows")
        with torch.no_grad():
            values = self.agent.value(self.agent.to_tensor(starts), self.agent.to_tensor(commands))
        result = np.asarray(values.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        if result.shape != (len(starts),) or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("agent.value must return finite non-negative distances")
        return result.copy()

    def _edge_prediction(self, starts: np.ndarray, commands: np.ndarray, context: Optional[np.ndarray]):
        if self._positive_centroids is None or self._negative_centroids is None:
            raise RuntimeError("plan must initialize centroids before edge prediction")
        if starts.shape != commands.shape or starts.ndim != 2 or starts.shape[1] != self._goal_dim:
            raise ValueError("edge callback requires paired raw goal rows")
        expected = self._context if context is None else _vector(context, "context", self.macro_model.context_dim)
        batch_context = np.repeat(expected[None, :], len(starts), axis=0)
        with torch.no_grad():
            z_start = self.ae.encode(self.agent.to_tensor(starts)).detach().to(self.device, dtype=torch.float32)
            z_command = self.ae.encode(self.agent.to_tensor(commands)).detach().to(self.device, dtype=torch.float32)
            output = self.macro_model(z_start, z_command, self.agent.to_tensor(batch_context).to(self.device),
                                      self._positive_centroids, self._negative_centroids)
        predictions = []
        for row in range(len(starts)):
            outcome = self._probabilities(output.outcome_probs[row])
            positive = self._probabilities(output.positive_probs[row])
            negative = self._probabilities(output.negative_probs[row])
            predictions.append(EdgePrediction(
                outcome_probs=outcome,
                positive_drift_distribution=None if positive.size == 0 else positive,
                negative_distribution=None if negative.size == 0 else negative,
                duration_by_outcome=output.duration_by_outcome[row].detach().cpu().numpy().astype(np.float32, copy=True),
            ))
        return predictions[0] if len(predictions) == 1 else predictions

    @staticmethod
    def _probabilities(values: torch.Tensor) -> np.ndarray:
        result = values.detach().cpu().numpy().astype(np.float64, copy=True)
        if result.ndim != 1 or not np.isfinite(result).all() or np.any(result < 0):
            raise ValueError("macro model probabilities must be finite and non-negative")
        if result.size:
            total = float(result.sum())
            if not isfinite(total) or total <= 0:
                raise ValueError("macro model probabilities must have positive mass")
            result /= total
        return result

    def _adaptive_horizon(self, state: Any, command_goal: Any) -> int:
        state_array, goal = _vector(state, "state"), _vector(command_goal, "command_goal")
        values = np.asarray(self.agent.distance_after_action(state_array, goal[None, :]), dtype=np.float64)
        if values.shape != (1,) or not np.isfinite(values).all() or values[0] < 0:
            raise ValueError("distance_after_action must return one finite non-negative distance")
        return int(np.clip(ceil(float(values[0])), self.cfg.pn_k_min, self.cfg.pn_k_max))

    def planned_command(self, state: Any, achieved_goal: Any, final_goal: Any, context: Any = None,
                        training: bool = False) -> CommandResult:
        result = self.plan(state, achieved_goal, final_goal, context, training)
        if result.status is PlanStatus.NO_SAFE_PLAN:
            return CommandResult(None, 0, "mcgs", result, dict(result.diagnostics))
        goal = result.command_goal.copy()
        return CommandResult(goal, self._adaptive_horizon(state, goal), "mcgs", result, dict(result.diagnostics))

    def direct_command(self, state: Any, final_goal: Any) -> CommandResult:
        goal = _vector(final_goal, "final_goal")
        return CommandResult(goal, self._adaptive_horizon(state, goal), "direct")

    def original_command(self, state: Any) -> CommandResult:
        if self.original_planner is None:
            raise RuntimeError("original planner is unavailable")
        if getattr(self.original_planner, "goal", None) is None:
            raise RuntimeError("original planner must be reset before requesting a command")
        goal, horizon = self.original_planner.select_command(_vector(state, "state"))
        return CommandResult(goal, int(horizon), "original")
