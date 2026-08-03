"""Risk-constrained Monte Carlo graph search over positive landmarks.

The planner deliberately depends only on callbacks.  Model and trainer adapters
live elsewhere so this search remains deterministic and straightforward to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from math import exp, isfinite, log, sqrt
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import torch


ROOT = -1


def _vector(value: Any, name: str, length: Optional[int] = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or (length is not None and result.size != length):
        raise ValueError(f"{name} must be a non-empty finite vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result.copy()


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability")
    return value


@dataclass(frozen=True, order=True)
class NodeKey:
    node_id: int
    remaining_macro_depth: int

    def __post_init__(self) -> None:
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, (int, np.integer)):
            raise ValueError("node_id must be an integer")
        if isinstance(self.remaining_macro_depth, bool) or not isinstance(self.remaining_macro_depth, (int, np.integer)):
            raise ValueError("remaining_macro_depth must be an integer")
        if self.remaining_macro_depth < 0:
            raise ValueError("remaining_macro_depth must be non-negative")


@dataclass(frozen=True)
class EdgePrediction:
    """Calibrated macro-outcome prediction for one directed edge.

    ``local_distance`` is optional model metadata.  The planner always replaces
    it with the trusted D/V callback value for the planning call.
    """

    p_target: float = 1.0
    p_drift: float = 0.0
    p_stuck: float = 0.0
    p_violation: float = 0.0
    positive_drift_distribution: Optional[Sequence[float]] = None
    negative_distribution: Optional[Sequence[float]] = None
    duration_by_outcome: Sequence[float] = (1.0, 1.0, 1.0, 1.0)
    local_distance: Optional[float] = None
    outcome_probs: Optional[Sequence[float]] = None

    def __post_init__(self) -> None:
        probabilities = (self.p_target, self.p_drift, self.p_stuck, self.p_violation)
        if self.outcome_probs is not None:
            probabilities = tuple(np.asarray(self.outcome_probs, dtype=np.float64).tolist())
            if len(probabilities) != 4:
                raise ValueError("outcome_probs must have four entries")
        probabilities = tuple(_probability(value, "outcome probability") for value in probabilities)
        if not np.isclose(sum(probabilities), 1.0, atol=1e-8):
            raise ValueError("outcome probabilities must sum to one")
        object.__setattr__(self, "p_target", probabilities[0])
        object.__setattr__(self, "p_drift", probabilities[1])
        object.__setattr__(self, "p_stuck", probabilities[2])
        object.__setattr__(self, "p_violation", probabilities[3])
        object.__setattr__(self, "outcome_probs", probabilities)
        object.__setattr__(self, "positive_drift_distribution", self._distribution(
            self.positive_drift_distribution, "positive_drift_distribution"))
        object.__setattr__(self, "negative_distribution", self._distribution(
            self.negative_distribution, "negative_distribution"))
        duration = np.asarray(self.duration_by_outcome, dtype=np.float64)
        if duration.shape != (4,) or not np.isfinite(duration).all() or np.any(duration < 1.0):
            raise ValueError("duration_by_outcome must be four finite values at least one")
        object.__setattr__(self, "duration_by_outcome", tuple(float(value) for value in duration))
        if self.local_distance is not None:
            value = float(self.local_distance)
            if not isfinite(value) or value < 0.0:
                raise ValueError("local_distance must be finite and non-negative")
            object.__setattr__(self, "local_distance", value)

    @staticmethod
    def _distribution(value: Optional[Sequence[float]], name: str) -> Optional[tuple[float, ...]]:
        if value is None:
            return None
        result = np.asarray(value, dtype=np.float64)
        if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all() or np.any(result < 0.0):
            raise ValueError(f"{name} must be a non-empty non-negative finite vector")
        if not np.isclose(result.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{name} must sum to one")
        return tuple(float(item) for item in result)


@dataclass
class EdgeStats:
    target_id: int
    prediction: EdgePrediction
    prior_score: float
    pseudocount: float
    risk_z: float
    visits: int = 0
    reward_sum: float = 0.0
    alpha: float = field(init=False)
    beta: float = field(init=False)

    def __post_init__(self) -> None:
        self.prior_score = float(self.prior_score)
        self.pseudocount = float(self.pseudocount)
        self.risk_z = float(self.risk_z)
        if not isfinite(self.prior_score) or not isfinite(self.pseudocount) or self.pseudocount < 0.0:
            raise ValueError("prior_score must be finite and pseudocount must be finite and non-negative")
        self.alpha = 1.0 + self.pseudocount * self.prediction.p_violation
        self.beta = 1.0 + self.pseudocount * (1.0 - self.prediction.p_violation)

    @property
    def q_reward(self) -> float:
        return self.reward_sum / self.visits if self.visits else 0.0

    @property
    def q_risk(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def risk_std(self) -> float:
        total = self.alpha + self.beta
        return sqrt(self.alpha * self.beta / (total * total * (total + 1.0)))

    @property
    def upper_risk(self) -> float:
        return min(1.0, self.q_risk + self.risk_z * self.risk_std)

    @property
    def feasibility_risk(self) -> float:
        return self.prediction.p_violation if self.visits == 0 else self.upper_risk

    @property
    def search_risk(self) -> float:
        # The conservative prior starts above the default search limit even
        # for p=0. Gather safe evidence until a rollout reports nonzero risk;
        # final root selection still always uses the posterior upper bound.
        prior_alpha = 1.0 + self.pseudocount * self.prediction.p_violation
        if self.alpha <= prior_alpha:
            return self.prediction.p_violation
        return self.upper_risk

    def update(self, reward_return: float, safety_return: float) -> None:
        self.visits += 1
        self.reward_sum += float(reward_return)
        self.alpha += float(safety_return)
        self.beta += 1.0 - float(safety_return)


@dataclass
class SearchNode:
    key: NodeKey
    edges: list[EdgeStats] = field(default_factory=list)
    visits: int = 0
    reward_sum: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0

    def update(self, reward_return: float, safety_return: float) -> None:
        self.visits += 1
        self.reward_sum += float(reward_return)
        self.alpha += float(safety_return)
        self.beta += 1.0 - float(safety_return)


class PlanStatus(str, Enum):
    ACTION = "ACTION"
    NO_SAFE_PLAN = "NO_SAFE_PLAN"


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    command_goal: np.ndarray
    node_id: Optional[int]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_goal", _vector(self.command_goal, "command_goal"))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class RiskConstrainedMCGS:
    """Per-call MCGS over ROOT, positive nodes, and a final GOAL node."""

    def __init__(self, cfg: Any, graph_search: Any, root_distance_fn: Callable,
                 value_distance_fn: Callable, edge_prediction_fn: Callable,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.graph_search = graph_search
        self.root_distance_fn = root_distance_fn
        self.value_distance_fn = value_distance_fn
        self.edge_prediction_fn = edge_prediction_fn
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_table: Mapping[NodeKey, SearchNode] = MappingProxyType({})
        self.last_diagnostics: Mapping[str, Any] = MappingProxyType({})
        self.last_leaf_reward: Mapping[int, float] = MappingProxyType({})
        self.last_leaf_risk: Mapping[int, float] = MappingProxyType({})

    def plan(self, state: Any, achieved_goal: Any, final_goal: Any, positive_goals: Any,
             context: Any = None, training: bool = False) -> PlanResult:
        started = perf_counter()
        self._state = _vector(state, "state")
        self._achieved = _vector(achieved_goal, "achieved_goal")
        self._goal = _vector(final_goal, "final_goal", len(self._achieved))
        self._positives = np.asarray(positive_goals, dtype=np.float64)
        if self._positives.ndim != 2 or self._positives.shape[1] != len(self._goal) or not np.isfinite(self._positives).all():
            raise ValueError("positive_goals must be a finite [N, goal_dim] array")
        self._context = None if context is None else _vector(context, "context").astype(np.float32, copy=True)
        self._goal_id = len(self._positives)
        self._training = bool(training)
        self._prediction_cache: dict[tuple[int, int], EdgePrediction] = {}
        self._distance_cache: dict[tuple[int, int], float] = {}
        self._remaining_cache: dict[int, float] = {self._goal_id: 0.0}
        self._table: dict[NodeKey, SearchNode] = {}
        requested_simulations = int(self.cfg.pn_num_simulations)
        self._diagnostics = {"simulations": 0, "requested_simulations": requested_simulations,
                             "completed_simulations": 0, "cycles": 0, "leaf_uses": 0,
                             "simulated_safe_successes": 0, "simulated_violations": 0,
                             "branch_before": 0, "branch_eligible": 0, "branch_after": 0}
        self._prepare_callback_caches()
        self._build_leaf_tables()
        root_key = NodeKey(ROOT, int(self.cfg.pn_macro_depth))
        root = self._node(root_key)
        self._diagnostics["branch_before"] = self._root_raw_count()
        initial_eligible = [edge for edge in root.edges
                            if edge.feasibility_risk <= self.cfg.pn_search_risk_limit]
        self._diagnostics["branch_eligible"] = len(initial_eligible)
        self._diagnostics["branch_after"] = min(len(initial_eligible), int(self.cfg.pn_top_k))
        for _ in range(requested_simulations):
            reward, safety, selected, parents = self._simulate(root_key, frozenset({ROOT}), 0.0)
            for edge in selected:
                edge.update(reward, safety)
            for node in parents:
                node.update(reward, safety)
            self._diagnostics["simulations"] += 1
            self._diagnostics["completed_simulations"] += 1

        result = self._choose_root(root, training)
        diagnostics = dict(self._diagnostics)
        diagnostics["latency"] = perf_counter() - started
        diagnostics["transposition_count"] = len(self._table)
        diagnostics["no_safe_plan"] = result is None
        if result is not None:
            diagnostics.update({"chosen_p_violation": result.prediction.p_violation,
                                "chosen_q_risk": result.q_risk,
                                "chosen_upper_risk": result.upper_risk,
                                "chosen_node_id": result.target_id,
                                "chosen_command_goal": self._node_goal(result.target_id)})
            plan_result = PlanResult(PlanStatus.ACTION, self._node_goal(result.target_id), result.target_id, diagnostics)
        else:
            fallback = self._fallback(training)
            if fallback is not None:
                diagnostics.update({"unsafe_fallback": True,
                                    "chosen_p_violation": fallback.prediction.p_violation,
                                    "chosen_q_risk": fallback.q_risk,
                                    "chosen_upper_risk": fallback.upper_risk,
                                    "chosen_node_id": fallback.target_id,
                                    "chosen_command_goal": self._node_goal(fallback.target_id)})
                plan_result = PlanResult(PlanStatus.ACTION, self._node_goal(fallback.target_id), fallback.target_id, diagnostics)
            else:
                plan_result = PlanResult(PlanStatus.NO_SAFE_PLAN, self._goal, None, diagnostics)
        self.last_table = MappingProxyType(dict(self._table))
        self.last_diagnostics = MappingProxyType(dict(diagnostics))
        return plan_result

    def _node_goal(self, node_id: int) -> np.ndarray:
        if node_id == ROOT:
            return self._achieved.copy()
        if node_id == self._goal_id:
            return self._goal.copy()
        return self._positives[node_id].copy()

    def _prepare_callback_caches(self) -> None:
        """Evaluate the complete directed planning graph once for this plan.

        The distance callbacks return one finite non-negative scalar per batch
        row. ``edge_prediction_fn`` returns one EdgePrediction per paired
        ``starts``/``commands`` row; a scalar EdgePrediction is also accepted
        for a one-row batch.
        All callback inputs are owned float32 arrays; no cache is published
        until every callback result has passed validation.
        """
        targets = tuple(range(self._goal_id + 1))
        root_pairs = [(ROOT, target) for target in targets]
        internal_pairs = [(source, target) for source in range(self._goal_id)
                          for target in targets if target != source]
        root_targets = np.asarray([self._node_goal(target) for _, target in root_pairs],
                                  dtype=np.float32).copy()
        root_values = self._validated_distances(
            self.root_distance_fn(self._state.astype(np.float32, copy=True), root_targets),
            len(root_pairs), "root_distance_fn")

        if internal_pairs:
            sources = np.asarray([self._node_goal(source) for source, _ in internal_pairs],
                                 dtype=np.float32).copy()
            targets_array = np.asarray([self._node_goal(target) for _, target in internal_pairs],
                                       dtype=np.float32).copy()
            internal_values = self._validated_distances(
                self.value_distance_fn(sources, targets_array), len(internal_pairs), "value_distance_fn")
        else:
            internal_values = np.empty(0, dtype=np.float64)

        pairs = root_pairs + internal_pairs
        starts = np.asarray([self._node_goal(source) for source, _ in pairs], dtype=np.float32).copy()
        commands = np.asarray([self._node_goal(target) for _, target in pairs], dtype=np.float32).copy()
        predictions = self._validated_predictions(
            self.edge_prediction_fn(starts, commands,
                                    None if self._context is None else self._context.copy()), len(pairs))

        distances = {pair: float(value) for pair, value in zip(root_pairs, root_values)}
        distances.update({pair: float(value) for pair, value in zip(internal_pairs, internal_values)})
        prepared_predictions: dict[tuple[int, int], EdgePrediction] = {}
        for pair, prediction in zip(pairs, predictions):
            if (prediction.positive_drift_distribution is not None
                    and len(prediction.positive_drift_distribution) != self._goal_id):
                raise ValueError("positive_drift_distribution must match positive_goals")
            prepared_predictions[pair] = replace(prediction, local_distance=distances[pair])
        self._distance_cache = distances
        self._prediction_cache = prepared_predictions

    @staticmethod
    def _validated_distances(values: Any, length: int, callback_name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (length,):
            raise ValueError(f"{callback_name} must return one finite non-negative distance per input row")
        if not np.isfinite(result).all() or np.any(result < 0.0):
            raise ValueError(f"{callback_name} must return finite non-negative distances")
        return result.copy()

    def _validated_predictions(self, values: Any, length: int) -> list[EdgePrediction]:
        if isinstance(values, EdgePrediction):
            if length == 1:
                return [values]
            raise ValueError("edge_prediction_fn must return one prediction per input row")
        try:
            predictions = list(values)
        except TypeError as error:
            raise ValueError("edge_prediction_fn must return one prediction per input row") from error
        if len(predictions) != length:
            raise ValueError("edge_prediction_fn must return one prediction per input row")
        if not all(isinstance(prediction, EdgePrediction) for prediction in predictions):
            raise ValueError("edge_prediction_fn must return EdgePrediction values")
        return predictions

    def _distance(self, source_id: int, target_id: int) -> float:
        key = (source_id, target_id)
        try:
            return self._distance_cache[key]
        except KeyError as error:
            raise RuntimeError("distance cache missing prepared directed pair") from error

    def _remaining(self, node_id: int) -> float:
        if node_id not in self._remaining_cache:
            self._remaining_cache[node_id] = self._distance(node_id, self._goal_id)
        return self._remaining_cache[node_id]

    def _prediction(self, source_id: int, target_id: int) -> EdgePrediction:
        key = (source_id, target_id)
        try:
            return self._prediction_cache[key]
        except KeyError as error:
            raise RuntimeError("prediction cache missing prepared directed pair") from error

    def _candidate_edges(self, key: NodeKey) -> list[EdgeStats]:
        source = key.node_id
        items: list[tuple[float, int, EdgePrediction]] = []
        edge_limit = self.cfg.pn_epsilon_edge_train if self._training else self.cfg.pn_epsilon_edge_eval
        for target in range(self._goal_id + 1):
            if target == source:
                continue
            prediction = self._prediction(source, target)
            local = prediction.local_distance
            if local > self.cfg.d_max or prediction.p_violation > edge_limit:
                continue
            score = (log(prediction.p_target + self.cfg.pn_prior_epsilon)
                     - self.cfg.pn_beta_distance * (local + self._remaining(target))
                     - self.cfg.pn_beta_risk * prediction.p_violation)
            items.append((score, target, prediction))
        items.sort(key=lambda item: (-item[0], item[1]))
        return [EdgeStats(target, prediction, score, self.cfg.pn_risk_pseudocount, self.cfg.pn_risk_z)
                for score, target, prediction in items]

    def _node(self, key: NodeKey) -> SearchNode:
        node = self._table.get(key)
        if node is None:
            node = SearchNode(key, self._candidate_edges(key))
            self._table[key] = node
        return node

    def _selection_priors(self, node: SearchNode, path: frozenset[int]) -> list[tuple[EdgeStats, float]]:
        # A transposition can be reached through several histories, so this
        # check cannot be frozen when its reusable node object is created.
        feasible = [edge for edge in node.edges
                    if edge.target_id not in path
                    and edge.search_risk <= self.cfg.pn_search_risk_limit]
        feasible.sort(key=lambda edge: (-edge.prior_score, edge.target_id))
        feasible = feasible[:int(self.cfg.pn_top_k)]
        if not feasible:
            return []
        scores = np.asarray([edge.prior_score for edge in feasible], dtype=np.float64)
        priors = np.exp(scores - scores.max())
        priors /= priors.sum()
        return list(zip(feasible, priors.tolist()))

    def _select(self, node: SearchNode, path: frozenset[int]) -> Optional[EdgeStats]:
        local_priors = self._selection_priors(node, path)
        if not local_priors:
            return None
        scale = sqrt(max(node.visits, 1))
        def score(edge: EdgeStats, prior: float) -> float:
            return (edge.q_reward - self.cfg.pn_lambda_search_risk * edge.q_risk
                    + self.cfg.pn_c_puct * prior * scale / (1 + edge.visits))
        return min(local_priors, key=lambda item: (-score(*item), item[0].target_id))[0]

    def _simulate(self, key: NodeKey, path: frozenset[int], elapsed: float) -> tuple[float, float, list[EdgeStats], list[SearchNode]]:
        node = self._node(key)
        parents = [node]
        if key.node_id == self._goal_id:
            return self._goal_return(elapsed), 0.0, [], parents
        if key.remaining_macro_depth == 0:
            return self._leaf(key.node_id) + ([], parents)
        edge = self._select(node, path)
        if edge is None:
            return self._leaf(key.node_id) + ([], parents)
        outcome = int(self.rng.choice(4, p=edge.prediction.outcome_probs))
        elapsed += edge.prediction.duration_by_outcome[outcome]
        selected = [edge]
        if outcome == 3:
            self._diagnostics["simulated_violations"] += 1
            return -self.cfg.pn_violation_penalty, 1.0, selected, parents
        if outcome == 2 or (outcome == 1 and not edge.prediction.positive_drift_distribution):
            return -self.cfg.pn_stuck_penalty, 0.0, selected, parents
        if outcome == 0:
            target = edge.target_id
        else:
            distribution = np.asarray(edge.prediction.positive_drift_distribution, dtype=np.float64)
            if len(distribution) != self._goal_id:
                raise ValueError("positive_drift_distribution must match positive_goals")
            target = int(self.rng.choice(self._goal_id, p=distribution))
        if target in path:
            self._diagnostics["cycles"] += 1
            return -self.cfg.pn_loop_penalty, 0.0, selected, parents
        if target == self._goal_id:
            self._diagnostics["simulated_safe_successes"] += 1
            return self._goal_return(elapsed), 0.0, selected, parents
        child_key = NodeKey(target, key.remaining_macro_depth - 1)
        reward, safety, child_edges, child_parents = self._simulate(child_key, path | {target}, elapsed)
        return reward, safety, selected + child_edges, parents + child_parents

    def _goal_return(self, elapsed: float) -> float:
        horizon = max(float(getattr(self.cfg, "max_episode_steps", 1)), 1.0)
        return 1.0 - self.cfg.pn_time_penalty * elapsed / horizon

    def _leaf(self, node_id: int) -> tuple[float, float]:
        self._diagnostics["leaf_uses"] += 1
        return self.last_leaf_reward.get(node_id, 0.0), self.last_leaf_risk.get(node_id, 1.0)

    def _build_leaf_tables(self) -> None:
        n = self._goal_id + 1
        neg_inf = float(getattr(self.cfg, "neg_inf", -1e6))
        weights = np.full((n, n), neg_inf, dtype=np.float64)
        risk = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(weights, 0.0)
        np.fill_diagonal(risk, 0.0)
        for source in range(self._goal_id):
            for target in range(n):
                if source == target:
                    continue
                prediction = self._prediction(source, target)
                if prediction.local_distance > self.cfg.d_max:
                    continue
                expected = (np.dot(np.asarray(prediction.outcome_probs), np.asarray(prediction.duration_by_outcome))
                            + self.cfg.pn_lambda_stuck_leaf * prediction.p_stuck
                            + self.cfg.pn_lambda_violation_leaf * prediction.p_violation)
                weights[source, target] = -expected
                risk[source, target] = -log(max(1.0 - prediction.p_violation, self.cfg.pn_leaf_epsilon))
        weights[self._goal_id, :] = neg_inf
        weights[self._goal_id, self._goal_id] = 0.0
        soft = self.graph_search.soft_floyd(torch.as_tensor(weights, dtype=torch.float32)).detach().cpu().numpy()
        for middle in range(n):
            risk = np.minimum(risk, risk[:, middle, None] + risk[middle, None, :])
        rewards: dict[int, float] = {}
        risks: dict[int, float] = {}
        for source in range(n):
            value = float(soft[source, self._goal_id])
            reachable = isfinite(value) and value > neg_inf * 0.5
            rewards[source] = exp(-max(-value, 0.0) / self.cfg.pn_leaf_temperature) if reachable else 0.0
            minimum = risk[source, self._goal_id]
            risks[source] = 1.0 - exp(-minimum) if isfinite(minimum) else 1.0
        self.last_leaf_reward = MappingProxyType(rewards)
        self.last_leaf_risk = MappingProxyType(risks)

    def _root_raw_count(self) -> int:
        return sum(1 for target in range(self._goal_id + 1)
                   if target != ROOT and self._distance(ROOT, target) <= self.cfg.d_max)

    def _choose_root(self, root: SearchNode, training: bool) -> Optional[EdgeStats]:
        self._training = training
        eligible = [edge for edge in root.edges if edge.visits and edge.upper_risk <= self.cfg.pn_root_risk_limit]
        if not eligible:
            return None
        return min(eligible, key=lambda edge: (-edge.visits, -edge.q_reward, edge.q_risk, edge.target_id))

    def _fallback(self, training: bool) -> Optional[EdgeStats]:
        if not training or not self.cfg.pn_training_unsafe_fallback:
            return None
        root = self._table[NodeKey(ROOT, int(self.cfg.pn_macro_depth))]
        return min(root.edges, key=lambda edge: (edge.prediction.p_violation, edge.prediction.local_distance, edge.target_id), default=None)
