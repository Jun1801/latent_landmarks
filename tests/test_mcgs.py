"""Deterministic contracts for the risk-constrained macro graph search."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from l3p.config import get_config
from l3p.pn_lmcgs.mcgs import (
    EdgePrediction,
    EdgeStats,
    NodeKey,
    PlanStatus,
    RiskConstrainedMCGS,
)


ROOT, SHORT, LONG, JOIN, GOAL = -1, 0, 1, 2, 3


class _IdentityGraphSearch:
    """A soft-Floyd double which leaves the supplied weights inspectable."""

    def __init__(self):
        self.calls = 0
        self.weights = None

    def soft_floyd(self, weights):
        self.calls += 1
        self.weights = weights.detach().cpu().numpy().copy()
        result = weights.clone()
        for middle in range(result.shape[0]):
            result = torch.maximum(result, result[:, middle, None] + result[middle, None, :])
        return result


class _Callbacks:
    def __init__(self, distances, predictions):
        self.distances = distances
        self.predictions = predictions
        self.d_calls = []
        self.v_calls = []

    def root(self, state, targets):
        self.d_calls.append(np.asarray(targets).copy())
        return np.array([self.distances[(ROOT, int(goal[0]))] for goal in targets])

    def value(self, sources, targets):
        self.v_calls.append((np.asarray(sources).copy(), np.asarray(targets).copy()))
        return np.array([
            self.distances[(int(source[0]), int(target[0]))]
            for source, target in zip(sources, targets)
        ])

    def edge(self, starts, commands, context):
        return [self.predictions[(int(start[0]), int(command[0]))] for start, command in zip(starts, commands)]


def _prediction(*, target=1.0, drift=0.0, stuck=0.0, violation=0.0,
                drift_dist=None, duration=(1, 1, 1, 1)):
    return EdgePrediction(
        p_target=target,
        p_drift=drift,
        p_stuck=stuck,
        p_violation=violation,
        positive_drift_distribution=drift_dist,
        duration_by_outcome=duration,
    )


def _planner(distances, predictions, **overrides):
    cfg = get_config(
        "PointMaze", d_max=overrides.pop("d_max", 20.0), neg_inf=-1e6, soft_iters=1, beta=1.0,
        pn_num_simulations=overrides.pop("pn_num_simulations", 32),
        pn_macro_depth=overrides.pop("pn_macro_depth", 3),
        pn_top_k=overrides.pop("pn_top_k", 5),
        pn_search_risk_limit=overrides.pop("pn_search_risk_limit", 0.99),
        pn_root_risk_limit=overrides.pop("pn_root_risk_limit", 0.99),
        pn_epsilon_edge_eval=overrides.pop("pn_epsilon_edge_eval", 1.0),
        pn_epsilon_edge_train=overrides.pop("pn_epsilon_edge_train", 1.0),
        pn_time_penalty=overrides.pop("pn_time_penalty", 0.0),
        **overrides,
    )
    callbacks = _Callbacks(distances, predictions)
    graph = _IdentityGraphSearch()
    return RiskConstrainedMCGS(cfg, graph, callbacks.root, callbacks.value, callbacks.edge,
                               rng=np.random.default_rng(7)), callbacks, graph


def _inputs(positives=(0, 1, 2), goal=3):
    return (np.array([99.0]), np.array([-1.0]), np.array([goal], dtype=np.float64),
            np.array([[node] for node in positives], dtype=np.float64))


def _complete_distances(nodes=(-1, 0, 1, 2, 3), default=10.0):
    return {(source, target): (0.0 if source == target else default)
            for source in nodes for target in nodes}


def test_zero_risk_chooses_shortest_route_and_returns_goal_copy():
    distances = _complete_distances()
    distances.update({(ROOT, SHORT): 1, (ROOT, LONG): 3, (ROOT, GOAL): 9,
                      (SHORT, GOAL): 1, (LONG, GOAL): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions)
    state, achieved, goal, positives = _inputs()

    result = planner.plan(state, achieved, goal, positives)

    assert result.status is PlanStatus.ACTION
    assert result.node_id == SHORT
    assert np.array_equal(result.command_goal, positives[SHORT])
    assert result.command_goal is not positives[SHORT]


def test_risky_short_route_yields_safe_long_route_under_root_threshold():
    distances = _complete_distances()
    distances.update({(ROOT, SHORT): 1, (ROOT, LONG): 3, (SHORT, GOAL): 1, (LONG, GOAL): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    predictions[(ROOT, SHORT)] = _prediction(target=0.2, violation=0.8)
    planner, _, _ = _planner(distances, predictions, pn_root_risk_limit=0.2,
                              pn_search_risk_limit=0.9)
    state, achieved, goal, positives = _inputs()

    assert planner.plan(state, achieved, goal, positives).node_id == LONG


def test_violation_backup_marks_every_selected_edge_unsafe():
    distances = _complete_distances(nodes=(-1, 0, 3))
    distances.update({(ROOT, 0): 1, (0, 3): 1})
    predictions = {(ROOT, 0): _prediction(target=0.0, violation=1.0),
                   (ROOT, 3): _prediction(target=0.0, violation=1.0),
                   (0, 3): _prediction()}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1,
                              pn_root_risk_limit=1.0, pn_search_risk_limit=1.0)
    state, achieved, goal, positives = _inputs((0,), 3)

    planner.plan(state, achieved, goal, positives)

    edge = planner.last_table[NodeKey(ROOT, 3)].edges[0]
    assert edge.visits == 1 and edge.alpha > 1.0 and edge.q_risk > 0.0
    assert planner.last_diagnostics["simulated_violations"] == 1


def test_drift_moves_to_sampled_positive_and_stuck_terminates():
    distances = _complete_distances(nodes=(-1, 0, 1, 3))
    distances.update({(ROOT, 0): 1, (ROOT, 1): 99, (ROOT, 3): 99, (0, 1): 1, (1, 3): 1})
    predictions = {(ROOT, 0): _prediction(target=0, drift=1, drift_dist=[0, 1]),
                   (ROOT, 1): _prediction(), (ROOT, 3): _prediction(),
                   (0, 1): _prediction(), (0, 3): _prediction(), (1, 0): _prediction(),
                   (1, 3): _prediction()}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0, 1), 3)
    planner.plan(state, achieved, goal, positives)
    assert planner.last_diagnostics["simulated_safe_successes"] == 1

    predictions[(ROOT, 0)] = _prediction(target=0, stuck=1)
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1)
    planner.plan(state, achieved, goal, positives)
    assert planner.last_diagnostics["simulated_safe_successes"] == 0


def test_repeated_drift_path_gets_loop_penalty():
    distances = _complete_distances(nodes=(-1, 0, 3))
    distances.update({(ROOT, 0): 1, (ROOT, 3): 99, (0, 3): 1})
    predictions = {(ROOT, 0): _prediction(),
                   (ROOT, 3): _prediction(), (0, 3): _prediction(target=0, drift=1, drift_dist=[1])}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0,), 3)
    planner.plan(state, achieved, goal, positives)
    assert planner.last_diagnostics["cycles"] == 1


def test_root_uses_d_only_and_internal_edges_use_v_only():
    distances = _complete_distances()
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, callbacks, _ = _planner(distances, predictions)
    state, achieved, goal, positives = _inputs()

    planner.plan(state, achieved, goal, positives)

    assert callbacks.d_calls
    assert callbacks.v_calls
    assert all(np.all(source != -1) and np.all(target != -1)
               for source, target in callbacks.v_calls)


def test_diamond_transposition_shares_node_stats_and_resets_per_plan():
    distances = _complete_distances()
    distances.update({(ROOT, 0): 1, (ROOT, 1): 1, (0, JOIN): 1, (1, JOIN): 1,
                      (JOIN, GOAL): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=20,
                              pn_macro_depth=3, pn_c_puct=20.0)
    state, achieved, goal, positives = _inputs((0, 1, JOIN), GOAL)
    planner.plan(state, achieved, goal, positives)
    assert planner.last_table[NodeKey(JOIN, 1)].visits > 1
    first_table = planner.last_table
    planner.plan(state, achieved, goal, positives)
    assert planner.last_table is not first_table


def test_no_feasible_root_and_training_fallback_rules():
    distances = _complete_distances(nodes=(-1, 0, 3))
    predictions = {(ROOT, 0): _prediction(target=0.1, violation=0.9),
                   (ROOT, 3): _prediction(target=0.1, violation=0.9),
                   (0, 3): _prediction()}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=2,
                              pn_epsilon_edge_eval=1.0, pn_epsilon_edge_train=1.0,
                              pn_root_risk_limit=0.1, pn_search_risk_limit=1.0,
                              pn_training_unsafe_fallback=True)
    state, achieved, goal, positives = _inputs((0,), 3)
    assert planner.plan(state, achieved, goal, positives, training=False).status is PlanStatus.NO_SAFE_PLAN
    assert planner.plan(state, achieved, goal, positives, training=True).status is PlanStatus.ACTION


def test_cold_start_uses_immediate_risk_but_root_uses_posterior_upper_bound():
    distances = _complete_distances(nodes=(-1, 0, 3))
    predictions = {(ROOT, 0): _prediction(), (ROOT, 3): _prediction(), (0, 3): _prediction()}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1,
                              pn_search_risk_limit=0.2, pn_root_risk_limit=0.2,
                              pn_risk_pseudocount=4.0)
    state, achieved, goal, positives = _inputs((0,), 3)
    planner.plan(state, achieved, goal, positives)
    edge = planner.last_table[NodeKey(ROOT, 3)].edges[0]
    assert edge.visits == 1 and edge.feasibility_risk > 0.2
    assert edge.upper_risk == pytest.approx(edge.q_risk + planner.cfg.pn_risk_z * edge.risk_std)


def test_candidates_topk_prior_final_goal_and_cycle_filtering():
    distances = _complete_distances(nodes=(-1, 0, 1, 2, 3))
    distances.update({(ROOT, 0): 3, (ROOT, 1): 1, (ROOT, 2): 2, (ROOT, 3): 4,
                      (0, 3): 1, (1, 3): 1, (2, 3): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, pn_top_k=2, pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0, 1, 2), 3)
    planner.plan(state, achieved, goal, positives)
    root_edges = planner.last_table[NodeKey(ROOT, 3)].edges
    assert [edge.target_id for edge in root_edges] == [1, 2]
    assert sum(edge.prior for edge in root_edges) == pytest.approx(1.0)


def test_leaf_soft_floyd_once_and_exact_risk_no_path_values():
    distances = _complete_distances(nodes=(-1, 0, 1, 3), default=99)
    distances.update({(ROOT, 0): 1, (ROOT, 1): 1, (0, 1): 1, (1, 3): 1})
    predictions = {(ROOT, 0): _prediction(), (ROOT, 1): _prediction(), (ROOT, 3): _prediction(),
                   (0, 1): _prediction(violation=0.1, target=0.9),
                   (1, 3): _prediction(), (0, 3): _prediction(), (1, 0): _prediction()}
    planner, _, graph = _planner(distances, predictions, pn_num_simulations=1, d_max=10)
    state, achieved, goal, positives = _inputs((0, 1), 3)
    planner.plan(state, achieved, goal, positives)
    assert graph.calls == 1
    assert planner.last_leaf_reward[0] == pytest.approx(np.exp(-2.1))
    assert planner.last_leaf_risk[0] == pytest.approx(0.1)
    assert planner.last_leaf_reward[1] == pytest.approx(np.exp(-1.0))


def test_prediction_validation_and_seeded_outcomes_are_deterministic():
    with pytest.raises(ValueError):
        EdgePrediction(p_target=0.5, p_drift=0.5, p_stuck=0.5, p_violation=0.0)
    with pytest.raises(ValueError):
        EdgePrediction(p_target=1.0, duration_by_outcome=(0, 1, 1, 1))

    distances = _complete_distances(nodes=(-1, 0, 3))
    predictions = {(ROOT, 0): _prediction(target=0.5, stuck=0.5),
                   (ROOT, 3): _prediction(target=0.5, stuck=0.5), (0, 3): _prediction()}
    state, achieved, goal, positives = _inputs((0,), 3)
    first, _, _ = _planner(distances, predictions, pn_num_simulations=8)
    second, _, _ = _planner(distances, predictions, pn_num_simulations=8)
    first_diagnostics = dict(first.plan(state, achieved, goal, positives).diagnostics)
    second_diagnostics = dict(second.plan(state, achieved, goal, positives).diagnostics)
    first_diagnostics.pop("latency")
    second_diagnostics.pop("latency")
    assert first_diagnostics == second_diagnostics


def test_edge_stats_posterior_math():
    edge = EdgeStats(target_id=0, prediction=_prediction(target=0.75, violation=0.25), prior=1.0,
                     pseudocount=4.0, risk_z=1.0)
    assert edge.alpha == pytest.approx(2.0)
    assert edge.beta == pytest.approx(4.0)
    edge.update(0.5, 1.0)
    assert edge.q_reward == pytest.approx(0.5)
    assert edge.q_risk == pytest.approx(3 / 7)
