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
    SearchNode,
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
        self.e_calls = []
        self.e_batches = []

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
        pairs = [(int(start[0]), int(command[0])) for start, command in zip(starts, commands)]
        self.e_calls.extend(pairs)
        self.e_batches.append(pairs)
        return [self.predictions[pair] for pair in pairs]


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


def test_violation_backup_updates_every_edge_and_node_in_a_three_edge_rollout():
    distances = _complete_distances(nodes=(-1, 0, 1, 2), default=99)
    distances.update({(ROOT, 0): 1, (0, 1): 1, (1, 2): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    predictions[(1, 2)] = _prediction(target=0, violation=1)
    planner, _, _ = _planner(distances, predictions, d_max=1, pn_num_simulations=1,
                              pn_macro_depth=3, pn_search_risk_limit=1,
                              pn_root_risk_limit=1)
    state, achieved, goal, positives = _inputs((0, 1), 2)

    planner.plan(state, achieved, goal, positives)

    for key, target in ((NodeKey(ROOT, 3), 0), (NodeKey(0, 2), 1), (NodeKey(1, 1), 2)):
        node = planner.last_table[key]
        edge = next(edge for edge in node.edges if edge.target_id == target)
        assert edge.visits == 1 and edge.alpha > 1.0
        assert node.visits == 1 and node.alpha == pytest.approx(2.0)


def test_drift_uses_its_sampled_destination_and_outcome_durations_in_goal_reward():
    distances = _complete_distances(nodes=(-1, 0, 1, 2), default=99)
    distances.update({(ROOT, 0): 1, (1, 2): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    predictions[(ROOT, 0)] = _prediction(target=0, drift=1, drift_dist=[0, 1],
                                          duration=(1, 2, 3, 4))
    predictions[(1, 2)] = _prediction(duration=(4, 1, 1, 1))
    planner, _, _ = _planner(distances, predictions, d_max=1, pn_num_simulations=1,
                              pn_macro_depth=3, pn_time_penalty=0.5,
                              max_episode_steps=10)
    state, achieved, goal, positives = _inputs((0, 1), 2)

    planner.plan(state, achieved, goal, positives)

    assert planner.last_diagnostics["simulated_safe_successes"] == 1
    assert planner.last_table[NodeKey(ROOT, 3)].edges[0].q_reward == pytest.approx(0.7)


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


def test_transposition_candidates_are_path_independent_but_selection_is_not():
    """A shared node must not retain the first path's cycle mask."""
    distances = _complete_distances(default=99)
    distances.update({(ROOT, 0): 1, (ROOT, 1): 1, (0, JOIN): 1,
                      (1, JOIN): 1, (JOIN, 0): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, d_max=1, pn_num_simulations=12,
                              pn_macro_depth=3, pn_c_puct=20.0)
    state, achieved, goal, positives = _inputs((0, 1, JOIN), GOAL)

    planner.plan(state, achieved, goal, positives)

    join = planner.last_table[NodeKey(JOIN, 1)]
    assert join.visits >= 2
    assert [edge.target_id for edge in join.edges] == [0]
    selected = planner._select(join, frozenset({ROOT, 1, JOIN}))
    assert selected is not None and selected.target_id == 0
    assert planner._select(join, frozenset({ROOT, 0, JOIN})) is None
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

    result = planner.plan(state, achieved, goal, positives, training=True)

    assert result.status is PlanStatus.ACTION
    assert {"unsafe_fallback", "chosen_p_violation", "chosen_q_risk", "chosen_upper_risk",
            "chosen_node_id", "chosen_command_goal"} <= result.diagnostics.keys()
    assert result.diagnostics["unsafe_fallback"] is True
    assert result.diagnostics["chosen_node_id"] == result.node_id
    assert np.array_equal(result.diagnostics["chosen_command_goal"], result.command_goal)
    assert all(np.isfinite(result.diagnostics[name])
               for name in ("chosen_p_violation", "chosen_q_risk", "chosen_upper_risk"))


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


def test_root_selection_breaks_ties_by_visits_reward_posterior_risk_then_id():
    distances = _complete_distances(nodes=(-1, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, pn_root_risk_limit=1,
                              pn_search_risk_limit=1)

    def edge(target, visits, reward_sum, posterior_risk, immediate_risk):
        result = EdgeStats(target, _prediction(target=1 - immediate_risk, violation=immediate_risk),
                           prior_score=0, pseudocount=0, risk_z=0)
        result.visits = visits
        result.reward_sum = reward_sum
        result.alpha, result.beta = posterior_risk * 10, (1 - posterior_risk) * 10
        return result

    root = SearchNode(NodeKey(ROOT, 1), [
        edge(4, visits=5, reward_sum=-5, posterior_risk=0.9, immediate_risk=0),
        edge(3, visits=4, reward_sum=40, posterior_risk=0.1, immediate_risk=1),
    ])
    assert planner._choose_root(root, training=False).target_id == 4

    root.edges = [
        edge(3, visits=5, reward_sum=15, posterior_risk=0.9, immediate_risk=0),
        edge(1, visits=5, reward_sum=20, posterior_risk=0.9, immediate_risk=0),
        edge(2, visits=5, reward_sum=20, posterior_risk=0.1, immediate_risk=1),
        edge(0, visits=5, reward_sum=20, posterior_risk=0.1, immediate_risk=1),
    ]
    assert planner._choose_root(root, training=False).target_id == 0


def test_candidates_topk_prior_final_goal_and_cycle_filtering():
    distances = _complete_distances(nodes=(-1, 0, 1, 2, 3))
    distances.update({(ROOT, 0): 3, (ROOT, 1): 1, (ROOT, 2): 2, (ROOT, 3): 4,
                      (0, 3): 1, (1, 3): 1, (2, 3): 1})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, callbacks, _ = _planner(distances, predictions, pn_top_k=2, pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0, 1, 2), 3)
    planner.plan(state, achieved, goal, positives)
    root_edges = planner.last_table[NodeKey(ROOT, 3)].edges
    assert [edge.target_id for edge in root_edges] == [1, 2, 0, 3]
    assert (ROOT, GOAL) in callbacks.e_calls


def test_top_k_is_applied_after_path_masking_with_local_normalized_priors():
    distances = _complete_distances(nodes=(-1, 0, 1, 2, 3), default=99)
    distances.update({(JOIN, 0): 1, (JOIN, 1): 1, (JOIN, GOAL): 1,
                      (0, GOAL): 0, (1, GOAL): 0})
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    predictions[(JOIN, GOAL)] = _prediction(target=0.5, stuck=0.5)
    planner, _, _ = _planner(distances, predictions, d_max=1, pn_top_k=2,
                              pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0, 1, JOIN), GOAL)

    planner.plan(state, achieved, goal, positives)
    join = planner._node(NodeKey(JOIN, 1))
    masked_path = frozenset({ROOT, 0, 1, JOIN})

    assert [edge.target_id for edge in join.edges] == [0, 1, GOAL]
    scores = [edge.prior_score for edge in join.edges]
    partially_masked = planner._selection_priors(join, frozenset({ROOT, 0, JOIN}))
    assert [edge.target_id for edge, _ in partially_masked] == [1, GOAL]
    assert sum(prior for _, prior in partially_masked) == pytest.approx(1.0)
    assert partially_masked[0][1] > partially_masked[1][1]
    assert planner._selection_priors(join, masked_path) == [(join.edges[-1], 1.0)]
    assert planner._select(join, masked_path).target_id == GOAL
    assert [edge.prior_score for edge in join.edges] == scores


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


def test_unreachable_leaf_uses_zero_reward_and_unit_risk():
    distances = _complete_distances(nodes=(-1, 0, 1, 2), default=99)
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, graph = _planner(distances, predictions, d_max=1, pn_num_simulations=1)
    state, achieved, goal, positives = _inputs((0, 1), 2)

    planner.plan(state, achieved, goal, positives)

    assert graph.calls == 1
    assert planner.last_leaf_reward[0] == 0.0
    assert planner.last_leaf_risk[0] == 1.0


def test_callback_batches_cover_every_directed_edge_and_reset_per_plan():
    positives = tuple(range(20))
    goal_id = len(positives)
    nodes = (ROOT, *positives, goal_id)
    distances = _complete_distances(nodes=nodes)
    predictions = {(source, target): _prediction()
                   for source in nodes for target in nodes
                   if target != source and target != ROOT}
    planner, callbacks, _ = _planner(distances, predictions, pn_num_simulations=20)
    state, achieved, goal, positive_goals = _inputs(positives, goal_id)

    planner.plan(state, achieved, goal, positive_goals)

    root_pairs = {(ROOT, target) for target in (*positives, goal_id)}
    internal_pairs = {(source, target) for source in positives
                      for target in (*positives, goal_id) if target != source}
    expected_pairs = root_pairs | internal_pairs
    assert len(callbacks.d_calls) == 1
    assert len(callbacks.v_calls) == 1
    assert len(callbacks.e_batches) == 1
    assert callbacks.d_calls[0].dtype == np.float32
    assert callbacks.v_calls[0][0].dtype == callbacks.v_calls[0][1].dtype == np.float32
    assert {(ROOT, int(target[0])) for target in callbacks.d_calls[0]} == root_pairs
    assert {(int(source[0]), int(target[0])) for source, target in zip(*callbacks.v_calls[0])} == internal_pairs
    assert set(callbacks.e_calls) == expected_pairs
    assert len(callbacks.e_calls) == len(expected_pairs)
    assert planner._distance(ROOT, 7) == pytest.approx(distances[(ROOT, 7)])
    assert planner._distance(7, 11) == pytest.approx(distances[(7, 11)])
    assert planner._prediction(7, 11).p_target == pytest.approx(predictions[(7, 11)].p_target)

    first_batches = (len(callbacks.d_calls), len(callbacks.v_calls), len(callbacks.e_batches))
    for _ in range(20):
        planner._distance(ROOT, 7)
        planner._distance(7, 11)
        planner._prediction(7, 11)
    assert (len(callbacks.d_calls), len(callbacks.v_calls), len(callbacks.e_batches)) == first_batches

    planner.plan(state, achieved, goal, positive_goals)
    assert (len(callbacks.d_calls), len(callbacks.v_calls), len(callbacks.e_batches)) == (2, 2, 2)


def test_callback_batches_reject_wrong_shapes_before_publishing_caches():
    distances = _complete_distances(nodes=(ROOT, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions)
    planner.edge_prediction_fn = lambda starts, commands, context: _prediction()

    with pytest.raises(ValueError, match="one prediction per input row"):
        planner.plan(*_inputs((0,), 1))

    assert planner._distance_cache == {}
    assert planner._prediction_cache == {}


def test_zero_positive_plan_accepts_single_scalar_edge_prediction():
    distances = {(ROOT, 0): 1.0}
    predictions = {(ROOT, 0): _prediction()}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1)
    state = np.array([99.0])
    achieved = np.array([-1.0])
    goal = np.array([0.0])
    positives = np.empty((0, 1))
    planner.edge_prediction_fn = lambda starts, commands, context: predictions[(ROOT, 0)]

    result = planner.plan(state, achieved, goal, positives)

    assert result.status is PlanStatus.ACTION
    assert result.node_id == 0
    assert np.array_equal(result.command_goal, goal)


def test_edge_callback_rejects_scalar_prediction_for_multirow_batch():
    distances = _complete_distances(nodes=(ROOT, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions)
    planner.edge_prediction_fn = lambda starts, commands, context: _prediction()

    with pytest.raises(ValueError, match="one prediction per input row"):
        planner.plan(*_inputs((0,), 1))


def test_callbacks_receive_owned_contiguous_float32_inputs_and_cannot_mutate_plan_arguments():
    distances = _complete_distances(nodes=(ROOT, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1)
    state = np.array([99.0], dtype=np.float64)
    achieved = np.array([-1.0], dtype=np.float64)
    goal = np.array([1.0], dtype=np.float64)
    positives = np.array([[0.0]], dtype=np.float64)
    context = np.array([7.0], dtype=np.float64)
    originals = tuple(value.copy() for value in (state, achieved, goal, positives, context))

    def assert_callback_array(value):
        assert value.dtype == np.float32
        assert value.flags.c_contiguous and value.flags.owndata

    def root(callback_state, targets):
        assert_callback_array(callback_state)
        assert_callback_array(targets)
        result = np.array([distances[(ROOT, int(target[0]))] for target in targets])
        callback_state.fill(-99.0)
        targets.fill(-99.0)
        return result

    def value(sources, targets):
        assert_callback_array(sources)
        assert_callback_array(targets)
        result = np.array([distances[(int(source[0]), int(target[0]))]
                           for source, target in zip(sources, targets)])
        sources.fill(-99.0)
        targets.fill(-99.0)
        return result

    def edge(starts, commands, callback_context):
        assert_callback_array(starts)
        assert_callback_array(commands)
        assert_callback_array(callback_context)
        result = [predictions[(int(start[0]), int(command[0]))]
                  for start, command in zip(starts, commands)]
        starts.fill(-99.0)
        commands.fill(-99.0)
        callback_context.fill(-99.0)
        return result

    planner.root_distance_fn = root
    planner.value_distance_fn = value
    planner.edge_prediction_fn = edge

    result = planner.plan(state, achieved, goal, positives, context=context)

    assert result.status is PlanStatus.ACTION
    assert np.array_equal(planner._context, context.astype(np.float32))
    for value, original in zip((state, achieved, goal, positives, context), originals):
        assert np.array_equal(value, original)


@pytest.mark.parametrize("argument, value, message", [
    ("state", np.array([np.inf]), "state"),
    ("achieved_goal", np.array([np.nan]), "achieved_goal"),
    ("final_goal", np.array([np.inf]), "final_goal"),
    ("positive_goals", np.array([[np.nan]]), "positive_goals"),
    ("context", np.array([np.inf]), "context"),
])
def test_plan_rejects_nonfinite_inputs_before_callback_side_effects(argument, value, message):
    distances = _complete_distances(nodes=(-1, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, callbacks, _ = _planner(distances, predictions)
    state, achieved, goal, positives = _inputs((0,), 1)
    arguments = dict(state=state, achieved_goal=achieved, final_goal=goal,
                     positive_goals=positives, context=np.array([0.0]))
    arguments[argument] = value

    with pytest.raises(ValueError, match=message):
        planner.plan(**arguments)

    assert not callbacks.d_calls and not callbacks.v_calls and not callbacks.e_calls


def test_plan_rejects_invalid_callback_shapes_and_drift_distributions():
    distances = _complete_distances(nodes=(-1, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, graph = _planner(distances, predictions)
    state, achieved, goal, positives = _inputs((0,), 1)
    planner.root_distance_fn = lambda state, targets: np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="root_distance_fn"):
        planner.plan(state, achieved, goal, positives)
    assert planner.last_table == {}

    planner, _, _ = _planner(distances, predictions)
    planner.edge_prediction_fn = lambda starts, commands, context: [_prediction(), _prediction()]
    with pytest.raises(ValueError, match="one prediction"):
        planner.plan(state, achieved, goal, positives)

    bad_distances = _complete_distances(nodes=(-1, 0, 1, 2))
    bad = _prediction(target=0, drift=1, drift_dist=[1])
    planner, _, _ = _planner(bad_distances, {key: bad for key in bad_distances if key[0] != key[1]})
    with pytest.raises(ValueError, match="positive_drift_distribution"):
        planner.plan(*_inputs((0, 1), 2))


def test_diagnostics_report_requested_and_completed_simulations_with_finite_metrics():
    distances = _complete_distances(nodes=(-1, 0, 1))
    predictions = {(source, target): _prediction()
                   for source, target in distances if source != target}
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=3)
    state, achieved, goal, positives = _inputs((0,), 1)

    result = planner.plan(state, achieved, goal, positives)

    diagnostics = result.diagnostics
    assert diagnostics["requested_simulations"] == diagnostics["completed_simulations"] == 3
    for name in ("latency", "branch_before", "branch_eligible", "branch_after", "transposition_count",
                 "simulated_safe_successes", "simulated_violations", "cycles", "leaf_uses"):
        assert np.isfinite(diagnostics[name]) and diagnostics[name] >= 0
    assert diagnostics["no_safe_plan"] is False


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
    edge = EdgeStats(target_id=0, prediction=_prediction(target=0.75, violation=0.25), prior_score=0.0,
                     pseudocount=4.0, risk_z=1.0)
    assert edge.alpha == pytest.approx(2.0)
    assert edge.beta == pytest.approx(4.0)
    edge.update(0.5, 1.0)
    assert edge.q_reward == pytest.approx(0.5)
    assert edge.q_risk == pytest.approx(3 / 7)


@pytest.mark.parametrize("p_violation", [0.0, 0.25, 0.75, 1.0])
@pytest.mark.parametrize("outcomes", [(), (0.0,), (1.0,), (0.0, 1.0, 1.0)])
def test_edge_upper_risk_never_understates_posterior_mean(p_violation, outcomes):
    edge = EdgeStats(target_id=0, prediction=_prediction(target=1.0 - p_violation,
                                                           violation=p_violation),
                     prior_score=0.0, pseudocount=4.0, risk_z=1.645)
    for safety in outcomes:
        edge.update(reward_return=0.0, safety_return=safety)
    assert edge.upper_risk >= edge.q_risk


def test_root_branch_after_reflects_risk_filtered_dynamic_top_k_frontier():
    positives = (0, 1, 2, 3, 4)
    goal_id = len(positives)
    nodes = (ROOT, *positives, goal_id)
    distances = _complete_distances(nodes=nodes)
    predictions = {(source, target): _prediction()
                   for source in nodes for target in nodes
                   if target != source and target != ROOT}
    predictions[(ROOT, 3)] = _prediction(target=0.5, violation=0.5)
    predictions[(ROOT, 4)] = _prediction(target=0.5, violation=0.5)
    planner, _, _ = _planner(distances, predictions, pn_num_simulations=1, pn_top_k=2,
                              pn_search_risk_limit=0.2, pn_root_risk_limit=0.2,
                              pn_epsilon_edge_eval=1.0)

    result = planner.plan(*_inputs(positives, goal_id))

    assert result.diagnostics["branch_before"] == 6
    assert result.diagnostics["branch_eligible"] == 4
    assert result.diagnostics["branch_after"] == 2
