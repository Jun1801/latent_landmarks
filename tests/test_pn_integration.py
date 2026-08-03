"""Contracts for the baseline command API and PN-LMCGS model adapter."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from l3p.config import get_config
from l3p.models.landmarks import LatentLandmarks
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.pn_lmcgs.mcgs import PlanStatus
from l3p.pn_lmcgs.macro_transition_model import MacroTransitionModel
from l3p.pn_lmcgs.planner_adapter import PNPlannerAdapter


class _BaselineAgent:
    def __init__(self):
        self.distance_calls = 0
        self.act_calls = []

    def distance_after_action(self, _obs, goals):
        self.distance_calls += 1
        return np.asarray([2.0, 3.0, 8.0, 20.0], dtype=np.float32)

    def act(self, obs, goal, noise_scale=0.0, random_prob=0.0):
        self.act_calls.append((np.asarray(obs).copy(), np.asarray(goal).copy(), noise_scale, random_prob))
        return np.asarray(goal, dtype=np.float32) + 10


class _IdentityAE:
    def decode(self, values):
        return values


def _baseline_planner():
    cfg = get_config("PointMaze")
    agent = _BaselineAgent()
    planner = LatentPlanner(agent, LatentLandmarks(3, 2), _IdentityAE(), GraphSearch(cfg), cfg)
    planner.n_landmarks = 3
    planner.landmark_goals = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    planner.goal = np.asarray([3, 3], dtype=np.float32)
    planner.d_c2g = np.asarray([-1.0, -0.5, -6.0, 0.0], dtype=np.float32)
    return planner, agent


def test_baseline_command_api_preserves_commitment_mask_and_action_arguments():
    planner, agent = _baseline_planner()
    obs = np.zeros(2, dtype=np.float32)

    command, horizon = planner.select_command(obs)
    np.testing.assert_array_equal(command, [0, 0])
    assert horizon == 2 and planner.cnt == 2
    command[:] = -9
    np.testing.assert_array_equal(planner.current_command(), [0, 0])
    action = planner.act_for_current_command(obs, noise_scale=0.25, random_prob=0.5)
    np.testing.assert_array_equal(action, [10, 10])
    assert planner.cnt == 2
    np.testing.assert_array_equal(planner.act(obs, 0.25, 0.5), [10, 10])
    assert planner.cnt == 1 and agent.distance_calls == 1
    planner.cnt = 0
    command, _ = planner.select_command(obs)
    np.testing.assert_array_equal(command, [1, 1])
    assert planner.subg_idx == 1 and agent.distance_calls == 2


def test_baseline_direct_command_never_creates_a_commitment():
    planner, agent = _baseline_planner()
    planner.n_landmarks = 0
    command, horizon = planner.select_command(np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(command, [3, 3])
    assert horizon == 0 and planner.cnt == 0 and agent.distance_calls == 0


class _AdapterAgent:
    device = torch.device("cpu")

    def __init__(self, distance=2.1):
        self.distance = float(distance)
        self.d_calls = []
        self.v_calls = []
        self.actor_calls = 0

        class _Value(nn.Module):
            def __init__(inner, parent):
                super().__init__()
                inner.parent = parent

            def forward(inner, starts, commands):
                inner.parent.v_calls.append((starts.detach().clone(), commands.detach().clone()))
                return torch.linalg.vector_norm(commands - starts, dim=-1) + 1.0

        self.value = _Value(self)

    def to_tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def distance_after_action(self, state, targets):
        array = np.asarray(targets, dtype=np.float32)
        self.d_calls.append((np.asarray(state, dtype=np.float32).copy(), array.copy()))
        return np.full(len(np.atleast_2d(array)), self.distance, dtype=np.float32)

    def actor(self, _state, _goal):
        self.actor_calls += 1
        return torch.zeros((len(_goal), 1), dtype=torch.float32)


class _TrackingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode_requires_grad = []

    def encode(self, goals):
        self.encode_requires_grad.append(torch.is_grad_enabled())
        return goals * 2.0

    def decode(self, centroids):
        return centroids + 10.0


class _TrackingModel(MacroTransitionModel):
    def __init__(self):
        super().__init__(embedding_dim=2, context_dim=2, hidden_dim=5, k_max=9)
        self.calls = []
        self.set_temperature(2.5)

    def forward(self, z_start, z_command, context, positive_centroids, negative_centroids):
        self.calls.append((z_start.detach().clone(), z_command.detach().clone(), context.detach().clone(),
                           positive_centroids.detach().clone(), negative_centroids.detach().clone(),
                           float(self.temperature)))
        return super().forward(z_start, z_command, context, positive_centroids, negative_centroids)


def _adapter(*, distance=2.1, positives=((1.0, 2.0), (3.0, 4.0)), negatives=((9.0, 8.0),), original=None):
    cfg = get_config("PointMaze", soft_iters=1, pn_num_simulations=2, pn_macro_depth=1,
                     pn_top_k=2, pn_k_min=2, pn_k_max=4, pn_epsilon_edge_eval=1.0,
                     pn_epsilon_edge_train=1.0, pn_search_risk_limit=1.0,
                     pn_root_risk_limit=1.0, pn_risk_pseudocount=0.0)
    agent, ae, model = _AdapterAgent(distance), _TrackingAE(), _TrackingModel()
    positive = LatentLandmarks(len(positives), 2)
    negative = LatentLandmarks(len(negatives), 2) if negatives is not None else None
    with torch.no_grad():
        positive.centroids.copy_(torch.as_tensor(positives, dtype=torch.float32).reshape(-1, 2))
        if negative is not None:
            negative.centroids.copy_(torch.as_tensor(negatives, dtype=torch.float32).reshape(-1, 2))
    adapter = PNPlannerAdapter(agent, ae, positive, negative, model, GraphSearch(cfg), cfg,
                               rng=np.random.default_rng(4), original_planner=original)
    return adapter, agent, ae, model


def test_adapter_uses_root_d_internal_v_and_current_decoded_positive_goals_only():
    adapter, agent, _ae, _model = _adapter()
    result = adapter.plan(np.array([0., 0., 1.]), np.array([0., 0.]), np.array([4., 4.]))
    assert result.status is PlanStatus.ACTION
    assert len(agent.d_calls) == 1
    assert len(agent.v_calls) == 1
    assert agent.d_calls[0][1].shape == (3, 2)  # decoded positives plus final goal, never negative
    assert agent.v_calls[0][0].shape == (4, 2)
    np.testing.assert_array_equal(agent.d_calls[0][1][0], [11., 12.])
    with torch.no_grad():
        adapter.positive_landmarks.centroids[0].add_(5)
    moved = adapter.plan(np.array([0., 0., 1.]), np.array([0., 0.]), np.array([4., 4.]))
    assert moved.status is PlanStatus.ACTION
    np.testing.assert_array_equal(agent.d_calls[-1][1][0], [16., 17.])


def test_adapter_model_callback_detaches_encodings_passes_context_temperature_and_negative_distribution():
    adapter, _agent, ae, model = _adapter()
    was_training = model.training
    adapter.plan(np.array([0., 0., 1.]), np.array([0., 0.]), np.array([4., 4.]), context=np.array([3., 4.]))
    assert model.training is was_training
    assert ae.encode_requires_grad and not any(ae.encode_requires_grad)
    starts, commands, context, positives, negatives, temperature = model.calls[-1]
    assert not starts.requires_grad and not commands.requires_grad
    assert starts.shape[0] == commands.shape[0] and np.allclose(context.cpu().numpy(), [3., 4.])
    assert temperature == 2.5 and positives.shape == (2, 2) and negatives.shape == (1, 2)
    prediction = next(iter(adapter.search._prediction_cache.values()))
    assert prediction.negative_distribution == pytest.approx((1.0,))
    adapter.plan(np.array([0., 0., 1.]), np.array([0., 0.]), np.array([4., 4.]))
    assert model.calls[-1][2].shape[1] == 2 and torch.count_nonzero(model.calls[-1][2]) == 0


def test_adaptive_horizon_clamps_and_rejects_nonfinite_distances():
    adapter, _agent, _ae, _model = _adapter(distance=2.1)
    direct = adapter.direct_command(np.array([0., 0., 1.]), np.array([4., 4.]))
    assert direct.horizon == 3 and direct.source == "direct"
    assert adapter._adaptive_horizon(np.zeros(3), np.zeros(2)) == 3
    adapter.agent.distance = 99
    assert adapter._adaptive_horizon(np.zeros(3), np.zeros(2)) == 4
    adapter.agent.distance = float("nan")
    with pytest.raises(ValueError, match="finite"):
        adapter.direct_command(np.zeros(3), np.zeros(2))


def test_zero_positives_uses_root_to_goal_scalar_model_callback_and_no_safe_command_is_empty():
    adapter, _agent, _ae, model = _adapter(positives=(), negatives=None)
    result = adapter.plan(np.zeros(3), np.zeros(2), np.ones(2))
    assert result.status is PlanStatus.ACTION and result.node_id == 0
    assert model.calls[-1][0].shape[0] == 1
    adapter.cfg.pn_root_risk_limit = -1.0
    command = adapter.planned_command(np.zeros(3), np.zeros(2), np.ones(2))
    assert command.goal is None and command.horizon == 0 and command.plan_result.status is PlanStatus.NO_SAFE_PLAN


def test_training_flag_forwards_to_mcgs_unsafe_fallback():
    adapter, _agent, _ae, _model = _adapter(positives=(), negatives=None)
    adapter.cfg.pn_root_risk_limit = -1.0
    adapter.cfg.pn_training_unsafe_fallback = True
    assert adapter.plan(np.zeros(3), np.zeros(2), np.ones(2), training=False).status is PlanStatus.NO_SAFE_PLAN
    assert adapter.plan(np.zeros(3), np.zeros(2), np.ones(2), training=True).status is PlanStatus.ACTION


def test_command_results_are_copied_and_original_helper_preserves_low_level_action_state():
    planner, baseline_agent = _baseline_planner()
    adapter, _agent, _ae, _model = _adapter(original=planner)
    planner.select_command(np.zeros(2, dtype=np.float32))
    original = adapter.original_command(np.zeros(2, dtype=np.float32))
    assert original.source == "original" and original.horizon == 1
    original.goal[:] = -4
    np.testing.assert_array_equal(planner.current_command(), [0, 0])
    assert not baseline_agent.act_calls
    with pytest.raises(RuntimeError, match="original planner"):
        _adapter()[0].original_command(np.zeros(2))


@pytest.mark.parametrize("state,achieved,goal,context", [
    (np.zeros((1, 3)), np.zeros(2), np.ones(2), None),
    (np.zeros(3), np.zeros(3), np.ones(2), None),
    (np.zeros(3), np.zeros(2), np.array([np.nan, 1.]), None),
    (np.zeros(3), np.zeros(2), np.ones(2), np.ones(3)),
])
def test_adapter_validates_raw_inputs_and_context(state, achieved, goal, context):
    adapter, _agent, _ae, _model = _adapter()
    with pytest.raises(ValueError):
        adapter.plan(state, achieved, goal, context)
