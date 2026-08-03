"""Contracts for the baseline command API and PN-LMCGS model adapter."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

import l3p.trainer as trainer_module
from l3p.config import get_config
from l3p.envs.vec_env import make_vec_env
from l3p.models.landmarks import LatentLandmarks
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.pn_lmcgs.mcgs import EdgePrediction, PlanStatus
from l3p.pn_lmcgs.macro_replay import MacroAttempt
from l3p.pn_lmcgs.macro_transition_model import MacroTransitionModel
from l3p.pn_lmcgs.planner_adapter import CommandResult, PNPlannerAdapter
from l3p.trainer import L3PTrainer, PNPhase


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


class _TinySafetyEnv:
    obs_dim = 2
    goal_dim = 2
    act_dim = 2
    max_action = 1.0
    max_episode_steps = 1

    def reset(self):
        return self._observation()

    def step(self, _action):
        return self._observation(), -1.0, True, False, {"safety_cost": 1.0}

    def compute_reward(self, _achieved_goal, _desired_goal, info=None):
        return -1.0

    def set_eval(self, _flag):
        pass

    @staticmethod
    def _observation():
        return {
            "observation": np.zeros(2, dtype=np.float32),
            "achieved_goal": np.zeros(2, dtype=np.float32),
            "desired_goal": np.ones(2, dtype=np.float32),
        }


class _TinyVecEnv:
    def __init__(self):
        self.envs = [_TinySafetyEnv()]
        self.n = 1
        env = self.envs[0]
        self.obs_dim = env.obs_dim
        self.goal_dim = env.goal_dim
        self.act_dim = env.act_dim
        self.max_action = env.max_action
        self.max_episode_steps = env.max_episode_steps

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.envs[0].compute_reward(achieved_goal, desired_goal, info)


def _tiny_trainer(*, pn_lmcgs_enabled, **overrides):
    config = dict(
        pn_lmcgs_enabled=pn_lmcgs_enabled,
        max_episode_steps=1,
        test_episode_steps=1,
        hidden_units=4,
        hidden_layers=1,
        ae_hidden_units=4,
        ae_hidden_layers=1,
        embedding_size=2,
        n_landmarks=2,
        pn_num_positive_landmarks=2,
        batch_size=2,
        gls_batch_size=2,
        landmark_batch_size=2,
    )
    config.update(overrides)
    cfg = get_config("PointMaze", **config)
    return L3PTrainer(_TinyVecEnv(), cfg)


def test_disabled_trainer_uses_legacy_collection_path():
    trainer = _tiny_trainer(pn_lmcgs_enabled=False)

    episode = trainer.collect_episode(trainer.env.envs[0], use_planning=False, random_actions=True)

    assert set(episode) == {"obs", "ag", "g", "act"}


def test_enabled_collection_records_macro_attempt_and_endpoint_cost():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)

    episode = trainer.collect_episode(trainer.env.envs[0], use_planning=False, random_actions=True)
    trainer.collect()

    assert "cost" in episode and "cmd_g" in episode
    assert episode["cost"][-1] == 1.0
    assert trainer.macro_buffer.size > 0


def test_phase_gates_do_not_enable_mcgs_early():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)

    assert trainer.pn_phase != PNPhase.JOINT


def test_joint_collection_with_default_limits_keeps_zero_risk_plan_available(
        monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=1,
        pn_probability_mcg_search=1.0,
        pn_probability_original_planner=0.0,
        pn_probability_direct_goal=0.0,
    )
    trainer.pn_phase = PNPhase.JOINT
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    prediction = EdgePrediction()
    monkeypatch.setattr(
        trainer.pn_planner.search,
        "edge_prediction_fn",
        lambda starts, _commands, _context: [prediction] * len(starts),
    )
    env = trainer.env.envs[0]
    monkeypatch.setattr(
        env,
        "step",
        lambda _action: (
            env._observation(), -1.0, True, False, {"safety_cost": 0.0},
        ),
    )

    episode = trainer.collect_episode(
        env, use_planning=True, random_actions=False,
    )

    assert episode is not None
    assert episode["length"] == 1
    assert trainer.pn_last_collection_status == "completed"
    assert trainer.pn_no_safe_plan_count == 0


def test_old_checkpoint_loads_and_new_checkpoint_round_trips(tmp_path):
    baseline_checkpoint = tmp_path / "baseline.pt"
    upgraded_checkpoint = tmp_path / "upgraded.pt"
    baseline = _tiny_trainer(pn_lmcgs_enabled=False)
    baseline.save(baseline_checkpoint)

    upgraded = _tiny_trainer(pn_lmcgs_enabled=True)
    upgraded.load(baseline_checkpoint)
    upgraded.save(upgraded_checkpoint)

    restored = _tiny_trainer(pn_lmcgs_enabled=True)
    restored.load(upgraded_checkpoint)

    assert upgraded.pn_state_summary() == restored.pn_state_summary()


def _ready_joint_candidate():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    trainer.cfg.pn_min_positive_samples = 1
    trainer.cfg.pn_min_positive_episodes = 1
    trainer.cfg.pn_min_macro_attempts = 1
    trainer.cfg.pn_min_attempts_per_common_action_bucket = 2
    trainer.centroids_initialized = True
    trainer.pn_positive_episode_count = 1
    trainer.pn_calibration_ready = True
    trainer.landmark_memory.ingest_episode({
        "ag": np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        "cmd_g": np.asarray([[0.0, 0.0]], dtype=np.float32),
        "goal_reached": np.asarray([True]),
        "cost": np.asarray([0.0], dtype=np.float32),
        "length": 1,
    }, episode_id=0)
    attempt = MacroAttempt(
        start_goal=np.zeros(2, dtype=np.float32), command_goal=np.zeros(2, dtype=np.float32),
        end_goal=np.zeros(2, dtype=np.float32), violation_goal=None, duration=1,
        context=np.empty(0, dtype=np.float32), target_reached=True, violation=False,
        episode_id=0, start_t=0, end_t=0,
    )
    trainer.macro_buffer.add(attempt)
    while trainer.macro_buffer.validation_indices.size == 0:
        trainer.macro_buffer.add(attempt)
    return trainer


@pytest.mark.parametrize(
    ("episodes_offset", "expected_phase"),
    ((-1, PNPhase.WARMUP), (1, PNPhase.LANDMARKS)),
)
def test_uninitialized_positive_centroids_block_joint_and_mcgs(
        monkeypatch, episodes_offset, expected_phase):
    trainer = _ready_joint_candidate()
    trainer.cfg.pn_min_attempts_per_common_action_bucket = 0
    trainer.cfg.pn_probability_mcg_search = 1.0
    trainer.cfg.pn_probability_original_planner = 0.0
    trainer.cfg.pn_probability_direct_goal = 0.0
    trainer.episodes_collected = trainer.cfg.n_warmup_trajs + episodes_offset
    trainer.centroids_initialized = False
    mcgs_calls = []
    monkeypatch.setattr(
        trainer.pn_planner,
        "planned_command",
        lambda *_args, **_kwargs: mcgs_calls.append(1) or CommandResult(
            np.ones(2, dtype=np.float32), 1, "mcgs",
        ),
    )

    trainer._refresh_pn_phase()
    command = trainer._pn_select_command(
        _TinySafetyEnv._observation(), np.ones(2, dtype=np.float32), True, False, False,
    )

    assert (trainer.pn_phase, command.source, mcgs_calls) == \
        (expected_phase, "direct", [])


def test_common_action_bucket_counts_current_positive_commands_and_direct_goals():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    trainer.landmarks.centroids.data.copy_(torch.as_tensor([[0.0, 0.0], [2.0, 0.0]]))
    trainer.ae.encode = lambda goals: goals
    for command in ([0.0, 0.0], [2.0, 0.0], [9.0, 9.0]):
        trainer.macro_buffer.add(MacroAttempt(
            start_goal=np.zeros(2, dtype=np.float32), command_goal=np.asarray(command, dtype=np.float32),
            end_goal=np.zeros(2, dtype=np.float32), violation_goal=None, duration=1,
            context=np.empty(0, dtype=np.float32), target_reached=False, violation=False,
            episode_id=1, start_t=0, end_t=0,
        ))

    assert trainer._pn_common_action_bucket_counts().tolist() == [1, 1, 1]


def test_joint_phase_requires_all_common_action_buckets_and_never_plans_early(monkeypatch):
    trainer = _ready_joint_candidate()
    monkeypatch.setattr(trainer, "_pn_common_action_bucket_counts", lambda: np.asarray([2, 1, 2]),
                        raising=False)

    trainer._refresh_pn_phase()

    assert trainer.pn_phase is PNPhase.MACRO_BOOTSTRAP
    monkeypatch.setattr(trainer.pn_planner, "planned_command", lambda *_args, **_kwargs: pytest.fail(
        "planned_command must not run before JOINT"))
    command = trainer._pn_select_command(
        _TinySafetyEnv._observation(), np.ones(2, dtype=np.float32), True, False, False,
    )
    assert command.source == "direct"

    monkeypatch.setattr(trainer, "_pn_common_action_bucket_counts", lambda: np.asarray([2, 2, 2]),
                        raising=False)
    trainer._refresh_pn_phase()
    assert trainer.pn_phase is PNPhase.JOINT


def test_negative_activation_requires_samples_and_distinct_episodes():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    trainer.cfg.pn_min_negative_samples = 2
    trainer.cfg.pn_min_negative_episodes = 2
    first = {
        "ag": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        "cost": np.asarray([1.0, 1.0], dtype=np.float32), "length": 2,
    }
    trainer.landmark_memory.ingest_episode(first, episode_id=10)
    trainer._refresh_pn_phase()
    assert not trainer.pn_negative_active

    trainer.landmark_memory.ingest_episode({
        "ag": np.asarray([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        "cost": np.asarray([1.0], dtype=np.float32), "length": 1,
    }, episode_id=11)
    trainer._refresh_pn_phase()
    assert trainer.pn_negative_active


class _ThreeStepEnv(_TinySafetyEnv):
    max_episode_steps = 3

    def __init__(self):
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self._observation()

    def step(self, _action):
        self.step_count += 1
        return self._observation(), -1.0, False, False, {"safety_cost": 0.0}

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0

    def _observation(self):
        goal = np.asarray([float(self.step_count), 0.0], dtype=np.float32)
        return {"observation": goal.copy(), "achieved_goal": goal, "desired_goal": np.asarray([9.0, 0.0], dtype=np.float32)}


class _ThreeStepVecEnv(_TinyVecEnv):
    def __init__(self):
        self.envs = [_ThreeStepEnv()]
        self.n = 1
        env = self.envs[0]
        self.obs_dim, self.goal_dim, self.act_dim = env.obs_dim, env.goal_dim, env.act_dim
        self.max_action, self.max_episode_steps = env.max_action, env.max_episode_steps


def _three_step_trainer(**overrides):
    config = dict(
        pn_lmcgs_enabled=True, max_episode_steps=3, test_episode_steps=3,
        hidden_units=4, hidden_layers=1, ae_hidden_units=4, ae_hidden_layers=1,
        embedding_size=2, n_landmarks=2, batch_size=2, gls_batch_size=2,
        landmark_batch_size=2, pn_k_min=1, pn_k_max=1,
    )
    config.update(overrides)
    cfg = get_config("PointMaze", **config)
    return L3PTrainer(_ThreeStepVecEnv(), cfg)


def test_macro_attempts_replan_each_primitive_step_with_chained_metadata(monkeypatch):
    trainer = _three_step_trainer()
    command = CommandResult(np.asarray([7.0, 0.0], dtype=np.float32), 1, "direct")
    calls = []
    monkeypatch.setattr(trainer, "_pn_select_command", lambda *_args: calls.append(1) or command)

    episode = trainer.collect_episode(trainer.env.envs[0], use_planning=False, random_actions=False)
    attempts = trainer.macro_buffer.attempts

    assert len(calls) == 3 and episode["length"] == 3 and len(attempts) == 3
    assert [attempt.duration for attempt in attempts] == [1, 1, 1]
    assert [attempt.episode_id for attempt in attempts] == [0, 0, 0]
    assert [(attempt.start_t, attempt.end_t) for attempt in attempts] == [(0, 0), (1, 1), (2, 2)]
    np.testing.assert_array_equal(episode["cmd_g"], np.tile(command.goal, (3, 1)))
    for earlier, later in zip(attempts, attempts[1:]):
        np.testing.assert_array_equal(earlier.end_goal, later.start_goal)


def test_command_success_ends_macro_and_replans_without_ending_episode(monkeypatch):
    trainer = _three_step_trainer(pn_k_min=3, pn_k_max=3)
    commands = iter((
        CommandResult(np.asarray([1.0, 0.0], dtype=np.float32), 3, "direct"),
        CommandResult(np.asarray([7.0, 0.0], dtype=np.float32), 3, "direct"),
    ))
    monkeypatch.setattr(trainer, "_pn_select_command", lambda *_args: next(commands))

    episode = trainer.collect_episode(trainer.env.envs[0], use_planning=False, random_actions=False)
    attempts = trainer.macro_buffer.attempts

    assert episode["length"] == 3 and len(attempts) == 2
    assert attempts[0].target_reached and attempts[0].duration == 1
    assert attempts[1].duration == 2


def test_no_safe_plan_returns_partial_episode_without_overwriting_status(monkeypatch):
    trainer = _three_step_trainer()
    commands = iter((
        CommandResult(np.asarray([7.0, 0.0], dtype=np.float32), 1, "direct"),
        CommandResult(None, 0, "mcgs"),
    ))
    monkeypatch.setattr(trainer, "_pn_select_command", lambda *_args: next(commands))

    episode = trainer.collect_episode(trainer.env.envs[0], use_planning=False, random_actions=False)

    assert trainer.env.envs[0].step_count == 1 and episode["length"] == 1
    assert trainer.pn_last_collection_status == "no_safe_plan_partial"


def test_train_raises_after_empty_no_safe_plan_collection(monkeypatch):
    trainer = _three_step_trainer()
    trainer.pn_last_collection_status = "no_safe_plan_empty"
    calls = 0

    def collect_once():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("train retried zero-progress collection")
        return 0

    monkeypatch.setattr(trainer, "collect", collect_once)
    monkeypatch.setattr(trainer, "evaluate", lambda *_args, **_kwargs: 0.0)
    with pytest.raises(RuntimeError, match="no safe plan"):
        trainer.train(total_steps=1)
    assert calls == 1


def _metrics_trainer(*, pn_lmcgs_enabled, metrics_callback):
    source = _tiny_trainer(
        pn_lmcgs_enabled=pn_lmcgs_enabled,
        train_after=10,
        log_interval=1,
        eval_interval=1,
        eval_episodes=1,
    )
    return L3PTrainer(
        _TinyVecEnv(), source.cfg, metrics_callback=metrics_callback,
    )


def _advance_one_train_step(trainer):
    def collect_once():
        trainer.total_env_steps += 1
        trainer.episodes_collected += 1
        return 1

    return collect_once


def test_train_callback_emits_scalar_train_and_checkpoint_metrics(monkeypatch, tmp_path):
    events = []
    trainer = _metrics_trainer(
        pn_lmcgs_enabled=False,
        metrics_callback=lambda event, step, metrics: events.append(
            (event, step, dict(metrics)),
        ),
    )
    monkeypatch.setattr(trainer, "collect", _advance_one_train_step(trainer))
    monkeypatch.setattr(trainer, "evaluate", lambda _episodes: 0.5)
    monkeypatch.setattr(trainer, "save", lambda _path: None)

    trainer.train(total_steps=1, checkpoint_path=str(tmp_path / "checkpoint.pt"),
                  checkpoint_every=1)

    train_event = next(event for event in events if event[0] == "train")
    assert train_event[1] == 1
    assert train_event[2]["episodes_collected"] == 1.0
    assert train_event[2]["elapsed_seconds"] >= 0.0
    assert all(isinstance(value, float) for value in train_event[2].values())
    assert ("checkpoint", 1, {"checkpoint_saved": 1.0}) in events


def test_pn_evaluation_callback_includes_safety_metrics(monkeypatch):
    events = []
    trainer = _metrics_trainer(
        pn_lmcgs_enabled=True,
        metrics_callback=lambda event, step, metrics: events.append(
            (event, step, dict(metrics)),
        ),
    )
    trainer.last_eval_metrics.update(
        safe_success_rate=0.75,
        episode_violation_rate=0.25,
    )
    monkeypatch.setattr(trainer, "collect", _advance_one_train_step(trainer))
    monkeypatch.setattr(trainer, "evaluate", lambda _episodes: 0.8)

    trainer.train(total_steps=1)

    evaluation_event = next(event for event in events if event[0] == "evaluation")
    assert evaluation_event[1] == 1
    assert evaluation_event[2]["success_rate"] == 0.8
    assert evaluation_event[2]["safe_success_rate"] == 0.75
    assert evaluation_event[2]["episode_violation_rate"] == 0.25


def _store_primitive_pn_episode(trainer):
    trainer.buffer.store_episode({
        "obs": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "ag": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "g": np.asarray([[2.0, 2.0]], dtype=np.float32),
        "cmd_g": np.asarray([[2.0, 2.0]], dtype=np.float32),
        "act": np.zeros((1, 2), dtype=np.float32),
        "cost": np.zeros(1, dtype=np.float32),
        "goal_reached": np.ones(1, dtype=bool),
        "length": 1,
    })


def _neutralize_unrelated_update_terms(trainer, monkeypatch):
    for name in ("update_value", "update_critic", "update_actor", "update_targets"):
        monkeypatch.setattr(trainer.agent, name, lambda *_args, **_kwargs: 0.0)
    ae_parameter = next(trainer.ae.parameters())
    monkeypatch.setattr(
        trainer_module,
        "ae_losses",
        lambda *_args, **_kwargs: (ae_parameter.sum() * 0.0,) * 3,
    )


def test_pn_centroids_train_only_from_safety_memories_and_negative_gate(monkeypatch):
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    _store_primitive_pn_episode(trainer)
    _neutralize_unrelated_update_terms(trainer, monkeypatch)
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    trainer.pn_negative_active = True
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    trainer.ae.encode = lambda goals: goals

    replay_goal = np.asarray([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32)
    positive_goal = np.asarray([[20.0, 20.0], [20.0, 20.0]], dtype=np.float32)
    negative_goal = np.asarray([[30.0, 30.0], [30.0, 30.0]], dtype=np.float32)
    monkeypatch.setattr(trainer.buffer, "sample_achieved_goals", lambda *_args: replay_goal)
    monkeypatch.setattr(trainer.landmark_memory, "sample_positive", lambda *_args: positive_goal)
    monkeypatch.setattr(
        trainer.landmark_memory,
        "sample_negative",
        lambda *_args: {"goals": negative_goal, "episode_id": np.zeros(2, dtype=np.int64),
                         "is_preimpact": np.zeros(2, dtype=bool)},
    )
    captured = {}

    def positive_elbo(z):
        captured["positive"] = z.detach().clone()
        return trainer.landmarks.centroids.sum() * 0.0

    def negative_elbo(z):
        captured["negative"] = z.detach().clone()
        return trainer.negative_landmarks.centroids.sum() * 0.0

    monkeypatch.setattr(trainer.landmarks, "elbo_loss", positive_elbo)
    monkeypatch.setattr(trainer.negative_landmarks, "elbo_loss", negative_elbo)

    trainer.update(1)

    np.testing.assert_array_equal(captured["positive"].cpu().numpy(), positive_goal)
    np.testing.assert_array_equal(captured["negative"].cpu().numpy(), negative_goal)

    trainer.pn_negative_active = False
    monkeypatch.setattr(trainer.landmark_memory, "sample_negative", lambda *_args: pytest.fail(
        "inactive negative landmarks must not sample violation memory"))
    trainer.update(1)

    disabled = _tiny_trainer(pn_lmcgs_enabled=False)
    _store_primitive_pn_episode(disabled)
    _neutralize_unrelated_update_terms(disabled, monkeypatch)
    disabled.centroids_initialized = True
    disabled.ae.encode = lambda goals: goals
    monkeypatch.setattr(disabled.buffer, "sample_achieved_goals", lambda *_args: replay_goal)
    observed = {}

    def replay_elbo(z):
        observed["replay"] = z.detach().clone()
        return disabled.landmarks.centroids.sum() * 0.0

    monkeypatch.setattr(
        disabled.landmarks,
        "elbo_loss",
        replay_elbo,
    )
    disabled.update(1)
    np.testing.assert_array_equal(observed["replay"].cpu().numpy(), replay_goal)


def test_pn_update_trains_macro_model_and_calibrates_natural_validation(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=2,
        pn_macro_batch_size=2,
        pn_calibration_interval=1,
        pn_positive_assignment_radius=100.0,
    )
    _store_primitive_pn_episode(trainer)
    _neutralize_unrelated_update_terms(trainer, monkeypatch)
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    trainer.pn_negative_active = True
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    trainer.ae.encode = lambda goals: goals
    monkeypatch.setattr(
        trainer.landmarks,
        "elbo_loss",
        lambda _z: trainer.landmarks.centroids.sum() * 0.0,
    )
    monkeypatch.setattr(
        trainer.negative_landmarks,
        "elbo_loss",
        lambda _z: trainer.negative_landmarks.centroids.sum() * 0.0,
    )
    trainer.landmarks.centroids.data.copy_(torch.as_tensor([[0.0, 0.0], [5.0, 5.0]]))
    trainer.negative_landmarks.centroids.data.copy_(torch.as_tensor([[9.0, 9.0]]))

    attempts = (
        (True, False, [0.0, 0.0], None),       # UID 0: TARGET, train
        (False, False, [4.0, 4.0], None),      # UID 1: safe failure, train
        (False, True, [8.0, 8.0], [9.0, 9.0]), # UID 2: VIOLATION, train
        (True, False, [1.0, 1.0], None),       # UID 3: TARGET, validation
    )
    for uid, (target, violation, end, violation_goal) in enumerate(attempts):
        trainer.macro_buffer.add(MacroAttempt(
            start_goal=np.zeros(2, dtype=np.float32), command_goal=np.asarray([2.0, 2.0], dtype=np.float32),
            end_goal=np.asarray(end, dtype=np.float32), violation_goal=violation_goal, duration=1,
            context=np.empty(0, dtype=np.float32), target_reached=target, violation=violation,
            episode_id=uid, start_t=0, end_t=0,
        ))
    assert trainer.macro_buffer.train_indices.tolist() == [0, 1, 2]
    assert trainer.macro_buffer.validation_indices.tolist() == [3]

    before = [parameter.detach().clone() for parameter in trainer.macro_model.parameters()]
    logs = trainer.update(1)

    assert any(not torch.equal(parameter, original)
               for parameter, original in zip(trainer.macro_model.parameters(), before))
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in trainer.macro_model.parameters())
    assert all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
               for parameter in tuple(trainer.ae.parameters()) + tuple(trainer.landmarks.parameters()))
    assert trainer.pn_calibration_ready
    assert np.isfinite(float(trainer.macro_model.temperature)) and float(trainer.macro_model.temperature) > 0.0
    assert trainer.pn_calibration_temperature == float(trainer.macro_model.temperature)
    for name in ("macro", "macro_outcome", "macro_positive", "macro_negative", "macro_duration"):
        assert np.isfinite(logs[name])
    assert logs["macro_validation_target_frequency"] == 1.0
    assert logs["macro_validation_violation_frequency"] == 0.0


def test_macro_update_bounds_current_centroid_relabeling_to_candidate_pool(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=2,
        pn_macro_batch_size=4,
        pn_positive_assignment_radius=0.25,
    )
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    trainer.pn_negative_active = True
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    trainer.ae.encode = lambda goals: goals
    trainer.landmarks.centroids.data.copy_(torch.as_tensor([[0.0, 0.0], [5.0, 5.0]]))
    trainer.negative_landmarks.centroids.data.copy_(torch.as_tensor([[9.0, 9.0]]))
    for uid in range(80):
        outcome = uid % 3
        trainer.macro_buffer.add(MacroAttempt(
            start_goal=np.asarray([uid, -uid], dtype=np.float32),
            command_goal=np.asarray([2.0, 2.0], dtype=np.float32),
            end_goal=np.asarray([5.0, 5.0], dtype=np.float32),
            violation_goal=(
                np.asarray([9.0, 9.0], dtype=np.float32) if outcome == 2 else None
            ),
            duration=1,
            context=np.empty(0, dtype=np.float32),
            target_reached=outcome == 0,
            violation=outcome == 2,
            episode_id=uid,
            start_t=0,
            end_t=0,
        ))
    train_size = trainer.macro_buffer.train_indices.size
    assert train_size > 4 * trainer.cfg.pn_macro_batch_size
    original_label = trainer._pn_label_macro_entries
    observed = {}

    def track_labels(attempts):
        labels = original_label(attempts)
        observed["count"] = len(attempts)
        observed["start_goals"] = [attempt.start_goal.copy() for attempt in attempts]
        observed["drift_ids"] = labels.positive_id[
            labels.outcome == int(trainer_module.Outcome.DRIFT)
        ].detach().cpu().tolist()
        attempts[0].start_goal.fill(999.0)
        return labels

    monkeypatch.setattr(trainer, "_pn_label_macro_entries", track_labels)

    logs = trainer._pn_update_macro_model()

    assert observed["count"] <= 4 * trainer.cfg.pn_macro_batch_size
    assert observed["count"] < train_size
    assert observed["drift_ids"] and set(observed["drift_ids"]) == {1}
    assert all(np.isfinite(value) for value in logs.values())
    assert any(not np.all(goal == 999.0) for goal in observed["start_goals"])
    assert all(not np.all(attempt.start_goal == 999.0)
               for attempt in trainer.macro_buffer.attempts)


def test_pn_update_returns_complete_finite_scalar_diagnostics():
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=2,
        pn_macro_batch_size=2,
        pn_calibration_interval=1,
        pn_cost_critic_train_after=0,
        pn_positive_assignment_radius=100.0,
    )
    _store_primitive_pn_episode(trainer)
    trainer.centroids_initialized = True
    trainer.total_env_steps = 1
    trainer.pn_positive_active = True
    trainer.pn_negative_active = True
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    trainer.landmarks.centroids.data.copy_(torch.as_tensor([[0.0, 0.0], [5.0, 5.0]]))
    trainer.negative_landmarks.centroids.data.copy_(torch.as_tensor([[9.0, 9.0]]))
    trainer.landmark_memory.ingest_episode({
        "ag": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "cmd_g": np.asarray([[1.0, 1.0]], dtype=np.float32),
        "cost": np.zeros(1, dtype=np.float32),
        "goal_reached": np.ones(1, dtype=bool),
        "length": 1,
    }, episode_id=0)
    trainer.landmark_memory.ingest_episode({
        "ag": np.asarray([[8.0, 8.0], [9.0, 9.0]], dtype=np.float32),
        "cmd_g": np.asarray([[2.0, 2.0]], dtype=np.float32),
        "cost": np.ones(1, dtype=np.float32),
        "goal_reached": np.zeros(1, dtype=bool),
        "length": 1,
    }, episode_id=1)

    attempts = (
        (True, False, [0.0, 0.0], None),       # UID 0: TARGET, train
        (False, False, [4.0, 4.0], None),      # UID 1: safe failure, train
        (False, True, [8.0, 8.0], [9.0, 9.0]), # UID 2: VIOLATION, train
        (True, False, [1.0, 1.0], None),       # UID 3: TARGET, validation
    )
    for uid, (target, violation, end, violation_goal) in enumerate(attempts):
        trainer.macro_buffer.add(MacroAttempt(
            start_goal=np.zeros(2, dtype=np.float32),
            command_goal=np.asarray([2.0, 2.0], dtype=np.float32),
            end_goal=np.asarray(end, dtype=np.float32),
            violation_goal=violation_goal, duration=1,
            context=np.empty(0, dtype=np.float32), target_reached=target,
            violation=violation, episode_id=uid, start_t=0, end_t=0,
        ))
    assert trainer.macro_buffer.train_indices.tolist() == [0, 1, 2]
    assert trainer.macro_buffer.validation_indices.tolist() == [3]

    logs = trainer.update(1)

    for value in logs.values():
        assert np.isscalar(value)
        assert np.isfinite(float(value))
    required = {
        "primitive_violation_rate", "episode_violation_rate", "safe_success_rate",
        "positive_memory_size", "negative_memory_size", "negative_episode_count",
        "positive_landmarks_active", "negative_landmarks_active",
        "positive_centroid_movement", "negative_centroid_movement", "positive_coverage",
        "negative_samples_per_centroid", "cost_critic_bce", "cost_critic_auroc",
        "cost_predicted_violation", "cost_empirical_violation",
        "macro_loss", "macro_outcome_loss", "macro_positive_loss", "macro_negative_loss",
        "macro_duration_loss", "macro_calibration_ece", "macro_temperature",
        "macro_target_frequency", "macro_drift_frequency", "macro_stuck_frequency",
        "macro_violation_frequency", "macro_predicted_violation", "macro_empirical_violation",
        "planner_latency", "planner_branch_before", "planner_branch_after",
        "planner_no_safe_plan_rate", "planner_chosen_p_violation",
        "planner_chosen_q_risk", "planner_chosen_upper_risk",
        "planner_simulated_safe_success_rate", "planner_cycles", "planner_leaf_uses",
    }
    assert required.issubset(logs)
    for prefix in (
            "macro_confusion_", "macro_precision_", "macro_recall_",
            "macro_duration_mae_", "macro_reliability_",
    ):
        matches = [value for name, value in logs.items() if name.startswith(prefix)]
        assert matches and all(np.isfinite(float(value)) for value in matches)
    for name in (
            "planner_latency", "planner_branch_before", "planner_branch_after",
            "planner_no_safe_plan_rate", "planner_chosen_p_violation",
            "planner_chosen_q_risk", "planner_chosen_upper_risk",
            "planner_simulated_safe_success_rate", "planner_cycles", "planner_leaf_uses",
    ):
        assert logs[name] == 0.0

    disabled = _tiny_trainer(pn_lmcgs_enabled=False)
    _store_primitive_pn_episode(disabled)
    assert set(disabled.update(1)) == {"critic", "value", "actor", "ae_rec", "ae_latent", "elbo"}


def test_pn_collection_counters_use_real_final_goal_and_safety_outcomes(monkeypatch):
    class _ScriptedCounterEnv(_TinySafetyEnv):
        def __init__(self, desired_goal, achieved_goal, safety_cost):
            self.desired_goal = np.asarray(desired_goal, dtype=np.float32)
            self.achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
            self.safety_cost = float(safety_cost)
            self.current_goal = np.zeros(2, dtype=np.float32)

        def reset(self):
            self.current_goal = np.zeros(2, dtype=np.float32)
            return self._observation()

        def step(self, _action):
            self.current_goal = self.achieved_goal.copy()
            return self._observation(), -1.0, True, False, {"safety_cost": self.safety_cost}

        def compute_reward(self, achieved_goal, desired_goal, info=None):
            return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0

        def _observation(self):
            return {
                "observation": self.current_goal.copy(),
                "achieved_goal": self.current_goal.copy(),
                "desired_goal": self.desired_goal.copy(),
            }

    def collect_scripted(desired_goal, achieved_goal, safety_cost, command_goal):
        trainer = _tiny_trainer(pn_lmcgs_enabled=True, initial_random_trajs=0)
        trainer.env.envs = [_ScriptedCounterEnv(desired_goal, achieved_goal, safety_cost)]
        command = CommandResult(np.asarray(command_goal, dtype=np.float32), 1, "direct")
        monkeypatch.setattr(trainer, "_pn_select_command", lambda *_args: command)
        assert trainer.collect() == 1
        return trainer

    safe = collect_scripted([1.0, 0.0], [1.0, 0.0], 0.0, [1.0, 0.0])
    assert safe.pn_primitive_transition_count == 1
    assert safe.pn_primitive_violation_count == 0
    assert safe.pn_episode_goal_success_count == 1
    assert safe.pn_episode_safe_success_count == 1
    assert safe.pn_violation_episode_count == 0
    assert safe.pn_path_length_total == pytest.approx(1.0)

    violation = collect_scripted([2.0, 0.0], [1.0, 0.0], 1.0, [1.0, 0.0])
    assert violation.pn_primitive_transition_count == 1
    assert violation.pn_primitive_violation_count == 1
    assert violation.pn_episode_goal_success_count == 0
    assert violation.pn_episode_safe_success_count == 0
    assert violation.pn_violation_episode_count == 1
    assert violation.pn_path_length_total == pytest.approx(1.0)


class _ScriptedEvalEnv(_TinySafetyEnv):
    max_episode_steps = 2

    def __init__(self, *, safety_cost, reaches_goal=True):
        self.safety_cost = float(safety_cost)
        self.reaches_goal = bool(reaches_goal)
        self.step_calls = 0
        self.current_goal = np.zeros(2, dtype=np.float32)
        self.desired_goal = np.asarray([2.0, 0.0], dtype=np.float32)

    def reset(self):
        self.step_calls = 0
        self.current_goal = np.zeros(2, dtype=np.float32)
        return self._observation()

    def step(self, _action):
        self.step_calls += 1
        if self.reaches_goal and self.step_calls == 2:
            self.current_goal = self.desired_goal.copy()
        else:
            self.current_goal = np.asarray([float(self.step_calls), 0.0], dtype=np.float32)
        return self._observation(), -1.0, self.step_calls >= 2, False, {
            "safety_cost": self.safety_cost,
        }

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return 0.0 if np.array_equal(achieved_goal, desired_goal) else -1.0

    def _observation(self):
        return {
            "observation": self.current_goal.copy(),
            "achieved_goal": self.current_goal.copy(),
            "desired_goal": self.desired_goal.copy(),
        }


class _ScriptedEvalVecEnv(_TinyVecEnv):
    def __init__(self, env):
        self.envs = [env]
        self.n = 1
        self.obs_dim = env.obs_dim
        self.goal_dim = env.goal_dim
        self.act_dim = env.act_dim
        self.max_action = env.max_action
        self.max_episode_steps = env.max_episode_steps
        self.eval_flags = []

    def set_eval(self, flag):
        self.eval_flags.append(bool(flag))


def _scripted_eval_trainer(*, safety_cost, reaches_goal=True, pn_lmcgs_enabled=True):
    env = _ScriptedEvalEnv(safety_cost=safety_cost, reaches_goal=reaches_goal)
    cfg = get_config(
        "PointMaze",
        pn_lmcgs_enabled=pn_lmcgs_enabled,
        max_episode_steps=2,
        test_episode_steps=2,
        hidden_units=4,
        hidden_layers=1,
        ae_hidden_units=4,
        ae_hidden_layers=1,
        embedding_size=2,
        n_landmarks=2,
        batch_size=2,
        gls_batch_size=2,
        landmark_batch_size=2,
        pn_k_min=1,
        pn_k_max=1,
    )
    return L3PTrainer(_ScriptedEvalVecEnv(env), cfg)


def test_pn_evaluate_reports_safe_metrics_and_disables_unsafe_fallback(monkeypatch):
    trainer = _scripted_eval_trainer(safety_cost=0.0)
    trainer.centroids_initialized = True
    trainer.pn_phase = PNPhase.JOINT
    trainer.cfg.pn_training_unsafe_fallback = True
    planning_calls = []
    command = CommandResult(
        np.asarray([2.0, 0.0], dtype=np.float32), 1, "mcgs", diagnostics={"latency": 0.125},
    )
    monkeypatch.setattr(
        trainer.pn_planner,
        "planned_command",
        lambda *_args, training=False, **_kwargs: planning_calls.append(training) or command,
    )
    before = (trainer.macro_buffer.size, trainer.pn_planning_count, trainer.pn_no_safe_plan_count)

    assert trainer.evaluate(1, use_planning=True) == 1.0

    metrics = trainer.last_eval_metrics
    required = {
        "goal_success_rate", "safe_success_rate", "episode_violation_rate",
        "path_length", "no_safe_plan_rate", "planning_latency",
    }
    assert required.issubset(metrics)
    assert all(np.isscalar(value) and np.isfinite(float(value)) for value in metrics.values())
    assert metrics["goal_success_rate"] == 1.0
    assert metrics["safe_success_rate"] == 1.0
    assert metrics["episode_violation_rate"] == 0.0
    assert metrics["no_safe_plan_rate"] == 0.0
    assert metrics["path_length"] == 2.0
    assert planning_calls == [False, False]
    assert trainer.env.envs[0].step_calls == 2
    assert (trainer.macro_buffer.size, trainer.pn_planning_count, trainer.pn_no_safe_plan_count) == before
    assert trainer.env.eval_flags == [True, False]

    violation = _scripted_eval_trainer(safety_cost=1.0)
    violation.centroids_initialized = True
    violation.pn_phase = PNPhase.JOINT
    monkeypatch.setattr(violation.pn_planner, "planned_command", lambda *_args, **_kwargs: command)
    assert violation.evaluate(1, use_planning=True) == 1.0
    assert violation.last_eval_metrics["safe_success_rate"] == 0.0
    assert violation.last_eval_metrics["episode_violation_rate"] == 1.0

    no_safe = _scripted_eval_trainer(safety_cost=0.0, reaches_goal=False)
    no_safe.centroids_initialized = True
    no_safe.pn_phase = PNPhase.JOINT
    no_safe.cfg.pn_training_unsafe_fallback = True
    no_safe_calls = []
    monkeypatch.setattr(
        no_safe.pn_planner,
        "planned_command",
        lambda *_args, training=False, **_kwargs: no_safe_calls.append(training) or CommandResult(
            None, 0, "mcgs", diagnostics={"latency": 0.25},
        ),
    )
    assert no_safe.evaluate(1, use_planning=True) == 0.0
    assert no_safe.env.envs[0].step_calls == 0
    assert no_safe_calls == [False]
    assert no_safe.last_eval_metrics["no_safe_plan_rate"] == 1.0
    assert all(np.isscalar(value) and np.isfinite(float(value))
               for value in no_safe.last_eval_metrics.values())

    disabled = _scripted_eval_trainer(safety_cost=0.0, pn_lmcgs_enabled=False)
    assert isinstance(disabled.evaluate(1, use_planning=False), float)
    assert not hasattr(disabled, "pn_planner")


def test_pn_checkpoint_includes_target_config_eval_and_round_trips(tmp_path):
    checkpoint = tmp_path / "pn-complete.pt"
    source = _ready_joint_candidate()
    source.cfg.pn_min_attempts_per_common_action_bucket = 0
    source._refresh_pn_phase()
    assert source.pn_phase is PNPhase.JOINT
    source.pn_calibration_temperature = 1.5
    source.pn_calibration_before_nll = 0.8
    source.pn_calibration_after_nll = 0.6
    source.pn_calibration_ece = 0.1
    source.pn_last_collection_status = "complete"
    source.pn_last_diagnostics["planning_latency"] = 0.5
    source.last_eval_metrics = {
        "goal_success_rate": 0.5,
        "safe_success_rate": 0.25,
        "episode_violation_rate": 0.5,
        "path_length": 2.0,
        "no_safe_plan_rate": 0.25,
        "planning_latency": 0.125,
    }
    source.total_env_steps = 7
    source.episodes_collected = 3
    source.pn_planning_count = 2
    source.pn_no_safe_plan_count = 1
    source.macro_buffer.add(MacroAttempt(
        start_goal=np.zeros(2, dtype=np.float32), command_goal=np.ones(2, dtype=np.float32),
        end_goal=np.ones(2, dtype=np.float32), violation_goal=None, duration=1,
        context=np.empty(0, dtype=np.float32), target_reached=True, violation=False,
        episode_id=0, start_t=0, end_t=0,
    ))
    with torch.no_grad():
        for parameter in source.agent.cost_critic_target.parameters():
            parameter.add_(0.25)
    source.save(checkpoint)

    raw = torch.load(checkpoint, weights_only=False)
    pn = raw["pn"]
    assert raw["version"] >= 2
    assert isinstance(pn["config_snapshot"], dict) and pn["config_snapshot"]
    assert "cost_critic_target" in pn
    assert pn["diagnostics"]["evaluation"] == source.last_eval_metrics
    assert {
        "positive_landmark_opt", "negative_landmark_opt", "macro_opt", "cost_critic_opt",
        "landmark_memory", "macro_buffer", "rng",
    }.issubset(pn)

    restored = _tiny_trainer(pn_lmcgs_enabled=True)
    restored.load(checkpoint)

    for saved, loaded in zip(
            source.agent.cost_critic_target.parameters(), restored.agent.cost_critic_target.parameters()):
        assert torch.equal(saved, loaded)
    assert any(not torch.equal(target, online) for target, online in zip(
        restored.agent.cost_critic_target.parameters(), restored.agent.cost_critic.parameters(),
    ))
    assert restored._pn_checkpoint_state()["config_snapshot"] == pn["config_snapshot"]
    assert restored.last_eval_metrics == source.last_eval_metrics
    assert restored.pn_state_summary() == source.pn_state_summary()

    disabled_checkpoint = tmp_path / "legacy.pt"
    disabled = _tiny_trainer(pn_lmcgs_enabled=False)
    disabled.save(disabled_checkpoint)
    legacy = torch.load(disabled_checkpoint, weights_only=False)
    assert set(legacy) == {"agent", "ae", "landmarks", "centroids_initialized"}


def test_checkpoint_phase_cannot_bypass_restored_scheduler_gates(tmp_path):
    valid_path = tmp_path / "valid-derived-phase.pt"
    tampered_path = tmp_path / "tampered-derived-phase.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    source._refresh_pn_phase()
    expected_phase = source.pn_phase
    assert expected_phase is not PNPhase.JOINT
    source.save(valid_path)
    tampered = torch.load(valid_path, weights_only=False)
    assert tampered["pn"]["phase"] == expected_phase.value
    tampered["pn"]["phase"] = PNPhase.JOINT.value
    torch.save(tampered, tampered_path)
    restored = _tiny_trainer(pn_lmcgs_enabled=True)

    restored.load(tampered_path)

    assert restored.pn_phase is expected_phase
    assert restored.pn_phase is not PNPhase.JOINT


def test_partial_versioned_pn_checkpoint_is_rejected(tmp_path):
    valid_checkpoint = tmp_path / "valid.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    source.save(valid_checkpoint)

    for missing_key in ("cost_critic_target", "config_snapshot", "diagnostics.evaluation"):
        malformed = torch.load(valid_checkpoint, weights_only=False)
        if missing_key == "diagnostics.evaluation":
            malformed["pn"]["diagnostics"].pop("evaluation", None)
        else:
            malformed["pn"].pop(missing_key, None)
        path = tmp_path / f"missing-{missing_key.replace('.', '-')}.pt"
        torch.save(malformed, path)

        with pytest.raises(ValueError, match="incomplete|missing|checkpoint"):
            _tiny_trainer(pn_lmcgs_enabled=True).load(path)


class _FixedChoiceRng:
    def __init__(self, *, choice=0, integer=0):
        self.choice_value = int(choice)
        self.integer_value = int(integer)
        self.choice_probabilities = []

    def choice(self, _values, p=None):
        self.choice_probabilities.append(None if p is None else np.asarray(p).copy())
        return self.choice_value

    def integers(self, _high, *args, **kwargs):
        return self.integer_value


def test_enabled_collect_bypasses_legacy_search_probability(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        initial_random_trajs=0,
        search_prob_train=0.0,
    )
    observed = []
    monkeypatch.setattr(
        trainer,
        "collect_episode",
        lambda _env, use_planning, random_actions: observed.append(
            (use_planning, random_actions)
        ) or None,
    )

    assert trainer.collect() == 0
    assert observed == [(True, False)]


def test_macro_bootstrap_explicitly_samples_a_current_positive_command(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=4,
    )
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    trainer.landmarks.centroids.data.copy_(
        torch.as_tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    )
    trainer.ae.decode = lambda centroids: centroids + 10.0
    trainer.rng = _FixedChoiceRng(choice=0, integer=0)
    monkeypatch.setattr(
        trainer.pn_planner,
        "direct_command",
        lambda _state, goal: CommandResult(np.asarray(goal), 2, "direct"),
    )
    monkeypatch.setattr(
        trainer.pn_planner,
        "planned_command",
        lambda *_args, **_kwargs: pytest.fail("MCGS must not run during bootstrap"),
    )

    command = trainer._pn_select_command(
        _TinySafetyEnv._observation(), np.ones(2, dtype=np.float32),
        use_planning=True, random_actions=False, original_ready=False,
    )

    np.testing.assert_array_equal(command.goal, [11.0, 12.0])
    assert command.source == "direct"


def test_enabled_positive_cardinality_and_initialization_use_only_safe_memory(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        n_landmarks=2,
        pn_num_positive_landmarks=3,
        pn_min_positive_samples=3,
        pn_min_positive_episodes=1,
        gls_batch_size=3,
    )
    assert trainer.landmarks.n_landmarks == 3
    monkeypatch.setattr(
        trainer.buffer,
        "sample_achieved_goals",
        lambda *_args: pytest.fail("enabled initialization must not sample HER replay"),
    )

    trainer._init_centroids()
    assert not trainer.centroids_initialized

    safe_goals = np.asarray(
        [[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32,
    )
    trainer.landmark_memory._positive.extend(goal.copy() for goal in safe_goals)
    trainer.pn_positive_episode_count = 1
    trainer.ae.encode = lambda goals: goals
    monkeypatch.setattr(
        trainer.landmark_memory,
        "sample_positive",
        lambda *_args: safe_goals.copy(),
    )

    trainer._init_centroids()

    assert trainer.centroids_initialized
    assert {tuple(row) for row in trainer.landmarks.centroids.detach().cpu().numpy()} == {
        tuple(row) for row in safe_goals
    }


def test_macro_learning_is_phase_gated_and_uncalibrated_validation_can_unlock_joint(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_calibrate_temperature=False,
        pn_calibration_interval=1,
        pn_min_macro_attempts=1,
        pn_min_attempts_per_common_action_bucket=0,
        pn_min_positive_samples=1,
        pn_min_positive_episodes=1,
        pn_k_min=1,
        pn_k_max=2,
    )
    _store_primitive_pn_episode(trainer)
    _neutralize_unrelated_update_terms(trainer, monkeypatch)
    trainer.centroids_initialized = True
    trainer.pn_positive_active = True
    trainer.pn_positive_episode_count = 1
    trainer.ae.encode = lambda goals: goals
    trainer.landmark_memory._positive.append(np.zeros(2, dtype=np.float32))
    attempt = MacroAttempt(
        start_goal=np.zeros(2, dtype=np.float32),
        command_goal=np.ones(2, dtype=np.float32),
        end_goal=np.ones(2, dtype=np.float32),
        violation_goal=None,
        duration=1,
        context=np.empty(0, dtype=np.float32),
        target_reached=True,
        violation=False,
        episode_id=0,
        start_t=0,
        end_t=0,
    )
    while trainer.macro_buffer.validation_indices.size == 0:
        trainer.macro_buffer.add(attempt)

    trainer.pn_phase = PNPhase.LANDMARKS
    monkeypatch.setattr(
        trainer,
        "_pn_update_macro_model",
        lambda: pytest.fail("macro model must not train before bootstrap"),
    )
    monkeypatch.setattr(
        trainer,
        "_pn_calibrate_macro_model",
        lambda: pytest.fail("macro validation must not run before bootstrap"),
    )
    trainer.update(1)

    monkeypatch.undo()
    _neutralize_unrelated_update_terms(trainer, monkeypatch)
    trainer.pn_phase = PNPhase.MACRO_BOOTSTRAP
    diagnostics = trainer._pn_calibrate_macro_model()
    assert trainer.pn_calibration_ready
    assert trainer.pn_calibration_temperature == 1.0
    assert diagnostics["macro_validation_target_frequency"] == 1.0

    trainer._refresh_pn_phase()
    assert trainer.pn_phase is PNPhase.JOINT
    trainer.cfg.pn_calibration_interval = 5
    trainer.grad_step_count = 1
    snapshot = trainer._pn_calibrate_macro_model()
    assert snapshot["macro_validation_target_frequency"] == 1.0


def test_common_action_coverage_excludes_validation_attempts():
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_positive_assignment_radius=0.1,
    )
    trainer.landmarks.centroids.data.copy_(
        torch.as_tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    )
    trainer.ae.encode = lambda goals: goals
    commands = ([0.0, 0.0], [9.0, 9.0], [0.0, 0.0], [2.0, 0.0])
    for uid, command in enumerate(commands):
        trainer.macro_buffer.add(MacroAttempt(
            start_goal=np.zeros(2, dtype=np.float32),
            command_goal=np.asarray(command, dtype=np.float32),
            end_goal=np.zeros(2, dtype=np.float32),
            violation_goal=None,
            duration=1,
            context=np.empty(0, dtype=np.float32),
            target_reached=False,
            violation=False,
            episode_id=uid,
            start_t=0,
            end_t=0,
        ))
    assert trainer.macro_buffer.train_indices.tolist() == [0, 1, 2]
    assert trainer.macro_buffer.validation_indices.tolist() == [3]

    assert trainer._pn_common_action_bucket_counts().tolist() == [2, 0, 1]


def test_ae_update_invalidates_cached_common_action_assignments(monkeypatch):
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    _store_primitive_pn_episode(trainer)
    _neutralize_unrelated_update_terms(trainer, monkeypatch)
    trainer._pn_common_action_bucket_cache_key = (1, 2, 3, 4.0)
    trainer._pn_common_action_bucket_cache = np.ones(3, dtype=np.int64)

    trainer.update(1)

    assert trainer._pn_common_action_bucket_cache_key is None
    assert trainer._pn_common_action_bucket_cache is None


class _ReachThenLeaveEnv(_ThreeStepEnv):
    def _observation(self):
        goal = np.asarray([float(self.step_count), 0.0], dtype=np.float32)
        return {
            "observation": goal.copy(),
            "achieved_goal": goal,
            "desired_goal": np.asarray([1.0, 0.0], dtype=np.float32),
        }


def _reach_then_leave_trainer():
    env = _ReachThenLeaveEnv()
    vec = _ThreeStepVecEnv()
    vec.envs = [env]
    vec.set_eval = lambda _flag: None
    cfg = get_config(
        "PointMaze",
        pn_lmcgs_enabled=True,
        max_episode_steps=3,
        test_episode_steps=3,
        hidden_units=4,
        hidden_layers=1,
        ae_hidden_units=4,
        ae_hidden_layers=1,
        embedding_size=2,
        n_landmarks=2,
        batch_size=2,
        gls_batch_size=2,
        landmark_batch_size=2,
        pn_k_min=1,
        pn_k_max=3,
    )
    return L3PTrainer(vec, cfg)


def test_collection_stops_at_task_goal_without_marking_command_success(monkeypatch):
    trainer = _reach_then_leave_trainer()
    command = CommandResult(np.asarray([9.0, 0.0], dtype=np.float32), 3, "direct")
    monkeypatch.setattr(trainer, "_pn_select_command", lambda *_args: command)

    episode = trainer.collect_episode(
        trainer.env.envs[0], use_planning=False, random_actions=False,
    )

    assert episode["length"] == 1
    assert episode["task_goal_reached"].tolist() == [True]
    assert not trainer.macro_buffer.attempts[0].target_reached


def test_evaluation_stops_at_first_task_goal_reach(monkeypatch):
    trainer = _reach_then_leave_trainer()
    monkeypatch.setattr(
        trainer.pn_planner,
        "direct_command",
        lambda _state, goal: CommandResult(np.asarray(goal), 3, "direct"),
    )

    assert trainer.evaluate(1, use_planning=False) == 1.0
    assert trainer.env.envs[0].step_count == 1
    assert trainer.last_eval_metrics["goal_success_rate"] == 1.0


def _numpy_rng_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_pn_evaluation_restores_rng_planner_search_and_module_modes(monkeypatch):
    trainer = _scripted_eval_trainer(safety_cost=0.0, reaches_goal=False)
    trainer.centroids_initialized = True
    trainer.pn_phase = PNPhase.JOINT
    trainer.planner.cnt = 7.0
    trainer.pn_planner._goal_dim = 17
    trainer.pn_planner.search.last_diagnostics = {"before": 1}
    trainer.macro_model.train(True)
    np.random.seed(1234)
    trainer_rng_before = copy.deepcopy(trainer.rng.bit_generator.state)
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()
    planner_cnt_before = trainer.planner.cnt
    adapter_goal_dim_before = trainer.pn_planner._goal_dim
    search_diagnostics_before = trainer.pn_planner.search.last_diagnostics

    def mutate_state(*_args, **_kwargs):
        trainer.rng.random()
        np.random.random()
        torch.rand(1)
        trainer.planner.cnt = 99.0
        trainer.pn_planner._goal_dim = 99
        trainer.pn_planner.search.last_diagnostics = {"after": 1}
        trainer.macro_model.eval()
        return CommandResult(None, 0, "mcgs", diagnostics={})

    monkeypatch.setattr(trainer.pn_planner, "planned_command", mutate_state)
    trainer.evaluate(1, use_planning=True)

    assert trainer.rng.bit_generator.state == trainer_rng_before
    assert _numpy_rng_states_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert trainer.planner.cnt == planner_cnt_before
    assert trainer.pn_planner._goal_dim == adapter_goal_dim_before
    assert trainer.pn_planner.search.last_diagnostics is search_diagnostics_before
    assert trainer.macro_model.training


def _real_pointmaze_trainer(seed, n_workers=2):
    cfg = get_config(
        "PointMaze",
        pn_lmcgs_enabled=True,
        n_workers=n_workers,
        max_episode_steps=1,
        test_episode_steps=1,
        hidden_units=4,
        hidden_layers=1,
        ae_hidden_units=4,
        ae_hidden_layers=1,
        embedding_size=2,
        n_landmarks=2,
        pn_num_positive_landmarks=2,
        batch_size=2,
        gls_batch_size=2,
        landmark_batch_size=2,
    )
    return L3PTrainer(make_vec_env(cfg, cfg.n_workers, seed), cfg)


def _pointmaze_rng_fingerprints(trainer):
    return [
        trainer_module._state_fingerprint(env.rng.bit_generator.state)
        for env in trainer.env.envs
    ]


def test_pn_evaluation_restores_real_pointmaze_environment_rng():
    trainer = _real_pointmaze_trainer(seed=17)
    before = _pointmaze_rng_fingerprints(trainer)

    trainer.evaluate(1, use_planning=False)

    assert _pointmaze_rng_fingerprints(trainer) == before


def test_checkpoint_exactly_restores_real_vec_env_rng(tmp_path):
    checkpoint = tmp_path / "real-environment-rng.pt"
    source = _real_pointmaze_trainer(seed=23)
    for index, env in enumerate(source.env.envs):
        env.reset()
        env.rng.random(index + 1)
    expected = _pointmaze_rng_fingerprints(source)
    source.save(checkpoint)
    raw = torch.load(checkpoint, weights_only=False)
    restored = _real_pointmaze_trainer(seed=91)

    restored.load(checkpoint)

    assert raw["pn"]["rng"]["environment_numpy"]
    assert _pointmaze_rng_fingerprints(restored) == expected
    for source_env, restored_env in zip(source.env.envs, restored.env.envs):
        np.testing.assert_array_equal(source_env.rng.random(4), restored_env.rng.random(4))


@pytest.mark.parametrize("source_workers, restored_workers", ((2, 1), (1, 2)))
def test_checkpoint_restores_overlapping_environment_rng_across_worker_counts(
        tmp_path, source_workers, restored_workers):
    checkpoint = tmp_path / "runtime-worker-count.pt"
    source = _real_pointmaze_trainer(seed=29, n_workers=source_workers)
    for index, env in enumerate(source.env.envs):
        env.reset()
        env.rng.random(index + 1)
    source.save(checkpoint)
    restored = _real_pointmaze_trainer(seed=101, n_workers=restored_workers)

    restored.load(checkpoint)

    np.testing.assert_array_equal(
        source.env.envs[0].rng.random(4), restored.env.envs[0].rng.random(4),
    )


def test_malformed_overlapping_environment_rng_is_rejected_atomically(tmp_path):
    valid_path = tmp_path / "valid-environment-rng.pt"
    malformed_path = tmp_path / "malformed-environment-rng.pt"
    source = _real_pointmaze_trainer(seed=31, n_workers=2)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["rng"]["environment_numpy"][0][0]["path"] = ("wrong",)
    torch.save(malformed, malformed_path)
    destination = _real_pointmaze_trainer(seed=103, n_workers=1)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="environment RNG path"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_overlapping_environment_bit_generator_type_is_prevalidated_atomically(tmp_path):
    valid_path = tmp_path / "valid-bit-generator.pt"
    malformed_path = tmp_path / "wrong-bit-generator.pt"
    source = _real_pointmaze_trainer(seed=41, n_workers=2)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["rng"]["environment_numpy"][0][0]["state"] = \
        np.random.Philox().state
    torch.save(malformed, malformed_path)
    destination = _real_pointmaze_trainer(seed=109, n_workers=1)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="environment RNG type|environment NumPy RNG state"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_malformed_unmatched_environment_rng_record_is_rejected_atomically(tmp_path):
    valid_path = tmp_path / "valid-unmatched-environment-rng.pt"
    malformed_path = tmp_path / "malformed-unmatched-environment-rng.pt"
    source = _real_pointmaze_trainer(seed=37, n_workers=2)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["rng"]["environment_numpy"][1][0]["state"] = {}
    torch.save(malformed, malformed_path)
    destination = _real_pointmaze_trainer(seed=107, n_workers=1)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="environment NumPy RNG state"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_original_pn_command_recomputes_adaptive_ceiling_horizon(monkeypatch):
    trainer = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_k_min=1,
        pn_k_max=5,
    )
    goal = np.asarray([4.0, 5.0], dtype=np.float32)
    monkeypatch.setattr(trainer.planner, "select_command", lambda _obs: (goal, 1))
    monkeypatch.setattr(
        trainer.agent,
        "distance_after_action",
        lambda _obs, goals: np.full(len(np.atleast_2d(goals)), 2.1, dtype=np.float32),
    )

    command = trainer._pn_original_command(np.zeros(2, dtype=np.float32))

    np.testing.assert_array_equal(command.goal, goal)
    assert command.horizon == 3


def test_planner_diagnostics_are_aggregated_and_include_active_counts(monkeypatch):
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    trainer.pn_phase = PNPhase.JOINT
    trainer.pn_positive_active = True
    trainer.pn_negative_active = True
    trainer.rng = _FixedChoiceRng(choice=0)
    diagnostics = iter((
        {"latency": 1.0, "branch_before": 2, "branch_after": 1,
         "chosen_p_violation": 0.1, "chosen_q_risk": 0.2,
         "chosen_upper_risk": 0.3, "completed_simulations": 4,
         "simulated_safe_successes": 2, "cycles": 1, "leaf_uses": 3},
        {"latency": 3.0, "branch_before": 6, "branch_after": 3,
         "chosen_p_violation": 0.3, "chosen_q_risk": 0.4,
         "chosen_upper_risk": 0.5, "completed_simulations": 8,
         "simulated_safe_successes": 6, "cycles": 5, "leaf_uses": 7},
    ))
    monkeypatch.setattr(
        trainer.pn_planner,
        "planned_command",
        lambda *_args, **_kwargs: CommandResult(
            np.ones(2, dtype=np.float32), 1, "mcgs", diagnostics=next(diagnostics),
        ),
    )
    for _ in range(2):
        trainer._pn_select_command(
            _TinySafetyEnv._observation(), np.ones(2, dtype=np.float32),
            use_planning=True, random_actions=False, original_ready=False,
        )

    logs = trainer._pn_cumulative_diagnostics()

    assert logs["active_positive_centroid_count"] == trainer.landmarks.n_landmarks
    assert logs["active_negative_centroid_count"] == trainer.negative_landmarks.n_landmarks
    assert logs["planner_latency"] == 2.0
    assert logs["planner_branch_before"] == 4.0
    assert logs["planner_simulation_count"] == 6.0
    assert logs["planner_simulated_safe_success_rate"] == pytest.approx(8 / 12)
    assert logs["planner_cycles"] == 3.0
    assert logs["planner_leaf_uses"] == 5.0


def _prime_optimizer(module, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().mean() for parameter in module.parameters())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _baseline_trainer_fingerprint(trainer):
    state = {
        "agent": trainer.agent.state_dict(),
        "ae": trainer.ae.state_dict(),
        "landmarks": trainer.landmarks.state_dict(),
        "centroids_initialized": trainer.centroids_initialized,
        "config": copy.deepcopy(vars(trainer.cfg)),
    }
    if trainer.cfg.pn_lmcgs_enabled:
        had_snapshot = hasattr(trainer, "pn_config_snapshot")
        previous_snapshot = copy.deepcopy(getattr(trainer, "pn_config_snapshot", None))
        try:
            state["version"] = trainer_module.PN_CHECKPOINT_VERSION
            state["pn"] = trainer._pn_checkpoint_state()
        finally:
            if had_snapshot:
                trainer.pn_config_snapshot = previous_snapshot
            elif hasattr(trainer, "pn_config_snapshot"):
                del trainer.pn_config_snapshot
    return trainer_module._state_fingerprint(state)


def test_atomic_fingerprint_covers_enabled_pn_state_and_rng_without_mutation():
    trainer = _tiny_trainer(pn_lmcgs_enabled=True)
    disabled = _tiny_trainer(pn_lmcgs_enabled=False)
    trainer.pn_config_snapshot = {"sentinel": "preserve"}
    config_snapshot = copy.deepcopy(trainer.pn_config_snapshot)
    trainer_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    negative_centroids = trainer.negative_landmarks.centroids.detach().clone()
    try:
        baseline = _baseline_trainer_fingerprint(trainer)
        assert _baseline_trainer_fingerprint(trainer) == baseline
        assert trainer.rng.bit_generator.state == trainer_rng
        assert _numpy_rng_states_equal(np.random.get_state(), numpy_rng)
        assert torch.equal(torch.get_rng_state(), torch_rng)
        assert trainer.pn_config_snapshot == config_snapshot

        with torch.no_grad():
            trainer.negative_landmarks.centroids.add_(1.0)
        assert _baseline_trainer_fingerprint(trainer) != baseline
        trainer.negative_landmarks.centroids.data.copy_(negative_centroids)
        assert _baseline_trainer_fingerprint(trainer) == baseline

        trainer.rng.random()
        assert _baseline_trainer_fingerprint(trainer) != baseline
        trainer.rng.bit_generator.state = copy.deepcopy(trainer_rng)
        assert _baseline_trainer_fingerprint(trainer) == baseline

        np.random.random()
        assert _baseline_trainer_fingerprint(trainer) != baseline
        np.random.set_state(copy.deepcopy(numpy_rng))
        assert _baseline_trainer_fingerprint(trainer) == baseline

        torch.rand(1)
        assert _baseline_trainer_fingerprint(trainer) != baseline
        torch.set_rng_state(torch_rng.clone())
        assert _baseline_trainer_fingerprint(trainer) == baseline

        disabled_baseline = _baseline_trainer_fingerprint(disabled)
        np.random.random()
        torch.rand(1)
        assert _baseline_trainer_fingerprint(disabled) == disabled_baseline
    finally:
        trainer.rng.bit_generator.state = copy.deepcopy(trainer_rng)
        np.random.set_state(copy.deepcopy(numpy_rng))
        torch.set_rng_state(torch_rng)


@pytest.mark.parametrize(
    ("invalid_state", "message"),
    (
        pytest.param(
            torch.zeros(32, dtype=torch.int64),
            "torch.uint8",
            id="wrong-dtype",
        ),
        pytest.param(
            torch.zeros((4, 4), dtype=torch.uint8),
            "one-dimensional",
            id="wrong-rank",
        ),
        pytest.param(
            torch.zeros(1, dtype=torch.uint8),
            "at least 16 bytes",
            id="implausibly-short",
        ),
    ),
)
def test_malformed_cuda_rng_state_is_rejected_before_trainer_mutation(
        tmp_path, invalid_state, message):
    valid_path = tmp_path / "valid-cuda-rng.pt"
    malformed_path = tmp_path / "malformed-cuda-rng.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["rng"]["cuda"] = [invalid_state]
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)
    assert _baseline_trainer_fingerprint(source) != before

    with pytest.raises(ValueError, match=message):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def _install_fake_cuda_rng_runtime(
        monkeypatch, current_device_count, rejected_state=None):
    real_generator = torch.Generator
    probed = []
    restored = []
    set_all_calls = []

    class _CudaGeneratorProbe:
        def __init__(self, device):
            self.device = str(device)

        def set_state(self, state):
            state = state.clone()
            probed.append((self.device, state))
            if rejected_state is not None and torch.equal(state, rejected_state):
                raise RuntimeError("invalid opaque CUDA RNG state")
            return self

    def generator(*args, **kwargs):
        device = kwargs.get("device", args[0] if args else "cpu")
        if str(device).startswith("cuda"):
            return _CudaGeneratorProbe(device)
        return real_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: current_device_count)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device=None: restored.append((device, state.clone())),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: set_all_calls.append([state.clone() for state in states]),
    )
    return probed, restored, set_all_calls


def test_live_cuda_rng_validation_is_atomic_for_each_overlapping_device(
        tmp_path, monkeypatch):
    valid_path = tmp_path / "valid-live-cuda-rng.pt"
    malformed_path = tmp_path / "invalid-live-cuda-rng.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    valid_state = torch.ones(16, dtype=torch.uint8)
    rejected_state = torch.arange(16, dtype=torch.uint8)
    malformed["pn"]["rng"]["cuda"] = [valid_state, rejected_state]
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)
    probed, restored, set_all_calls = _install_fake_cuda_rng_runtime(
        monkeypatch, current_device_count=2, rejected_state=rejected_state,
    )

    with pytest.raises(ValueError, match="CUDA RNG state 1.*current device"):
        destination.load(malformed_path)

    assert [device for device, _state in probed] == ["cuda:0", "cuda:1"]
    assert not restored and not set_all_calls
    assert _baseline_trainer_fingerprint(destination) == before


def test_cuda_rng_restore_tolerates_saved_and_current_device_count_difference(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "different-cuda-device-count.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    source.save(checkpoint)
    raw = torch.load(checkpoint, weights_only=False)
    saved_states = [
        torch.arange(16, dtype=torch.uint8),
        torch.arange(16, dtype=torch.uint8) + 16,
    ]
    raw["pn"]["rng"]["cuda"] = saved_states
    torch.save(raw, checkpoint)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    probed, restored, set_all_calls = _install_fake_cuda_rng_runtime(
        monkeypatch, current_device_count=1,
    )

    destination.load(checkpoint)

    assert len(probed) == 1 and probed[0][0] == "cuda:0"
    assert torch.equal(probed[0][1], saved_states[0])
    assert len(restored) == 1 and restored[0][0] == 0
    assert torch.equal(restored[0][1], saved_states[0])
    assert not set_all_calls


def test_versioned_checkpoint_exactly_restores_training_replay_config_and_rng(tmp_path):
    checkpoint = tmp_path / "exact-resume.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    _store_primitive_pn_episode(source)
    for module, optimizer in (
        (source.agent.actor, source.agent.actor_opt),
        (source.agent.critic, source.agent.critic_opt),
        (source.agent.value, source.agent.value_opt),
        (source.ae, source.ae_opt),
    ):
        _prime_optimizer(module, optimizer)
    with torch.no_grad():
        for parameter in source.agent.actor_target.parameters():
            parameter.add_(0.125)
        for parameter in source.agent.critic_target.parameters():
            parameter.sub_(0.25)
    source.cfg.pn_probability_mcg_search = 0.4
    source.cfg.pn_probability_original_planner = 0.2
    source.cfg.pn_probability_direct_goal = 0.4
    source.rng.random(3)
    np.random.seed(8675309)
    np.random.random(2)
    torch.manual_seed(4242)
    torch.rand(2)
    trainer_rng = copy.deepcopy(source.rng.bit_generator.state)
    global_numpy_rng = copy.deepcopy(np.random.get_state())
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else []
    )

    source.save(checkpoint)
    raw = torch.load(checkpoint, weights_only=False)

    pn = raw["pn"]
    assert {"baseline_training", "primitive_replay"}.issubset(pn)
    assert {
        "actor_target", "critic_target", "actor_opt", "critic_opt",
        "value_opt", "ae_opt",
    }.issubset(pn["baseline_training"])
    assert {"trainer_numpy", "global_numpy", "torch", "cuda"}.issubset(pn["rng"])
    assert len(pn["primitive_replay"]["episodes"]) == 1
    assert pn["primitive_replay"]["episodes"][0]["obs"].shape == (2, 2)
    assert pn["config_snapshot"]["pn_probability_direct_goal"] == 0.4

    restored = _tiny_trainer(pn_lmcgs_enabled=True)
    restored.load(checkpoint)

    for name in ("actor_target", "critic_target"):
        source_module = getattr(source.agent, name)
        restored_module = getattr(restored.agent, name)
        assert trainer_module._state_fingerprint(source_module.state_dict()) == \
            trainer_module._state_fingerprint(restored_module.state_dict())
    for name in ("actor_opt", "critic_opt", "value_opt"):
        assert trainer_module._state_fingerprint(getattr(source.agent, name).state_dict()) == \
            trainer_module._state_fingerprint(getattr(restored.agent, name).state_dict())
    assert trainer_module._state_fingerprint(source.ae_opt.state_dict()) == \
        trainer_module._state_fingerprint(restored.ae_opt.state_dict())
    assert restored.buffer.ptr == source.buffer.ptr
    assert restored.buffer.n_episodes == source.buffer.n_episodes
    assert restored.buffer.n_transitions == source.buffer.n_transitions
    np.testing.assert_array_equal(restored.buffer.obs[0, :2], source.buffer.obs[0, :2])
    np.testing.assert_array_equal(restored.buffer.cmd_g[0, :1], source.buffer.cmd_g[0, :1])
    assert restored.cfg.pn_probability_mcg_search == 0.4
    assert restored.cfg.pn_probability_original_planner == 0.2
    assert restored.cfg.pn_probability_direct_goal == 0.4
    assert restored.rng.bit_generator.state == trainer_rng
    assert _numpy_rng_states_equal(np.random.get_state(), global_numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    if cuda_rng:
        assert all(torch.equal(actual, expected) for actual, expected in zip(
            torch.cuda.get_rng_state_all(), cuda_rng,
        ))


def test_checkpoint_schema_is_prevalidated_before_any_trainer_mutation(tmp_path):
    valid_path = tmp_path / "valid-preflight.pt"
    malformed_path = tmp_path / "missing-online-cost.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["agent"].pop("cost_critic")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="cost_critic|incomplete|checkpoint"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_macro_module_config_mismatch_is_rejected_before_trainer_mutation(tmp_path):
    valid_path = tmp_path / "valid-module-config.pt"
    malformed_path = tmp_path / "invalid-module-config.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["module_config"]["hidden_dim"] += 1
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(
        ValueError, match="module|config|macro|incompatible|checkpoint",
    ):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_macro_module_config_must_match_config_snapshot_atomically(tmp_path):
    valid_path = tmp_path / "valid-macro-snapshot.pt"
    malformed_path = tmp_path / "invalid-macro-snapshot.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    module_config = malformed["pn"]["module_config"]
    altered_model = MacroTransitionModel(
        embedding_dim=module_config["embedding_dim"],
        context_dim=module_config["context_dim"],
        hidden_dim=module_config["hidden_dim"] + 1,
        k_max=module_config["k_max"],
    )
    module_config["hidden_dim"] += 1
    malformed["pn"]["macro_model"] = altered_model.state_dict()
    malformed["pn"]["macro_opt"] = torch.optim.Adam(
        altered_model.parameters(), lr=source.cfg.pn_macro_lr,
    ).state_dict()
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="module_config.*config_snapshot"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_unversioned_partial_pn_and_incompatible_config_are_rejected_atomically(tmp_path):
    baseline_path = tmp_path / "legacy-with-pn.pt"
    baseline = _tiny_trainer(pn_lmcgs_enabled=False)
    baseline.save(baseline_path)
    malformed = torch.load(baseline_path, weights_only=False)
    malformed["pn"] = {"partial": True}
    torch.save(malformed, baseline_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="version|pn|checkpoint"):
        destination.load(baseline_path)
    assert _baseline_trainer_fingerprint(destination) == before

    incompatible_path = tmp_path / "incompatible.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    source.save(incompatible_path)
    incompatible = _tiny_trainer(pn_lmcgs_enabled=True, embedding_size=3)
    incompatible_before = _baseline_trainer_fingerprint(incompatible)

    with pytest.raises(ValueError, match="config|embedding|incompatible|checkpoint"):
        incompatible.load(incompatible_path)
    assert _baseline_trainer_fingerprint(incompatible) == incompatible_before


def test_legacy_upgrade_reinitializes_pn_positive_landmarks_from_safe_data(tmp_path):
    checkpoint = tmp_path / "legacy-landmarks.pt"
    baseline = _tiny_trainer(pn_lmcgs_enabled=False, n_landmarks=4)
    baseline.centroids_initialized = True
    with torch.no_grad():
        baseline.landmarks.centroids.fill_(99.0)
    baseline.save(checkpoint)

    upgraded = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_num_positive_landmarks=2,
        pn_min_positive_samples=1,
        pn_min_positive_episodes=1,
    )
    upgraded.load(checkpoint)

    assert upgraded.landmarks.n_landmarks == 2
    assert not upgraded.centroids_initialized
    assert not upgraded.pn_positive_active
    assert not torch.all(upgraded.landmarks.centroids == 99.0)


def test_nonfinite_nested_checkpoint_state_is_rejected_before_mutation(tmp_path):
    valid_path = tmp_path / "valid-finite.pt"
    malformed_path = tmp_path / "invalid-calibration.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["calibration"]["temperature"] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="calibration|finite|checkpoint"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


@pytest.mark.parametrize("corruption", ("module", "optimizer"))
def test_nonfinite_module_and_optimizer_tensors_are_rejected_atomically(
        tmp_path, corruption):
    valid_path = tmp_path / "valid-tensor-state.pt"
    malformed_path = tmp_path / f"nonfinite-{corruption}.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    _prime_optimizer(source.macro_model, source.macro_opt)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    if corruption == "module":
        tensor = next(iter(malformed["agent"]["actor"].values()))
        tensor.reshape(-1)[0] = float("nan")
    else:
        optimizer_state = malformed["pn"]["macro_opt"]["state"]
        parameter_state = next(iter(optimizer_state.values()))
        parameter_state["exp_avg"].reshape(-1)[0] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match=f"{corruption}|actor|macro_opt|finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_nonfinite_unused_pn_optimizer_tensor_is_rejected_atomically(tmp_path):
    valid_path = tmp_path / "valid-unused-optimizer.pt"
    malformed_path = tmp_path / "nonfinite-unused-optimizer.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    _prime_optimizer(source.macro_model, source.macro_opt)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    optimizer_state = malformed["pn"]["macro_opt"]["state"]
    parameter_state = next(iter(optimizer_state.values()))
    parameter_state["exp_avg"].reshape(-1)[0] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=False)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="macro_opt.*finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def _populated_replay_and_landmark_memory_source():
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    _store_primitive_pn_episode(source)
    source.landmark_memory.ingest_episode({
        "ag": np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "cmd_g": np.asarray([[1.0, 1.0]], dtype=np.float32),
        "cost": np.zeros(1, dtype=np.float32),
        "goal_reached": np.ones(1, dtype=bool),
        "length": 1,
    }, episode_id=11)
    source.landmark_memory.ingest_episode({
        "ag": np.asarray([[8.0, 8.0], [9.0, 9.0]], dtype=np.float32),
        "cost": np.ones(1, dtype=np.float32),
        "length": 1,
    }, episode_id=12)
    assert source.buffer.n_episodes == 1
    assert source.landmark_memory.positive_size == 1
    assert source.landmark_memory.negative_size == 1
    return source


@pytest.mark.parametrize(
    "field",
    ("obs", "ag", "g", "cmd_g", "act", "cost"),
)
def test_nonfinite_primitive_replay_is_rejected_before_trainer_mutation(
        tmp_path, field):
    malformed_path = tmp_path / f"nonfinite-primitive-replay-{field}.pt"
    source = _populated_replay_and_landmark_memory_source()
    source.save(malformed_path)
    malformed = torch.load(malformed_path, weights_only=False)
    replay_episode = malformed["pn"]["primitive_replay"]["episodes"][0]
    replay_episode[field].reshape(-1)[0] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)
    assert _baseline_trainer_fingerprint(source) != before

    with pytest.raises(ValueError, match=rf"primitive replay {field}.*finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


@pytest.mark.parametrize("field", ("positive_goals", "negative_goals"))
def test_nonfinite_landmark_memory_goals_are_rejected_before_trainer_mutation(
        tmp_path, field):
    malformed_path = tmp_path / f"nonfinite-landmark-memory-{field}.pt"
    source = _populated_replay_and_landmark_memory_source()
    source.save(malformed_path)
    malformed = torch.load(malformed_path, weights_only=False)
    malformed["pn"]["landmark_memory"][field].reshape(-1)[0] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)
    assert _baseline_trainer_fingerprint(source) != before

    with pytest.raises(ValueError, match=rf"landmark memory {field}.*finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("negative_episode_ids", np.asarray([1.5]), "episode IDs"),
        ("negative_is_preimpact", np.asarray([1], dtype=np.int64), "pre-impact metadata"),
    ),
)
def test_landmark_memory_metadata_is_rejected_before_trainer_mutation(
        tmp_path, field, invalid, message):
    malformed_path = tmp_path / f"invalid-landmark-memory-{field}.pt"
    source = _populated_replay_and_landmark_memory_source()
    source.save(malformed_path)
    malformed = torch.load(malformed_path, weights_only=False)
    malformed["pn"]["landmark_memory"][field] = invalid
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)
    assert _baseline_trainer_fingerprint(source) != before

    with pytest.raises(ValueError, match=rf"landmark memory {message}.*invalid"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


@pytest.mark.parametrize(
    "corruption",
    (
        "cost_critic_nan",
        "cost_critic_shape",
        "cost_critic_target_nan",
        "cost_critic_target_shape",
        "negative_landmarks_nan",
        "negative_landmarks_shape",
        "positive_landmark_opt_empty_groups",
        "negative_landmark_opt_empty_groups",
        "macro_opt_empty_groups",
        "cost_critic_opt_empty_groups",
    ),
)
def test_disabled_destination_prevalidates_all_serialized_pn_state_atomically(
        tmp_path, corruption):
    valid_path = tmp_path / "valid-disabled-preflight.pt"
    malformed_path = tmp_path / f"invalid-disabled-preflight-{corruption}.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    for module, optimizer in (
        (source.landmarks, source.landmark_opt),
        (source.negative_landmarks, source.negative_landmark_opt),
        (source.macro_model, source.macro_opt),
        (source.agent.cost_critic, source.agent.cost_critic_opt),
    ):
        _prime_optimizer(module, optimizer)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)

    state_locations = {
        "cost_critic": malformed["agent"]["cost_critic"],
        "cost_critic_target": malformed["pn"]["cost_critic_target"],
        "negative_landmarks": malformed["pn"]["negative_landmarks"],
    }
    state_name = corruption.removesuffix("_nan").removesuffix("_shape")
    if corruption.endswith("_nan"):
        next(iter(state_locations[state_name].values())).reshape(-1)[0] = float("nan")
    elif corruption.endswith("_shape"):
        key = next(iter(state_locations[state_name]))
        value = state_locations[state_name][key]
        state_locations[state_name][key] = torch.zeros(
            *value.shape[:-1], value.shape[-1] + 1,
            dtype=value.dtype, device=value.device,
        )
    else:
        optimizer_name = corruption.removesuffix("_empty_groups")
        malformed["pn"][optimizer_name]["param_groups"] = []
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=False)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="cost_critic|negative_landmarks|optimizer|shape|finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_restored_config_is_validated_before_checkpoint_mutation(tmp_path):
    valid_path = tmp_path / "valid-config.pt"
    malformed_path = tmp_path / "invalid-restored-config.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    malformed["pn"]["config_snapshot"]["pn_probability_mcg_search"] = float("nan")
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="pn_probability_mcg_search|config|finite"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_valid_pn_checkpoint_loads_into_disabled_trainer_as_baseline(tmp_path):
    checkpoint = tmp_path / "pn-to-disabled.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(0.5)
    source.save(checkpoint)
    disabled = _tiny_trainer(pn_lmcgs_enabled=False)
    torch.manual_seed(2718)
    torch_rng = torch.get_rng_state().clone()

    disabled.load(checkpoint)

    assert not hasattr(disabled, "pn_planner")
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert trainer_module._state_fingerprint(disabled.agent.actor.state_dict()) == \
        trainer_module._state_fingerprint(source.agent.actor.state_dict())


@pytest.mark.parametrize("corruption", ("normalizer", "optimizer"))
def test_nested_training_schema_is_prevalidated_atomically(tmp_path, corruption):
    valid_path = tmp_path / "valid-nested.pt"
    malformed_path = tmp_path / f"invalid-{corruption}.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    with torch.no_grad():
        for parameter in source.agent.actor.parameters():
            parameter.add_(1.0)
    _prime_optimizer(source.agent.actor, source.agent.actor_opt)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    if corruption == "normalizer":
        malformed["agent"]["o_norm"].pop("mean")
    else:
        malformed["pn"]["baseline_training"]["actor_opt"]["param_groups"] = []
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="normalizer|optimizer|incomplete|checkpoint"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_landmark_optimizer_tensor_shapes_are_prevalidated_atomically(tmp_path):
    valid_path = tmp_path / "valid-landmark-optimizer.pt"
    malformed_path = tmp_path / "invalid-landmark-optimizer.pt"
    source = _tiny_trainer(pn_lmcgs_enabled=True)
    _prime_optimizer(source.landmarks, source.landmark_opt)
    _prime_optimizer(source.negative_landmarks, source.negative_landmark_opt)
    source.save(valid_path)
    malformed = torch.load(valid_path, weights_only=False)
    optimizer_state = malformed["pn"]["positive_landmark_opt"]["state"]
    parameter_state = next(iter(optimizer_state.values()))
    parameter_state["exp_avg"] = torch.zeros(1)
    torch.save(malformed, malformed_path)
    destination = _tiny_trainer(pn_lmcgs_enabled=True)
    before = _baseline_trainer_fingerprint(destination)

    with pytest.raises(ValueError, match="positive_landmark_opt.*tensor shape"):
        destination.load(malformed_path)

    assert _baseline_trainer_fingerprint(destination) == before


def test_valid_checkpoint_rebuilds_different_landmark_counts_with_optimizer_state(tmp_path):
    checkpoint = tmp_path / "different-landmark-counts.pt"
    source = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_num_positive_landmarks=3,
        pn_num_negative_landmarks=4,
    )
    _prime_optimizer(source.landmarks, source.landmark_opt)
    _prime_optimizer(source.negative_landmarks, source.negative_landmark_opt)
    source.save(checkpoint)
    destination = _tiny_trainer(
        pn_lmcgs_enabled=True,
        pn_num_positive_landmarks=2,
        pn_num_negative_landmarks=2,
    )

    destination.load(checkpoint)

    assert destination.landmarks.n_landmarks == 3
    assert destination.negative_landmarks.n_landmarks == 4
    assert trainer_module._state_fingerprint(destination.landmark_opt.state_dict()) == \
        trainer_module._state_fingerprint(source.landmark_opt.state_dict())
    assert trainer_module._state_fingerprint(destination.negative_landmark_opt.state_dict()) == \
        trainer_module._state_fingerprint(source.negative_landmark_opt.state_dict())
