"""Goal-conditioned violation critic and safe actor regression tests."""

import numpy as np
import pytest
import torch

from l3p.agent.ddpg import DDPGAgent
from l3p.config import get_config
from l3p.pn_lmcgs.safety_critic import ViolationCritic, violation_td_target
from l3p.trainer import L3PTrainer


def make_cfg(enabled=True, **overrides):
    cfg = get_config("PointMaze", device="cpu", pn_lmcgs_enabled=enabled,
                     hidden_units=16, hidden_layers=1, batch_size=4,
                     target_update_interval=1, **overrides)
    return cfg


def make_agent(enabled=True, **overrides):
    torch.manual_seed(7)
    return DDPGAgent(2, 1, 1, 1.0, make_cfg(enabled, **overrides))


def make_batch(agent, batch_size=4):
    raw = {
        "obs": np.linspace(-1, 1, batch_size * 2, dtype=np.float32).reshape(batch_size, 2),
        "next_obs": np.linspace(1, -1, batch_size * 2, dtype=np.float32).reshape(batch_size, 2),
        "g": np.linspace(-0.5, 0.5, batch_size, dtype=np.float32).reshape(batch_size, 1),
        "act": np.linspace(-0.25, 0.25, batch_size, dtype=np.float32).reshape(batch_size, 1),
    }
    return {
        "obs": agent.norm_obs(raw["obs"]),
        "next_obs": agent.norm_obs(raw["next_obs"]),
        "g": agent.norm_goal(raw["g"]),
        "act": agent.to_tensor(raw["act"]),
        "reward": agent.to_tensor(np.zeros(batch_size, dtype=np.float32)),
        "next_ag": agent.to_tensor(raw["g"]),
        "future_ag": agent.to_tensor(raw["g"]),
        "future_ag_norm": agent.norm_goal(raw["g"]),
        "cost": agent.to_tensor(np.array([0, 1, 0, 0][:batch_size], dtype=np.float32)),
        "stop": agent.to_tensor(np.array([0, 0, 1, 0][:batch_size], dtype=np.float32)),
    }


def test_violation_target_stops_on_cost_goal_and_termination():
    next_probability = torch.tensor([0.8, 0.8, 0.8, 0.8])
    target = violation_td_target(
        cost=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        stop=torch.tensor([False, True, True, False]),
        next_probability=next_probability,
    )
    assert torch.allclose(target, torch.tensor([1.0, 0.0, 0.0, 0.8]))
    assert torch.all((target >= 0) & (target <= 1))


def test_violation_critic_outputs_batched_probabilities_in_unit_interval():
    critic = ViolationCritic(obs_dim=2, goal_dim=1, act_dim=1, hidden_units=8, hidden_layers=2)
    probability = critic(torch.randn(3, 2), torch.randn(3, 1), torch.randn(3, 1))
    assert probability.shape == (3,)
    assert torch.isfinite(probability).all()
    assert torch.all((probability >= 0) & (probability <= 1))


def test_cost_update_changes_online_critic_but_not_target():
    agent = make_agent(True)
    batch = make_batch(agent)
    online_before = [p.detach().clone() for p in agent.cost_critic.parameters()]
    target_before = [p.detach().clone() for p in agent.cost_critic_target.parameters()]
    loss = agent.update_cost_critic(batch)
    assert np.isfinite(loss)
    assert any(not torch.equal(before, after) for before, after in zip(online_before, agent.cost_critic.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(target_before, agent.cost_critic_target.parameters()))
    assert all(p.grad is None for p in agent.cost_critic.parameters())
    assert all(p.grad is None for p in agent.cost_critic_target.parameters())


def test_actor_safety_gradient_updates_actor_not_cost_critic():
    agent = make_agent(True, pn_cost_critic_weight=100.0)
    batch = make_batch(agent)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    cost_before = [p.detach().clone() for p in agent.cost_critic.parameters()]
    target_before = [p.detach().clone() for p in agent.cost_critic_target.parameters()]
    agent.update_actor(batch)
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, agent.actor.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(cost_before, agent.cost_critic.parameters()))
    assert all(torch.equal(before, after) for before, after in zip(target_before, agent.cost_critic_target.parameters()))
    assert all(p.grad is None for p in agent.cost_critic.parameters())
    assert all(p.grad is None for p in agent.cost_critic_target.parameters())


def test_polyak_updates_cost_critic_target():
    agent = make_agent(True, polyak=0.25)
    with torch.no_grad():
        for parameter in agent.cost_critic.parameters():
            parameter.fill_(1.0)
        for parameter in agent.cost_critic_target.parameters():
            parameter.zero_()
    agent.update_targets()
    assert all(torch.allclose(parameter, torch.full_like(parameter, 0.75))
               for parameter in agent.cost_critic_target.parameters())


def test_cost_after_action_broadcasts_single_observation():
    agent = make_agent(True)
    obs = np.array([0.2, -0.4], dtype=np.float32)
    goals = np.array([[-0.5], [0.0], [0.5]], dtype=np.float32)
    actual = agent.cost_after_action(obs, goals)
    with torch.no_grad():
        normalized_obs = agent.norm_obs(np.repeat(obs[None], len(goals), axis=0))
        normalized_goals = agent.norm_goal(goals)
        expected = agent.cost_critic(normalized_obs, agent.actor(normalized_obs, normalized_goals),
                                     normalized_goals).cpu().numpy()
    assert actual.shape == (3,)
    np.testing.assert_allclose(actual, expected)


def test_disabled_agent_preserves_baseline_actor_update_and_state_keys():
    agent = make_agent(False)
    batch = make_batch(agent)
    with torch.no_grad():
        action = agent.actor(batch["obs"], batch["g"])
        expected_loss = (-agent.critic(batch["obs"], action, batch["g"], agent.gamma).mean()
                         + agent.cfg.action_l2 * (action / agent.max_action).pow(2).mean())
    returned_loss = agent.update_actor(batch)
    assert returned_loss == expected_loss.item()
    assert "cost_critic" not in agent.state_dict()
    assert not hasattr(agent, "cost_critic")


def test_legacy_and_pn_agent_state_loading_semantics():
    legacy = make_agent(False).state_dict()
    enabled = make_agent(True)
    cost_before = [p.detach().clone() for p in enabled.cost_critic.parameters()]
    enabled.load_state_dict(legacy)
    assert all(torch.equal(before, after) for before, after in zip(cost_before, enabled.cost_critic.parameters()))
    assert all(not parameter.requires_grad for parameter in enabled.cost_critic_target.parameters())

    pn_state = enabled.state_dict()
    round_trip = make_agent(True)
    round_trip.load_state_dict(pn_state)
    assert all(torch.equal(a, b) for a, b in zip(enabled.cost_critic.parameters(), round_trip.cost_critic.parameters()))
    disabled = make_agent(False)
    disabled.load_state_dict(pn_state)
    assert not hasattr(disabled, "cost_critic")


def test_trainer_batch_and_update_integrate_replay_cost_and_stop_fields():
    agent = make_agent(True)
    trainer = L3PTrainer.__new__(L3PTrainer)
    trainer.agent = agent
    raw = {
        "obs": np.zeros((2, 2), dtype=np.float32),
        "next_obs": np.ones((2, 2), dtype=np.float32),
        "g": np.zeros((2, 1), dtype=np.float32),
        "act": np.zeros((2, 1), dtype=np.float32),
        "reward": np.zeros(2, dtype=np.float32),
        "next_ag": np.zeros((2, 1), dtype=np.float32),
        "future_ag": np.ones((2, 1), dtype=np.float32),
        "ag": np.zeros((2, 1), dtype=np.float32),
        "cmd_g": np.full((2, 1), 2.0, dtype=np.float32),
        "cost": np.array([0.0, 1.0], dtype=np.float32),
        "stop": np.array([False, True]),
    }
    batch = trainer._make_batch(raw)
    assert set(("cost", "stop", "ag", "cmd_g", "ag_norm", "cmd_g_norm")) <= set(batch)
    assert torch.equal(batch["cost"], torch.tensor([0.0, 1.0]))
    assert torch.equal(batch["stop"], torch.tensor([0.0, 1.0]))


def _goal_observation(achieved, goal=5.0):
    return {
        "observation": np.array([achieved, -achieved], dtype=np.float32),
        "achieved_goal": np.array([achieved], dtype=np.float32),
        "desired_goal": np.array([goal], dtype=np.float32),
    }


class _ScriptedRawEnv:
    obs_dim, goal_dim, act_dim, max_action = 2, 1, 1, 1.0

    def __init__(self, steps):
        self.steps = list(steps)
        self.step_calls = 0

    def reset(self):
        self.step_calls = 0
        return _goal_observation(0.0)

    def step(self, action):
        spec = self.steps[self.step_calls]
        self.step_calls += 1
        observation = _goal_observation(spec["achieved"])
        info = dict(spec.get("info", {}))
        kind = spec["kind"]
        if kind == "legacy":
            if spec.get("truncated", False):
                info["TimeLimit.truncated"] = True
            return observation, -1.0, spec.get("done", False), info
        if kind == "gymnasium":
            return observation, -1.0, spec.get("terminated", False), spec.get("truncated", False), info
        return (observation, -1.0, spec.get("cost", 0.0),
                spec.get("terminated", False), spec.get("truncated", False), info)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        return np.where(np.all(np.isclose(achieved_goal, desired_goal), axis=-1), 0.0, -1.0)


class _ScriptedVecEnv:
    def __init__(self, env, horizon=3):
        self.envs = [env]
        self.n = 1
        self.obs_dim = env.obs_dim
        self.goal_dim = env.goal_dim
        self.act_dim = env.act_dim
        self.max_action = env.max_action
        self.max_episode_steps = horizon

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        return self.envs[0].compute_reward(achieved_goal, desired_goal, info)


def _collection_trainer(env, *, enabled=True, horizon=3, **overrides):
    cfg = make_cfg(
        enabled,
        max_episode_steps=horizon,
        test_episode_steps=horizon,
        her_ratio=0.0,
        n_warmup_trajs=999,
        random_landmarks_train=0,
        pn_collision_cost_enabled=True,
        pn_collision_cost_info_key="collision",
        pn_collision_cost_unsafe_value="hit",
        **overrides,
    )
    return L3PTrainer(_ScriptedVecEnv(env, horizon), cfg)


@pytest.mark.parametrize(
    "step",
    [
        {"kind": "legacy", "achieved": 5.0, "done": True, "info": {"collision": "hit"}},
        {"kind": "gymnasium", "achieved": 5.0, "terminated": True, "info": {"cost": 1.0}},
        {"kind": "safety", "achieved": 5.0, "terminated": True, "cost": 1.0},
    ],
)
def test_enabled_collection_persists_normalized_safety_metadata_and_real_length(step):
    env = _ScriptedRawEnv([step])
    trainer = _collection_trainer(env)

    trainer.collect()

    assert env.step_calls == 1
    assert trainer.total_env_steps == 1
    assert trainer.buffer.episode_lengths[0] == 1
    assert trainer.buffer.valid[0, 0]
    assert not trainer.buffer.valid[0, 1:].any()
    assert trainer.buffer.cost[0, 0] == 1.0
    assert trainer.buffer.violation[0, 0]
    assert trainer.buffer.goal_reached[0, 0]
    assert trainer.buffer.termination_reason[0, 0] == trainer.buffer.REASON_VIOLATION
    assert trainer.agent.o_norm.count == pytest.approx(1.0001)
    assert trainer.agent.g_norm.count == pytest.approx(1.0001)
    sampled = trainer.buffer.sample(8, np.random.default_rng(7))
    assert np.all(sampled["cost"] == 1.0)
    assert np.all(sampled["termination_reason"] == trainer.buffer.REASON_VIOLATION)


def test_enabled_collection_records_planner_command_goal_and_command_success():
    env = _ScriptedRawEnv([{"kind": "gymnasium", "achieved": 2.0, "terminated": True}])
    trainer = _collection_trainer(env)

    class _Planner:
        def reset(self, goal, extra=None):
            pass

        def act(self, observation, noise_scale, random_prob):
            return np.zeros(1, dtype=np.float32)

        def _current_subgoal(self):
            return np.array([2.0], dtype=np.float32)

    trainer.planner = _Planner()
    trainer.centroids_initialized = True
    episode = trainer.collect_episode(env, use_planning=True, random_actions=False)

    assert episode["length"] == 1
    np.testing.assert_array_equal(episode["g"][0], np.array([5.0], dtype=np.float32))
    np.testing.assert_array_equal(episode["cmd_g"][0], np.array([2.0], dtype=np.float32))
    assert episode["goal_reached"][0]
    assert episode["termination_reason"][0] == "goal"


def test_disabled_collection_preserves_legacy_keys_and_fixed_horizon_behavior():
    env = _ScriptedRawEnv([
        {"kind": "legacy", "achieved": 1.0, "done": True},
        {"kind": "legacy", "achieved": 2.0, "done": True},
        {"kind": "legacy", "achieved": 3.0, "done": True},
    ])
    trainer = _collection_trainer(env, enabled=False)

    episode = trainer.collect_episode(env, use_planning=False, random_actions=False)
    trainer.collect()

    assert set(episode) == {"obs", "ag", "g", "act"}
    assert env.step_calls == trainer.T
    assert trainer.total_env_steps == trainer.T
    assert trainer.buffer.episode_lengths[0] == trainer.T
    assert trainer.buffer.valid[0].all()


def _update_raw_batch(batch_size):
    return {
        "obs": np.zeros((batch_size, 2), dtype=np.float32),
        "next_obs": np.ones((batch_size, 2), dtype=np.float32),
        "g": np.zeros((batch_size, 1), dtype=np.float32),
        "act": np.zeros((batch_size, 1), dtype=np.float32),
        "reward": np.zeros(batch_size, dtype=np.float32),
        "next_ag": np.zeros((batch_size, 1), dtype=np.float32),
        "future_ag": np.ones((batch_size, 1), dtype=np.float32),
        "ag": np.zeros((batch_size, 1), dtype=np.float32),
        "cmd_g": np.zeros((batch_size, 1), dtype=np.float32),
        "cost": np.zeros(batch_size, dtype=np.float32),
        "stop": np.zeros(batch_size, dtype=bool),
    }


def test_trainer_cost_critic_threshold_batch_size_and_actor_risk_gate(monkeypatch):
    import l3p.trainer as trainer_module

    agent = make_agent(True)
    trainer = L3PTrainer.__new__(L3PTrainer)
    trainer.cfg = make_cfg(True, pn_cost_critic_batch_size=7, pn_cost_critic_train_after=5)
    trainer.agent = agent
    trainer.rng = np.random.default_rng(0)
    trainer.total_env_steps = 4
    trainer.centroids_initialized = False
    trainer.grad_step_count = 0

    class _Buffer:
        def __init__(self):
            self.sample_sizes = []

        def sample(self, size, rng):
            self.sample_sizes.append(size)
            return _update_raw_batch(size)

        def sample_achieved_goals(self, size, rng):
            return np.zeros((size, 1), dtype=np.float32)

    trainer.buffer = _Buffer()
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    trainer.ae_opt = torch.optim.SGD([parameter], lr=0.1)
    trainer.landmark_opt = torch.optim.SGD([torch.nn.Parameter(torch.tensor(0.0))], lr=0.1)
    trainer.ae = object()
    trainer.landmarks = object()
    monkeypatch.setattr(trainer_module, "ae_losses", lambda *args: (
        parameter * 0, parameter * 0, parameter * 0,
    ))
    agent.update_value = lambda batch: 0.0
    agent.update_critic = lambda batch: 0.0
    cost_batch_sizes = []
    actor_cost_flags = []
    agent.update_cost_critic = lambda batch: cost_batch_sizes.append(batch["obs"].shape[0]) or 1.0
    agent.update_actor = lambda batch, use_cost_critic=True: actor_cost_flags.append(use_cost_critic) or 1.0
    agent.update_targets = lambda: None

    cold_logs = trainer.update(1)
    assert trainer.buffer.sample_sizes == [trainer.cfg.batch_size]
    assert cost_batch_sizes == []
    assert actor_cost_flags == [False]
    assert cold_logs["cost_critic"] == 0.0

    trainer.total_env_steps = 5
    trainer.buffer.sample_sizes.clear()
    warm_logs = trainer.update(1)
    assert trainer.buffer.sample_sizes == [trainer.cfg.batch_size, 7]
    assert cost_batch_sizes == [7]
    assert actor_cost_flags == [False, True]
    assert warm_logs["cost_critic"] == 1.0


def test_enabled_train_scales_updates_by_early_done_transitions(monkeypatch):
    env = _ScriptedRawEnv([
        {"kind": "gymnasium", "achieved": 1.0, "terminated": True},
    ])
    trainer = _collection_trainer(
        env,
        train_after=0,
        env_steps_per_opt=1,
        eval_interval=100,
        log_interval=100,
    )
    requested_updates = []
    monkeypatch.setattr(trainer, "update", lambda n_steps: requested_updates.append(n_steps) or {})
    monkeypatch.setattr(trainer, "evaluate", lambda *args: 0.0)

    trainer.train(total_steps=2)

    assert trainer.total_env_steps == 2
    assert requested_updates == [1]


def test_disabled_collect_returns_fixed_horizon_transitions():
    env = _ScriptedRawEnv([
        {"kind": "legacy", "achieved": 1.0, "done": True},
        {"kind": "legacy", "achieved": 2.0, "done": True},
        {"kind": "legacy", "achieved": 3.0, "done": True},
    ])
    trainer = _collection_trainer(env, enabled=False)

    collected = trainer.collect()

    assert collected == trainer.env.n * trainer.T
    assert trainer.total_env_steps == collected


@pytest.mark.parametrize("env_steps_per_opt", [0, -1])
def test_train_avoids_nonpositive_optimizer_step_divisors(monkeypatch, env_steps_per_opt):
    env = _ScriptedRawEnv([
        {"kind": "gymnasium", "achieved": 1.0, "terminated": True},
    ])
    trainer = _collection_trainer(
        env,
        train_after=0,
        env_steps_per_opt=env_steps_per_opt,
        eval_interval=100,
        log_interval=100,
    )
    requested_updates = []
    monkeypatch.setattr(trainer, "update", lambda n_steps: requested_updates.append(n_steps) or {})
    monkeypatch.setattr(trainer, "evaluate", lambda *args: 0.0)

    trainer.train(total_steps=2)

    assert requested_updates == [1]
