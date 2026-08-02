"""Goal-conditioned violation critic and safe actor regression tests."""

import numpy as np
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
