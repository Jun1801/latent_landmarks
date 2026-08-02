"""Goal-conditioned DDPG with a distance-parameterized critic (paper Section 3-4.1).

The agent owns:
  * an actor  pi(s, g)
  * a distance-parameterized critic D(s, a, g)  ->  Q via Eq. 3
  * target copies of both (Polyak-averaged)
  * a goal-to-goal value function V(g1, g2)  (Eq. 4)
  * running normalizers for observations and goals

Losses implemented here:
  * critic TD loss (Eq. 1)          -> update_critic
  * actor loss (maximize Q + L2)    -> update_actor
  * value regression (Eq. 4)        -> update_value

The critic / actor consume *normalized* obs and goals. V, the auto-encoder and
the landmarks all operate in *raw* goal space; the planner normalizes decoded
landmark goals before querying the critic, so representations stay consistent.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from l3p.agent.normalizer import Normalizer
from l3p.models.networks import Actor, Critic, ValueFunction
from l3p.pn_lmcgs.safety_critic import (
    ViolationCritic,
    violation_binary_cross_entropy,
    violation_td_target,
)


class DDPGAgent:
    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int, max_action: float, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.obs_dim, self.goal_dim, self.act_dim = obs_dim, goal_dim, act_dim
        self.max_action = max_action
        self.gamma = cfg.gamma

        self.actor = Actor(obs_dim, goal_dim, act_dim, cfg.hidden_units,
                           cfg.hidden_layers, max_action).to(self.device)
        self.critic = Critic(obs_dim, goal_dim, act_dim, cfg.hidden_units,
                             cfg.hidden_layers).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.value = ValueFunction(goal_dim, cfg.hidden_units, cfg.hidden_layers).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.critic_lr)
        self.pn_lmcgs_enabled = bool(getattr(cfg, "pn_lmcgs_enabled", False))
        if self.pn_lmcgs_enabled:
            self.cost_critic = ViolationCritic(
                obs_dim, goal_dim, act_dim, cfg.hidden_units, cfg.hidden_layers,
            ).to(self.device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            for p in self.cost_critic_target.parameters():
                p.requires_grad_(False)
            cost_critic_lr = getattr(cfg, "pn_cost_critic_lr", None)
            self.cost_critic_opt = torch.optim.Adam(
                self.cost_critic.parameters(),
                lr=cfg.critic_lr if cost_critic_lr is None else cost_critic_lr,
            )

        self.o_norm = Normalizer(obs_dim)
        self.g_norm = Normalizer(goal_dim)
        # V normalizes its goal inputs with the running goal statistics.
        self.value.normalizer = self.g_norm

    # ------------------------------------------------------------------ utils
    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _no(self, obs: np.ndarray) -> torch.Tensor:
        return self._t(self.o_norm.normalize(obs))

    def _ng(self, goal: np.ndarray) -> torch.Tensor:
        return self._t(self.g_norm.normalize(goal))

    def update_normalizers(self, obs: np.ndarray, goal: np.ndarray) -> None:
        self.o_norm.update(obs)
        self.g_norm.update(goal)

    # public tensor helpers (used by the trainer to build batches)
    def to_tensor(self, x) -> torch.Tensor:
        return self._t(x)

    def norm_obs(self, obs: np.ndarray) -> torch.Tensor:
        return self._no(obs)

    def norm_goal(self, goal: np.ndarray) -> torch.Tensor:
        return self._ng(goal)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray, noise_scale: float = 0.0,
            random_prob: float = 0.0) -> np.ndarray:
        """Return an action for a (batch of) obs/goal in raw space."""
        single = obs.ndim == 1
        obs = np.atleast_2d(obs)
        goal = np.atleast_2d(goal)
        a = self.actor(self._no(obs), self._ng(goal)).cpu().numpy()
        if noise_scale > 0:
            a = a + noise_scale * self.max_action * np.random.randn(*a.shape)
        a = np.clip(a, -self.max_action, self.max_action)
        if random_prob > 0:
            rand = np.random.uniform(-self.max_action, self.max_action, size=a.shape)
            mask = (np.random.rand(a.shape[0], 1) < random_prob)
            a = np.where(mask, rand, a)
        return a[0] if single else a

    @torch.no_grad()
    def distance_after_action(self, obs: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Eq. 7 building block:  D(s, pi(s, g), g)  for each row (obs, g).

        `obs` may be a single obs broadcast against many goals, or matched rows.
        Returns a 1-D array of distances.
        """
        obs = np.atleast_2d(obs)
        goals = np.atleast_2d(goals)
        if obs.shape[0] == 1 and goals.shape[0] > 1:
            obs = np.repeat(obs, goals.shape[0], axis=0)
        no, ng = self._no(obs), self._ng(goals)
        a = self.actor(no, ng)
        d = self.critic.distance(no, a, ng)
        return d.cpu().numpy()

    @torch.no_grad()
    def cost_after_action(self, obs: np.ndarray, goals: np.ndarray) -> np.ndarray:
        """Return ``C(s, pi(s, g), g)`` for raw observation/goal rows."""
        if not self.pn_lmcgs_enabled:
            raise RuntimeError("cost critic is unavailable when pn_lmcgs_enabled=False")
        obs = np.atleast_2d(obs)
        goals = np.atleast_2d(goals)
        if obs.shape[0] == 1 and goals.shape[0] > 1:
            obs = np.repeat(obs, goals.shape[0], axis=0)
        no, ng = self._no(obs), self._ng(goals)
        a = self.actor(no, ng)
        return self.cost_critic(no, a, ng).cpu().numpy()

    # ------------------------------------------------------------------ learning
    def update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Critic TD loss, Eq. 1 (with the distance parameterization of Eq. 3)."""
        obs, act, next_obs = batch["obs"], batch["act"], batch["next_obs"]
        g, reward = batch["g"], batch["reward"]

        with torch.no_grad():
            next_a = self.actor_target(next_obs, g)
            q_next = self.critic_target(next_obs, next_a, g, self.gamma)
            target_q = reward + self.gamma * q_next
            # Q is bounded in [-1/(1-gamma), 0]; clamp the target to that range.
            target_q = torch.clamp(target_q, -1.0 / (1.0 - self.gamma), 0.0)

        q = self.critic(obs, act, g, self.gamma)
        loss = ((q - target_q) ** 2).mean()

        self.critic_opt.zero_grad()
        loss.backward()
        if self.cfg.grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_norm_clip)
        self.critic_opt.step()
        return float(loss.item())

    def update_actor(self, batch: Dict[str, torch.Tensor], use_cost_critic: bool = True) -> float:
        """Actor loss: maximize Q(s, pi(s,g), g) with an L2 penalty on actions."""
        obs, g = batch["obs"], batch["g"]
        a = self.actor(obs, g)
        q = self.critic(obs, a, g, self.gamma)
        loss = -q.mean() + self.cfg.action_l2 * (a / self.max_action).pow(2).mean()
        if self.pn_lmcgs_enabled and use_cost_critic:
            self.cost_critic_opt.zero_grad(set_to_none=True)
            with self._freeze_cost_critic():
                risk = self.cost_critic(obs, a, g)
            loss = loss + self.cfg.pn_cost_critic_weight * risk.mean()

        self.actor_opt.zero_grad()
        loss.backward()
        if self.cfg.grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_norm_clip)
        self.actor_opt.step()
        return float(loss.item())

    @contextmanager
    def _freeze_cost_critic(self):
        """Keep actor-to-action gradients while avoiding cost-critic gradients."""
        parameters = tuple(self.cost_critic.parameters())
        requires_grad = [p.requires_grad for p in parameters]
        for p in parameters:
            p.requires_grad_(False)
            p.grad = None
        try:
            yield
        finally:
            for p, enabled in zip(parameters, requires_grad):
                p.requires_grad_(enabled)
                p.grad = None

    def update_cost_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Train ``C`` against the detached safety-violation TD target."""
        if not self.pn_lmcgs_enabled:
            raise RuntimeError("cost critic is unavailable when pn_lmcgs_enabled=False")
        obs, act, next_obs, g = batch["obs"], batch["act"], batch["next_obs"], batch["g"]
        with torch.no_grad():
            next_act = self.actor_target(next_obs, g)
            next_probability = self.cost_critic_target(next_obs, next_act, g)
            target = violation_td_target(batch["cost"], batch["stop"], next_probability).detach()

        prediction = self.cost_critic(obs, act, g)
        loss = violation_binary_cross_entropy(prediction, target)
        self.cost_critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.cfg.grad_norm_clip)
        self.cost_critic_opt.step()
        self.cost_critic_opt.zero_grad(set_to_none=True)
        return float(loss.item())

    def update_value(self, batch: Dict[str, torch.Tensor]) -> float:
        """Value regression toward the distance function, Eq. 4.

        target = D(s_t, a_t, Psi(s_k))     (the online distance D, detached)
        pred   = V(Psi(s_{t+1}), Psi(s_k))

        Eq. 4 regresses V toward the distance function D itself (Eq. 3), so we
        use the online critic's distance (detached) as the target — not the
        target network (V is a separate head, no bootstrapping self-reference).
        The critic consumes normalized obs/goals (`future_ag_norm`), while V
        operates in raw goal space (`next_ag`, `future_ag`).
        """
        obs, act = batch["obs"], batch["act"]
        next_ag, future_ag = batch["next_ag"], batch["future_ag"]
        with torch.no_grad():
            target = self.critic.distance(obs, act, batch["future_ag_norm"])
        pred = self.value(next_ag, future_ag)
        loss = ((pred - target) ** 2).mean()

        self.value_opt.zero_grad()
        loss.backward()
        if self.cfg.grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.grad_norm_clip)
        self.value_opt.step()
        return float(loss.item())

    def update_targets(self) -> None:
        tau = self.cfg.polyak
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.mul_(tau).add_((1 - tau) * p)
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(tau).add_((1 - tau) * p)
            if self.pn_lmcgs_enabled:
                for p, tp in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
                    tp.mul_(tau).add_((1 - tau) * p)

    # ------------------------------------------------------------------ (de)serialize
    def state_dict(self) -> dict:
        state = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                     value=self.value.state_dict(),
                     o_norm=self.o_norm.state_dict(), g_norm=self.g_norm.state_dict())
        if self.pn_lmcgs_enabled:
            state["cost_critic"] = self.cost_critic.state_dict()
        return state

    def load_state_dict(self, d: dict) -> None:
        self.actor.load_state_dict(d["actor"])
        self.critic.load_state_dict(d["critic"])
        self.value.load_state_dict(d["value"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        if self.pn_lmcgs_enabled:
            if "cost_critic" in d:
                self.cost_critic.load_state_dict(d["cost_critic"])
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            for p in self.cost_critic_target.parameters():
                p.requires_grad_(False)
        self.o_norm.load_state_dict(d["o_norm"])
        self.g_norm.load_state_dict(d["g_norm"])
