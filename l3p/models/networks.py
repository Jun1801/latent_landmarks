"""Neural network modules for L3P.

Contains the goal-conditioned Actor, the distance-parameterized Critic, and the
goal-to-goal Value function V(g1, g2).

Distance parameterization (paper Eq. 3):

    Q(s, a, g) = -(1 - gamma^D) / (1 - gamma),   with D(s, a, g) >= 0

where D is the (positive) number-of-steps estimate produced by the critic.
Parameterizing Q through D disentangles gamma from multi-goal Q-learning and
gives us a direct estimate of reachability distance for planning.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def mlp(sizes: List[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    """Build a simple fully-connected network from a list of layer sizes."""
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic goal-conditioned policy pi(s, g) -> a in [-max_action, max_action]."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 hidden_units: int = 256, hidden_layers: int = 3,
                 max_action: float = 1.0):
        super().__init__()
        sizes = [obs_dim + goal_dim] + [hidden_units] * hidden_layers + [act_dim]
        self.net = mlp(sizes, activation=nn.ReLU, output_activation=nn.Tanh)
        self.max_action = max_action

    def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, goal], dim=-1)
        return self.max_action * self.net(x)


class Critic(nn.Module):
    """Distance-parameterized critic.

    The network maps (s, a, g) to a raw scalar; softplus makes the distance
    D(s, a, g) >= 0. Q is then recovered analytically via Eq. 3.
    """

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 hidden_units: int = 256, hidden_layers: int = 3):
        super().__init__()
        sizes = [obs_dim + goal_dim + act_dim] + [hidden_units] * hidden_layers + [1]
        self.net = mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity)

    def distance(self, obs: torch.Tensor, act: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Return D(s, a, g) >= 0 (expected number of steps to reach g)."""
        x = torch.cat([obs, act, goal], dim=-1)
        return nn.functional.softplus(self.net(x)).squeeze(-1)

    @staticmethod
    def q_from_distance(distance: torch.Tensor, gamma: float) -> torch.Tensor:
        """Eq. 3:  Q = -(1 - gamma^D) / (1 - gamma)."""
        return -(1.0 - torch.pow(torch.as_tensor(gamma, dtype=distance.dtype,
                                                  device=distance.device), distance)) / (1.0 - gamma)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, goal: torch.Tensor,
                gamma: float) -> torch.Tensor:
        return self.q_from_distance(self.distance(obs, act, goal), gamma)


class ValueFunction(nn.Module):
    """Goal-to-goal value V(g1, g2): estimated number of steps for the policy to
    go from goal g1 to goal g2 (paper Eq. 4). Output is made non-negative with
    softplus, mirroring the distance semantics of the critic."""

    def __init__(self, goal_dim: int, hidden_units: int = 256, hidden_layers: int = 3):
        super().__init__()
        sizes = [2 * goal_dim] + [hidden_units] * hidden_layers + [1]
        self.net = mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity)
        # Optional goal normalizer: normalizing inputs stabilizes V on envs with
        # large-magnitude goal coordinates (e.g. AntMaze). Set externally.
        self.normalizer = None

    def forward(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            g1 = self.normalizer.normalize_torch(g1)
            g2 = self.normalizer.normalize_torch(g2)
        x = torch.cat([g1, g2], dim=-1)
        return nn.functional.softplus(self.net(x)).squeeze(-1)
