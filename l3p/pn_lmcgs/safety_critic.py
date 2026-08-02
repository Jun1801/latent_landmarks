"""Goal-conditioned probability critic for primitive safety violations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from l3p.models.networks import mlp


class ViolationCritic(nn.Module):
    """Estimate the probability of a future violation from ``(s, a, g)``."""

    def __init__(self, obs_dim: int, goal_dim: int, act_dim: int,
                 hidden_units: int = 256, hidden_layers: int = 3):
        super().__init__()
        sizes = [obs_dim + act_dim + goal_dim] + [hidden_units] * hidden_layers + [1]
        self.net = mlp(sizes, activation=nn.ReLU, output_activation=nn.Sigmoid)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act, goal], dim=-1)).squeeze(-1)


def _probability_vector(value, name: str, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    elif tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    elif tensor.ndim != 1:
        raise ValueError(f"{name} must be scalar, [B], or [B, 1], got {tuple(tensor.shape)}")
    tensor = tensor.to(dtype=torch.float32)
    if not torch.isfinite(tensor).all() or torch.any((tensor < 0) | (tensor > 1)):
        raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
    return tensor


def violation_td_target(cost, stop, next_probability) -> torch.Tensor:
    """Return ``cost + (1-cost) * (1-stop) * next_probability`` in ``[0, 1]``."""
    next_probability = _probability_vector(next_probability, "next_probability",
                                            device=torch.as_tensor(next_probability).device)
    cost = _probability_vector(cost, "cost", device=next_probability.device)
    stop = _probability_vector(stop, "stop", device=next_probability.device)
    try:
        cost, stop, next_probability = torch.broadcast_tensors(cost, stop, next_probability)
    except RuntimeError as exc:
        raise ValueError("cost, stop, and next_probability must have broadcast-compatible shapes") from exc
    return cost + (1.0 - cost) * (1.0 - stop) * next_probability


def violation_binary_cross_entropy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Numerically safe BCE for probability critic predictions and bounded targets."""
    prediction = _probability_vector(prediction, "prediction", device=prediction.device)
    target = _probability_vector(target, "target", device=prediction.device)
    try:
        prediction, target = torch.broadcast_tensors(prediction, target)
    except RuntimeError as exc:
        raise ValueError("prediction and target must have broadcast-compatible shapes") from exc
    eps = torch.finfo(prediction.dtype).eps
    return F.binary_cross_entropy(prediction.clamp(eps, 1.0 - eps), target)
