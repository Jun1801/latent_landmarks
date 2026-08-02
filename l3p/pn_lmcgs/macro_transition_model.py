"""Factorized stochastic transition model over current landmark centroids."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from l3p.pn_lmcgs.macro_replay import Outcome


@dataclass
class MacroLabels:
    outcome: torch.Tensor
    positive_id: torch.Tensor
    negative_id: torch.Tensor
    duration: torch.Tensor


@dataclass
class MacroTransitionOutput:
    outcome_logits: torch.Tensor
    outcome_probs: torch.Tensor
    positive_logits: torch.Tensor
    positive_probs: torch.Tensor
    negative_logits: torch.Tensor
    negative_probs: torch.Tensor
    duration_by_outcome: torch.Tensor


@dataclass
class MacroModelLoss:
    total: torch.Tensor
    outcome: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    duration: torch.Tensor
    positive_count: int
    negative_count: int


class MacroTransitionModel(nn.Module):
    def __init__(self, embedding_dim: int, context_dim: int, hidden_dim: int, k_max: int):
        super().__init__()
        for value, name, minimum in ((embedding_dim, "embedding_dim", 1), (context_dim, "context_dim", 0),
                                     (hidden_dim, "hidden_dim", 1), (k_max, "k_max", 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer at least {minimum}")
        self.embedding_dim, self.context_dim, self.k_max = embedding_dim, context_dim, k_max
        self.backbone = nn.Sequential(
            nn.Linear(embedding_dim * 3 + context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.outcome_head = nn.Linear(hidden_dim, 4)
        self.positive_query_head = nn.Linear(hidden_dim, embedding_dim)
        self.negative_query_head = nn.Linear(hidden_dim, embedding_dim)
        self.duration_head = nn.Linear(hidden_dim, 4)
        self.register_buffer("temperature", torch.tensor(1.0, dtype=torch.float32))

    def set_temperature(self, temperature: float) -> None:
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        with torch.no_grad():
            self.temperature.fill_(float(temperature))

    def forward(self, z_start: torch.Tensor, z_command: torch.Tensor, context: torch.Tensor,
                positive_centroids: torch.Tensor, negative_centroids: torch.Tensor) -> MacroTransitionOutput:
        device = self.temperature.device
        z_start = _batch_embedding(z_start, "z_start", self.embedding_dim, device)
        z_command = _batch_embedding(z_command, "z_command", self.embedding_dim, device)
        context = _batch_embedding(context, "context", self.context_dim, device)
        if z_start.shape[0] != z_command.shape[0] or z_start.shape[0] != context.shape[0]:
            raise ValueError("z_start, z_command, and context must share batch dimension")
        positive = _centroids(positive_centroids, "positive_centroids", self.embedding_dim, device)
        negative = _centroids(negative_centroids, "negative_centroids", self.embedding_dim, device)
        features = torch.cat([z_start.detach(), z_command.detach(), (z_command - z_start).detach(), context.detach()], dim=-1)
        hidden = self.backbone(features)
        outcome_logits = self.outcome_head(hidden)
        positive_logits = _query_logits(self.positive_query_head(hidden), positive)
        negative_logits = _query_logits(self.negative_query_head(hidden), negative)
        return MacroTransitionOutput(
            outcome_logits=outcome_logits,
            outcome_probs=F.softmax(outcome_logits / self.temperature, dim=-1),
            positive_logits=positive_logits,
            positive_probs=_probabilities(positive_logits),
            negative_logits=negative_logits,
            negative_probs=_probabilities(negative_logits),
            duration_by_outcome=1.0 + torch.sigmoid(self.duration_head(hidden)) * (self.k_max - 1.0),
        )


def macro_model_loss(output: MacroTransitionOutput, labels: MacroLabels, beta_duration: float = 0.1,
                     k_max: int = 1) -> MacroModelLoss:
    if not isinstance(output, MacroTransitionOutput) or not isinstance(labels, MacroLabels):
        raise ValueError("output and labels must have the macro model types")
    if not isinstance(k_max, int) or isinstance(k_max, bool) or k_max < 1:
        raise ValueError("k_max must be a positive integer")
    if not isinstance(beta_duration, (float, int)) or not math.isfinite(beta_duration) or beta_duration < 0:
        raise ValueError("beta_duration must be finite and non-negative")
    batch = output.outcome_logits.shape[0]
    if batch == 0:
        raise ValueError("batch size must be positive")
    if output.outcome_logits.shape != (batch, 4) or output.duration_by_outcome.shape != (batch, 4):
        raise ValueError("output has invalid outcome or duration shapes")
    device = output.outcome_logits.device
    outcome = _label_vector(labels.outcome, "outcome", batch, device, torch.long)
    positive_id = _label_vector(labels.positive_id, "positive_id", batch, device, torch.long)
    negative_id = _label_vector(labels.negative_id, "negative_id", batch, device, torch.long)
    duration = _label_vector(labels.duration, "duration", batch, device, torch.float32)
    if torch.any((outcome < 0) | (outcome > 3)):
        raise ValueError("outcome labels must be in [0, 3]")
    if not torch.isfinite(duration).all() or torch.any((duration < 1) | (duration > k_max)):
        raise ValueError("duration labels must be finite and in [1, k_max]")
    outcome_loss = F.cross_entropy(output.outcome_logits, outcome)
    zero = output.outcome_logits.sum() * 0.0
    positive_mask = outcome == int(Outcome.DRIFT)
    # A negative ID of -1 records the pre-activation generic-violation case.
    negative_mask = (outcome == int(Outcome.VIOLATION)) & (negative_id >= 0)
    _validate_candidate_ids(positive_id, positive_mask, output.positive_logits.shape[1], "positive")
    _validate_candidate_ids(negative_id, negative_mask, output.negative_logits.shape[1], "negative")
    positive_loss = _conditional_ce(output.positive_logits, positive_id, positive_mask, "positive")
    negative_loss = _conditional_ce(output.negative_logits, negative_id, negative_mask, "negative")
    selected_duration = output.duration_by_outcome.gather(1, outcome[:, None]).squeeze(1)
    duration_loss = F.smooth_l1_loss(selected_duration / k_max, duration / k_max)
    return MacroModelLoss(outcome_loss + positive_loss + negative_loss + float(beta_duration) * duration_loss,
                          outcome_loss, positive_loss if positive_mask.any() else zero,
                          negative_loss if negative_mask.any() else zero, duration_loss,
                          int(positive_mask.sum()), int(negative_mask.sum()))


def _batch_embedding(value, name: str, width: int, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device != device or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}] on the model device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value.to(dtype=torch.float32)


def _centroids(value, name: str, width: int, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device != device or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [N, {width}] on the model device")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value.detach().to(dtype=torch.float32)


def _query_logits(query: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if centroids.shape[0] == 0:
        return query.new_empty((query.shape[0], 0))
    return query @ centroids.T / math.sqrt(query.shape[1])


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    return logits if logits.shape[1] == 0 else F.softmax(logits, dim=-1)


def _label_vector(value, name: str, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device != device or value.ndim != 1 or value.shape[0] != batch:
        raise ValueError(f"{name} labels must be a [B] tensor on the model device")
    if dtype == torch.long and (value.dtype.is_floating_point or value.dtype == torch.bool):
        raise ValueError(f"{name} labels must be integer tensors")
    return value.to(dtype=dtype)


def _conditional_ce(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, name: str) -> torch.Tensor:
    if not mask.any():
        return logits.sum() * 0.0
    if logits.ndim != 2 or logits.shape[0] != len(ids) or logits.shape[1] == 0:
        raise ValueError(f"{name} logits must be available for active {name} labels")
    selected = ids[mask]
    if torch.any((selected < 0) | (selected >= logits.shape[1])):
        raise ValueError(f"active {name} labels must be valid centroid IDs")
    return F.cross_entropy(logits[mask], selected)


def _validate_candidate_ids(ids: torch.Tensor, active: torch.Tensor, available: int, name: str) -> None:
    if torch.any(ids < -1):
        raise ValueError(f"{name} labels must be -1 or valid centroid IDs")
    if torch.any(ids[~active] != -1):
        raise ValueError(f"inactive {name} labels must be -1")
    if active.any() and (available == 0 or torch.any((ids[active] < 0) | (ids[active] >= available))):
        raise ValueError(f"active {name} labels must be valid centroid IDs")
