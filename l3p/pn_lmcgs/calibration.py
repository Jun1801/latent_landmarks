"""Calibration and diagnostics for macro outcome predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from l3p.pn_lmcgs.macro_replay import Outcome


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    before_nll: float
    after_nll: float


def fit_temperature(outcome_logits, outcome_labels) -> TemperatureFit:
    logits, labels = _logits_and_labels(outcome_logits, outcome_labels)
    # An empty validation partition provides no calibration evidence.
    if logits.shape[0] == 0:
        return TemperatureFit(1.0, 0.0, 0.0)
    grid = torch.linspace(0.5, 5.0, 91, device=logits.device)
    nlls = torch.stack([F.cross_entropy(logits / temperature, labels) for temperature in grid])
    base = F.cross_entropy(logits, labels)
    best_index = int(torch.argmin(nlls))
    best_nll = nlls[best_index]
    if best_nll > base:
        return TemperatureFit(1.0, float(base), float(base))
    return TemperatureFit(float(grid[best_index]), float(base), float(best_nll))


def expected_calibration_error(outcome_logits, outcome_labels, temperature: float = 1.0, n_bins: int = 10) -> float:
    logits, labels = _logits_and_labels(outcome_logits, outcome_labels)
    _positive_scalar(temperature, "temperature")
    n_bins = _positive_int(n_bins, "n_bins")
    if logits.shape[0] == 0:
        return 0.0
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    accuracy = prediction.eq(labels).float()
    result = torch.zeros((), device=logits.device)
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        mask = (confidence >= low) & ((confidence < high) if index + 1 < n_bins else (confidence <= high))
        if mask.any():
            result += mask.float().mean() * (confidence[mask].mean() - accuracy[mask].mean()).abs()
    return float(result)


def violation_reliability_bins(outcome_logits, outcome_labels, n_bins: int = 10, temperature: float = 1.0) -> dict[str, np.ndarray]:
    """Return violation calibration bins; empty bins use predicted=empirical=count=0."""
    logits, labels = _logits_and_labels(outcome_logits, outcome_labels)
    n_bins = _positive_int(n_bins, "n_bins")
    _positive_scalar(temperature, "temperature")
    probability = torch.softmax(logits / temperature, dim=-1)[:, int(Outcome.VIOLATION)]
    actual = labels.eq(int(Outcome.VIOLATION)).float()
    count = np.zeros(n_bins, dtype=np.int64)
    predicted = np.zeros(n_bins, dtype=np.float64)
    empirical = np.zeros(n_bins, dtype=np.float64)
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        mask = (probability >= low) & ((probability < high) if index + 1 < n_bins else (probability <= high))
        count[index] = int(mask.sum())
        if count[index]:
            predicted[index] = float(probability[mask].mean())
            empirical[index] = float(actual[mask].mean())
    return {"count": count, "predicted": predicted, "empirical": empirical,
            "lower": np.arange(n_bins, dtype=np.float64) / n_bins,
            "upper": np.arange(1, n_bins + 1, dtype=np.float64) / n_bins}


def confusion_matrix_and_classification_metrics(prediction, target) -> dict[str, np.ndarray]:
    pred = _outcome_vector(prediction, "prediction")
    target = _outcome_vector(target, "target")
    if len(pred) != len(target):
        raise ValueError("prediction and target must have equal length")
    matrix = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(target.cpu().numpy(), pred.cpu().numpy()):
        matrix[actual, predicted] += 1
    diagonal = np.diag(matrix).astype(np.float64)
    precision = np.divide(diagonal, matrix.sum(axis=0), out=np.zeros(4), where=matrix.sum(axis=0) != 0)
    recall = np.divide(diagonal, matrix.sum(axis=1), out=np.zeros(4), where=matrix.sum(axis=1) != 0)
    return {"matrix": matrix, "precision": precision, "recall": recall}


def duration_mae_by_outcome(outcomes, predicted_duration, true_duration) -> np.ndarray:
    """Return outcome MAEs; outcomes without observations use the neutral value 0."""
    labels = _outcome_vector(outcomes, "outcomes")
    predicted = _duration_vector(predicted_duration, "predicted_duration", len(labels))
    actual = _duration_vector(true_duration, "true_duration", len(labels))
    result = np.zeros(4, dtype=np.float64)
    for outcome in range(4):
        mask = labels == outcome
        if mask.any():
            result[outcome] = float((predicted[mask] - actual[mask]).abs().mean())
    return result


def _logits_and_labels(logits, labels) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("outcome_logits must have shape [B, 4]")
    if not torch.isfinite(logits).all():
        raise ValueError("outcome_logits must contain finite values")
    values = _outcome_vector(labels, "outcome_labels", device=logits.device)
    if len(values) != logits.shape[0]:
        raise ValueError("outcome_labels must have length B")
    return logits.detach().to(dtype=torch.float32), values


def _outcome_vector(value, name: str, device=None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be an integer vector")
    if tensor.dtype.is_floating_point and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain finite values")
    if tensor.dtype == torch.bool or tensor.dtype.is_floating_point:
        raise ValueError(f"{name} must be an integer vector")
    tensor = tensor.to(dtype=torch.long)
    if torch.any((tensor < 0) | (tensor > 3)):
        raise ValueError(f"{name} values must be in [0, 3]")
    return tensor


def _duration_vector(value, name: str, length: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1 or len(tensor) != length or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite vector matching outcomes")
    return tensor


def _positive_scalar(value, name: str) -> None:
    if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


outcome_confusion_metrics = confusion_matrix_and_classification_metrics
