import copy

import numpy as np
import pytest
import torch

from l3p.pn_lmcgs.calibration import (
    confusion_matrix_and_classification_metrics,
    duration_mae_by_outcome,
    expected_calibration_error,
    fit_temperature,
    violation_reliability_bins,
)
from l3p.pn_lmcgs.macro_replay import (
    MacroAttempt,
    MacroAttemptBuffer,
    Outcome,
    label_macro_attempts,
    stratified_macro_indices,
)
from l3p.pn_lmcgs.macro_transition_model import MacroLabels, MacroTransitionModel, macro_model_loss


def attempt(value=0.0, *, target=False, violation=False, violation_goal=None, episode=1):
    return MacroAttempt(
        start_goal=np.array([value, 0.0], dtype=np.float32),
        command_goal=np.array([value + 1.0, 0.0], dtype=np.float32),
        end_goal=np.array([value + 2.0, 0.0], dtype=np.float32),
        violation_goal=violation_goal,
        duration=2,
        context=np.array([value], dtype=np.float32),
        target_reached=target,
        violation=violation,
        episode_id=episode,
        start_t=3,
        end_t=4,
    )


def test_attempt_validation_and_buffer_fifo_copy_split_and_atomic_load():
    with pytest.raises(ValueError, match="finite"):
        attempt(np.nan)
    with pytest.raises(ValueError, match="duration"):
        MacroAttempt(np.zeros(2), np.zeros(2), np.zeros(2), None, 0, np.zeros(1), False, False, 1, 0, 0)
    with pytest.raises(ValueError, match="end_t"):
        MacroAttempt(np.zeros(2), np.zeros(2), np.zeros(2), None, 1, np.zeros(1), False, False, 1, 2, 1)

    first, second, third = attempt(1), attempt(2), attempt(3)
    replay = MacroAttemptBuffer(capacity=2, validation_fraction=0.5, split_seed=5)
    replay.extend([first, second, third])
    first.start_goal[:] = 99
    assert len(replay) == replay.size == 2
    assert replay.attempts[0].start_goal[0] == 2
    leaked = replay.attempts
    leaked[0].start_goal[:] = 88
    assert replay.attempts[0].start_goal[0] == 2
    train, validation = replay.train_indices, replay.validation_indices
    assert set(train).isdisjoint(validation)
    assert sorted(np.concatenate([train, validation]).tolist()) == [0, 1]
    np.testing.assert_array_equal(replay.sample_train(4, np.random.default_rng(3))["start_goal"][:, 0], [2, 2, 2, 2])
    np.testing.assert_array_equal(replay.sample_validation(4, np.random.default_rng(3))["start_goal"][:, 0], [3, 3, 3, 3])

    clone = MacroAttemptBuffer(capacity=2, validation_fraction=0.5, split_seed=5)
    state = replay.state_dict()
    clone.load_state_dict(state)
    np.testing.assert_array_equal(clone.state_dict()["attempts"][0]["start_goal"], state["attempts"][0]["start_goal"])
    before = clone.state_dict()
    bad = copy.deepcopy(state)
    bad["capacity"] = 3
    with pytest.raises(ValueError, match="capacity"):
        clone.load_state_dict(bad)
    np.testing.assert_array_equal(clone.state_dict()["attempts"][0]["start_goal"], before["attempts"][0]["start_goal"])


def test_stratification_is_exact_sized_and_handles_absent_classes():
    labels = np.array([Outcome.TARGET, Outcome.TARGET, Outcome.STUCK])
    indices = stratified_macro_indices(labels, 12, np.random.default_rng(4))
    assert indices.shape == (12,)
    assert np.all((indices >= 0) & (indices < len(labels)))
    assert set(labels[indices]) <= {Outcome.TARGET, Outcome.STUCK}
    with pytest.raises(ValueError, match="batch"):
        stratified_macro_indices(labels, 0, np.random.default_rng(0))


def test_current_centroid_labels_prioritize_violation_and_change_when_centroids_move():
    attempts = [
        attempt(0, target=True, violation=True, violation_goal=np.array([9.0, 0.0], dtype=np.float32)),
        attempt(1, target=True),
        attempt(2),
        attempt(20),
    ]
    encoder = lambda x: x
    labels = label_macro_attempts(
        attempts, encoder, torch.tensor([[4.0, 0.0]]), torch.tensor([[9.0, 0.0]]),
        assignment_radius=0.1, negative_active=True,
    )
    assert labels.outcome.tolist() == [Outcome.VIOLATION, Outcome.TARGET, Outcome.DRIFT, Outcome.STUCK]
    assert labels.positive_id.tolist() == [-1, -1, 0, -1]
    assert labels.negative_id.tolist() == [0, -1, -1, -1]
    moved = label_macro_attempts(attempts[2:3], encoder, torch.tensor([[40.0, 0.0]]), None, 0.1, False)
    assert moved.outcome.item() == Outcome.STUCK
    zero_positive = label_macro_attempts(attempts[2:3], encoder, torch.empty(0, 2), None, 1.0, False)
    assert zero_positive.outcome.item() == Outcome.STUCK
    with pytest.raises(ValueError, match="positive_centroids"):
        label_macro_attempts(attempts[2:3], encoder, torch.empty(0, 1), None, 1.0, False)


def test_model_factorization_shapes_gradients_moving_centroids_and_duration_bounds():
    torch.manual_seed(2)
    model = MacroTransitionModel(embedding_dim=2, context_dim=1, hidden_dim=7, k_max=5)
    z_start = torch.randn(3, 2, requires_grad=True)
    z_command = torch.randn(3, 2, requires_grad=True)
    context = torch.randn(3, 1, requires_grad=True)
    positive = torch.randn(3, 2, requires_grad=True)
    negative = torch.randn(2, 2, requires_grad=True)
    output = model(z_start, z_command, context, positive, negative)
    assert output.outcome_logits.shape == (3, 4)
    assert output.positive_logits.shape == (3, 3)
    assert output.negative_logits.shape == (3, 2)
    assert torch.allclose(output.outcome_probs.sum(-1), torch.ones(3))
    assert torch.allclose(output.positive_probs.sum(-1), torch.ones(3))
    assert torch.allclose(output.negative_probs.sum(-1), torch.ones(3))
    assert torch.all((output.duration_by_outcome >= 1) & (output.duration_by_outcome <= 5))
    output.outcome_logits.sum().backward()
    assert z_start.grad is None and z_command.grad is None and context.grad is None
    assert positive.grad is None and negative.grad is None
    assert model.outcome_head.weight.grad is not None

    heads = tuple(id(head) for head in (model.outcome_head, model.positive_query_head, model.negative_query_head, model.duration_head))
    output = model(torch.zeros(2, 2), torch.ones(2, 2), torch.zeros(2, 1), torch.empty(0, 2), torch.empty(0, 2))
    assert output.positive_probs.shape == output.negative_probs.shape == (2, 0)
    assert heads == tuple(id(head) for head in (model.outcome_head, model.positive_query_head, model.negative_query_head, model.duration_head))
    model.set_temperature(2.0)
    assert model.temperature.item() == 2.0
    with pytest.raises(ValueError, match="temperature"):
        model.set_temperature(0)


def test_loss_uses_conditional_terms_and_rejects_invalid_active_ids():
    model = MacroTransitionModel(2, 0, 4, 4)
    output = model(torch.zeros(3, 2), torch.ones(3, 2), torch.empty(3, 0), torch.randn(1, 2), torch.randn(1, 2))
    labels = MacroLabels(
        outcome=torch.tensor([Outcome.TARGET, Outcome.DRIFT, Outcome.VIOLATION]),
        positive_id=torch.tensor([-1, 0, -1]),
        negative_id=torch.tensor([-1, -1, 0]),
        duration=torch.tensor([1.0, 2.0, 4.0]),
    )
    loss = macro_model_loss(output, labels, beta_duration=0.1, k_max=4)
    assert loss.positive_count == loss.negative_count == 1
    assert loss.total.requires_grad
    bad = MacroLabels(labels.outcome, torch.tensor([-1, -1, -1]), labels.negative_id, labels.duration)
    with pytest.raises(ValueError, match="positive"):
        macro_model_loss(output, bad, k_max=4)
    invalid_duration = MacroLabels(labels.outcome, labels.positive_id, labels.negative_id, torch.tensor([0.0, 2.0, 4.0]))
    with pytest.raises(ValueError, match="duration"):
        macro_model_loss(output, invalid_duration, k_max=4)
    inactive_output = model(torch.zeros(1, 2), torch.ones(1, 2), torch.empty(1, 0), torch.empty(0, 2), torch.empty(0, 2))
    inactive_labels = MacroLabels(torch.tensor([Outcome.VIOLATION]), torch.tensor([-1]), torch.tensor([-1]), torch.tensor([2.0]))
    inactive_loss = macro_model_loss(inactive_output, inactive_labels, k_max=4)
    assert inactive_loss.negative_count == 0


def test_calibration_and_diagnostics_are_finite_and_handle_missing_classes():
    logits = torch.tensor([[8.0, 0, 0, 0], [7.0, 0, 0, 0], [0, 5.0, 0, 0]], dtype=torch.float32)
    labels = torch.tensor([1, 1, 1])
    result = fit_temperature(logits, labels)
    assert 0.5 <= result.temperature <= 5.0
    assert result.after_nll <= result.before_nll + 1e-7
    ece = expected_calibration_error(logits, labels, temperature=result.temperature, n_bins=3)
    assert np.isfinite(ece)
    reliability = violation_reliability_bins(logits, labels, n_bins=4)
    assert reliability["count"].sum() == 3
    metrics = confusion_matrix_and_classification_metrics(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]))
    assert metrics["matrix"].shape == (4, 4)
    assert np.isfinite(metrics["precision"]).all()
    mae = duration_mae_by_outcome(torch.tensor([0, 0, 1]), torch.tensor([1.0, 3.0, 4.0]), torch.tensor([2.0, 2.0, 1.0]))
    assert mae.shape == (4,)
    assert mae[0] == pytest.approx(1.0)
