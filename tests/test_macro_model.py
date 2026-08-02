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
from l3p.pn_lmcgs.macro_transition_model import (
    MacroLabels,
    MacroTransitionModel,
    MacroTransitionOutput,
    macro_model_loss,
)


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
    np.testing.assert_array_equal(replay.sample_train(4, np.random.default_rng(3))["start_goal"][:, 0], [3, 3, 3, 3])
    np.testing.assert_array_equal(replay.sample_validation(4, np.random.default_rng(3))["start_goal"][:, 0], [2, 2, 2, 2])

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
    nested_bad = copy.deepcopy(state)
    nested_bad["attempts"][0]["duration"] = 0
    with pytest.raises(ValueError, match="duration"):
        clone.load_state_dict(nested_bad)
    np.testing.assert_array_equal(clone.state_dict()["attempts"][0]["start_goal"], before["attempts"][0]["start_goal"])


def test_replay_split_membership_survives_fifo_eviction_and_state_roundtrip():
    replay = MacroAttemptBuffer(capacity=3, validation_fraction=0.5, split_seed=11)
    memberships = {}
    for value in range(6):
        replay.add(attempt(value))
        state = replay.state_dict()
        current = {
            int(item["start_goal"][0]): is_validation
            for item, is_validation in zip(state["attempts"], state["validation_membership"])
        }
        for identity, is_validation in current.items():
            if identity in memberships:
                assert is_validation == memberships[identity]
        memberships.update(current)
        expected_validation_size = min(
            max(1, int(np.floor(len(replay) * 0.5))) if len(replay) > 1 else 0,
            len(replay) - 1,
        )
        assert sum(state["validation_membership"]) == expected_validation_size

    restored = MacroAttemptBuffer(capacity=3, validation_fraction=0.5, split_seed=11)
    restored.load_state_dict(replay.state_dict())
    restored_state, replay_state = restored.state_dict(), replay.state_dict()
    assert restored_state["validation_membership"] == replay_state["validation_membership"]
    for restored_item, replay_item in zip(restored_state["attempts"], replay_state["attempts"]):
        np.testing.assert_array_equal(restored_item["start_goal"], replay_item["start_goal"])
    for sampler in ("sample_train", "sample_validation"):
        np.testing.assert_array_equal(
            getattr(restored, sampler)(8, np.random.default_rng(7))["start_goal"],
            getattr(replay, sampler)(8, np.random.default_rng(7))["start_goal"],
        )


def test_replay_sampling_uses_cached_partition_slots_and_load_is_atomic():
    replay = MacroAttemptBuffer(capacity=4, validation_fraction=0.5, split_seed=3)
    replay.extend([attempt(value) for value in range(4)])
    assert len(replay.train_indices) == len(replay.validation_indices) == 2

    class SamplingRng(np.random.Generator):
        def __init__(self):
            super().__init__(np.random.PCG64(1))

        def integers(self, high, size):
            return np.zeros(size, dtype=np.int64)

        def permutation(self, *_args, **_kwargs):
            raise AssertionError("sampling must not rebuild a split permutation")

    replay.sample_train(3, SamplingRng())
    replay.sample_validation(3, SamplingRng())

    class NoMaterializeSlots(list):
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("sampling must not materialize the whole partition")

    replay._train_slots = NoMaterializeSlots(replay._train_slots[:])
    replay.sample_train(3, SamplingRng())
    before = replay.state_dict()
    malformed = copy.deepcopy(before)
    malformed["validation_membership"][0] = "not-a-boolean"
    with pytest.raises(ValueError, match="membership"):
        replay.load_state_dict(malformed)
    np.testing.assert_array_equal(replay.state_dict()["attempts"][0]["start_goal"], before["attempts"][0]["start_goal"])
    assert replay.state_dict()["validation_membership"] == before["validation_membership"]
    malformed_cache = copy.deepcopy(before)
    malformed_cache["partition_order"]["train"].append(0)
    with pytest.raises(ValueError, match="partition order"):
        replay.load_state_dict(malformed_cache)
    assert replay.state_dict()["validation_membership"] == before["validation_membership"]

    replay.add(attempt(4))
    state = replay.state_dict()
    assert len(replay.train_indices) == len(replay.validation_indices) == 2
    assert sum(state["validation_membership"]) == 2


def test_stratification_is_exact_sized_and_handles_absent_classes():
    labels = np.array([Outcome.TARGET, Outcome.TARGET, Outcome.STUCK])
    indices = stratified_macro_indices(labels, 12, np.random.default_rng(4))
    assert indices.shape == (12,)
    assert np.all((indices >= 0) & (indices < len(labels)))
    assert set(labels[indices]) <= {Outcome.TARGET, Outcome.STUCK}
    with pytest.raises(ValueError, match="batch"):
        stratified_macro_indices(labels, 0, np.random.default_rng(0))


def test_stratification_uses_exact_50_25_25_mix_and_deterministic_valid_reallocation():
    labels = np.array([Outcome.TARGET, Outcome.DRIFT, Outcome.STUCK, Outcome.VIOLATION])
    indices = stratified_macro_indices(labels, 20, np.random.default_rng(6))
    sampled = labels[indices]
    assert np.count_nonzero(sampled == Outcome.TARGET) == 10
    assert np.count_nonzero(np.isin(sampled, [Outcome.DRIFT, Outcome.STUCK])) == 5
    assert np.count_nonzero(sampled == Outcome.VIOLATION) == 5

    sparse = np.array([Outcome.TARGET, Outcome.STUCK])
    first = stratified_macro_indices(sparse, 13, np.random.default_rng(9))
    second = stratified_macro_indices(sparse, 13, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (13,)
    assert np.all(np.isin(sparse[first], sparse))


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


def test_labels_use_boundary_radius_violation_fallback_and_no_grad_encoder_output():
    boundary, outside, violated = attempt(0), attempt(0), attempt(0, violation=True)
    boundary.end_goal[:] = [1.0, 0.0]
    outside.end_goal[:] = [1.0001, 0.0]
    violated.end_goal[:] = [3.0, 0.0]
    calls: list[bool] = []

    def encoder(goals):
        calls.append(torch.is_grad_enabled())
        return goals * 2

    labels = label_macro_attempts(
        [boundary, outside, violated], encoder, torch.tensor([[2.0, 0.0]]), torch.tensor([[6.0, 0.0]]),
        assignment_radius=0.0, negative_active=True,
    )
    assert labels.outcome.tolist() == [Outcome.DRIFT, Outcome.STUCK, Outcome.VIOLATION]
    assert labels.positive_id.tolist() == [0, -1, -1]
    assert labels.negative_id.tolist() == [-1, -1, 0]
    assert calls and not any(calls)


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
    empty = model(torch.empty(0, 2), torch.empty(0, 2), torch.empty(0, 1), torch.empty(0, 2), torch.empty(0, 2))
    assert empty.outcome_probs.shape == (0, 4)
    assert empty.positive_probs.shape == empty.negative_probs.shape == (0, 0)
    assert empty.duration_by_outcome.shape == (0, 4)
    model.set_temperature(2.0)
    assert model.temperature.item() == 2.0
    with pytest.raises(ValueError, match="temperature"):
        model.set_temperature(0)


def test_model_temperature_persists_and_moved_centroids_change_candidate_probabilities():
    model = MacroTransitionModel(2, 0, 5, 4)
    with torch.no_grad():
        model.positive_query_head.weight.zero_()
        model.positive_query_head.bias.copy_(torch.tensor([1.0, 0.0]))
    inputs = (torch.zeros(1, 2), torch.ones(1, 2), torch.empty(1, 0))
    heads = tuple(id(head) for head in (model.outcome_head, model.positive_query_head, model.negative_query_head, model.duration_head))
    first = model(*inputs, torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), torch.empty(0, 2))
    second = model(*inputs, torch.tensor([[-1.0, 0.0], [1.0, 0.0]]), torch.empty(0, 2))
    assert not torch.allclose(first.positive_logits, second.positive_logits)
    assert not torch.allclose(first.positive_probs, second.positive_probs)
    assert heads == tuple(id(head) for head in (model.outcome_head, model.positive_query_head, model.negative_query_head, model.duration_head))
    uncalibrated = first.outcome_probs
    model.set_temperature(2.0)
    assert not torch.allclose(uncalibrated, model(*inputs, torch.empty(0, 2), torch.empty(0, 2)).outcome_probs)
    restored = MacroTransitionModel(2, 0, 5, 4)
    restored.load_state_dict(model.state_dict())
    assert restored.temperature.item() == pytest.approx(2.0)


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
    assert inactive_loss.negative.item() == 0.0
    invalid_negative = MacroLabels(labels.outcome, labels.positive_id, torch.tensor([-1, -1, 1]), labels.duration)
    with pytest.raises(ValueError, match="negative"):
        macro_model_loss(output, invalid_negative, k_max=4)


def test_macro_model_loss_rejects_empty_batch_before_cross_entropy():
    output = MacroTransitionOutput(
        outcome_logits=torch.empty(0, 4), outcome_probs=torch.empty(0, 4),
        positive_logits=torch.empty(0, 0), positive_probs=torch.empty(0, 0),
        negative_logits=torch.empty(0, 0), negative_probs=torch.empty(0, 0),
        duration_by_outcome=torch.empty(0, 4),
    )
    labels = MacroLabels(
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0),
    )
    with pytest.raises(ValueError, match="batch size must be positive"):
        macro_model_loss(output, labels)


def test_mixed_outcome_loss_composes_components_and_reaches_every_trainable_head():
    model = MacroTransitionModel(2, 1, 6, 4)
    z_start = torch.zeros(4, 2, requires_grad=True)
    z_command = torch.ones(4, 2, requires_grad=True)
    context = torch.zeros(4, 1, requires_grad=True)
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    negative = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], requires_grad=True)
    output = model(z_start, z_command, context, positive, negative)
    labels = MacroLabels(
        torch.tensor([Outcome.TARGET, Outcome.DRIFT, Outcome.VIOLATION, Outcome.STUCK]),
        torch.tensor([-1, 1, -1, -1]), torch.tensor([-1, -1, 0, -1]),
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    loss = macro_model_loss(output, labels, beta_duration=0.3, k_max=4)
    assert loss.positive.item() > 0 and loss.negative.item() > 0
    assert torch.allclose(loss.total, loss.outcome + loss.positive + loss.negative + 0.3 * loss.duration)
    loss.total.backward()
    for head in (model.backbone, model.outcome_head, model.positive_query_head, model.negative_query_head, model.duration_head):
        assert any(parameter.grad is not None for parameter in head.parameters())
    assert z_start.grad is None and z_command.grad is None and context.grad is None
    assert positive.grad is None and negative.grad is None


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


def test_empty_calibration_diagnostics_use_documented_neutral_values():
    logits = torch.empty((0, 4), dtype=torch.float32)
    labels = torch.empty(0, dtype=torch.long)

    fit = fit_temperature(logits, labels)
    assert (fit.temperature, fit.before_nll, fit.after_nll) == (1.0, 0.0, 0.0)
    assert expected_calibration_error(logits, labels) == 0.0

    reliability = violation_reliability_bins(logits, labels, n_bins=4)
    np.testing.assert_array_equal(reliability["count"], np.zeros(4, dtype=np.int64))
    np.testing.assert_array_equal(reliability["predicted"], np.zeros(4))
    np.testing.assert_array_equal(reliability["empirical"], np.zeros(4))

    metrics = confusion_matrix_and_classification_metrics(labels, labels)
    np.testing.assert_array_equal(metrics["matrix"], np.zeros((4, 4), dtype=np.int64))
    np.testing.assert_array_equal(metrics["precision"], np.zeros(4))
    np.testing.assert_array_equal(metrics["recall"], np.zeros(4))
    np.testing.assert_array_equal(duration_mae_by_outcome(labels, torch.empty(0), torch.empty(0)), np.zeros(4))


def test_calibration_diagnostics_report_exact_sparse_values():
    logits = torch.zeros((2, 4), dtype=torch.float32)
    labels = torch.tensor([Outcome.VIOLATION, Outcome.TARGET])
    reliability = violation_reliability_bins(logits, labels, n_bins=4)
    np.testing.assert_array_equal(reliability["count"], [0, 2, 0, 0])
    np.testing.assert_allclose(reliability["predicted"], [0.0, 0.25, 0.0, 0.0])
    np.testing.assert_allclose(reliability["empirical"], [0.0, 0.5, 0.0, 0.0])

    metrics = confusion_matrix_and_classification_metrics(
        torch.tensor([0, 0, 1, 3]), torch.tensor([0, 1, 1, 3])
    )
    np.testing.assert_array_equal(metrics["matrix"], [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(metrics["precision"], [0.5, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(metrics["recall"], [1.0, 0.5, 0.0, 1.0])
    np.testing.assert_allclose(
        duration_mae_by_outcome(
            torch.tensor([0, 0, 1, 2, 3]),
            torch.tensor([1.0, 5.0, 6.0, 8.0, 10.0]),
            torch.tensor([2.0, 3.0, 2.0, 5.0, 6.0]),
        ),
        [1.5, 4.0, 3.0, 4.0],
    )


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (fit_temperature, (torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]), torch.tensor([0]))),
        (expected_calibration_error, (torch.zeros((1, 4)), torch.tensor([float("nan")]))),
        (violation_reliability_bins, (torch.zeros((1, 4)), torch.tensor([float("inf")]))),
        (confusion_matrix_and_classification_metrics, (torch.tensor([0]), torch.tensor([float("nan")]))),
        (duration_mae_by_outcome, (torch.tensor([0]), torch.tensor([float("nan")]), torch.tensor([1.0]))),
    ],
)
def test_calibration_diagnostics_reject_nonfinite_inputs(function, args):
    with pytest.raises(ValueError, match="finite"):
        function(*args)
