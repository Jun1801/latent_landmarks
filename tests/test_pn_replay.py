"""Safety-aware HER replay and PN-LMCGS landmark-memory tests."""

import numpy as np
import pytest

from l3p.replay.her_buffer import HERReplayBuffer
from l3p.pn_lmcgs.landmark_memories import LandmarkMemoryManager


def reward(next_ag, g):
    return (np.linalg.norm(next_ag - g, axis=-1) > 1e-6).astype(np.float32) * -1


def episode(length=3, goal_dim=1, **fields):
    ep = {
        "obs": np.arange(length + 1, dtype=np.float32).reshape(-1, 1),
        "ag": np.arange(length + 1, dtype=np.float32).reshape(-1, 1),
        "g": np.full((length, goal_dim), 9, dtype=np.float32),
        "act": np.zeros((length, 1), dtype=np.float32),
    }
    ep.update(fields)
    return ep


def buffer(size=3, horizon=5, her_ratio=0.0):
    return HERReplayBuffer(size, horizon, 1, 1, 1, reward,
                           her_ratio=her_ratio, hindsight_range=2)


def test_legacy_episode_defaults_and_old_outputs_are_preserved():
    replay = buffer(horizon=3)
    replay.store_episode(episode(3))
    batch = replay.sample(4, np.random.default_rng(0))
    assert {"obs", "next_obs", "ag", "next_ag", "act", "g", "future_ag", "reward"} <= set(batch)
    assert np.all(batch["cmd_g"] == batch["g"])
    for key in ("cost", "violation", "goal_reached", "terminated", "truncated", "stop"):
        assert not batch[key].any()
    assert np.all(batch["termination_reason"] == HERReplayBuffer.REASON_OTHER)
    assert replay.episode_lengths[0] == 3


def test_her_preserves_cost_and_recomputes_reached_stop():
    replay = buffer(horizon=3, her_ratio=1.0)
    replay.hindsight_range = 1
    ep = episode(3, cost=np.array([0, 1, 0], dtype=np.float32),
                 violation=np.array([0, 1, 0], dtype=bool),
                 terminated=np.array([0, 1, 0], dtype=bool))
    replay.store_episode(ep)
    batch = replay.sample(64, np.random.default_rng(4))
    assert np.array_equal(batch["cost"], batch["violation"].astype(np.float32))
    assert np.all(batch["goal_reached"])
    assert np.all(batch["stop"])
    assert np.all(batch["reward"] == 0)


def test_shortened_episode_future_indices_and_achieved_goals_exclude_padding():
    replay = buffer(horizon=5, her_ratio=1.0)
    replay.store_episode(episode(2))
    batch = replay.sample(100, np.random.default_rng(3))
    assert np.all(batch["obs"].ravel() < 2)
    assert np.all(batch["future_ag"].ravel() > batch["ag"].ravel())
    assert np.all(batch["future_ag"].ravel() <= 2)
    achieved = replay.sample_achieved_goals(100, np.random.default_rng(2))
    assert np.all((achieved >= 0) & (achieved <= 2))
    with pytest.raises(ValueError, match="empty"):
        buffer().sample(1)


def test_overwriting_long_episode_with_short_clears_stale_storage():
    replay = buffer(size=1, horizon=5)
    replay.store_episode(episode(5))
    short = episode(2)
    short["obs"][:] = 50
    short["ag"][:] = 60
    replay.store_episode(short)
    assert replay.episode_lengths[0] == 2
    assert not replay.valid[0, 2:].any()
    assert not replay.obs[0, 3:].any()
    assert not replay.ag[0, 3:].any()


def manager(**kwargs):
    return LandmarkMemoryManager(goal_dim=1, positive_capacity=8,
                                 negative_capacity=3, **kwargs)


def test_safe_successful_command_segment_adds_only_eligible_positive_goals():
    memory = manager()
    ep = episode(4, cmd_g=np.array([[2], [2], [2], [7]], dtype=np.float32),
                 goal_reached=np.array([0, 1, 0, 0], dtype=bool),
                 cost=np.zeros(4, dtype=np.float32))
    memory.ingest_episode(ep, episode_id=10)
    assert np.array_equal(memory.positive_goals, np.array([[0.], [1.]], dtype=np.float32))
    assert memory.negative_size == 0


def test_timeout_or_failure_is_neutral():
    memory = manager()
    ep = episode(3, goal_reached=np.zeros(3, dtype=bool),
                 truncated=np.array([0, 0, 1], dtype=bool), cost=np.zeros(3))
    memory.ingest_episode(ep, episode_id=1)
    assert memory.positive_size == memory.negative_size == 0


def test_violation_stores_exactly_post_transition_endpoint():
    memory = manager()
    ep = episode(3, ag=np.array([[10.], [11.], [12.], [13.]], dtype=np.float32),
                 cost=np.array([0, 1, 0], dtype=np.float32))
    memory.ingest_episode(ep, episode_id=7)
    assert np.array_equal(memory.negative_goals, np.array([[12.]], dtype=np.float32))
    sample = memory.sample_negative(1, np.random.default_rng(0))
    assert sample["episode_id"].item() == 7
    assert not sample["is_preimpact"].item()


def test_violation_fallback_metadata_and_distinct_episode_gate_survives_eviction():
    memory = LandmarkMemoryManager(goal_dim=1, positive_capacity=2, negative_capacity=2)
    for ident in (1, 2, 3):
        ep = episode(1, ag=np.array([[ident], [ident + 10]], dtype=np.float32),
                     cost=np.array([1.]), post_state_available=np.array([ident != 1]))
        memory.ingest_episode(ep, episode_id=ident)
    assert memory.negative_size == 2
    assert memory.distinct_negative_episodes == 2
    assert memory.negative_landmarks_active(2, 2)
    sample = memory.sample_negative(2, np.random.default_rng(3))
    assert not np.any(sample["is_preimpact"]), "evicted fallback must not leave stale metadata"


def test_memory_copies_inputs_and_roundtrips_with_validation():
    memory = manager()
    ep = episode(2, goal_reached=np.array([1, 0], dtype=bool), cost=np.array([0., 1.]))
    memory.ingest_episode(ep, episode_id=5)
    ep["ag"][:] = 999
    assert not np.any(memory.positive_goals == 999)
    assert not np.any(memory.negative_goals == 999)
    clone = manager()
    clone.load_state_dict(memory.state_dict())
    assert np.array_equal(clone.positive_goals, memory.positive_goals)
    assert np.array_equal(clone.negative_goals, memory.negative_goals)
    bad = memory.state_dict()
    bad["goal_dim"] = 2
    with pytest.raises(ValueError, match="goal_dim"):
        clone.load_state_dict(bad)
    bad = memory.state_dict()
    bad["negative_capacity"] = 99
    with pytest.raises(ValueError, match="capacities"):
        clone.load_state_dict(bad)
    with pytest.raises(ValueError, match="zero"):
        memory.ingest_episode(episode(0), 6)


def test_memory_rejects_malformed_commanded_goal_shape():
    memory = manager()
    ep = episode(2, cmd_g=np.array([1., 1.]), goal_reached=np.array([0, 1], dtype=bool))
    with pytest.raises(ValueError, match="cmd_g"):
        memory.ingest_episode(ep, episode_id=1)
