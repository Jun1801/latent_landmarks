"""Safety-cost interface and hazardous PointMaze regression tests."""

import os
import sys
from collections import deque

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from l3p.config import get_config, validate_pn_config
from l3p.envs.mujoco import GymGoalEnvWrapper
from l3p.envs.point_maze import PointMazeEnv, _build_serpentine
from l3p.envs.safety import CollisionCostAdapter, normalize_step_result


def test_normalize_legacy_step_copies_info_and_sets_contract_fields():
    info = {"is_success": 1.0}
    result = normalize_step_result(("obs", -1.0, True, info))

    assert result.observation == "obs"
    assert result.reward == -1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.done is True
    assert result.safety_cost == 0.0
    assert result.info is not info
    assert info == {"is_success": 1.0}
    assert result.info["goal_reached"] is True
    assert result.info["termination_reason"] == "goal"
    assert result.info["safety_cost"] == 0.0


def test_normalize_gymnasium_success_metadata_sets_goal_termination():
    result = normalize_step_result(("obs", -1.0, True, False, {"success": 1.0}))

    assert result.info["goal_reached"] is True
    assert result.info["termination_reason"] == "goal"


def test_normalize_gymnasium_timeout_is_not_a_violation():
    result = normalize_step_result(("obs", -1.0, False, True, {}))

    assert result.terminated is False
    assert result.truncated is True
    assert result.done is True
    assert result.safety_cost == 0.0
    assert result.info["termination_reason"] == "timeout"


def test_point_maze_legacy_timeout_normalizes_as_truncation():
    env = PointMazeEnv(cfg=get_config("PointMaze", max_episode_steps=1), seed=0)
    env.reset()

    result = normalize_step_result(env.step(np.zeros(2, dtype=np.float32)))

    assert result.info["termination_reason"] == "timeout"
    assert result.info["TimeLimit.truncated"] is True
    assert result.terminated is False
    assert result.truncated is True


def test_normalize_safety_gymnasium_uses_separate_cost_before_info():
    result = normalize_step_result(
        ("obs", -1.0, 2.0, False, False, {"safety_cost": 0.0, "cost": 0.0})
    )

    assert result.safety_cost == 1.0
    assert result.info["safety_cost"] == 1.0


@pytest.mark.parametrize(
    ("raw_cost", "expected"),
    [(-2.0, 0.0), (0.0, 0.0), (0.01, 1.0), (True, 1.0)],
)
def test_safety_cost_values_are_binarized(raw_cost, expected):
    result = normalize_step_result(("obs", 0.0, raw_cost, False, False, {}))

    assert result.safety_cost == expected


def test_cost_precedence_prefers_safety_info_then_generic_cost_then_adapter():
    adapter = CollisionCostAdapter(enabled=True, info_key="collision", unsafe_value="hit")

    safety_info = normalize_step_result(
        ("obs", 0.0, False, False, {"safety_cost": 0.0, "cost": 3.0, "collision": "hit"}),
        collision_adapter=adapter,
    )
    generic_cost = normalize_step_result(
        ("obs", 0.0, False, False, {"cost": 3.0, "collision": "hit"}),
        collision_adapter=adapter,
    )
    adapter_cost = normalize_step_result(
        ("obs", 0.0, False, False, {"collision": "hit"}),
        collision_adapter=adapter,
    )

    assert safety_info.safety_cost == 0.0
    assert generic_cost.safety_cost == 1.0
    assert adapter_cost.safety_cost == 1.0


def test_collision_adapter_is_opt_in_and_supports_unsafe_value():
    disabled = CollisionCostAdapter(info_key="collision", unsafe_value="hit")
    enabled = CollisionCostAdapter(enabled=True, info_key="collision", unsafe_value="hit")

    assert normalize_step_result(("obs", 0.0, False, {}), collision_adapter=disabled).safety_cost == 0.0
    assert normalize_step_result(("obs", 0.0, False, {"collision": "hit"}), collision_adapter=enabled).safety_cost == 1.0
    assert normalize_step_result(("obs", 0.0, False, {"collision": "clear"}), collision_adapter=enabled).safety_cost == 0.0


def test_missing_cost_and_ordinary_collision_info_default_to_zero():
    result = normalize_step_result(("obs", 0.0, False, {"collision": True}))

    assert result.safety_cost == 0.0


@pytest.mark.parametrize("raw", [("obs", 0.0, False), ("obs", 0.0, False, False, False, {}, "extra")])
def test_normalize_rejects_unknown_step_tuple_lengths(raw):
    with pytest.raises(ValueError, match="step result.*4, 5, or 6"):
        normalize_step_result(raw)


def _shortest_path_length(maze, start, goal, blocked=None):
    blocked = np.zeros_like(maze, dtype=bool) if blocked is None else blocked
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (row + drow, col + dcol)
            if (
                0 <= nxt[0] < maze.shape[0]
                and 0 <= nxt[1] < maze.shape[1]
                and maze[nxt] == 0
                and not blocked[nxt]
                and nxt not in visited
            ):
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def test_hazardous_point_maze_has_short_risky_route_and_long_safe_detour():
    cfg = get_config("PointMaze", pn_lmcgs_enabled=True, pn_pointmaze_hazard_enabled=True)
    env = PointMazeEnv(cfg=cfg, seed=0)
    middle = env.H // 2
    start = (middle, 1)
    goal = (middle, env.W - 2)

    direct = _shortest_path_length(env.maze, start, goal)
    safe_detour = _shortest_path_length(env.maze, start, goal, blocked=env.hazard_mask)

    assert np.all(env.maze[0, :] == 1)
    assert np.all(env.maze[-1, :] == 1)
    assert env.hazard_mask[middle, 1:env.W - 1].any()
    assert direct is not None and safe_detour is not None
    assert direct < safe_detour


@pytest.mark.parametrize("eval_mode", [False, True])
def test_hazardous_point_maze_reset_never_samples_hazardous_start_or_goal(eval_mode):
    cfg = get_config("PointMaze", pn_lmcgs_enabled=True, pn_pointmaze_hazard_enabled=True)
    env = PointMazeEnv(cfg=cfg, seed=17)
    env.set_eval(eval_mode)

    for _ in range(100):
        obs = env.reset()
        for key in ("achieved_goal", "desired_goal"):
            col, row = np.floor(obs[key]).astype(int)
            assert not env.hazard_mask[row, col]


def test_hazardous_point_maze_emits_cost_without_terminal_violation():
    cfg = get_config("PointMaze", pn_lmcgs_enabled=True, pn_pointmaze_hazard_enabled=True)
    env = PointMazeEnv(cfg=cfg, seed=0)
    env.reset()
    hazard_row, hazard_col = np.argwhere(env.hazard_mask)[0]
    env._pos = env._cell_to_pos((hazard_row, hazard_col))

    _, _, done, info = env.step(np.zeros(2, dtype=np.float32))

    assert done is False
    assert info["safety_cost"] == 1.0
    assert info["hazard"] is True
    assert info["termination_reason"] == "other"


def test_standard_point_maze_layout_and_step_behavior_remain_default():
    env = PointMazeEnv(cfg=get_config("PointMaze"), seed=0)

    assert np.array_equal(env.maze, _build_serpentine(env.W))
    assert not env.hazard_mask.any()
    env.reset()
    _, _, done, info = env.step(np.zeros(2, dtype=np.float32))
    assert done is False
    assert info["safety_cost"] == 0.0
    assert info["hazard"] is False
    assert "goal_reached" in info and "termination_reason" in info


class _ActionSpace:
    shape = (2,)
    high = np.array([1.0, 1.0], dtype=np.float32)


class _TupleBackend:
    action_space = _ActionSpace()

    def __init__(self, step_result):
        self.step_result = step_result
        self.unwrapped = self

    def reset(self):
        return {
            "observation": np.zeros(2, dtype=np.float32),
            "achieved_goal": np.zeros(2, dtype=np.float32),
            "desired_goal": np.ones(2, dtype=np.float32),
        }

    def step(self, action):
        return self.step_result

    def compute_reward(self, achieved_goal, desired_goal, info):
        return float(np.allclose(achieved_goal, desired_goal)) - 1.0


@pytest.mark.parametrize(
    "step_result",
    [
        ("legacy", -1.0, True, {}),
        ("gymnasium", -1.0, False, True, {}),
        ("safety", -1.0, 1.0, False, False, {}),
    ],
)
def test_gym_wrapper_returns_normalized_legacy_tuple_for_every_backend_shape(step_result):
    wrapper = GymGoalEnvWrapper(_TupleBackend(step_result), get_config("PointMaze"))

    obs, reward, done, info = wrapper.step(np.zeros(2, dtype=np.float32))

    assert obs in {"legacy", "gymnasium", "safety"}
    assert reward == -1.0
    assert isinstance(done, bool)
    assert "safety_cost" in info


def test_pn_config_defaults_and_validation():
    cfg = get_config("PointMaze")
    assert cfg.pn_lmcgs_enabled is False
    assert cfg.pn_pointmaze_hazard_enabled is False
    assert cfg.pn_collision_cost_enabled is False
    validate_pn_config(cfg)

    invalid_overrides = [
        {"pn_num_positive_landmarks": 0},
        {"pn_probability_mcg_search": 0.8},
        {"pn_search_risk_limit": 1.1},
        {"pn_k_min": 51, "pn_k_max": 50},
        {"pn_calibration_validation_split": 1.0},
        {"pn_pointmaze_hazard_enabled": True},
    ]
    for overrides in invalid_overrides:
        with pytest.raises(ValueError):
            get_config("PointMaze", **overrides)


@pytest.mark.parametrize("risk_z", [-1.0, -0.0, 0.0, 1.645])
def test_pn_config_requires_nonnegative_risk_z(risk_z):
    if risk_z < 0.0:
        with pytest.raises(ValueError, match="pn_risk_z"):
            get_config("PointMaze", pn_risk_z=risk_z)
    else:
        assert get_config("PointMaze", pn_risk_z=risk_z).pn_risk_z == risk_z


@pytest.mark.parametrize("info_key", ["", "   "])
def test_pn_config_requires_nonempty_collision_key_when_enabled(info_key):
    with pytest.raises(ValueError, match="pn_collision_cost_info_key"):
        get_config(
            "PointMaze",
            pn_collision_cost_enabled=True,
            pn_collision_cost_info_key=info_key,
        )


def test_pn_config_accepts_nonempty_collision_key_when_enabled():
    cfg = get_config(
        "PointMaze",
        pn_collision_cost_enabled=True,
        pn_collision_cost_info_key="collision",
    )

    assert cfg.pn_collision_cost_info_key == "collision"


@pytest.mark.parametrize(
    "overrides",
    [
        {"pn_num_simulations": 1.5},
        {"pn_num_simulations": True},
        {"pn_k_max": float("nan")},
        {"pn_k_max": float("inf")},
        {"pn_probability_mcg_search": float("nan")},
        {"pn_search_risk_limit": float("nan")},
    ],
)
def test_pn_config_rejects_non_integral_and_nonfinite_values(overrides):
    with pytest.raises(ValueError):
        get_config("PointMaze", **overrides)
