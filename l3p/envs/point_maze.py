"""PointMaze-Hard: a self-contained, pure-NumPy multi-goal maze environment.

A point mass navigates a width-1 serpentine corridor maze. It follows the gym
GoalEnv convention: observations are dicts with ``observation``,
``achieved_goal`` and ``desired_goal``, and ``compute_reward`` returns the
sparse reward  r = 0 if within ``goal_threshold`` of the goal else -1.

Training uses a uniform initial-state and goal distribution over free space.
Evaluation (``eval_mode``) always starts at one end of the maze and sets the
goal at the far end (the longest path), reproducing the long-horizon
generalization test of the paper (Figure 5) with no prior knowledge of the map.

No MuJoCo / gym dependency — this env is what makes L3P runnable out of the box.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _build_serpentine(size: int = 11) -> np.ndarray:
    """A width-1 zig-zag corridor: vertical corridors joined alternately at top
    and bottom, forcing a long path from one end to the other."""
    maze = np.ones((size, size), dtype=np.int8)     # 1 = wall
    corridors = list(range(1, size - 1, 2))          # odd columns are corridors
    for c in corridors:
        maze[1:size - 1, c] = 0
    # connectors between adjacent corridors, alternating top / bottom
    top, bot = 1, size - 2
    for i, c in enumerate(corridors[:-1]):
        row = top if i % 2 == 0 else bot
        maze[row, c + 1] = 0
    return maze


def _build_two_route(size: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Build a short hazardous middle route and a longer upper detour."""
    if size < 7:
        raise ValueError("two-route PointMaze requires maze_size >= 7")
    maze = np.ones((size, size), dtype=np.int8)
    hazard_mask = np.zeros((size, size), dtype=bool)
    middle, upper = size // 2, 1
    left, right = 1, size - 2

    # The two horizontal corridors share endpoint regions only through the two
    # side connectors, yielding exactly a direct path and an upper detour.
    maze[middle, left:right + 1] = 0
    maze[upper, left:right + 1] = 0
    maze[upper:middle + 1, left] = 0
    maze[upper:middle + 1, right] = 0
    hazard_mask[middle, left + 2:right - 1] = True
    return maze, hazard_mask


class PointMazeEnv:
    metadata = {"render.modes": []}

    def __init__(self, cfg=None, goal_threshold: float = 0.15, max_vel: float = 0.95,
                 maze_size: int = 11, seed: Optional[int] = None):
        if cfg is not None:
            maze_size = getattr(cfg, "maze_size", maze_size)
        hazard_enabled = bool(getattr(cfg, "pn_pointmaze_hazard_enabled", False))
        if hazard_enabled:
            self.maze, self.hazard_mask = _build_two_route(maze_size)
        else:
            self.maze = _build_serpentine(maze_size)
            self.hazard_mask = np.zeros_like(self.maze, dtype=bool)
        self.H, self.W = self.maze.shape
        self.goal_threshold = getattr(cfg, "goal_threshold", goal_threshold) if cfg else goal_threshold
        self.max_vel = max_vel
        self.max_episode_steps = getattr(cfg, "max_episode_steps", 200) if cfg else 200
        self.rng = np.random.default_rng(seed)

        self.free_cells = np.argwhere(self.maze == 0)  # (row, col)
        self.safe_free_cells = self.free_cells[
            ~self.hazard_mask[self.free_cells[:, 0], self.free_cells[:, 1]]
        ]
        # Longest-path endpoints for evaluation: first and last corridor.
        corridors = list(range(1, self.W - 1, 2))
        if hazard_enabled:
            endpoint_row = self.H // 2
            self._start_cell = (endpoint_row, 1)
            self._goal_cell = (endpoint_row, self.W - 2)
        else:
            self._start_cell = (1, corridors[0])
            self._goal_cell = (1, corridors[-1])
        # For evaluation we sample start/goal from the two end corridors (rather
        # than two fixed points), so repeated eval episodes genuinely differ and
        # the success rate is a smooth average rather than all-or-nothing.
        self._start_region = self.safe_free_cells[
            self.safe_free_cells[:, 1] == self._start_cell[1]
        ]
        self._goal_region = self.safe_free_cells[
            self.safe_free_cells[:, 1] == self._goal_cell[1]
        ]

        self.obs_dim = 2
        self.goal_dim = 2
        self.act_dim = 2
        self.max_action = 1.0
        self.eval_mode = False

        self._pos = None
        self._goal = None
        self._steps = 0

    # ------------------------------------------------------------------ helpers
    def _is_free(self, x: float, y: float) -> bool:
        c, r = int(np.floor(x)), int(np.floor(y))
        if r < 0 or r >= self.H or c < 0 or c >= self.W:
            return False
        return self.maze[r, c] == 0

    def _cell_to_pos(self, cell) -> np.ndarray:
        r, c = cell
        return np.array([c + 0.5, r + 0.5], dtype=np.float32)

    def _sample_free_pos(self, cells=None) -> np.ndarray:
        cells = self.free_cells if cells is None else cells
        idx = self.rng.integers(0, len(cells))
        r, c = cells[idx]
        offset = self.rng.uniform(-0.35, 0.35, size=2)
        return np.array([c + 0.5 + offset[0], r + 0.5 + offset[1]], dtype=np.float32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        pos = self._pos.astype(np.float32)
        return dict(observation=pos.copy(), achieved_goal=pos.copy(),
                    desired_goal=self._goal.copy())

    # ------------------------------------------------------------------ gym API
    def set_eval(self, flag: bool) -> None:
        self.eval_mode = flag

    def reset(self) -> Dict[str, np.ndarray]:
        self._steps = 0
        if self.eval_mode:
            # long-horizon test: start in the first corridor, goal in the last —
            # randomized within each end region so eval episodes differ.
            self._pos = self._sample_free_pos(self._start_region)
            self._goal = self._sample_free_pos(self._goal_region)
        else:
            self._pos = self._sample_free_pos(self.safe_free_cells)
            self._goal = self._sample_free_pos(self.safe_free_cells)
        return self._get_obs()

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        delta = action * self.max_vel
        # Sub-step with per-axis sliding to prevent tunneling through thin walls.
        n_sub = max(1, int(np.ceil(np.max(np.abs(delta)) / 0.1)))
        step = delta / n_sub
        pos = self._pos.copy()
        for _ in range(n_sub):
            nx = pos[0] + step[0]
            if self._is_free(nx, pos[1]):
                pos[0] = nx
            ny = pos[1] + step[1]
            if self._is_free(pos[0], ny):
                pos[1] = ny
        self._pos = pos

        self._steps += 1
        obs = self._get_obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], None))
        success = reward == 0.0
        done = self._steps >= self.max_episode_steps
        row, col = int(np.floor(self._pos[1])), int(np.floor(self._pos[0]))
        hazard = bool(self.hazard_mask[row, col])
        safety_cost = float(hazard)
        info = {
            "is_success": float(success),
            "goal_reached": bool(success),
            "safety_cost": safety_cost,
            "hazard": hazard,
            "termination_reason": "timeout" if done else ("goal" if success else "other"),
        }
        if done:
            info["TimeLimit.truncated"] = True
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       info=None) -> np.ndarray:
        """Sparse reward, vectorized for HER: 0 if reached else -1."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d > self.goal_threshold).astype(np.float32)
