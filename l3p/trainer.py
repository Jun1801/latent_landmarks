"""Overall training procedure for L3P (paper Algorithm 3).

Alternates between (a) collecting episodes with the L3P planner into a
centralized HER replay buffer and (b) taking gradient steps on every module:

    critic Q via distance D (Eq. 1)        -> agent.update_critic
    value function V (Eq. 4)               -> agent.update_value
    policy pi (max Q + action L2)          -> agent.update_actor
    auto-encoder (L_rec + lambda*L_latent) -> ae_losses (Eq. 2)
    latent centroids (ELBO)                -> landmarks.elbo_loss (Eq. 5)

Warm-up: the first `initial_random_trajs` episodes per worker use random
actions; after `n_warmup_trajs` episodes total the centroids are initialized
via GLS and planning is enabled. During training half the episodes use the
planner (search_prob_train) and each planning episode adds a fresh set of random
GLS landmarks for exploration.
"""

from __future__ import annotations

import copy
import hashlib
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from l3p.config import validate_pn_config
from l3p.agent.ddpg import DDPGAgent
from l3p.losses import ae_losses
from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.landmarks import LatentLandmarks, greedy_latent_sparsification
from l3p.envs.safety import CollisionCostAdapter, normalize_step_result
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.pn_lmcgs.calibration import (
    confusion_matrix_and_classification_metrics,
    duration_mae_by_outcome,
    expected_calibration_error,
    fit_temperature,
    violation_reliability_bins,
)
from l3p.pn_lmcgs.landmark_memories import LandmarkMemoryManager
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
    macro_model_loss,
)
from l3p.pn_lmcgs.planner_adapter import CommandResult, PNPlannerAdapter
from l3p.pn_lmcgs.safety_critic import ViolationCritic
from l3p.replay.her_buffer import HERReplayBuffer


class PNPhase(str, Enum):
    WARMUP = "warmup"
    LANDMARKS = "landmarks"
    MACRO_BOOTSTRAP = "macro_bootstrap"
    JOINT = "joint"


PN_CHECKPOINT_VERSION = 3
# CUDA generator checkpoints carry at least a 64-bit seed and 64-bit offset.
_PN_MIN_CUDA_RNG_STATE_BYTES = 16

_PN_STRUCTURAL_CONFIG_FIELDS = {
    "hidden_units", "hidden_layers", "ae_hidden_units", "ae_hidden_layers",
    "embedding_size", "max_episode_steps", "pn_macro_hidden_dim", "pn_k_max",
    "pn_calibration_validation_split", "pn_positive_memory_capacity",
    "pn_negative_memory_capacity", "pn_macro_replay_capacity",
    "pn_pointmaze_hazard_enabled", "pn_collision_cost_enabled",
    "pn_collision_cost_info_key", "pn_collision_cost_unsafe_value",
}

_PN_RUNTIME_CONFIG_FIELDS = {
    "device", "total_steps", "eval_interval", "eval_episodes", "log_interval",
    "n_workers", "pn_lmcgs_enabled",
}


class _LegacyStepAPIProxy:
    """Present Gymnasium termination values to the unchanged legacy collector."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    def step(self, action):
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
            info = dict(info)
            if truncated:
                info["TimeLimit.truncated"] = True
            return observation, reward, bool(terminated or truncated), info
        return result


def _state_fingerprint(value: Any) -> str:
    """Return a deterministic digest for nested checkpoint-compatible state."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for entry in item:
                visit(entry)
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def _episode_success(info: dict, reward: float) -> float:
    """Robustly read a success signal across env conventions.

    NumPy PointMaze and Fetch use info['is_success']; gymnasium-robotics maze
    envs (AntMaze / PointMaze) use info['success']. Fall back to the sparse
    reward (0 == goal reached, -1 otherwise).
    """
    for key in ("is_success", "success"):
        if key in info and info[key] is not None:
            return float(info[key])
    return float(reward == 0.0)


class L3PTrainer:
    def __init__(self, vec_env, cfg,
                 metrics_callback: Optional[Callable[[str, int, Dict[str, float]], None]] = None):
        self.env = vec_env
        self.cfg = cfg
        self.metrics_callback = metrics_callback
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        od, gd, ad = vec_env.obs_dim, vec_env.goal_dim, vec_env.act_dim
        self.T = vec_env.max_episode_steps

        self.agent = DDPGAgent(od, gd, ad, vec_env.max_action, cfg)
        self.ae = ReachabilityAutoEncoder(gd, cfg.embedding_size, cfg.ae_hidden_units,
                                          cfg.ae_hidden_layers).to(self.device)
        positive_landmark_count = (
            cfg.pn_num_positive_landmarks if cfg.pn_lmcgs_enabled else cfg.n_landmarks
        )
        self.landmarks = LatentLandmarks(
            positive_landmark_count, cfg.embedding_size,
        ).to(self.device)
        # The AE encoder normalizes its goal inputs with the running goal stats
        # (stabilizes the latent space for large-coordinate goal spaces).
        self.ae.normalizer = self.agent.g_norm
        self.graph_search = GraphSearch(cfg)
        self.planner = LatentPlanner(self.agent, self.landmarks, self.ae, self.graph_search, cfg)

        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=cfg.ae_lr)
        self.landmark_opt = torch.optim.Adam(self.landmarks.parameters(), lr=cfg.landmark_lr)

        self.buffer = HERReplayBuffer(
            size_episodes=100_000, horizon=self.T, obs_dim=od, goal_dim=gd, act_dim=ad,
            compute_reward=vec_env.compute_reward, her_ratio=cfg.her_ratio,
            hindsight_range=cfg.hindsight_range)

        self.total_env_steps = 0
        self.episodes_collected = 0
        self.centroids_initialized = False
        self.grad_step_count = 0

        if cfg.pn_lmcgs_enabled:
            cfg_state = vars(cfg)
            self.pn_config_snapshot = copy.deepcopy({
                name: cfg_state[name] for name in sorted(cfg_state)
            })
            if not self.pn_config_snapshot:
                raise ValueError("enabled PN trainer requires a nonempty configuration state")
            self.landmark_memory = LandmarkMemoryManager(
                gd,
                positive_capacity=cfg.pn_positive_memory_capacity,
                negative_capacity=cfg.pn_negative_memory_capacity,
            )
            self.negative_landmarks = LatentLandmarks(
                cfg.pn_num_negative_landmarks, cfg.embedding_size,
            ).to(self.device)
            self.negative_landmark_opt = torch.optim.Adam(
                self.negative_landmarks.parameters(), lr=cfg.landmark_lr,
            )
            self.macro_buffer = MacroAttemptBuffer(
                cfg.pn_macro_replay_capacity,
                validation_fraction=cfg.pn_calibration_validation_split,
                split_seed=cfg.seed,
            )
            self.macro_model = MacroTransitionModel(
                cfg.embedding_size,
                context_dim=0,
                hidden_dim=cfg.pn_macro_hidden_dim,
                k_max=cfg.pn_k_max,
            ).to(self.device)
            self.macro_opt = torch.optim.Adam(
                self.macro_model.parameters(), lr=cfg.pn_macro_lr,
            )
            self.pn_phase = PNPhase.WARMUP
            self.pn_positive_active = False
            self.pn_negative_active = False
            self.pn_calibration_ready = False
            self.pn_calibration_temperature = 1.0
            self.pn_calibration_before_nll = 0.0
            self.pn_calibration_after_nll = 0.0
            self.pn_calibration_ece = 0.0
            self.pn_positive_episode_count = 0
            self.pn_primitive_transition_count = 0
            self.pn_primitive_violation_count = 0
            self.pn_episode_goal_success_count = 0
            self.pn_episode_safe_success_count = 0
            self.pn_violation_episode_count = 0
            self.pn_path_length_total = 0.0
            self.pn_planning_count = 0
            self.pn_no_safe_plan_count = 0
            self.pn_ingested_episode_ids = set()
            self.pn_last_collection_status = "initialized"
            self.pn_last_diagnostics = {
                "planning_latency": 0.0,
                "branch_before": 0.0,
                "branch_after": 0.0,
                "chosen_p_violation": 0.0,
                "chosen_q_risk": 0.0,
                "chosen_upper_risk": 0.0,
                "completed_simulations": 0.0,
                "simulated_safe_successes": 0.0,
                "cycles": 0.0,
                "leaf_uses": 0.0,
            }
            self.pn_planner_diagnostic_sums = {
                name: 0.0 for name in self.pn_last_diagnostics
            }
            self._pn_common_action_bucket_cache_key = None
            self._pn_common_action_bucket_cache = None
            self.last_eval_metrics = {
                "goal_success_rate": 0.0,
                "safe_success_rate": 0.0,
                "episode_violation_rate": 0.0,
                "path_length": 0.0,
                "no_safe_plan_rate": 0.0,
                "planning_latency": 0.0,
            }
            self.pn_last_validation_diagnostics = self._pn_neutral_validation_logs()
            self._bind_pn_planner()

    def _bind_pn_planner(self) -> None:
        self.pn_planner = PNPlannerAdapter(
            self.agent,
            self.ae,
            self.landmarks,
            self.negative_landmarks,
            self.macro_model,
            self.graph_search,
            self.cfg,
            rng=self.rng,
            original_planner=self.planner,
        )

    def _refresh_pn_phase(self) -> None:
        self.pn_positive_active = (
            self.landmark_memory.positive_size >= self.cfg.pn_min_positive_samples
            and self.pn_positive_episode_count >= self.cfg.pn_min_positive_episodes
        )
        self.pn_negative_active = self.landmark_memory.negative_landmarks_active(
            self.cfg.pn_min_negative_samples,
            self.cfg.pn_min_negative_episodes,
        )
        if not self.centroids_initialized:
            self.pn_phase = PNPhase.WARMUP \
                if self.episodes_collected < self.cfg.n_warmup_trajs else PNPhase.LANDMARKS
            return
        if not self.pn_positive_active:
            self.pn_phase = PNPhase.LANDMARKS
            return
        if (self.macro_buffer.size < self.cfg.pn_min_macro_attempts
                or self.macro_buffer.validation_indices.size == 0
                or not self.pn_calibration_ready):
            self.pn_phase = PNPhase.MACRO_BOOTSTRAP
            return
        min_common_action_attempts = self.cfg.pn_min_attempts_per_common_action_bucket
        if (min_common_action_attempts > 0
                and np.any(self._pn_common_action_bucket_counts()
                           < min_common_action_attempts)):
            self.pn_phase = PNPhase.MACRO_BOOTSTRAP
            return
        self.pn_phase = PNPhase.JOINT

    def _pn_common_action_bucket_counts(self) -> np.ndarray:
        """Count macro commands assigned to current positive landmarks.

        Commands outside the current positive-centroid assignment radius share
        the final direct/other-goal bucket.
        """
        n_landmarks = self.landmarks.n_landmarks
        counts = np.zeros(n_landmarks + 1, dtype=np.int64)
        all_attempts = self.macro_buffer.attempts
        train_indices = self.macro_buffer.train_indices
        if train_indices.size == 0:
            return counts

        assignment_radius = float(self.cfg.pn_positive_assignment_radius)
        if not np.isfinite(assignment_radius) or assignment_radius < 0.0:
            raise ValueError("pn_positive_assignment_radius must be finite and non-negative")

        cache_key = (
            int(self.macro_buffer.size),
            int(getattr(self.macro_buffer, "_next_uid", self.macro_buffer.size)),
            int(getattr(self.landmarks.centroids, "_version", 0)),
            assignment_radius,
        )
        if (self._pn_common_action_bucket_cache_key == cache_key
                and self._pn_common_action_bucket_cache is not None):
            return self._pn_common_action_bucket_cache.copy()

        attempts = [all_attempts[int(index)] for index in train_indices]
        command_goals = np.asarray(
            [attempt.command_goal for attempt in attempts], dtype=np.float32,
        )
        with torch.no_grad():
            positive_centroids = self.landmarks.centroids.detach()
            if (positive_centroids.ndim != 2
                    or positive_centroids.shape[0] != n_landmarks
                    or not torch.isfinite(positive_centroids).all()):
                raise ValueError("positive centroids must be finite [N, latent_dim] values")
            command_latents = torch.as_tensor(
                self.ae.encode(self.agent.to_tensor(command_goals)),
                dtype=torch.float32,
                device=self.device,
            )
            if (command_latents.ndim != 2
                    or command_latents.shape[0] != len(attempts)
                    or command_latents.shape[1] != positive_centroids.shape[1]
                    or not torch.isfinite(command_latents).all()):
                raise ValueError("AE encoder must return finite [B, latent_dim] command embeddings")
            distances = torch.cdist(command_latents, positive_centroids)
            nearest_distances, nearest_indices = torch.min(distances, dim=1)
            assigned = nearest_distances <= assignment_radius
            if assigned.any():
                assigned_indices = nearest_indices[assigned].cpu().numpy()
                counts[:n_landmarks] = np.bincount(
                    assigned_indices, minlength=n_landmarks,
                )[:n_landmarks]
            counts[-1] = int((~assigned).sum().item())
        self._pn_common_action_bucket_cache_key = cache_key
        self._pn_common_action_bucket_cache = counts.copy()
        return counts

    # ------------------------------------------------------------------ collection
    def _sample_random_landmarks(self, n: int) -> Optional[torch.Tensor]:
        """GLS-select `n` extra latent landmarks from the replay (train-time
        exploration; Appendix D)."""
        if n <= 0 or len(self.buffer) == 0:
            return None
        goals = self.buffer.sample_achieved_goals(max(n * 4, self.cfg.gls_batch_size), self.rng)
        with torch.no_grad():
            z = self.ae.encode(self.agent.to_tensor(goals))
        idx = greedy_latent_sparsification(z, n, self.rng)
        return z[idx].detach()

    def _pn_original_command(self, observation: np.ndarray) -> CommandResult:
        if hasattr(self.planner, "select_command"):
            goal, _legacy_horizon = self.planner.select_command(observation)
        else:
            goal = self.planner._current_subgoal()
        goal = np.asarray(goal, dtype=np.float32)
        adaptive = self.pn_planner.direct_command(observation, goal)
        return CommandResult(goal, adaptive.horizon, "original")

    def _pn_positive_command(self, observation: np.ndarray) -> CommandResult:
        if not self.centroids_initialized or not self.pn_positive_active:
            raise RuntimeError("positive landmark exploration requires active centroids")
        centroids = self.landmarks.centroids.detach()
        if centroids.ndim != 2 or centroids.shape[0] == 0:
            raise RuntimeError("positive landmark exploration requires nonempty centroids")
        index = int(self.rng.integers(centroids.shape[0]))
        with torch.no_grad():
            decoded = self.ae.decode(centroids[index:index + 1])
        goal = np.asarray(decoded.detach().cpu().numpy()[0], dtype=np.float32)
        return self.pn_planner.direct_command(observation, goal)

    def _pn_record_planner_diagnostics(self, diagnostics: dict) -> None:
        for source, target in (
            ("latency", "planning_latency"),
            ("branch_before", "branch_before"),
            ("branch_after", "branch_after"),
            ("chosen_p_violation", "chosen_p_violation"),
            ("chosen_q_risk", "chosen_q_risk"),
            ("chosen_upper_risk", "chosen_upper_risk"),
            ("completed_simulations", "completed_simulations"),
            ("simulated_safe_successes", "simulated_safe_successes"),
            ("cycles", "cycles"),
            ("leaf_uses", "leaf_uses"),
        ):
            value = self._pn_optional_finite_scalar(diagnostics.get(source, 0.0))
            self.pn_last_diagnostics[target] = value
            self.pn_planner_diagnostic_sums[target] += value

    def _pn_select_command(self, obs_dict: dict, final_goal: np.ndarray,
                           use_planning: bool, random_actions: bool,
                           original_ready: bool) -> CommandResult:
        observation = np.asarray(obs_dict["observation"], dtype=np.float32)
        achieved = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
        if random_actions or not use_planning:
            return self.pn_planner.direct_command(observation, final_goal)
        if self.pn_phase in (PNPhase.WARMUP, PNPhase.LANDMARKS):
            return self._pn_original_command(observation) if original_ready else \
                self.pn_planner.direct_command(observation, final_goal)
        if self.pn_phase == PNPhase.MACRO_BOOTSTRAP:
            modes = []
            if self.centroids_initialized and self.pn_positive_active:
                modes.append(lambda: self._pn_positive_command(observation))
            if original_ready:
                modes.append(lambda: self._pn_original_command(observation))
            modes.append(lambda: self.pn_planner.direct_command(observation, final_goal))
            return modes[int(self.rng.integers(len(modes)))]()

        probabilities = np.asarray([
            self.cfg.pn_probability_mcg_search,
            self.cfg.pn_probability_original_planner,
            self.cfg.pn_probability_direct_goal,
        ], dtype=np.float64)
        probabilities /= probabilities.sum()
        mode = int(self.rng.choice(3, p=probabilities))
        if mode == 0:
            self.pn_planning_count = int(getattr(self, "pn_planning_count", 0)) + 1
            command = self.pn_planner.planned_command(
                observation, achieved, final_goal,
                context=None, training=True,
            )
            diagnostics = command.diagnostics or {}
            self._pn_record_planner_diagnostics(diagnostics)
            if command.goal is None:
                self.pn_no_safe_plan_count += 1
            return command
        if mode == 1 and original_ready:
            return self._pn_original_command(observation)
        return self.pn_planner.direct_command(observation, final_goal)

    def _pn_action(self, env, obs_dict: dict, command: CommandResult,
                   random_actions: bool) -> np.ndarray:
        if random_actions:
            return self.rng.uniform(-env.max_action, env.max_action, size=self.env.act_dim)
        if command.source == "original":
            if hasattr(self.planner, "act_for_current_command"):
                return self.planner.act_for_current_command(
                    obs_dict["observation"], self.cfg.action_noise,
                    self.cfg.random_action_prob,
                )
            return self.planner.act(
                obs_dict["observation"], self.cfg.action_noise,
                self.cfg.random_action_prob,
            )
        return self.agent.act(
            obs_dict["observation"], command.goal, self.cfg.action_noise,
            self.cfg.random_action_prob,
        )

    def collect_episode(self, env, use_planning: bool,
                        random_actions: bool) -> Optional[Dict[str, np.ndarray]]:
        if not self.cfg.pn_lmcgs_enabled:
            return self._collect_baseline_episode(
                _LegacyStepAPIProxy(env), use_planning, random_actions,
            )

        self.pn_last_collection_status = "collecting"
        obs_dict = env.reset()
        goal = obs_dict["desired_goal"].astype(np.float32)
        obs_buf = np.zeros((self.T + 1, self.env.obs_dim), dtype=np.float32)
        ag_buf = np.zeros((self.T + 1, self.env.goal_dim), dtype=np.float32)
        g_buf = np.zeros((self.T, self.env.goal_dim), dtype=np.float32)
        cmd_g_buf = np.zeros((self.T, self.env.goal_dim), dtype=np.float32)
        act_buf = np.zeros((self.T, self.env.act_dim), dtype=np.float32)
        cost_buf = np.zeros(self.T, dtype=np.float32)
        violation_buf = np.zeros(self.T, dtype=bool)
        goal_reached_buf = np.zeros(self.T, dtype=bool)
        task_goal_reached_buf = np.zeros(self.T, dtype=bool)
        terminated_buf = np.zeros(self.T, dtype=bool)
        truncated_buf = np.zeros(self.T, dtype=bool)
        reason_buf = np.full(self.T, "other", dtype=object)
        collision_adapter = CollisionCostAdapter(
            enabled=getattr(self.cfg, "pn_collision_cost_enabled", False),
            info_key=getattr(self.cfg, "pn_collision_cost_info_key", ""),
            unsafe_value=getattr(self.cfg, "pn_collision_cost_unsafe_value", None),
        )

        original_ready = use_planning and self.centroids_initialized
        if original_ready:
            extra = self._sample_random_landmarks(self.cfg.random_landmarks_train)
            self.planner.reset(goal, extra)

        episode_id = int(self.episodes_collected)
        t = 0
        stop_episode = False
        while t < self.T and not stop_episode:
            command = self._pn_select_command(
                obs_dict, goal, use_planning, random_actions, original_ready,
            )
            if command.goal is None:
                self.pn_last_collection_status = (
                    "no_safe_plan_empty" if t == 0 else "no_safe_plan_partial"
                )
                break
            command_goal = np.asarray(command.goal, dtype=np.float32).copy()
            horizon = int(np.clip(max(1, command.horizon),
                                  self.cfg.pn_k_min, self.cfg.pn_k_max))
            macro_start_t = t
            macro_start_goal = np.asarray(obs_dict["achieved_goal"], dtype=np.float32).copy()
            macro_violation = False
            macro_target_reached = False
            violation_goal = None

            for _ in range(horizon):
                if t >= self.T:
                    break
                obs_buf[t] = obs_dict["observation"]
                ag_buf[t] = obs_dict["achieved_goal"]
                g_buf[t] = goal
                cmd_g_buf[t] = command_goal
                action = self._pn_action(env, obs_dict, command, random_actions)
                act_buf[t] = action

                step = normalize_step_result(
                    env.step(action), collision_adapter=collision_adapter,
                )
                obs_dict = step.observation
                cost_buf[t] = step.safety_cost
                violation_buf[t] = step.safety_cost > 0.0
                goal_reached_buf[t] = bool(np.all(
                    np.asarray(env.compute_reward(
                        obs_dict["achieved_goal"], command_goal,
                    )) == 0.0
                ))
                task_goal_reached_buf[t] = bool(np.all(
                    np.asarray(env.compute_reward(
                        obs_dict["achieved_goal"], goal,
                    )) == 0.0
                ))
                terminated_buf[t] = step.terminated
                truncated_buf[t] = step.truncated
                if violation_buf[t]:
                    reason_buf[t] = "violation"
                elif task_goal_reached_buf[t] or goal_reached_buf[t]:
                    reason_buf[t] = "goal"
                else:
                    reason_buf[t] = step.info["termination_reason"]

                macro_violation = macro_violation or bool(violation_buf[t])
                macro_target_reached = macro_target_reached or bool(goal_reached_buf[t])
                if violation_buf[t]:
                    violation_goal = np.asarray(
                        obs_dict["achieved_goal"], dtype=np.float32,
                    ).copy()
                t += 1
                if (macro_target_reached or macro_violation
                        or task_goal_reached_buf[t - 1] or step.done or t >= self.T):
                    stop_episode = bool(
                        task_goal_reached_buf[t - 1] or step.done or t >= self.T
                    )
                    break

            macro_end_t = t - 1
            self.macro_buffer.add(MacroAttempt(
                start_goal=macro_start_goal,
                command_goal=command_goal,
                end_goal=np.asarray(obs_dict["achieved_goal"], dtype=np.float32).copy(),
                violation_goal=violation_goal,
                duration=macro_end_t - macro_start_t + 1,
                context=np.empty(0, dtype=np.float32),
                target_reached=macro_target_reached,
                violation=macro_violation,
                episode_id=episode_id,
                start_t=macro_start_t,
                end_t=macro_end_t,
            ))
            if command.source == "original" and hasattr(self.planner, "cnt"):
                self.planner.cnt = 0.0

        length = t
        if length == 0:
            return None
        if self.pn_last_collection_status == "collecting":
            self.pn_last_collection_status = "completed"
        obs_buf[length] = obs_dict["observation"]
        ag_buf[length] = obs_dict["achieved_goal"]
        return dict(
            obs=obs_buf[:length + 1], ag=ag_buf[:length + 1], g=g_buf[:length],
            cmd_g=cmd_g_buf[:length], act=act_buf[:length], cost=cost_buf[:length],
            violation=violation_buf[:length], goal_reached=goal_reached_buf[:length],
            task_goal_reached=task_goal_reached_buf[:length],
            terminated=terminated_buf[:length], truncated=truncated_buf[:length],
            termination_reason=reason_buf[:length], length=length,
            episode_id=episode_id,
        )

    def _collect_baseline_episode(self, env, use_planning: bool,
                                  random_actions: bool) -> Dict[str, np.ndarray]:
        obs_dict = env.reset()
        goal = obs_dict["desired_goal"].astype(np.float32)

        obs_buf = np.zeros((self.T + 1, self.env.obs_dim), dtype=np.float32)
        ag_buf = np.zeros((self.T + 1, self.env.goal_dim), dtype=np.float32)
        g_buf = np.zeros((self.T, self.env.goal_dim), dtype=np.float32)
        act_buf = np.zeros((self.T, self.env.act_dim), dtype=np.float32)

        planning = use_planning and self.centroids_initialized
        if planning:
            extra = self._sample_random_landmarks(self.cfg.random_landmarks_train)
            self.planner.reset(goal, extra)

        for t in range(self.T):
            obs_buf[t] = obs_dict["observation"]
            ag_buf[t] = obs_dict["achieved_goal"]
            g_buf[t] = goal
            if random_actions:
                a = self.rng.uniform(-env.max_action, env.max_action, size=self.env.act_dim)
            elif planning:
                a = self.planner.act(obs_dict["observation"], self.cfg.action_noise,
                                     self.cfg.random_action_prob)
            else:
                a = self.agent.act(obs_dict["observation"], goal, self.cfg.action_noise,
                                   self.cfg.random_action_prob)
            act_buf[t] = a
            obs_dict, _, done, _ = env.step(a)

        obs_buf[self.T] = obs_dict["observation"]
        ag_buf[self.T] = obs_dict["achieved_goal"]

        return dict(obs=obs_buf, ag=ag_buf, g=g_buf, act=act_buf)

    def collect(self) -> int:
        collected = 0
        for env in self.env.envs:
            random_actions = self.episodes_collected < self.cfg.initial_random_trajs * self.env.n
            use_planning = (
                True if self.cfg.pn_lmcgs_enabled
                else self.rng.random() < self.cfg.search_prob_train
            )
            ep = self.collect_episode(env, use_planning, random_actions)
            if ep is None:
                continue
            self.buffer.store_episode(ep)
            if self.cfg.pn_lmcgs_enabled:
                length = ep["length"]
                self.agent.update_normalizers(ep["obs"][:length], ep["g"][:length])
                episode_id = int(ep["episode_id"])
                if episode_id not in self.pn_ingested_episode_ids:
                    positive_contributor = self._pn_episode_contributes_positive(ep)
                    self.landmark_memory.ingest_episode(ep, episode_id)
                    self._pn_record_collected_episode(ep, env)
                    if positive_contributor:
                        self.pn_positive_episode_count += 1
                    self.pn_ingested_episode_ids.add(episode_id)
            else:
                self.agent.update_normalizers(ep["obs"][:-1], ep["g"])
            self.episodes_collected += 1
            env_steps = ep["length"] if self.cfg.pn_lmcgs_enabled else self.T
            self.total_env_steps += env_steps
            collected += env_steps
            if self.cfg.pn_lmcgs_enabled:
                self._refresh_pn_phase()

        # Initialize latent centroids via GLS once enough warm-up data is in.
        if not self.centroids_initialized and self.episodes_collected >= self.cfg.n_warmup_trajs:
            self._init_centroids()
            if self.cfg.pn_lmcgs_enabled:
                self._refresh_pn_phase()
        return collected

    def _pn_record_collected_episode(self, ep: Dict[str, np.ndarray], env) -> None:
        """Update cumulative diagnostics from one ingested real PN episode."""
        length = int(ep["length"])
        costs = np.asarray(ep["cost"], dtype=np.float32)[:length]
        achieved = np.asarray(ep["ag"], dtype=np.float32)[:length + 1]
        task_goal = np.asarray(ep["g"], dtype=np.float32)[length - 1]
        if "task_goal_reached" in ep:
            goal_success = bool(np.any(
                np.asarray(ep["task_goal_reached"], dtype=bool)[:length],
            ))
        else:
            final_reward = np.asarray(env.compute_reward(achieved[length], task_goal))
            goal_success = bool(np.all(final_reward == 0.0))
        episode_violation = bool(np.any(costs > 0.0))
        displacements = np.diff(achieved, axis=0)
        path_length = float(np.linalg.norm(displacements, axis=1).sum())

        self.pn_primitive_transition_count = int(getattr(
            self, "pn_primitive_transition_count", 0,
        )) + length
        self.pn_primitive_violation_count = int(getattr(
            self, "pn_primitive_violation_count", 0,
        )) + int(np.count_nonzero(costs > 0.0))
        self.pn_episode_goal_success_count = int(getattr(
            self, "pn_episode_goal_success_count", 0,
        )) + int(goal_success)
        self.pn_episode_safe_success_count = int(getattr(
            self, "pn_episode_safe_success_count", 0,
        )) + int(goal_success and not episode_violation)
        self.pn_violation_episode_count = int(getattr(
            self, "pn_violation_episode_count", 0,
        )) + int(episode_violation)
        self.pn_path_length_total = float(getattr(
            self, "pn_path_length_total", 0.0,
        )) + path_length

    @staticmethod
    def _pn_episode_contributes_positive(ep: Dict[str, np.ndarray]) -> bool:
        length = int(ep["length"])
        commands = np.asarray(ep["cmd_g"])[:length]
        reached = np.asarray(ep["goal_reached"], dtype=bool)[:length]
        costs = np.asarray(ep["cost"], dtype=np.float32)[:length]
        start = 0
        while start < length:
            end = start + 1
            while end < length and np.array_equal(commands[end], commands[start]):
                end += 1
            successful = np.flatnonzero(reached[start:end]) + start
            for transition in range(start, end):
                later = successful[successful >= transition]
                if len(later) and costs[transition:int(later[0]) + 1].sum() == 0.0:
                    return True
            start = end
        return False

    def _init_centroids(self) -> None:
        if self.cfg.pn_lmcgs_enabled:
            positive_ready = (
                self.landmark_memory.positive_size >= self.cfg.pn_min_positive_samples
                and self.pn_positive_episode_count >= self.cfg.pn_min_positive_episodes
            )
            if not positive_ready:
                return
            sample_size = max(self.cfg.gls_batch_size, self.landmarks.n_landmarks)
            goals = self.landmark_memory.sample_positive(sample_size, self.rng)
        else:
            goals = self.buffer.sample_achieved_goals(self.cfg.gls_batch_size, self.rng)
        with torch.no_grad():
            z = self.ae.encode(self.agent.to_tensor(goals))
            idx = greedy_latent_sparsification(z, self.landmarks.n_landmarks, self.rng)
            if len(idx) != self.landmarks.n_landmarks:
                raise ValueError("centroid initialization requires one sample per landmark")
            self.landmarks.centroids.data.copy_(z[idx])
        self.centroids_initialized = True
        if self.cfg.pn_lmcgs_enabled:
            self._pn_common_action_bucket_cache_key = None
            self._pn_common_action_bucket_cache = None

    # ------------------------------------------------------------------ optimization
    def _make_batch(self, raw: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        a = self.agent
        return dict(
            obs=a.norm_obs(raw["obs"]),
            next_obs=a.norm_obs(raw["next_obs"]),
            g=a.norm_goal(raw["g"]),
            act=a.to_tensor(raw["act"]),
            reward=a.to_tensor(raw["reward"]),
            next_ag=a.to_tensor(raw["next_ag"]),          # raw goal space for V
            future_ag=a.to_tensor(raw["future_ag"]),      # raw goal space for V
            future_ag_norm=a.norm_goal(raw["future_ag"]),  # normalized for the critic
            ag=a.to_tensor(raw["ag"]),                     # raw achieved goal for PN extensions
            cmd_g=a.to_tensor(raw["cmd_g"]),               # raw commanded goal for PN extensions
            ag_norm=a.norm_goal(raw["ag"]),
            cmd_g_norm=a.norm_goal(raw["cmd_g"]),
            cost=a.to_tensor(raw["cost"]),
            stop=a.to_tensor(raw["stop"]),
        )

    @staticmethod
    def _pn_finite_scalar(value, name: str) -> float:
        result = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @staticmethod
    def _pn_optional_finite_scalar(value) -> float:
        try:
            result = float(value.detach().item() if isinstance(value, torch.Tensor) else value)
        except (TypeError, ValueError, RuntimeError):
            return 0.0
        return result if np.isfinite(result) else 0.0

    @staticmethod
    def _pn_outcome_names() -> tuple[str, ...]:
        return ("target", "drift", "stuck", "violation")

    def _pn_neutral_validation_logs(self) -> Dict[str, float]:
        names = self._pn_outcome_names()
        result = {
            "macro_calibration_ece": 0.0,
            "macro_temperature": 0.0,
            "macro_predicted_violation": 0.0,
            "macro_empirical_violation": 0.0,
        }
        for name in names:
            result[f"macro_validation_{name}_frequency"] = 0.0
            result[f"macro_{name}_frequency"] = 0.0
            result[f"macro_precision_{name}"] = 0.0
            result[f"macro_recall_{name}"] = 0.0
            result[f"macro_duration_mae_{name}"] = 0.0
        for actual in names:
            for predicted in names:
                result[f"macro_confusion_{actual}_{predicted}"] = 0.0
        n_bins = max(1, int(getattr(self.cfg, "pn_calibration_bins", 10)))
        for index in range(n_bins):
            for suffix in ("count", "predicted", "empirical"):
                result[f"macro_reliability_{index}_{suffix}"] = 0.0
        return result

    def _pn_neutral_update_logs(self) -> Dict[str, float]:
        result = {
            "cost_critic": 0.0,
            "positive_elbo": 0.0,
            "negative_elbo": 0.0,
            "positive_centroid_movement": 0.0,
            "negative_centroid_movement": 0.0,
            "macro": 0.0,
            "macro_outcome": 0.0,
            "macro_positive": 0.0,
            "macro_negative": 0.0,
            "macro_duration": 0.0,
            "primitive_violation_rate": 0.0,
            "episode_violation_rate": 0.0,
            "safe_success_rate": 0.0,
            "positive_memory_size": 0.0,
            "negative_memory_size": 0.0,
            "negative_episode_count": 0.0,
            "positive_landmarks_active": 0.0,
            "negative_landmarks_active": 0.0,
            "active_positive_centroid_count": 0.0,
            "active_negative_centroid_count": 0.0,
            "positive_coverage": 0.0,
            "negative_samples_per_centroid": 0.0,
            "cost_critic_bce": 0.0,
            "cost_critic_auroc": 0.0,
            "cost_predicted_violation": 0.0,
            "cost_empirical_violation": 0.0,
            "macro_loss": 0.0,
            "macro_outcome_loss": 0.0,
            "macro_positive_loss": 0.0,
            "macro_negative_loss": 0.0,
            "macro_duration_loss": 0.0,
            "planner_latency": 0.0,
            "planner_branch_before": 0.0,
            "planner_branch_after": 0.0,
            "planner_no_safe_plan_rate": 0.0,
            "planner_chosen_p_violation": 0.0,
            "planner_chosen_q_risk": 0.0,
            "planner_chosen_upper_risk": 0.0,
            "planner_simulated_safe_success_rate": 0.0,
            "planner_simulation_count": 0.0,
            "planner_cycles": 0.0,
            "planner_leaf_uses": 0.0,
        }
        result.update(self._pn_neutral_validation_logs())
        return result

    @staticmethod
    def _pn_binary_auroc(prediction: np.ndarray, target: np.ndarray) -> float:
        scores = np.asarray(prediction, dtype=np.float64).reshape(-1)
        labels = np.asarray(target, dtype=bool).reshape(-1)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise ValueError("cost critic AUROC inputs must be paired finite vectors")
        positive = scores[labels]
        negative = scores[~labels]
        if positive.size == 0 or negative.size == 0:
            return 0.5
        comparisons = positive[:, None] - negative[None, :]
        return float((np.count_nonzero(comparisons > 0.0)
                      + 0.5 * np.count_nonzero(comparisons == 0.0))
                     / comparisons.size)

    def _pn_cost_diagnostics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with torch.no_grad():
            prediction = self.agent.cost_critic(
                batch["obs"], batch["act"], batch["g"],
            ).detach().cpu().numpy()
        empirical = (batch["cost"].detach().cpu().numpy() > 0.0)
        return {
            "cost_critic_auroc": self._pn_binary_auroc(prediction, empirical),
            "cost_predicted_violation": self._pn_finite_scalar(
                np.asarray(prediction).mean(), "cost predicted violation",
            ),
            "cost_empirical_violation": self._pn_finite_scalar(
                np.asarray(empirical, dtype=np.float32).mean(),
                "cost empirical violation",
            ),
        }

    def _pn_cumulative_diagnostics(self) -> Dict[str, float]:
        transitions = int(getattr(self, "pn_primitive_transition_count", 0))
        primitive_violations = int(getattr(self, "pn_primitive_violation_count", 0))
        episodes = int(getattr(self, "episodes_collected", 0))
        episode_violations = int(getattr(self, "pn_violation_episode_count", 0))
        safe_successes = int(getattr(self, "pn_episode_safe_success_count", 0))
        memory = getattr(self, "landmark_memory", None)
        positive_size = int(getattr(memory, "positive_size", 0))
        negative_size = int(getattr(memory, "negative_size", 0))
        negative_episodes = int(getattr(memory, "distinct_negative_episodes", 0))
        positive_sample_gate = max(1, int(getattr(
            self.cfg, "pn_min_positive_samples", 1,
        )))
        positive_episode_gate = max(1, int(getattr(
            self.cfg, "pn_min_positive_episodes", 1,
        )))
        positive_episodes = int(getattr(self, "pn_positive_episode_count", 0))
        positive_coverage = min(
            1.0,
            positive_size / positive_sample_gate,
            positive_episodes / positive_episode_gate,
        )
        negative_landmarks = getattr(self, "negative_landmarks", None)
        negative_centroids = max(1, int(getattr(negative_landmarks, "n_landmarks", 0)))
        planning_count = int(getattr(self, "pn_planning_count", 0))
        no_safe_plans = int(getattr(self, "pn_no_safe_plan_count", 0))
        planner = getattr(self, "pn_planner_diagnostic_sums", {})
        completed = self._pn_optional_finite_scalar(
            planner.get("completed_simulations", 0.0),
        )
        simulated_safe = self._pn_optional_finite_scalar(
            planner.get("simulated_safe_successes", 0.0),
        )
        return {
            "primitive_violation_rate": primitive_violations / max(1, transitions),
            "episode_violation_rate": episode_violations / max(1, episodes),
            "safe_success_rate": safe_successes / max(1, episodes),
            "positive_memory_size": float(positive_size),
            "negative_memory_size": float(negative_size),
            "negative_episode_count": float(negative_episodes),
            "positive_landmarks_active": float(bool(getattr(
                self, "pn_positive_active", False,
            ))),
            "negative_landmarks_active": float(bool(getattr(
                self, "pn_negative_active", False,
            ))),
            "active_positive_centroid_count": float(
                self.landmarks.n_landmarks if getattr(self, "pn_positive_active", False) else 0
            ),
            "active_negative_centroid_count": float(
                getattr(negative_landmarks, "n_landmarks", 0)
                if getattr(self, "pn_negative_active", False) else 0
            ),
            "positive_coverage": float(positive_coverage),
            "negative_samples_per_centroid": negative_size / negative_centroids,
            "macro_calibration_ece": self._pn_optional_finite_scalar(getattr(
                self, "pn_calibration_ece", 0.0,
            )),
            "macro_temperature": self._pn_optional_finite_scalar(getattr(
                self, "pn_calibration_temperature", 1.0,
            )),
            "planner_latency": self._pn_optional_finite_scalar(
                planner.get("planning_latency", 0.0),
            ) / max(1, planning_count),
            "planner_branch_before": self._pn_optional_finite_scalar(
                planner.get("branch_before", 0.0),
            ) / max(1, planning_count),
            "planner_branch_after": self._pn_optional_finite_scalar(
                planner.get("branch_after", 0.0),
            ) / max(1, planning_count),
            "planner_no_safe_plan_rate": no_safe_plans / max(1, planning_count),
            "planner_chosen_p_violation": self._pn_optional_finite_scalar(
                planner.get("chosen_p_violation", 0.0),
            ) / max(1, planning_count),
            "planner_chosen_q_risk": self._pn_optional_finite_scalar(
                planner.get("chosen_q_risk", 0.0),
            ) / max(1, planning_count),
            "planner_chosen_upper_risk": self._pn_optional_finite_scalar(
                planner.get("chosen_upper_risk", 0.0),
            ) / max(1, planning_count),
            "planner_simulated_safe_success_rate": simulated_safe / max(1.0, completed),
            "planner_simulation_count": completed / max(1, planning_count),
            "planner_cycles": self._pn_optional_finite_scalar(
                planner.get("cycles", 0.0),
            ) / max(1, planning_count),
            "planner_leaf_uses": self._pn_optional_finite_scalar(
                planner.get("leaf_uses", 0.0),
            ) / max(1, planning_count),
        }

    def _pn_flatten_validation_diagnostics(
            self, output, labels: MacroLabels, temperature: float,
    ) -> Dict[str, float]:
        result = self._pn_neutral_validation_logs()
        names = self._pn_outcome_names()
        probabilities = torch.softmax(output.outcome_logits / temperature, dim=-1)
        prediction = probabilities.argmax(dim=-1)
        metrics = confusion_matrix_and_classification_metrics(prediction, labels.outcome)
        selected_duration = output.duration_by_outcome.gather(
            1, labels.outcome[:, None],
        ).squeeze(1)
        duration_mae = duration_mae_by_outcome(
            labels.outcome, selected_duration, labels.duration,
        )
        reliability = violation_reliability_bins(
            output.outcome_logits,
            labels.outcome,
            n_bins=max(1, int(getattr(self.cfg, "pn_calibration_bins", 10))),
            temperature=temperature,
        )

        for index, name in enumerate(names):
            frequency = (labels.outcome == index).float().mean()
            value = self._pn_finite_scalar(frequency, f"macro validation {name} frequency")
            result[f"macro_validation_{name}_frequency"] = value
            result[f"macro_{name}_frequency"] = value
            result[f"macro_precision_{name}"] = self._pn_finite_scalar(
                metrics["precision"][index], f"macro {name} precision",
            )
            result[f"macro_recall_{name}"] = self._pn_finite_scalar(
                metrics["recall"][index], f"macro {name} recall",
            )
            result[f"macro_duration_mae_{name}"] = self._pn_finite_scalar(
                duration_mae[index], f"macro {name} duration MAE",
            )
            for predicted_index, predicted_name in enumerate(names):
                result[f"macro_confusion_{name}_{predicted_name}"] = float(
                    metrics["matrix"][index, predicted_index]
                )
        violation_probability = probabilities[:, int(Outcome.VIOLATION)]
        result["macro_predicted_violation"] = self._pn_finite_scalar(
            violation_probability.mean(), "macro predicted violation",
        )
        result["macro_empirical_violation"] = self._pn_finite_scalar(
            (labels.outcome == int(Outcome.VIOLATION)).float().mean(),
            "macro empirical violation",
        )
        for index in range(len(reliability["count"])):
            for suffix in ("count", "predicted", "empirical"):
                result[f"macro_reliability_{index}_{suffix}"] = self._pn_finite_scalar(
                    reliability[suffix][index], f"macro reliability {index} {suffix}",
                )
        return result

    def _pn_validate_logs(self, logs: Dict[str, float]) -> Dict[str, float]:
        result = {}
        for name, value in logs.items():
            if not np.isscalar(value):
                raise ValueError(f"update log {name!r} must be scalar")
            result[name] = self._pn_finite_scalar(value, f"update log {name}")
        return result

    def _pn_update_centroids(self) -> Dict[str, float]:
        result = {
            "positive_elbo": 0.0,
            "negative_elbo": 0.0,
            "positive_centroid_movement": 0.0,
            "negative_centroid_movement": 0.0,
        }
        required = ("landmark_memory", "landmarks", "landmark_opt", "ae", "agent", "rng")
        if not self.centroids_initialized or not all(hasattr(self, name) for name in required):
            return result

        updates = []
        if getattr(self, "pn_positive_active", False):
            updates.append((
                "positive", self.landmark_memory.sample_positive, self.landmarks,
                self.landmark_opt,
                "cannot sample positive landmarks from an empty memory",
            ))
        if (getattr(self, "pn_negative_active", False)
                and hasattr(self, "negative_landmarks")
                and hasattr(self, "negative_landmark_opt")):
            updates.append((
                "negative", lambda n, rng: self.landmark_memory.sample_negative(n, rng)["goals"],
                self.negative_landmarks, self.negative_landmark_opt,
                "cannot sample negative landmarks from an empty memory",
            ))

        for prefix, sample, landmarks, optimizer, empty_message in updates:
            try:
                goals = sample(self.cfg.gls_batch_size, self.rng)
            except ValueError as exc:
                if str(exc) == empty_message:
                    continue
                raise
            with torch.no_grad():
                z_all = self.ae.encode(self.agent.to_tensor(goals))
                indices = greedy_latent_sparsification(
                    z_all, self.cfg.landmark_batch_size, self.rng,
                )
                z = z_all[indices].detach()
            before = landmarks.centroids.detach().clone()
            loss = landmarks.elbo_loss(z)
            loss_value = self._pn_finite_scalar(loss, f"{prefix} landmark ELBO")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if prefix == "positive":
                self._pn_common_action_bucket_cache_key = None
                self._pn_common_action_bucket_cache = None
            movement = torch.linalg.vector_norm(landmarks.centroids.detach() - before)
            result[f"{prefix}_elbo"] = loss_value
            result[f"{prefix}_centroid_movement"] = self._pn_finite_scalar(
                movement, f"{prefix} centroid movement",
            )
        return result

    def _pn_macro_centroids(self) -> tuple[torch.Tensor, torch.Tensor]:
        positive = self.landmarks.centroids.detach()
        if getattr(self, "pn_negative_active", False) and hasattr(self, "negative_landmarks"):
            negative = self.negative_landmarks.centroids.detach()
        else:
            negative = positive.new_empty((0, positive.shape[1]))
        return positive, negative

    def _pn_label_macro_entries(self, attempts: list[MacroAttempt]) -> MacroLabels:
        positive, negative = self._pn_macro_centroids()
        return label_macro_attempts(
            attempts,
            self.ae.encode,
            positive,
            negative,
            self.cfg.pn_positive_assignment_radius,
            bool(getattr(self, "pn_negative_active", False)),
            device=self.device,
        )

    def _pn_macro_inputs(
            self, attempts: list[MacroAttempt],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = np.asarray([attempt.start_goal for attempt in attempts], dtype=np.float32)
        commands = np.asarray([attempt.command_goal for attempt in attempts], dtype=np.float32)
        context = np.asarray([attempt.context for attempt in attempts], dtype=np.float32)
        with torch.no_grad():
            z_start = self.ae.encode(self.agent.to_tensor(starts)).detach()
            z_command = self.ae.encode(self.agent.to_tensor(commands)).detach()
            context_t = torch.as_tensor(
                context, dtype=torch.float32, device=self.device,
            ).detach()
        return z_start, z_command, context_t

    def _pn_update_macro_model(self) -> Dict[str, float]:
        result = {
            "macro": 0.0,
            "macro_outcome": 0.0,
            "macro_positive": 0.0,
            "macro_negative": 0.0,
            "macro_duration": 0.0,
        }
        if getattr(self, "pn_phase", PNPhase.WARMUP) not in (
                PNPhase.MACRO_BOOTSTRAP, PNPhase.JOINT):
            return result
        required = (
            "macro_buffer", "macro_model", "macro_opt", "landmarks", "ae", "agent", "device",
        )
        if not all(hasattr(self, name) for name in required):
            return result

        train_size = self.macro_buffer.train_size
        if train_size == 0:
            return result
        candidate_pool_size = min(
            train_size, 4 * self.cfg.pn_macro_batch_size,
        )
        train_attempts = self.macro_buffer.sample_train_attempts(
            candidate_pool_size, self.rng,
        )
        train_labels = self._pn_label_macro_entries(train_attempts)
        selected = stratified_macro_indices(
            train_labels.outcome.detach().cpu().numpy(),
            self.cfg.pn_macro_batch_size,
            self.rng,
        )
        batch_attempts = [train_attempts[int(index)] for index in selected]
        index_t = torch.as_tensor(selected, dtype=torch.long, device=self.device)
        labels = MacroLabels(
            train_labels.outcome[index_t],
            train_labels.positive_id[index_t],
            train_labels.negative_id[index_t],
            train_labels.duration[index_t],
        )
        z_start, z_command, context = self._pn_macro_inputs(batch_attempts)
        positive, negative = self._pn_macro_centroids()
        was_training = self.macro_model.training
        self.macro_model.train()
        output = self.macro_model(
            z_start, z_command, context, positive.detach(), negative.detach(),
        )
        loss = macro_model_loss(
            output, labels,
            beta_duration=self.cfg.pn_beta_duration,
            k_max=self.cfg.pn_k_max,
        )
        loss_values = {
            name: self._pn_finite_scalar(value, name)
            for name, value in (
                ("macro", loss.total),
                ("macro_outcome", loss.outcome),
                ("macro_positive", loss.positive),
                ("macro_negative", loss.negative),
                ("macro_duration", loss.duration),
            )
        }
        self.macro_opt.zero_grad()
        loss.total.backward()
        self.macro_opt.step()
        self.macro_opt.zero_grad()
        self.macro_model.train(was_training)
        result.update(loss_values)
        return result

    def _pn_calibrate_macro_model(self) -> Dict[str, float]:
        result = dict(self.pn_last_validation_diagnostics)
        required = ("macro_buffer", "macro_model", "landmarks", "ae", "agent", "device")
        if (getattr(self, "pn_phase", PNPhase.WARMUP) not in (
                    PNPhase.MACRO_BOOTSTRAP, PNPhase.JOINT)
                or not all(hasattr(self, name) for name in required)
                or self.grad_step_count % self.cfg.pn_calibration_interval != 0):
            return result

        attempts = self.macro_buffer.attempts
        validation_indices = self.macro_buffer.validation_indices
        if validation_indices.size == 0:
            return result
        validation_attempts = [attempts[int(index)] for index in validation_indices]
        labels = self._pn_label_macro_entries(validation_attempts)
        z_start, z_command, context = self._pn_macro_inputs(validation_attempts)
        positive, negative = self._pn_macro_centroids()
        was_training = self.macro_model.training
        self.macro_model.eval()
        with torch.no_grad():
            output = self.macro_model(
                z_start, z_command, context, positive.detach(), negative.detach(),
            )
        self.macro_model.train(was_training)

        if self.cfg.pn_calibrate_temperature:
            calibration = fit_temperature(output.outcome_logits, labels.outcome)
            temperature = self._pn_finite_scalar(
                calibration.temperature, "calibration temperature",
            )
            before_nll = self._pn_finite_scalar(
                calibration.before_nll, "calibration before NLL",
            )
            after_nll = self._pn_finite_scalar(
                calibration.after_nll, "calibration after NLL",
            )
        else:
            temperature = 1.0
            natural_nll = torch.nn.functional.cross_entropy(
                output.outcome_logits, labels.outcome,
            )
            before_nll = self._pn_finite_scalar(
                natural_nll, "uncalibrated validation NLL",
            )
            after_nll = before_nll
        if temperature <= 0.0:
            raise ValueError("calibration temperature must be positive")
        ece = expected_calibration_error(
            output.outcome_logits,
            labels.outcome,
            temperature=temperature,
            n_bins=max(1, int(getattr(self.cfg, "pn_calibration_bins", 10))),
        )
        self.pn_calibration_ready = True
        self.pn_calibration_temperature = temperature
        self.pn_calibration_before_nll = before_nll
        self.pn_calibration_after_nll = after_nll
        self.pn_calibration_ece = self._pn_finite_scalar(ece, "calibration ECE")
        self.macro_model.set_temperature(temperature)
        result.update(self._pn_flatten_validation_diagnostics(
            output, labels, temperature,
        ))
        result["macro_calibration_ece"] = self.pn_calibration_ece
        result["macro_temperature"] = temperature
        self.pn_last_validation_diagnostics = dict(result)
        if all(hasattr(self, name) for name in (
                "landmark_memory", "macro_buffer", "episodes_collected",
                "pn_positive_episode_count")):
            self._refresh_pn_phase()
        return result

    def update(self, n_steps: int) -> Dict[str, float]:
        logs = {"critic": 0.0, "value": 0.0, "actor": 0.0, "ae_rec": 0.0,
                "ae_latent": 0.0, "elbo": 0.0}
        if self.cfg.pn_lmcgs_enabled:
            logs.update(self._pn_neutral_update_logs())
        for _ in range(n_steps):
            raw = self.buffer.sample(self.cfg.batch_size, self.rng)
            batch = self._make_batch(raw)
            cost_critic_ready = (
                self.cfg.pn_lmcgs_enabled
                and self.total_env_steps >= self.cfg.pn_cost_critic_train_after
            )

            logs["value"] += self.agent.update_value(batch)
            logs["critic"] += self.agent.update_critic(batch)
            if cost_critic_ready:
                if self.cfg.pn_cost_critic_batch_size == self.cfg.batch_size:
                    cost_batch = batch
                else:
                    cost_raw = self.buffer.sample(self.cfg.pn_cost_critic_batch_size, self.rng)
                    cost_batch = self._make_batch(cost_raw)
                logs["cost_critic"] += self.agent.update_cost_critic(cost_batch)
                for name, value in self._pn_cost_diagnostics(cost_batch).items():
                    logs[name] += value
            logs["actor"] += self.agent.update_actor(batch, use_cost_critic=cost_critic_ready)

            # Auto-encoder (Eq. 2) on a fresh batch of achieved goals.
            goals = self.buffer.sample_achieved_goals(self.cfg.batch_size, self.rng)
            goals_t = self.agent.to_tensor(goals)
            l_rec, l_latent, ae_total = ae_losses(self.ae, self.agent.value, goals_t,
                                                  self.cfg.ae_lambda)
            self.ae_opt.zero_grad()
            ae_total.backward()
            self.ae_opt.step()
            if self.cfg.pn_lmcgs_enabled:
                self._pn_common_action_bucket_cache_key = None
                self._pn_common_action_bucket_cache = None
            logs["ae_rec"] += float(l_rec.item())
            logs["ae_latent"] += float(l_latent.item())

            # Latent centroids (ELBO, Eq. 5) on a GLS-sparsified batch.
            if self.cfg.pn_lmcgs_enabled:
                centroid_logs = self._pn_update_centroids()
                for name, value in centroid_logs.items():
                    logs[name] += value
                logs["elbo"] += centroid_logs["positive_elbo"]
                if getattr(self, "pn_phase", PNPhase.WARMUP) in (
                        PNPhase.MACRO_BOOTSTRAP, PNPhase.JOINT):
                    macro_logs = self._pn_update_macro_model()
                    for name, value in macro_logs.items():
                        logs[name] += value
            elif self.centroids_initialized:
                cg = self.buffer.sample_achieved_goals(self.cfg.gls_batch_size, self.rng)
                with torch.no_grad():
                    z_all = self.ae.encode(self.agent.to_tensor(cg))
                idx = greedy_latent_sparsification(z_all, self.cfg.landmark_batch_size, self.rng)
                z = z_all[idx].detach()
                elbo = self.landmarks.elbo_loss(z)
                self.landmark_opt.zero_grad()
                elbo.backward()
                self.landmark_opt.step()
                logs["elbo"] += float(elbo.item())

            self.grad_step_count += 1
            if self.grad_step_count % self.cfg.target_update_interval == 0:
                self.agent.update_targets()
            if (self.cfg.pn_lmcgs_enabled
                    and getattr(self, "pn_phase", PNPhase.WARMUP) in (
                        PNPhase.MACRO_BOOTSTRAP, PNPhase.JOINT)):
                calibration_logs = self._pn_calibrate_macro_model()
                for name, value in calibration_logs.items():
                    logs[name] += value

        averaged = {k: v / max(1, n_steps) for k, v in logs.items()}
        if self.cfg.pn_lmcgs_enabled:
            for alias, source in (
                ("cost_critic_bce", "cost_critic"),
                ("macro_loss", "macro"),
                ("macro_outcome_loss", "macro_outcome"),
                ("macro_positive_loss", "macro_positive"),
                ("macro_negative_loss", "macro_negative"),
                ("macro_duration_loss", "macro_duration"),
            ):
                averaged[alias] = averaged[source]
            averaged.update(self._pn_cumulative_diagnostics())
            averaged.update(getattr(
                self, "pn_last_validation_diagnostics",
                self._pn_neutral_validation_logs(),
            ))
        return self._pn_validate_logs(averaged)

    # ------------------------------------------------------------------ evaluation
    def _pn_modules_with_modes(self) -> tuple[torch.nn.Module, ...]:
        modules = [
            self.agent.actor, self.agent.actor_target,
            self.agent.critic, self.agent.critic_target, self.agent.value,
            self.ae, self.landmarks, self.macro_model, self.negative_landmarks,
        ]
        if hasattr(self.agent, "cost_critic"):
            modules.extend((self.agent.cost_critic, self.agent.cost_critic_target))
        unique = {}
        for module in modules:
            unique[id(module)] = module
        return tuple(unique.values())

    def _pn_environment_numpy_rngs(self) -> list[list[tuple[tuple[str, ...], object]]]:
        environments = getattr(self.env, "envs", ())
        discovered = []
        for environment in environments:
            queue = [((), environment)]
            visited_objects = set()
            visited_rngs = set()
            handles = []
            while queue:
                path, current = queue.pop(0)
                if id(current) in visited_objects:
                    continue
                visited_objects.add(id(current))
                for attribute in ("rng", "np_random"):
                    try:
                        candidate = getattr(current, attribute)
                    except (AttributeError, RuntimeError):
                        continue
                    if (isinstance(candidate, (np.random.Generator, np.random.RandomState))
                            and id(candidate) not in visited_rngs):
                        visited_rngs.add(id(candidate))
                        handles.append((path + (attribute,), candidate))
                for attribute in ("env", "unwrapped"):
                    try:
                        child = getattr(current, attribute)
                    except (AttributeError, RuntimeError):
                        continue
                    if child is not current and id(child) not in visited_objects:
                        queue.append((path + (attribute,), child))
            handles.sort(key=lambda item: item[0])
            discovered.append(handles)
        return discovered

    def _pn_environment_rng_state(self) -> list[list[dict]]:
        state = []
        for handles in self._pn_environment_numpy_rngs():
            worker_state = []
            for path, rng in handles:
                if isinstance(rng, np.random.Generator):
                    kind = "Generator"
                    rng_state = rng.bit_generator.state
                else:
                    kind = "RandomState"
                    rng_state = rng.get_state()
                worker_state.append({
                    "path": path,
                    "kind": kind,
                    "state": copy.deepcopy(rng_state),
                })
            state.append(worker_state)
        return state

    def _pn_prevalidate_environment_rng_state(self, state: object) -> None:
        handles_by_worker = self._pn_environment_numpy_rngs()
        if not isinstance(state, (list, tuple)):
            raise ValueError("PN checkpoint environment RNG worker state is incompatible")
        for worker_index, worker_state in enumerate(state):
            if not isinstance(worker_state, (list, tuple)):
                raise ValueError(
                    f"PN checkpoint environment RNG worker {worker_index} state is incompatible"
                )
            for record in worker_state:
                if not isinstance(record, dict) or set(record) != {"path", "kind", "state"}:
                    raise ValueError("PN checkpoint environment RNG record is incomplete")
                path = record["path"]
                if (not isinstance(path, (list, tuple))
                        or not all(isinstance(part, str) for part in path)):
                    raise ValueError("PN checkpoint environment RNG path is incompatible")
                if record["kind"] not in {"Generator", "RandomState"}:
                    raise ValueError("PN checkpoint environment RNG type is incompatible")
                try:
                    rng_state = copy.deepcopy(record["state"])
                    if record["kind"] == "Generator":
                        bit_generator_name = (
                            rng_state.get("bit_generator")
                            if isinstance(rng_state, dict) else None
                        )
                        bit_generator_type = (
                            getattr(np.random, bit_generator_name, None)
                            if isinstance(bit_generator_name, str) else None
                        )
                        if (not isinstance(bit_generator_type, type)
                                or not issubclass(bit_generator_type, np.random.BitGenerator)):
                            raise ValueError
                        probe = np.random.Generator(bit_generator_type())
                        probe.bit_generator.state = rng_state
                    else:
                        probe = np.random.RandomState()
                        probe.set_state(rng_state)
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError("PN checkpoint environment NumPy RNG state is invalid") from exc
        for worker_index, (worker_state, handles) in enumerate(zip(state, handles_by_worker)):
            if len(worker_state) != len(handles):
                raise ValueError(
                    f"PN checkpoint environment RNG worker {worker_index} state is incompatible"
                )
            for record, (expected_path, rng) in zip(worker_state, handles):
                if tuple(record["path"]) != expected_path:
                    raise ValueError("PN checkpoint environment RNG path is incompatible")
                expected_kind = (
                    "Generator" if isinstance(rng, np.random.Generator) else "RandomState"
                )
                if record["kind"] != expected_kind:
                    raise ValueError("PN checkpoint environment RNG type is incompatible")
                if isinstance(rng, np.random.Generator):
                    saved_bit_generator = record["state"].get("bit_generator")
                    if saved_bit_generator != type(rng.bit_generator).__name__:
                        raise ValueError("PN checkpoint environment RNG type is incompatible")
                try:
                    probe = copy.deepcopy(rng)
                    if isinstance(probe, np.random.Generator):
                        probe.bit_generator.state = copy.deepcopy(record["state"])
                    else:
                        probe.set_state(copy.deepcopy(record["state"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "PN checkpoint environment NumPy RNG state is invalid"
                    ) from exc

    def _pn_restore_environment_rng_state(self, state: object) -> None:
        self._pn_prevalidate_environment_rng_state(state)
        for worker_state, handles in zip(state, self._pn_environment_numpy_rngs()):
            for record, (_path, rng) in zip(worker_state, handles):
                if isinstance(rng, np.random.Generator):
                    rng.bit_generator.state = copy.deepcopy(record["state"])
                else:
                    rng.set_state(copy.deepcopy(record["state"]))

    @contextmanager
    def _pn_isolated_evaluation_state(self):
        trainer_rng = copy.deepcopy(self.rng.bit_generator.state)
        global_numpy_rng = copy.deepcopy(np.random.get_state())
        environment_rng = self._pn_environment_rng_state()
        torch_rng = torch.get_rng_state().clone()
        cuda_rng = (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        )
        planner_state = self.planner.__dict__.copy()
        adapter_state = self.pn_planner.__dict__.copy()
        search = self.pn_planner.search
        search_state = search.__dict__.copy()
        modules = self._pn_modules_with_modes()
        module_modes = tuple(module.training for module in modules)
        try:
            yield
        finally:
            self.rng.bit_generator.state = copy.deepcopy(trainer_rng)
            np.random.set_state(global_numpy_rng)
            self._pn_restore_environment_rng_state(environment_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            self.planner.__dict__.clear()
            self.planner.__dict__.update(planner_state)
            search.__dict__.clear()
            search.__dict__.update(search_state)
            self.pn_planner.__dict__.clear()
            self.pn_planner.__dict__.update(adapter_state)
            for module, training in zip(modules, module_modes):
                module.train(training)

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 20, use_planning: bool = True) -> float:
        if not self.cfg.pn_lmcgs_enabled:
            self.env.set_eval(True)
            try:
                return self._evaluate_legacy(n_episodes, use_planning)
            finally:
                self.env.set_eval(False)
        with self._pn_isolated_evaluation_state():
            self.env.set_eval(True)
            try:
                return self._evaluate_pn(n_episodes, use_planning)
            finally:
                self.env.set_eval(False)

    def _evaluate_legacy(self, n_episodes: int, use_planning: bool) -> float:
        """Keep the baseline evaluation loop unchanged across Gym step APIs."""
        env = _LegacyStepAPIProxy(self.env.envs[0])
        successes = 0
        for _ in range(n_episodes):
            obs_dict = env.reset()
            goal = obs_dict["desired_goal"].astype(np.float32)
            planning = use_planning and self.centroids_initialized
            if planning:
                self.planner.reset(goal)
            success = 0.0
            for _ in range(self.cfg.test_episode_steps):
                if planning:
                    action = self.planner.act(obs_dict["observation"])
                else:
                    action = self.agent.act(obs_dict["observation"], goal)
                obs_dict, reward, _done, info = env.step(action)
                success = max(success, _episode_success(info, reward))
            successes += int(success > 0)
        return successes / n_episodes

    def _evaluate_pn(self, n_episodes: int, use_planning: bool) -> float:
        """Evaluate PN commands without changing training state or replay."""
        env = self.env.envs[0]
        collision_adapter = CollisionCostAdapter(
            enabled=getattr(self.cfg, "pn_collision_cost_enabled", False),
            info_key=getattr(self.cfg, "pn_collision_cost_info_key", ""),
            unsafe_value=getattr(self.cfg, "pn_collision_cost_unsafe_value", None),
        )
        goal_successes = 0
        safe_successes = 0
        violation_episodes = 0
        path_length_total = 0.0
        plan_calls = 0
        no_safe_plans = 0
        planning_latency_total = 0.0

        for _ in range(n_episodes):
            obs_dict = env.reset()
            final_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32).copy()
            original_ready = use_planning and self.centroids_initialized
            if original_ready:
                self.planner.reset(final_goal)

            episode_violation = False
            task_goal_success = False
            primitive_steps = 0
            while primitive_steps < self.cfg.test_episode_steps:
                state = np.asarray(obs_dict["observation"], dtype=np.float32)
                achieved = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
                if self.pn_phase == PNPhase.JOINT and original_ready:
                    plan_calls += 1
                    command = self.pn_planner.planned_command(
                        state, achieved, final_goal, context=None, training=False,
                    )
                    diagnostics = command.diagnostics or {}
                    try:
                        latency = float(diagnostics.get("latency", 0.0))
                    except (TypeError, ValueError):
                        latency = 0.0
                    planning_latency_total += latency if np.isfinite(latency) else 0.0
                elif original_ready:
                    command = self._pn_original_command(state)
                else:
                    command = self.pn_planner.direct_command(state, final_goal)

                if command.goal is None:
                    no_safe_plans += 1
                    break

                command_goal = np.asarray(command.goal, dtype=np.float32)
                horizon = min(
                    max(1, int(command.horizon)),
                    self.cfg.pn_k_max,
                    self.cfg.test_episode_steps - primitive_steps,
                )
                end_episode = False
                for _ in range(horizon):
                    start_goal = np.asarray(obs_dict["achieved_goal"], dtype=np.float32).copy()
                    if command.source == "original":
                        if hasattr(self.planner, "act_for_current_command"):
                            action = self.planner.act_for_current_command(
                                obs_dict["observation"], 0.0, 0.0,
                            )
                        else:
                            action = self.planner.act(obs_dict["observation"], 0.0, 0.0)
                    else:
                        action = self.agent.act(obs_dict["observation"], command_goal, 0.0, 0.0)

                    step = normalize_step_result(
                        env.step(action), collision_adapter=collision_adapter,
                    )
                    obs_dict = step.observation
                    primitive_steps += 1
                    end_goal = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
                    path_length_total += float(np.linalg.norm(end_goal - start_goal))
                    violation = step.safety_cost > 0.0
                    episode_violation = episode_violation or violation
                    command_success = bool(np.all(np.asarray(
                        env.compute_reward(end_goal, command_goal),
                    ) == 0.0))
                    task_goal_success = task_goal_success or bool(np.all(np.asarray(
                        env.compute_reward(end_goal, final_goal),
                    ) == 0.0))
                    if command_success or violation or task_goal_success or step.done:
                        end_episode = task_goal_success or step.done
                        break

                if end_episode:
                    break

            final_reward = np.asarray(env.compute_reward(
                np.asarray(obs_dict["achieved_goal"], dtype=np.float32), final_goal,
            ))
            goal_success = task_goal_success or bool(np.all(final_reward == 0.0))
            goal_successes += int(goal_success)
            safe_successes += int(goal_success and not episode_violation)
            violation_episodes += int(episode_violation)

        denominator = max(1, n_episodes)
        self.last_eval_metrics = {
            "goal_success_rate": float(goal_successes / denominator),
            "safe_success_rate": float(safe_successes / denominator),
            "episode_violation_rate": float(violation_episodes / denominator),
            "path_length": float(path_length_total / denominator),
            "no_safe_plan_rate": float(no_safe_plans / max(1, plan_calls)),
            "planning_latency": float(planning_latency_total / max(1, plan_calls)),
        }
        return float(goal_successes / denominator)

    # ------------------------------------------------------------------ main loop
    def _emit_metrics(self, event: str, metrics: Dict[str, float]) -> None:
        """Send finite scalar diagnostics to an optional external metrics sink."""
        if self.metrics_callback is None:
            return
        scalars = {}
        for name, value in metrics.items():
            scalar = float(value)
            if not np.isfinite(scalar):
                raise ValueError(f"metric {name!r} must be finite")
            scalars[str(name)] = scalar
        self.metrics_callback(str(event), int(self.total_env_steps), scalars)

    def train(self, total_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None, checkpoint_every: int = 0) -> None:
        total_steps = total_steps or self.cfg.total_steps
        t0 = time.time()
        last_log = 0
        last_ckpt = 0
        while self.total_env_steps < total_steps:
            collected = self.collect()
            if (self.cfg.pn_lmcgs_enabled and collected == 0
                    and self.pn_last_collection_status == "no_safe_plan_empty"):
                raise RuntimeError("no safe plan available for PN collection")

            if self.total_env_steps >= self.cfg.train_after and len(self.buffer) > 1:
                env_steps_per_opt = max(1, self.cfg.env_steps_per_opt)
                n_updates = max(1, collected // env_steps_per_opt)
                logs = self.update(n_updates)
            else:
                logs = {}

            if self.total_env_steps - last_log >= self.cfg.log_interval:
                last_log = self.total_env_steps
                msg = (f"[{self.total_env_steps:>8d} steps | {self.episodes_collected} eps | "
                       f"{time.time() - t0:5.1f}s] landmarks={'on' if self.centroids_initialized else 'off'}")
                if logs:
                    msg += " | " + " ".join(f"{k}={v:.3f}" for k, v in logs.items())
                print(msg, flush=True)
                train_metrics = dict(logs)
                train_metrics.update(
                    episodes_collected=float(self.episodes_collected),
                    centroids_initialized=float(self.centroids_initialized),
                    elapsed_seconds=float(time.time() - t0),
                )
                self._emit_metrics("train", train_metrics)

            if self.total_env_steps % self.cfg.eval_interval < self.env.n * self.T:
                sr = self.evaluate(self.cfg.eval_episodes)
                print(f"    >> eval success rate (long-horizon test): {sr:.2f}", flush=True)
                evaluation_metrics = {"success_rate": float(sr)}
                if self.cfg.pn_lmcgs_enabled:
                    evaluation_metrics.update(self.last_eval_metrics)
                self._emit_metrics("evaluation", evaluation_metrics)

            if checkpoint_path and checkpoint_every and \
                    self.total_env_steps - last_ckpt >= checkpoint_every:
                last_ckpt = self.total_env_steps
                self.save(checkpoint_path)
                print(f"    >> checkpoint saved @ {self.total_env_steps} steps", flush=True)
                self._emit_metrics("checkpoint", {"checkpoint_saved": 1.0})

    def _pn_current_config_snapshot(self) -> dict:
        state = vars(self.cfg)
        snapshot = copy.deepcopy({name: state[name] for name in sorted(state)})
        if not snapshot:
            raise ValueError("enabled PN trainer requires a nonempty configuration state")
        self.pn_config_snapshot = snapshot
        return copy.deepcopy(snapshot)

    def _pn_validate_config_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, dict) or not snapshot:
            raise ValueError("PN checkpoint config_snapshot is incomplete")
        current = vars(self.cfg)
        if set(snapshot) != set(current):
            missing = sorted(set(current).difference(snapshot))
            unexpected = sorted(set(snapshot).difference(current))
            raise ValueError(
                "PN checkpoint config_snapshot schema is incomplete; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, value in snapshot.items():
            current_value = current[name]
            if isinstance(current_value, bool):
                valid_type = isinstance(value, (bool, np.bool_))
            elif isinstance(current_value, int):
                valid_type = not isinstance(value, (bool, np.bool_)) and isinstance(
                    value, (int, np.integer),
                )
            elif isinstance(current_value, float):
                valid_type = not isinstance(value, (bool, np.bool_)) and isinstance(
                    value, (int, float, np.integer, np.floating),
                )
            elif isinstance(current_value, str):
                valid_type = isinstance(value, str)
            else:
                valid_type = True
            if not valid_type:
                raise ValueError(f"PN checkpoint config field {name!r} has invalid type")
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                raise ValueError(f"PN checkpoint config field {name!r} must be finite")

        checkpoint_cfg = copy.deepcopy(self.cfg)
        for name, value in snapshot.items():
            setattr(checkpoint_cfg, name, copy.deepcopy(value))
        validate_pn_config(checkpoint_cfg)

        if self.cfg.pn_lmcgs_enabled:
            restored_cfg = copy.deepcopy(self.cfg)
            for name, value in snapshot.items():
                if name not in _PN_RUNTIME_CONFIG_FIELDS:
                    setattr(restored_cfg, name, copy.deepcopy(value))
            validate_pn_config(restored_cfg)

        fields = set(_PN_STRUCTURAL_CONFIG_FIELDS)
        if not self.cfg.pn_lmcgs_enabled:
            fields = {
                "hidden_units", "hidden_layers", "ae_hidden_units",
                "ae_hidden_layers", "embedding_size", "max_episode_steps",
            }
        for name in sorted(fields):
            if name not in snapshot or name not in current:
                raise ValueError(f"PN checkpoint config is missing structural field {name!r}")
            if _state_fingerprint(snapshot[name]) != _state_fingerprint(current[name]):
                raise ValueError(
                    f"PN checkpoint config field {name!r} is incompatible with trainer"
                )

    def _pn_restore_config_snapshot(self, snapshot: dict) -> None:
        current = vars(self.cfg)
        for name, value in snapshot.items():
            if name in current and name not in _PN_RUNTIME_CONFIG_FIELDS:
                setattr(self.cfg, name, copy.deepcopy(value))
        self.pn_config_snapshot = copy.deepcopy(snapshot)

    def _pn_primitive_replay_state(self) -> dict:
        buffer = self.buffer
        slots = np.flatnonzero(buffer.episode_lengths > 0)
        episodes = []
        transition_fields = (
            "g", "cmd_g", "act", "cost", "violation", "goal_reached",
            "terminated", "truncated", "termination_reason",
        )
        for slot_value in slots:
            slot = int(slot_value)
            length = int(buffer.episode_lengths[slot])
            episode = {
                "slot": slot,
                "length": length,
                "obs": buffer.obs[slot, :length + 1].copy(),
                "ag": buffer.ag[slot, :length + 1].copy(),
            }
            for name in transition_fields:
                episode[name] = getattr(buffer, name)[slot, :length].copy()
            episodes.append(episode)
        if len(episodes) != buffer.n_episodes:
            raise ValueError("primitive replay metadata does not match occupied episodes")
        return {
            "size": int(buffer.size),
            "horizon": int(buffer.T),
            "obs_dim": int(buffer.obs.shape[2]),
            "goal_dim": int(buffer.ag.shape[2]),
            "act_dim": int(buffer.act.shape[2]),
            "her_ratio": float(buffer.her_ratio),
            "hindsight_range": int(buffer.hindsight_range),
            "ptr": int(buffer.ptr),
            "n_episodes": int(buffer.n_episodes),
            "episodes": episodes,
        }

    def _pn_validate_primitive_replay_state(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("PN checkpoint primitive replay state must be a dictionary")
        required = {
            "size", "horizon", "obs_dim", "goal_dim", "act_dim", "her_ratio",
            "hindsight_range", "ptr", "n_episodes", "episodes",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"PN checkpoint primitive replay state is incomplete; missing {sorted(missing)}"
            )
        buffer = self.buffer
        expected = {
            "size": buffer.size,
            "horizon": buffer.T,
            "obs_dim": buffer.obs.shape[2],
            "goal_dim": buffer.ag.shape[2],
            "act_dim": buffer.act.shape[2],
        }
        for name, value in expected.items():
            if int(state[name]) != int(value):
                raise ValueError(f"PN checkpoint primitive replay {name} is incompatible")
        ptr = int(state["ptr"])
        n_episodes = int(state["n_episodes"])
        if not 0 <= ptr < buffer.size or not 0 <= n_episodes <= buffer.size:
            raise ValueError("PN checkpoint primitive replay pointer/count is invalid")
        episodes = state["episodes"]
        if not isinstance(episodes, (list, tuple)) or len(episodes) != n_episodes:
            raise ValueError("PN checkpoint primitive replay episode list is invalid")
        episode_required = {
            "slot", "length", "obs", "ag", "g", "cmd_g", "act", "cost",
            "violation", "goal_reached", "terminated", "truncated",
            "termination_reason",
        }
        seen_slots = set()
        transition_shapes = {
            "g": (buffer.g.shape[2],),
            "cmd_g": (buffer.cmd_g.shape[2],),
            "act": (buffer.act.shape[2],),
            "cost": (), "violation": (), "goal_reached": (),
            "terminated": (), "truncated": (), "termination_reason": (),
        }
        for episode in episodes:
            if not isinstance(episode, dict) or episode_required.difference(episode):
                raise ValueError("PN checkpoint primitive replay episode is incomplete")
            slot = int(episode["slot"])
            length = int(episode["length"])
            if (slot in seen_slots or not 0 <= slot < buffer.size
                    or not 1 <= length <= buffer.T):
                raise ValueError("PN checkpoint primitive replay episode metadata is invalid")
            seen_slots.add(slot)
            if np.asarray(episode["obs"]).shape != (length + 1, buffer.obs.shape[2]):
                raise ValueError("PN checkpoint primitive replay obs shape is invalid")
            if np.asarray(episode["ag"]).shape != (length + 1, buffer.ag.shape[2]):
                raise ValueError("PN checkpoint primitive replay ag shape is invalid")
            for name, suffix in transition_shapes.items():
                if np.asarray(episode[name]).shape != (length,) + suffix:
                    raise ValueError(
                        f"PN checkpoint primitive replay {name} shape is invalid"
                    )
            for name in ("obs", "ag", "g", "cmd_g", "act", "cost"):
                values = np.asarray(episode[name])
                if (not np.issubdtype(values.dtype, np.number)
                        or np.issubdtype(values.dtype, np.complexfloating)
                        or not np.isfinite(values).all()):
                    raise ValueError(
                        f"PN checkpoint primitive replay {name} must contain finite real values"
                    )
        ratio = float(state["her_ratio"])
        if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("PN checkpoint primitive replay HER ratio is invalid")
        if int(state["hindsight_range"]) < 1:
            raise ValueError("PN checkpoint primitive replay hindsight range is invalid")

    def _pn_restore_primitive_replay_state(self, state: dict) -> None:
        buffer = self.buffer
        saved_slots = {int(episode["slot"]) for episode in state["episodes"]}
        existing_slots = set(np.flatnonzero(buffer.episode_lengths > 0).tolist())
        for slot in existing_slots | saved_slots:
            buffer._clear_slot(int(slot))
        for episode in state["episodes"]:
            slot = int(episode["slot"])
            length = int(episode["length"])
            buffer.obs[slot, :length + 1] = np.asarray(episode["obs"], dtype=buffer.obs.dtype)
            buffer.ag[slot, :length + 1] = np.asarray(episode["ag"], dtype=buffer.ag.dtype)
            for name in (
                "g", "cmd_g", "act", "cost", "violation", "goal_reached",
                "terminated", "truncated", "termination_reason",
            ):
                target = getattr(buffer, name)
                target[slot, :length] = np.asarray(episode[name], dtype=target.dtype)
            buffer.valid[slot, :length] = True
            buffer.episode_lengths[slot] = length
        buffer.ptr = int(state["ptr"])
        buffer.n_episodes = int(state["n_episodes"])
        buffer.her_ratio = float(state["her_ratio"])
        buffer.hindsight_range = int(state["hindsight_range"])

    @staticmethod
    def _pn_validate_module_state(module: torch.nn.Module, state: object,
                                  name: str, allow_centroid_count: bool = False) -> None:
        if not isinstance(state, dict):
            raise ValueError(f"PN checkpoint {name} state must be a dictionary")
        expected = module.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"PN checkpoint {name} state is incomplete")
        for key, expected_value in expected.items():
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"PN checkpoint {name}.{key} must be a tensor")
            if allow_centroid_count and key == "centroids":
                valid_shape = (
                    value.ndim == expected_value.ndim
                    and value.shape[1:] == expected_value.shape[1:]
                    and value.shape[0] > 0
                )
            else:
                valid_shape = value.shape == expected_value.shape
            if not valid_shape:
                raise ValueError(f"PN checkpoint {name}.{key} shape is incompatible")
            if ((torch.is_floating_point(value) or torch.is_complex(value))
                    and not torch.isfinite(value).all()):
                raise ValueError(f"PN checkpoint {name}.{key} must be finite")

    @staticmethod
    def _pn_validate_normalizer_state(normalizer: object, state: object,
                                      name: str) -> None:
        expected = normalizer.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError(f"PN checkpoint {name} normalizer state is incomplete")
        for key in ("sum", "sumsq", "mean", "std"):
            value = np.asarray(state[key])
            if value.shape != np.asarray(expected[key]).shape or not np.isfinite(value).all():
                raise ValueError(
                    f"PN checkpoint {name} normalizer {key} is incompatible"
                )
        count = float(state["count"])
        if not np.isfinite(count) or count <= 0.0:
            raise ValueError(f"PN checkpoint {name} normalizer count is invalid")

    @staticmethod
    def _pn_validate_optimizer_state(state: object, name: str,
                                     optimizer: Optional[torch.optim.Optimizer] = None,
                                     parameter_shapes: Optional[list[torch.Size]] = None) -> None:
        if (not isinstance(state, dict) or set(state) != {"state", "param_groups"}
                or not isinstance(state["state"], dict)
                or not isinstance(state["param_groups"], list)):
            raise ValueError(f"PN checkpoint {name} optimizer state is incomplete")
        for parameter_state in state["state"].values():
            if not isinstance(parameter_state, dict):
                raise ValueError(f"PN checkpoint {name} optimizer parameter state is invalid")
            for value in parameter_state.values():
                if (isinstance(value, torch.Tensor)
                        and (torch.is_floating_point(value) or torch.is_complex(value))
                        and not torch.isfinite(value).all()):
                    raise ValueError(
                        f"PN checkpoint {name} optimizer tensor must be finite"
                    )
        if optimizer is None:
            return
        try:
            probe = copy.deepcopy(optimizer)
            probe.load_state_dict(copy.deepcopy(state))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"PN checkpoint {name} optimizer state is incompatible") from exc
        saved_ids = []
        for group in state["param_groups"]:
            if not isinstance(group, dict) or not isinstance(group.get("params"), list):
                raise ValueError(f"PN checkpoint {name} optimizer groups are incomplete")
            saved_ids.extend(group["params"])
        parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        if len(saved_ids) != len(parameters):
            raise ValueError(f"PN checkpoint {name} optimizer parameter count is incompatible")
        expected_shapes = (
            [parameter.shape for parameter in parameters]
            if parameter_shapes is None else parameter_shapes
        )
        if len(expected_shapes) != len(parameters):
            raise ValueError(f"PN checkpoint {name} optimizer parameter count is incompatible")
        for saved_id, expected_shape in zip(saved_ids, expected_shapes):
            parameter_state = state["state"].get(saved_id, {})
            if not isinstance(parameter_state, dict):
                raise ValueError(f"PN checkpoint {name} optimizer parameter state is invalid")
            for value in parameter_state.values():
                if (isinstance(value, torch.Tensor) and value.ndim > 0
                        and value.shape != expected_shape):
                    raise ValueError(
                        f"PN checkpoint {name} optimizer tensor shape is incompatible"
                    )

    def _pn_prevalidate_checkpoint(self, checkpoint: object) -> tuple[Optional[int], Optional[dict]]:
        if not isinstance(checkpoint, dict):
            raise ValueError("trainer checkpoint must be a dictionary")
        root_required = {"agent", "ae", "landmarks", "centroids_initialized"}
        missing_root = root_required.difference(checkpoint)
        if missing_root:
            raise ValueError(f"trainer checkpoint is incomplete; missing {sorted(missing_root)}")
        agent_state = checkpoint["agent"]
        baseline_agent_required = {"actor", "critic", "value", "o_norm", "g_norm"}
        if not isinstance(agent_state, dict) or baseline_agent_required.difference(agent_state):
            raise ValueError("trainer checkpoint agent state is incomplete")
        self._pn_validate_module_state(self.agent.actor, agent_state["actor"], "agent.actor")
        self._pn_validate_module_state(self.agent.critic, agent_state["critic"], "agent.critic")
        self._pn_validate_module_state(self.agent.value, agent_state["value"], "agent.value")
        self._pn_validate_normalizer_state(
            self.agent.o_norm, agent_state["o_norm"], "observation",
        )
        self._pn_validate_normalizer_state(
            self.agent.g_norm, agent_state["g_norm"], "goal",
        )
        self._pn_validate_module_state(self.ae, checkpoint["ae"], "autoencoder")
        self._pn_validate_module_state(
            self.landmarks, checkpoint["landmarks"], "positive_landmarks",
            allow_centroid_count=True,
        )
        if not isinstance(checkpoint["centroids_initialized"], (bool, np.bool_)):
            raise ValueError("trainer checkpoint centroids_initialized must be boolean")

        version = checkpoint.get("version")
        if version is None:
            if "pn" in checkpoint:
                raise ValueError("unversioned checkpoint must not contain a partial 'pn' section")
            return None, None
        if version != PN_CHECKPOINT_VERSION:
            raise ValueError(f"unsupported PN checkpoint version {version!r}")
        pn = checkpoint.get("pn")
        if not isinstance(pn, dict):
            raise ValueError("versioned PN checkpoint requires a complete 'pn' section")
        required = {
            "config_snapshot", "module_config", "baseline_training",
            "primitive_replay", "positive_landmark_opt", "negative_landmarks",
            "negative_landmark_opt", "macro_model", "macro_opt",
            "cost_critic_target", "cost_critic_opt", "landmark_memory",
            "macro_buffer", "calibration", "phase", "activation", "counters",
            "diagnostics", "rng",
        }
        missing = required.difference(pn)
        if missing:
            raise ValueError(
                "versioned PN checkpoint has incomplete 'pn' section; "
                f"missing keys: {sorted(missing)}"
            )
        if "cost_critic" not in agent_state:
            raise ValueError("versioned PN checkpoint agent.cost_critic is missing")
        self._pn_validate_config_snapshot(pn["config_snapshot"])
        self._pn_validate_primitive_replay_state(pn["primitive_replay"])

        baseline_training = pn["baseline_training"]
        baseline_required = {
            "actor_target", "critic_target", "actor_opt", "critic_opt",
            "value_opt", "ae_opt",
        }
        if (not isinstance(baseline_training, dict)
                or baseline_required.difference(baseline_training)):
            raise ValueError("PN checkpoint baseline_training state is incomplete")
        self._pn_validate_module_state(
            self.agent.actor_target, baseline_training["actor_target"], "agent.actor_target",
        )
        self._pn_validate_module_state(
            self.agent.critic_target, baseline_training["critic_target"], "agent.critic_target",
        )
        for name, optimizer in (
            ("actor_opt", self.agent.actor_opt),
            ("critic_opt", self.agent.critic_opt),
            ("value_opt", self.agent.value_opt),
            ("ae_opt", self.ae_opt),
        ):
            self._pn_validate_optimizer_state(
                baseline_training[name], name, optimizer,
            )
        module_config = pn["module_config"]
        module_required = {"embedding_dim", "context_dim", "hidden_dim", "k_max"}
        if not isinstance(module_config, dict) or set(module_config) != module_required:
            raise ValueError("PN checkpoint module_config is incomplete")
        expected_module_config = {
            "embedding_dim": pn["config_snapshot"]["embedding_size"],
            "context_dim": 0,
            "hidden_dim": pn["config_snapshot"]["pn_macro_hidden_dim"],
            "k_max": pn["config_snapshot"]["pn_k_max"],
        }
        if any(module_config[name] != expected_module_config[name]
               for name in module_required):
            raise ValueError("PN checkpoint module_config does not match config_snapshot")
        negative_state = pn["negative_landmarks"]
        negative_centroids = (
            negative_state.get("centroids") if isinstance(negative_state, dict) else None
        )
        if (not isinstance(negative_centroids, torch.Tensor)
                or negative_centroids.ndim != 2
                or negative_centroids.shape[0] < 1):
            raise ValueError("PN checkpoint negative_landmarks.centroids shape is incompatible")
        positive_count = int(checkpoint["landmarks"]["centroids"].shape[0])
        negative_count = int(negative_centroids.shape[0])
        snapshot = pn["config_snapshot"]
        cost_critic_lr = snapshot["pn_cost_critic_lr"]
        if cost_critic_lr is None:
            cost_critic_lr = snapshot["critic_lr"]
        try:
            with torch.random.fork_rng(devices=[]):
                positive_landmarks = LatentLandmarks(
                    positive_count, int(snapshot["embedding_size"]),
                ).to(self.device)
                negative_landmarks = LatentLandmarks(
                    negative_count, int(snapshot["embedding_size"]),
                ).to(self.device)
                macro_model = MacroTransitionModel(**module_config)
                cost_critic = ViolationCritic(
                    self.agent.obs_dim,
                    self.agent.goal_dim,
                    self.agent.act_dim,
                    int(snapshot["hidden_units"]),
                    int(snapshot["hidden_layers"]),
                ).to(self.device)
                positive_landmark_opt = torch.optim.Adam(
                    positive_landmarks.parameters(), lr=float(snapshot["landmark_lr"]),
                )
                negative_landmark_opt = torch.optim.Adam(
                    negative_landmarks.parameters(), lr=float(snapshot["landmark_lr"]),
                )
                macro_opt = torch.optim.Adam(
                    macro_model.parameters(), lr=float(snapshot["pn_macro_lr"]),
                )
                cost_critic_opt = torch.optim.Adam(
                    cost_critic.parameters(), lr=float(cost_critic_lr),
                )
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint module_config values are invalid") from exc
        self._pn_validate_module_state(
            positive_landmarks, checkpoint["landmarks"], "positive_landmarks",
        )
        self._pn_validate_module_state(
            negative_landmarks, negative_state, "negative_landmarks",
        )
        self._pn_validate_module_state(
            macro_model, pn["macro_model"], "macro_model",
        )
        self._pn_validate_module_state(
            cost_critic, agent_state["cost_critic"], "agent.cost_critic",
        )
        self._pn_validate_module_state(
            cost_critic, pn["cost_critic_target"], "agent.cost_critic_target",
        )
        for name, optimizer in (
            ("positive_landmark_opt", positive_landmark_opt),
            ("negative_landmark_opt", negative_landmark_opt),
            ("macro_opt", macro_opt),
            ("cost_critic_opt", cost_critic_opt),
        ):
            self._pn_validate_optimizer_state(pn[name], name, optimizer)

        memory_state = pn["landmark_memory"]
        if not isinstance(memory_state, dict):
            raise ValueError("PN checkpoint landmark memory state is incomplete")
        memory_required = {
            "goal_dim", "positive_capacity", "negative_capacity", "positive_goals",
            "negative_goals", "negative_episode_ids", "negative_is_preimpact",
        }
        if memory_required.difference(memory_state):
            raise ValueError("PN checkpoint landmark memory state is incomplete")
        for name in ("positive_goals", "negative_goals"):
            values = np.asarray(memory_state[name])
            if (not np.issubdtype(values.dtype, np.number)
                    or np.issubdtype(values.dtype, np.complexfloating)
                    or not np.isfinite(values).all()):
                raise ValueError(
                    f"PN checkpoint landmark memory {name} must contain finite real values"
                )
        episode_ids = np.asarray(memory_state["negative_episode_ids"])
        if (not np.issubdtype(episode_ids.dtype, np.number)
                or np.issubdtype(episode_ids.dtype, np.complexfloating)
                or not np.isfinite(episode_ids).all()
                or not np.equal(episode_ids, np.floor(episode_ids)).all()):
            raise ValueError("PN checkpoint landmark memory episode IDs are invalid")
        preimpact = np.asarray(memory_state["negative_is_preimpact"])
        if not np.issubdtype(preimpact.dtype, np.bool_):
            raise ValueError("PN checkpoint landmark memory pre-impact metadata is invalid")
        memory = LandmarkMemoryManager(
            int(memory_state.get("goal_dim", -1)),
            positive_capacity=int(memory_state.get("positive_capacity", -1)),
            negative_capacity=int(memory_state.get("negative_capacity", -1)),
        )
        memory.load_state_dict(memory_state)
        macro_state = pn["macro_buffer"]
        if not isinstance(macro_state, dict):
            raise ValueError("PN checkpoint macro replay state is incomplete")
        macro_buffer = MacroAttemptBuffer(
            int(macro_state.get("capacity", -1)),
            validation_fraction=float(macro_state.get("validation_fraction", -1.0)),
            split_seed=int(macro_state.get("split_seed", -1)),
        )
        macro_buffer.load_state_dict(macro_state)

        if pn["phase"] not in {phase.value for phase in PNPhase}:
            raise ValueError("PN checkpoint phase is invalid")
        activation = pn["activation"]
        if not isinstance(activation, dict) or set(activation) != {"positive", "negative"}:
            raise ValueError("PN checkpoint activation state is incomplete")
        if not all(isinstance(activation[name], (bool, np.bool_)) for name in activation):
            raise ValueError("PN checkpoint activation values must be boolean")
        calibration = pn["calibration"]
        if not isinstance(calibration, dict) or set(calibration) != {
                "ready", "temperature", "before_nll", "after_nll", "ece"}:
            raise ValueError("PN checkpoint calibration state is incomplete")
        if not isinstance(calibration["ready"], (bool, np.bool_)):
            raise ValueError("PN checkpoint calibration readiness must be boolean")
        try:
            calibration_values = np.asarray([
                float(calibration["temperature"]), float(calibration["before_nll"]),
                float(calibration["after_nll"]), float(calibration["ece"]),
            ], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint calibration state must be finite") from exc
        if (not np.isfinite(calibration_values).all()
                or calibration_values[0] <= 0.0):
            raise ValueError("PN checkpoint calibration state must be finite")
        counter_required = {
            "total_env_steps", "episodes_collected", "grad_step_count",
            "positive_episodes", "primitive_transitions", "primitive_violations",
            "goal_success_episodes", "safe_success_episodes", "violation_episodes",
            "path_length_total", "planning", "no_safe_plan", "ingested_episode_ids",
        }
        if not isinstance(pn["counters"], dict) or counter_required.difference(pn["counters"]):
            raise ValueError("PN checkpoint counter state is incomplete")
        counters = pn["counters"]
        integer_counters = counter_required.difference({
            "path_length_total", "ingested_episode_ids",
        })
        for name in integer_counters:
            value = counters[name]
            if (isinstance(value, (bool, np.bool_)) or int(value) != value
                    or int(value) < 0):
                raise ValueError(f"PN checkpoint counter {name!r} is invalid")
        if not np.isfinite(float(counters["path_length_total"])):
            raise ValueError("PN checkpoint path length counter must be finite")
        try:
            ingested_ids = tuple(int(value) for value in counters["ingested_episode_ids"])
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint ingested episode IDs are invalid") from exc
        if any(value < 0 for value in ingested_ids):
            raise ValueError("PN checkpoint ingested episode IDs are invalid")
        diagnostics = pn["diagnostics"]
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
                "collection_status", "last", "sums", "validation", "evaluation"}:
            raise ValueError("PN checkpoint diagnostic state is incomplete")
        if not isinstance(diagnostics["collection_status"], str):
            raise ValueError("PN checkpoint collection status is invalid")
        for section in ("last", "sums", "validation", "evaluation"):
            values = diagnostics[section]
            if not isinstance(values, dict) or (
                    section in {"validation", "evaluation"} and not values):
                raise ValueError(f"PN checkpoint {section} diagnostics are incomplete")
            for name, value in values.items():
                if not isinstance(name, str) or not np.isscalar(value):
                    raise ValueError(f"PN checkpoint {section} diagnostics are invalid")
                try:
                    scalar = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"PN checkpoint {section} diagnostics must be finite"
                    ) from exc
                if not np.isfinite(scalar):
                    raise ValueError(f"PN checkpoint {section} diagnostics must be finite")
        if set(diagnostics["last"]) != set(diagnostics["sums"]):
            raise ValueError("PN checkpoint planner diagnostic sums are incomplete")

        rng = pn["rng"]
        if not isinstance(rng, dict) or set(rng) != {
                "trainer_numpy", "global_numpy", "environment_numpy", "torch", "cuda"}:
            raise ValueError("PN checkpoint RNG state is incomplete")
        try:
            test_generator = np.random.default_rng()
            test_generator.bit_generator.state = copy.deepcopy(rng["trainer_numpy"])
            test_random_state = np.random.RandomState()
            test_random_state.set_state(copy.deepcopy(rng["global_numpy"]))
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint NumPy RNG state is invalid") from exc
        self._pn_prevalidate_environment_rng_state(rng["environment_numpy"])
        if not isinstance(rng["torch"], torch.Tensor):
            raise ValueError("PN checkpoint Torch RNG state is invalid")
        try:
            torch.Generator(device="cpu").set_state(rng["torch"].cpu())
        except RuntimeError as exc:
            raise ValueError("PN checkpoint Torch RNG state is invalid") from exc
        cuda_rng = rng["cuda"]
        if not isinstance(cuda_rng, (list, tuple)):
            raise ValueError("PN checkpoint CUDA RNG state is invalid")
        for index, state in enumerate(cuda_rng):
            if not isinstance(state, torch.Tensor):
                raise ValueError(
                    f"PN checkpoint CUDA RNG state {index} must be a tensor"
                )
            if state.dtype != torch.uint8:
                raise ValueError(
                    f"PN checkpoint CUDA RNG state {index} must use torch.uint8"
                )
            if state.ndim != 1:
                raise ValueError(
                    f"PN checkpoint CUDA RNG state {index} must be one-dimensional"
                )
            if state.numel() < _PN_MIN_CUDA_RNG_STATE_BYTES:
                raise ValueError(
                    f"PN checkpoint CUDA RNG state {index} must contain at least "
                    f"{_PN_MIN_CUDA_RNG_STATE_BYTES} bytes"
                )
        if torch.cuda.is_available():
            overlap = min(len(cuda_rng), torch.cuda.device_count())
            for index, state in enumerate(cuda_rng[:overlap]):
                try:
                    torch.Generator(device=f"cuda:{index}").set_state(state.cpu())
                except (RuntimeError, TypeError) as exc:
                    raise ValueError(
                        f"PN checkpoint CUDA RNG state {index} is incompatible "
                        "with the current device"
                    ) from exc
        return PN_CHECKPOINT_VERSION, pn

    def save(self, path: str) -> None:
        state = dict(agent=self.agent.state_dict(), ae=self.ae.state_dict(),
                     landmarks=self.landmarks.state_dict(),
                     centroids_initialized=self.centroids_initialized)
        if self.cfg.pn_lmcgs_enabled:
            self.macro_model.set_temperature(self.pn_calibration_temperature)
            state["version"] = PN_CHECKPOINT_VERSION
            state["pn"] = self._pn_checkpoint_state()
        torch.save(state, path)

    def _pn_checkpoint_state(self) -> dict:
        evaluation = self.last_eval_metrics
        if not isinstance(evaluation, dict) or not evaluation:
            raise ValueError("PN checkpoint evaluation diagnostics must be a nonempty dictionary")
        evaluation_state = {}
        for name in sorted(evaluation):
            value = evaluation[name]
            if not np.isscalar(value):
                raise ValueError("PN checkpoint evaluation diagnostics must be scalar")
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("PN checkpoint evaluation diagnostics must be finite") from exc
            if not np.isfinite(scalar):
                raise ValueError("PN checkpoint evaluation diagnostics must be finite")
            evaluation_state[str(name)] = scalar
        return {
            "config_snapshot": self._pn_current_config_snapshot(),
            "module_config": {
                "embedding_dim": self.macro_model.embedding_dim,
                "context_dim": self.macro_model.context_dim,
                "hidden_dim": self.macro_model.backbone[0].out_features,
                "k_max": self.macro_model.k_max,
            },
            "baseline_training": {
                "actor_target": self.agent.actor_target.state_dict(),
                "critic_target": self.agent.critic_target.state_dict(),
                "actor_opt": self.agent.actor_opt.state_dict(),
                "critic_opt": self.agent.critic_opt.state_dict(),
                "value_opt": self.agent.value_opt.state_dict(),
                "ae_opt": self.ae_opt.state_dict(),
            },
            "primitive_replay": self._pn_primitive_replay_state(),
            "positive_landmark_opt": self.landmark_opt.state_dict(),
            "negative_landmarks": self.negative_landmarks.state_dict(),
            "negative_landmark_opt": self.negative_landmark_opt.state_dict(),
            "macro_model": self.macro_model.state_dict(),
            "macro_opt": self.macro_opt.state_dict(),
            "cost_critic_target": self.agent.cost_critic_target.state_dict(),
            "cost_critic_opt": self.agent.cost_critic_opt.state_dict(),
            "landmark_memory": self.landmark_memory.state_dict(),
            "macro_buffer": self.macro_buffer.state_dict(),
            "calibration": {
                "ready": self.pn_calibration_ready,
                "temperature": self.pn_calibration_temperature,
                "before_nll": self.pn_calibration_before_nll,
                "after_nll": self.pn_calibration_after_nll,
                "ece": self.pn_calibration_ece,
            },
            "phase": self.pn_phase.value,
            "activation": {
                "positive": self.pn_positive_active,
                "negative": self.pn_negative_active,
            },
            "counters": {
                "total_env_steps": self.total_env_steps,
                "episodes_collected": self.episodes_collected,
                "grad_step_count": self.grad_step_count,
                "positive_episodes": self.pn_positive_episode_count,
                "primitive_transitions": self.pn_primitive_transition_count,
                "primitive_violations": self.pn_primitive_violation_count,
                "goal_success_episodes": self.pn_episode_goal_success_count,
                "safe_success_episodes": self.pn_episode_safe_success_count,
                "violation_episodes": self.pn_violation_episode_count,
                "path_length_total": self.pn_path_length_total,
                "planning": self.pn_planning_count,
                "no_safe_plan": self.pn_no_safe_plan_count,
                "ingested_episode_ids": tuple(sorted(self.pn_ingested_episode_ids)),
            },
            "diagnostics": {
                "collection_status": self.pn_last_collection_status,
                "last": dict(self.pn_last_diagnostics),
                "sums": dict(self.pn_planner_diagnostic_sums),
                "validation": dict(self.pn_last_validation_diagnostics),
                "evaluation": evaluation_state,
            },
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    [state.clone() for state in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available() else []
                ),
                "trainer_numpy": copy.deepcopy(self.rng.bit_generator.state),
                "global_numpy": copy.deepcopy(np.random.get_state()),
                "environment_numpy": self._pn_environment_rng_state(),
            },
        }

    def pn_state_summary(self) -> tuple:
        if not self.cfg.pn_lmcgs_enabled:
            raise RuntimeError("PN state is unavailable when pn_lmcgs_enabled=False")
        pn = self._pn_checkpoint_state()
        counters = pn["counters"]
        calibration = pn["calibration"]
        return (
            f"pn-v{PN_CHECKPOINT_VERSION}",
            self.pn_phase.value,
            bool(self.pn_positive_active),
            bool(self.pn_negative_active),
            bool(self.pn_calibration_ready),
            int(self.landmark_memory.positive_size),
            int(self.landmark_memory.negative_size),
            int(self.landmark_memory.distinct_negative_episodes),
            int(self.macro_buffer.size),
            tuple((name, int(counters[name])) for name in (
                "total_env_steps", "episodes_collected", "grad_step_count",
                "positive_episodes", "primitive_transitions", "primitive_violations",
                "goal_success_episodes", "safe_success_episodes", "violation_episodes",
                "planning", "no_safe_plan",
            )),
            float(counters["path_length_total"]),
            tuple(int(value) for value in counters["ingested_episode_ids"]),
            tuple((name, float(calibration[name])) for name in (
                "temperature", "before_nll", "after_nll", "ece",
            )),
            str(self.pn_last_collection_status),
            tuple(sorted((str(key), float(value))
                         for key, value in self.pn_last_diagnostics.items())),
            tuple(sorted((str(key), float(value))
                         for key, value in self.pn_planner_diagnostic_sums.items())),
            tuple(sorted((str(key), float(value))
                         for key, value in self.pn_last_validation_diagnostics.items())),
            _state_fingerprint(pn["config_snapshot"]),
            _state_fingerprint(pn["baseline_training"]),
            _state_fingerprint(pn["primitive_replay"]),
            _state_fingerprint(self.landmarks.state_dict()),
            _state_fingerprint(self.landmark_opt.state_dict()),
            _state_fingerprint(self.negative_landmarks.state_dict()),
            _state_fingerprint(self.negative_landmark_opt.state_dict()),
            _state_fingerprint(self.macro_model.state_dict()),
            _state_fingerprint(self.macro_opt.state_dict()),
            _state_fingerprint(self.agent.cost_critic.state_dict()),
            _state_fingerprint(self.agent.cost_critic_target.state_dict()),
            _state_fingerprint(self.agent.cost_critic_opt.state_dict()),
            _state_fingerprint(self.landmark_memory.state_dict()),
            _state_fingerprint(self.macro_buffer.state_dict()),
            tuple(sorted((str(key), float(value))
                         for key, value in pn["diagnostics"]["evaluation"].items())),
            _state_fingerprint(pn["rng"]),
        )

    def load(self, path: str) -> None:
        # weights_only=False: the checkpoint stores numpy normalizer statistics.
        d = torch.load(path, map_location=self.device, weights_only=False)
        version, pn = self._pn_prevalidate_checkpoint(d)
        if version is not None and self.cfg.pn_lmcgs_enabled:
            self._pn_restore_config_snapshot(pn["config_snapshot"])
        self.agent.load_state_dict(d["agent"])
        self.ae.load_state_dict(d["ae"])
        # Rebuild the landmark module if the checkpoint used a different N.
        ckpt_n = d["landmarks"]["centroids"].shape[0]
        if ckpt_n != self.landmarks.n_landmarks:
            self.landmarks = LatentLandmarks(ckpt_n, self.cfg.embedding_size).to(self.device)
            self.planner.landmarks = self.landmarks
            self.landmark_opt = torch.optim.Adam(
                self.landmarks.parameters(), lr=self.cfg.landmark_lr,
            )
        self.landmarks.load_state_dict(d["landmarks"])
        self.centroids_initialized = d["centroids_initialized"]
        if not self.cfg.pn_lmcgs_enabled:
            return

        if version is None:
            self.landmarks = LatentLandmarks(
                self.cfg.pn_num_positive_landmarks, self.cfg.embedding_size,
            ).to(self.device)
            self.landmark_opt = torch.optim.Adam(
                self.landmarks.parameters(), lr=self.cfg.landmark_lr,
            )
            self.planner.landmarks = self.landmarks
            self.centroids_initialized = False
            self._bind_pn_planner()
            self._refresh_pn_phase()
            return
        try:
            self._load_pn_checkpoint_state(pn)
        except KeyError as exc:
            raise ValueError(
                "versioned PN checkpoint has incomplete nested state; "
                f"missing key: {exc.args[0]!r}"
            ) from exc

    def _load_pn_checkpoint_state(self, pn: dict) -> None:
        config_snapshot = pn["config_snapshot"]
        if not isinstance(config_snapshot, dict) or not config_snapshot:
            raise ValueError("PN checkpoint config_snapshot is incomplete")
        module_config = pn["module_config"]
        module_required = {"embedding_dim", "context_dim", "hidden_dim", "k_max"}
        if not isinstance(module_config, dict) or module_required.difference(module_config):
            raise ValueError("PN checkpoint module_config is incomplete")
        if int(module_config["embedding_dim"]) != self.cfg.embedding_size:
            raise ValueError("PN checkpoint embedding dimension does not match trainer")

        negative_n = int(pn["negative_landmarks"]["centroids"].shape[0])
        if negative_n != self.negative_landmarks.n_landmarks:
            self.negative_landmarks = LatentLandmarks(
                negative_n, self.cfg.embedding_size,
            ).to(self.device)
            self.negative_landmark_opt = torch.optim.Adam(
                self.negative_landmarks.parameters(), lr=self.cfg.landmark_lr,
            )
        self.negative_landmarks.load_state_dict(pn["negative_landmarks"])

        expected_macro = (
            int(module_config["embedding_dim"]), int(module_config["context_dim"]),
            int(module_config["hidden_dim"]), int(module_config["k_max"]),
        )
        current_macro = (
            self.macro_model.embedding_dim, self.macro_model.context_dim,
            self.macro_model.backbone[0].out_features, self.macro_model.k_max,
        )
        if current_macro != expected_macro:
            self.macro_model = MacroTransitionModel(
                embedding_dim=expected_macro[0],
                context_dim=expected_macro[1],
                hidden_dim=expected_macro[2],
                k_max=expected_macro[3],
            ).to(self.device)
            self.macro_opt = torch.optim.Adam(
                self.macro_model.parameters(), lr=self.cfg.pn_macro_lr,
            )
        self.macro_model.load_state_dict(pn["macro_model"])

        memory_state = pn["landmark_memory"]
        self.landmark_memory = LandmarkMemoryManager(
            int(memory_state["goal_dim"]),
            positive_capacity=int(memory_state["positive_capacity"]),
            negative_capacity=int(memory_state["negative_capacity"]),
        )
        self.landmark_memory.load_state_dict(memory_state)
        buffer_state = pn["macro_buffer"]
        self.macro_buffer = MacroAttemptBuffer(
            int(buffer_state["capacity"]),
            validation_fraction=float(buffer_state["validation_fraction"]),
            split_seed=int(buffer_state["split_seed"]),
        )
        self.macro_buffer.load_state_dict(buffer_state)

        self.landmark_opt.load_state_dict(pn["positive_landmark_opt"])
        self.negative_landmark_opt.load_state_dict(pn["negative_landmark_opt"])
        self.macro_opt.load_state_dict(pn["macro_opt"])
        self.agent.cost_critic_opt.load_state_dict(pn["cost_critic_opt"])
        baseline_training = pn["baseline_training"]
        self.agent.actor_target.load_state_dict(baseline_training["actor_target"])
        self.agent.critic_target.load_state_dict(baseline_training["critic_target"])
        self.agent.actor_opt.load_state_dict(baseline_training["actor_opt"])
        self.agent.critic_opt.load_state_dict(baseline_training["critic_opt"])
        self.agent.value_opt.load_state_dict(baseline_training["value_opt"])
        self.ae_opt.load_state_dict(baseline_training["ae_opt"])
        try:
            self.agent.cost_critic_target.load_state_dict(pn["cost_critic_target"])
        except (RuntimeError, TypeError) as exc:
            raise ValueError("PN checkpoint cost_critic_target state is invalid") from exc

        calibration = pn["calibration"]
        calibration_required = {"ready", "temperature", "before_nll", "after_nll", "ece"}
        if not isinstance(calibration, dict) or calibration_required.difference(calibration):
            raise ValueError("PN checkpoint calibration state is incomplete")
        self.pn_calibration_ready = bool(calibration["ready"])
        self.pn_calibration_temperature = float(calibration["temperature"])
        self.pn_calibration_before_nll = float(calibration["before_nll"])
        self.pn_calibration_after_nll = float(calibration["after_nll"])
        self.pn_calibration_ece = float(calibration["ece"])
        calibration_values = (
            self.pn_calibration_temperature, self.pn_calibration_before_nll,
            self.pn_calibration_after_nll, self.pn_calibration_ece,
        )
        if not np.isfinite(calibration_values).all() or self.pn_calibration_temperature <= 0.0:
            raise ValueError("PN checkpoint calibration state must be finite")
        self.macro_model.set_temperature(self.pn_calibration_temperature)

        activation = pn["activation"]
        counters = pn["counters"]
        diagnostics = pn["diagnostics"]
        if not isinstance(activation, dict) or set(activation) != {"positive", "negative"}:
            raise ValueError("PN checkpoint activation state is incomplete")
        counter_required = {
            "total_env_steps", "episodes_collected", "grad_step_count",
            "positive_episodes", "primitive_transitions", "primitive_violations",
            "goal_success_episodes", "safe_success_episodes", "violation_episodes",
            "path_length_total", "planning", "no_safe_plan", "ingested_episode_ids",
        }
        if not isinstance(counters, dict) or counter_required.difference(counters):
            raise ValueError("PN checkpoint counter state is incomplete")
        if not isinstance(diagnostics, dict) or set(diagnostics) != {
                "collection_status", "last", "sums", "validation", "evaluation"}:
            raise ValueError("PN checkpoint diagnostic state is incomplete")
        self.pn_phase = PNPhase(pn["phase"])
        self.pn_positive_active = bool(activation["positive"])
        self.pn_negative_active = bool(activation["negative"])
        self.total_env_steps = int(counters["total_env_steps"])
        self.episodes_collected = int(counters["episodes_collected"])
        self.grad_step_count = int(counters["grad_step_count"])
        self.pn_positive_episode_count = int(counters["positive_episodes"])
        self.pn_primitive_transition_count = int(counters["primitive_transitions"])
        self.pn_primitive_violation_count = int(counters["primitive_violations"])
        self.pn_episode_goal_success_count = int(counters["goal_success_episodes"])
        self.pn_episode_safe_success_count = int(counters["safe_success_episodes"])
        self.pn_violation_episode_count = int(counters["violation_episodes"])
        self.pn_path_length_total = float(counters["path_length_total"])
        self.pn_planning_count = int(counters["planning"])
        self.pn_no_safe_plan_count = int(counters["no_safe_plan"])
        self.pn_ingested_episode_ids = set(int(value) for value in counters["ingested_episode_ids"])
        last_diagnostics = diagnostics["last"]
        if not isinstance(last_diagnostics, dict):
            raise ValueError("PN checkpoint diagnostics must be a dictionary")
        self.pn_last_collection_status = str(diagnostics["collection_status"])
        try:
            self.pn_last_diagnostics = {
                str(key): float(value) for key, value in last_diagnostics.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint diagnostics must be finite") from exc
        if not np.isfinite(tuple(self.pn_last_diagnostics.values())).all():
            raise ValueError("PN checkpoint diagnostics must be finite")
        diagnostic_sums = diagnostics["sums"]
        if not isinstance(diagnostic_sums, dict):
            raise ValueError("PN checkpoint diagnostic sums must be a dictionary")
        try:
            self.pn_planner_diagnostic_sums = {
                str(key): float(value) for key, value in diagnostic_sums.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint diagnostic sums must be finite") from exc
        if (set(self.pn_planner_diagnostic_sums) != set(self.pn_last_diagnostics)
                or not np.isfinite(tuple(self.pn_planner_diagnostic_sums.values())).all()):
            raise ValueError("PN checkpoint diagnostic sums are incomplete or non-finite")
        validation = diagnostics["validation"]
        if not isinstance(validation, dict) or not validation:
            raise ValueError("PN checkpoint validation diagnostics are incomplete")
        try:
            self.pn_last_validation_diagnostics = {
                str(key): float(value) for key, value in validation.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("PN checkpoint validation diagnostics must be finite") from exc
        if not np.isfinite(tuple(self.pn_last_validation_diagnostics.values())).all():
            raise ValueError("PN checkpoint validation diagnostics must be finite")
        evaluation = diagnostics["evaluation"]
        if not isinstance(evaluation, dict) or not evaluation:
            raise ValueError("PN checkpoint evaluation diagnostics are incomplete")
        restored_evaluation = {}
        for name, value in evaluation.items():
            if not np.isscalar(value):
                raise ValueError("PN checkpoint evaluation diagnostics must be scalar")
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("PN checkpoint evaluation diagnostics must be finite") from exc
            if not np.isfinite(scalar):
                raise ValueError("PN checkpoint evaluation diagnostics must be finite")
            restored_evaluation[str(name)] = scalar
        self.last_eval_metrics = restored_evaluation
        self.pn_config_snapshot = copy.deepcopy(config_snapshot)
        self._pn_restore_primitive_replay_state(pn["primitive_replay"])

        rng = pn["rng"]
        if not isinstance(rng, dict) or set(rng) != {
                "trainer_numpy", "global_numpy", "environment_numpy", "torch", "cuda"}:
            raise ValueError("PN checkpoint RNG state is incomplete")
        self.rng.bit_generator.state = copy.deepcopy(rng["trainer_numpy"])
        np.random.set_state(copy.deepcopy(rng["global_numpy"]))
        self._pn_restore_environment_rng_state(rng["environment_numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if torch.cuda.is_available():
            overlap = min(len(rng["cuda"]), torch.cuda.device_count())
            for index, state in enumerate(rng["cuda"][:overlap]):
                torch.cuda.set_rng_state(state.cpu(), device=index)
        self._pn_common_action_bucket_cache_key = None
        self._pn_common_action_bucket_cache = None
        self.planner.landmarks = self.landmarks
        self._bind_pn_planner()
        self._refresh_pn_phase()
