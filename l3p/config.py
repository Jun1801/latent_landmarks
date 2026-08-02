"""Hyper-parameter configuration for L3P.

Values are taken from Appendix E of the paper (arXiv:2011.12491v3). The common
table applies to every environment; the per-environment table overrides a
handful of values for Point-Maze / Ant-Maze / Fetch tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral, Real
from typing import Optional


@dataclass
class Config:
    # ---- environment ----
    env_name: str = "PointMaze"
    max_episode_steps: int = 200        # PointMaze-Hard uses a short 200-step horizon during training
    goal_threshold: float = 0.15        # distance under which a goal counts as reached (Ψ(s)=g)
    maze_size: int = 11                 # PointMaze grid size (larger -> longer horizons)
    test_episode_steps: int = 200       # horizon at test time (paper uses a longer test horizon)
    # env-specific structural knobs (only used by the MuJoCo Fetch variants)
    place_inside_box_ratio: float = 0.2  # Place-Inside-Box curriculum: fraction of inside-box goals
    has_distractor: bool = False         # Box-Distractor: a box distractor sits in the workspace

    # ---- DDPG (common table, Appendix E) ----
    actor_lr: float = 2e-4              # per-env below
    critic_lr: float = 2e-4
    n_workers: int = 1                  # per-env below
    batch_size: int = 512               # per-env below
    action_l2: float = 0.5              # per-env below (L2 penalty on actions)
    gamma: float = 0.98                 # per-env below
    hindsight_range: int = 80           # per-env below; shortened HER future range
    hidden_layers: int = 3              # number of hidden layers (all DDPG nets)
    hidden_units: int = 256
    polyak: float = 0.995               # target-network Polyak coefficient (tau)
    target_update_interval: int = 10    # steps between target-net updates
    env_steps_per_opt: int = 2          # ratio between env steps and optimization steps
    random_action_prob: float = 0.2     # epsilon for random actions during collection
    initial_random_trajs: int = 100     # warm-up random trajectories per worker
    her_ratio: float = 0.85             # fraction of relabelled (hindsight) transitions
    action_noise: float = 0.2           # gaussian exploration noise std (per-env below)
    grad_norm_clip: Optional[float] = None  # Ant-Maze uses 15.0

    # ---- Latent Landmarks & Auto-encoder (common table) ----
    ae_hidden_layers: int = 2
    ae_hidden_units: int = 128
    embedding_size: int = 16            # latent dim of the reachability-constrained AE
    ae_lambda: float = 1.0              # lambda for the reachability constraint loss (Eq 2)
    ae_lr: float = 3e-4
    n_landmarks: int = 50               # per-env below
    n_warmup_trajs: int = 500           # per-env below
    landmark_batch_size: int = 256      # per-env below
    landmark_lr: float = 1e-3           # centroid / sigma learning rate
    gls_batch_size: int = 512           # K achieved goals sampled before sub-sampling (GLS)

    # ---- Graph Search (common table) ----
    search_prob_train: float = 0.5      # probability of using search during train
    soft_iters: int = 20                # S: number of soft value iterations (soft Floyd)
    beta: float = 1.1                   # softmax temperature for soft relaxation
    d_max: float = 20.0                 # per-env below; edge-weight cutoff threshold
    random_landmarks_train: int = 150   # per-env below; extra GLS landmarks added per episode
    neg_inf: float = -1e6               # value used to mask out edges beyond d_max (Appendix B)

    # ---- training loop (Algorithm 3) ----
    total_steps: int = 2_000_000
    k_env: int = 1                      # episodes collected per outer iteration before grad steps
    n_grad_steps: int = 40              # gradient steps per optimization phase
    train_after: int = 1000             # env steps before the first gradient step
    eval_interval: int = 20_000
    eval_episodes: int = 20
    log_interval: int = 2000
    seed: int = 0
    device: str = "cpu"

    # ---- Positive-Negative Landmark MCGS (opt-in) ----
    # Environment / feature flags. Collision labels remain opt-in because
    # ordinary contact is not a portable definition of a safety violation.
    pn_lmcgs_enabled: bool = False
    pn_pointmaze_hazard_enabled: bool = False
    pn_collision_cost_enabled: bool = False
    pn_collision_cost_info_key: str = ""
    pn_collision_cost_unsafe_value: Optional[object] = None

    # Positive / negative landmark memories and activation gates.
    pn_num_positive_landmarks: int = 50
    pn_num_negative_landmarks: int = 10
    pn_positive_memory_capacity: int = 100_000
    pn_negative_memory_capacity: int = 100_000
    pn_min_positive_samples: int = 500
    pn_min_positive_episodes: int = 20
    pn_min_negative_samples: int = 100
    pn_min_negative_episodes: int = 20
    pn_positive_assignment_radius: float = 1.0
    pn_landmark_refresh_interval: int = 1_000

    # Goal-conditioned violation critic.
    pn_cost_critic_weight: float = 5.0
    pn_cost_critic_lr: float = 3e-4
    pn_cost_critic_batch_size: int = 256
    pn_cost_critic_train_after: int = 1_000

    # Macro attempts, model, and calibration.
    pn_k_min: int = 5
    pn_k_max: int = 50
    pn_macro_replay_capacity: int = 100_000
    pn_macro_hidden_dim: int = 256
    pn_macro_batch_size: int = 256
    pn_macro_lr: float = 3e-4
    pn_beta_duration: float = 0.1
    pn_calibrate_temperature: bool = True
    pn_calibration_validation_split: float = 0.2
    pn_calibration_temperature_min: float = 0.5
    pn_calibration_temperature_max: float = 5.0
    pn_calibration_grid_size: int = 91
    pn_calibration_interval: int = 5_000
    pn_outcome_target_probability: float = 0.50
    pn_outcome_drift_or_stuck_probability: float = 0.25
    pn_outcome_violation_probability: float = 0.25

    # MCGS candidates, risk, PUCT, penalties, and leaf heuristic.
    pn_top_k: int = 5
    pn_macro_depth: int = 6
    pn_num_simulations: int = 256
    pn_c_puct: float = 1.5
    pn_prior_epsilon: float = 0.001
    pn_beta_distance: float = 0.05
    pn_beta_risk: float = 2.0
    pn_epsilon_edge_train: float = 0.30
    pn_epsilon_edge_eval: float = 0.20
    pn_search_risk_limit: float = 0.20
    pn_root_risk_limit: float = 0.10
    pn_risk_pseudocount: float = 4.0
    pn_risk_z: float = 1.645
    pn_lambda_search_risk: float = 1.0
    pn_stuck_penalty: float = 0.2
    pn_violation_penalty: float = 1.0
    pn_loop_penalty: float = 0.2
    pn_time_penalty: float = 0.05
    pn_lambda_stuck_leaf: float = 0.2
    pn_lambda_violation_leaf: float = 1.0
    pn_leaf_temperature: float = 1.0
    pn_leaf_epsilon: float = 1e-6

    # Training gates, collection mix, diagnostics, and explicit fallback.
    pn_min_macro_attempts: int = 5_000
    pn_min_attempts_per_common_action_bucket: int = 20
    pn_probability_mcg_search: float = 0.50
    pn_probability_original_planner: float = 0.25
    pn_probability_direct_goal: float = 0.25
    pn_training_unsafe_fallback: bool = False
    pn_diagnostics_enabled: bool = True
    pn_diagnostic_interval: int = 2_000
    pn_calibration_bins: int = 10

    def scaled_neg_reward(self) -> float:
        return -1.0


# --- per-environment overrides (Appendix E, second table) ---
_PER_ENV = {
    "PointMaze": dict(
        actor_lr=2e-4, critic_lr=2e-4, n_workers=1, batch_size=512,
        action_l2=0.5, gamma=0.98, hindsight_range=80,
        n_landmarks=50, n_warmup_trajs=500, landmark_batch_size=256,
        d_max=20.0, random_landmarks_train=150, action_noise=0.2,
        max_episode_steps=200, grad_norm_clip=None,
    ),
    "AntMaze": dict(
        actor_lr=2e-4, critic_lr=2e-4, n_workers=3, batch_size=1024,
        action_l2=0.05, gamma=0.98, hindsight_range=100,
        n_landmarks=50, n_warmup_trajs=500, landmark_batch_size=256,
        d_max=20.0, random_landmarks_train=150, action_noise=0.2,
        max_episode_steps=500, grad_norm_clip=15.0,
    ),
    "Fetch": dict(
        actor_lr=1e-3, critic_lr=1e-3, n_workers=12, batch_size=1024,
        action_l2=0.01, gamma=0.99, hindsight_range=50,
        n_landmarks=80, n_warmup_trajs=6000, landmark_batch_size=150,
        d_max=15.0, random_landmarks_train=20, action_noise=0.1,
        max_episode_steps=50, grad_norm_clip=None,
    ),
}


# --- the five environments from the paper (Section 5) ---
# `family` selects the Appendix-E hyper-parameter column; the rest are the
# structural / env-specific settings (gym id, horizons, curriculum, distractor).
ENV_SPECS = {
    "PointMaze": dict(
        family="PointMaze", gym_id=None, needs_mujoco=False,
        train_horizon=200, test_horizon=500,
        note="Point mass in a hard maze. Runnable here in pure NumPy; the paper "
             "uses a MuJoCo point-in-maze. Test starts one end, goal the other.",
    ),
    "PointMazeMuJoCo": dict(
        family="PointMaze", gym_id="PointMaze_Large-v3", needs_mujoco=True,
        train_horizon=200, test_horizon=500,
        note="Real MuJoCo point-in-maze (gymnasium-robotics PointMaze_Large) — the "
             "paper's PointMaze environment type, as opposed to the pure-NumPy proxy.",
    ),
    "AntMaze": dict(
        family="AntMaze", gym_id="AntMaze-v0", needs_mujoco=True,
        train_horizon=200, test_horizon=500,
        note="MuJoCo Ant navigating a hard maze; grad-norm clip 15 (Appendix D). "
             "Test generalizes to the longest, unseen path.",
    ),
    "FetchPickAndPlace": dict(
        family="Fetch", gym_id="FetchPickAndPlace-v1", needs_mujoco=True,
        train_horizon=50, test_horizon=50,
        note="Standard gym-robotics Fetch pick-and-place. Inputs normalized by "
             "running mean/std (Appendix D).",
    ),
    "BoxDistractorPickAndPlace": dict(
        family="Fetch", gym_id="FetchPickAndPlaceBoxDistractor-v1", needs_mujoco=True,
        train_horizon=50, test_horizon=50, has_distractor=True,
        note="Pick-and-place with a box distractor in the middle of the table; "
             "the arm must pick/place the block while avoiding collision with the box.",
    ),
    "PlaceInsideBox": dict(
        family="Fetch", gym_id="FetchPlaceInsideBox-v1", needs_mujoco=True,
        train_horizon=50, test_horizon=50, place_inside_box_ratio=0.2,
        note="Place the object inside a box at random locations. Curriculum: 80% "
             "regular pick-and-place goals, 20% inside-the-box goals (Section 5.3).",
    ),
}

_ALIASES = {
    "point": "PointMaze", "pointmaze": "PointMaze", "point-maze": "PointMaze",
    "ant": "AntMaze", "antmaze": "AntMaze", "ant-maze": "AntMaze",
    "fetch": "FetchPickAndPlace", "pick": "FetchPickAndPlace",
    "fetchpickandplace": "FetchPickAndPlace",
    "distractor": "BoxDistractorPickAndPlace",
    "boxdistractor": "BoxDistractorPickAndPlace",
    "boxdistractorpickandplace": "BoxDistractorPickAndPlace",
    "inside_box": "PlaceInsideBox", "insidebox": "PlaceInsideBox",
    "placeinsidebox": "PlaceInsideBox",
}


def resolve_env(env_name: str) -> str:
    """Map a user-supplied env name / alias to a canonical ENV_SPECS key."""
    key = env_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if env_name in ENV_SPECS:
        return env_name
    if key in _ALIASES:
        return _ALIASES[key]
    if key in {k.lower() for k in ENV_SPECS}:
        return {k.lower(): k for k in ENV_SPECS}[key]
    return "PointMaze"


def env_spec(env_name: str) -> dict:
    return ENV_SPECS[resolve_env(env_name)]


def list_envs() -> list:
    return list(ENV_SPECS.keys())


def validate_pn_config(cfg: Config) -> None:
    """Validate opt-in PN-LMCGS settings without changing baseline defaults."""
    positive_counts = (
        "pn_num_positive_landmarks", "pn_num_negative_landmarks",
        "pn_positive_memory_capacity", "pn_negative_memory_capacity",
        "pn_cost_critic_batch_size", "pn_macro_replay_capacity",
        "pn_macro_hidden_dim", "pn_macro_batch_size", "pn_top_k",
        "pn_macro_depth", "pn_num_simulations", "pn_min_macro_attempts",
        "pn_calibration_grid_size", "pn_calibration_interval",
        "pn_diagnostic_interval", "pn_calibration_bins",
    )
    nonnegative_counts = (
        "pn_min_positive_samples", "pn_min_positive_episodes",
        "pn_min_negative_samples", "pn_min_negative_episodes",
        "pn_cost_critic_train_after", "pn_landmark_refresh_interval",
        "pn_min_attempts_per_common_action_bucket",
    )
    integer_names = positive_counts + nonnegative_counts + ("pn_k_min", "pn_k_max")
    numeric_names = (
        *integer_names,
        "pn_positive_assignment_radius", "pn_cost_critic_weight", "pn_cost_critic_lr",
        "pn_macro_lr", "pn_beta_duration", "pn_calibration_validation_split",
        "pn_calibration_temperature_min", "pn_calibration_temperature_max",
        "pn_outcome_target_probability", "pn_outcome_drift_or_stuck_probability",
        "pn_outcome_violation_probability", "pn_c_puct", "pn_prior_epsilon",
        "pn_beta_distance", "pn_beta_risk", "pn_epsilon_edge_train",
        "pn_epsilon_edge_eval", "pn_search_risk_limit", "pn_root_risk_limit",
        "pn_risk_pseudocount", "pn_risk_z", "pn_lambda_search_risk",
        "pn_stuck_penalty", "pn_violation_penalty", "pn_loop_penalty",
        "pn_time_penalty", "pn_lambda_stuck_leaf", "pn_lambda_violation_leaf",
        "pn_leaf_temperature", "pn_leaf_epsilon", "pn_probability_mcg_search",
        "pn_probability_original_planner", "pn_probability_direct_goal",
    )
    for name in numeric_names:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError(f"{name} must be a finite real value")
    for name in integer_names:
        if not isinstance(getattr(cfg, name), Integral):
            raise ValueError(f"{name} must be an integer")
    for name in positive_counts:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in nonnegative_counts:
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    if cfg.pn_k_min <= 0 or cfg.pn_k_max < cfg.pn_k_min:
        raise ValueError("pn_k_min and pn_k_max must satisfy 0 < pn_k_min <= pn_k_max")
    if not 0.0 < cfg.pn_calibration_validation_split < 1.0:
        raise ValueError("pn_calibration_validation_split must be between 0 and 1")
    if not 0.0 < cfg.pn_calibration_temperature_min <= cfg.pn_calibration_temperature_max:
        raise ValueError("calibration temperature bounds must be positive and ordered")

    probability_names = (
        "pn_probability_mcg_search", "pn_probability_original_planner",
        "pn_probability_direct_goal", "pn_outcome_target_probability",
        "pn_outcome_drift_or_stuck_probability", "pn_outcome_violation_probability",
    )
    for name in probability_names:
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if abs(
        cfg.pn_probability_mcg_search + cfg.pn_probability_original_planner
        + cfg.pn_probability_direct_goal - 1.0
    ) > 1e-8:
        raise ValueError("PN collection probabilities must sum to one")
    if abs(
        cfg.pn_outcome_target_probability + cfg.pn_outcome_drift_or_stuck_probability
        + cfg.pn_outcome_violation_probability - 1.0
    ) > 1e-8:
        raise ValueError("PN macro outcome probabilities must sum to one")

    risk_bounds = (
        "pn_epsilon_edge_train", "pn_epsilon_edge_eval",
        "pn_search_risk_limit", "pn_root_risk_limit",
    )
    for name in risk_bounds:
        if not 0.0 <= getattr(cfg, name) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if cfg.pn_root_risk_limit > cfg.pn_search_risk_limit:
        raise ValueError("pn_root_risk_limit must not exceed pn_search_risk_limit")
    if cfg.pn_pointmaze_hazard_enabled and not cfg.pn_lmcgs_enabled:
        raise ValueError("pn_pointmaze_hazard_enabled requires pn_lmcgs_enabled=True")


def get_config(env_name: str = "PointMaze", **overrides) -> Config:
    """Build a Config for `env_name`: apply the Appendix-E hyper-parameters for
    its family, then the env-specific structural settings, then any explicit
    keyword `overrides`."""
    spec = env_spec(env_name)
    base = Config(env_name=resolve_env(env_name))
    base = replace(base, **_PER_ENV[spec["family"]])
    base = replace(base,
                   max_episode_steps=spec["train_horizon"],
                   test_episode_steps=spec["test_horizon"],
                   has_distractor=spec.get("has_distractor", False),
                   place_inside_box_ratio=spec.get("place_inside_box_ratio", 0.2))
    if overrides:
        base = replace(base, **overrides)
    validate_pn_config(base)
    return base
