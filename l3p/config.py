"""Hyper-parameter configuration for L3P.

Values are taken from Appendix E of the paper (arXiv:2011.12491v3). The common
table applies to every environment; the per-environment table overrides a
handful of values for Point-Maze / Ant-Maze / Fetch tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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

    # ---- optional value-function contrastive loss ----
    use_value_contrastive: bool = False
    value_contrastive_lambda: float = 0.1
    value_contrastive_temperature: float = 1.0
    n_value_negatives: int = 4
    negative_sampling_strategy: str = "random"  # random | cross_episode

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
    return base
