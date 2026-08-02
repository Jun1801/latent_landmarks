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

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from l3p.agent.ddpg import DDPGAgent
from l3p.losses import ae_losses
from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.landmarks import LatentLandmarks, greedy_latent_sparsification
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.replay.her_buffer import HERReplayBuffer


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
    def __init__(self, vec_env, cfg):
        self.env = vec_env
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)

        od, gd, ad = vec_env.obs_dim, vec_env.goal_dim, vec_env.act_dim
        self.T = vec_env.max_episode_steps

        self.agent = DDPGAgent(od, gd, ad, vec_env.max_action, cfg)
        self.ae = ReachabilityAutoEncoder(gd, cfg.embedding_size, cfg.ae_hidden_units,
                                          cfg.ae_hidden_layers).to(self.device)
        self.landmarks = LatentLandmarks(cfg.n_landmarks, cfg.embedding_size).to(self.device)
        # The AE encoder normalizes its goal inputs with the running goal stats
        # (stabilizes the latent space for large-coordinate goal spaces).
        self.ae.normalizer = self.agent.g_norm
        self.graph_search = GraphSearch(cfg)
        self.planner = LatentPlanner(self.agent, self.landmarks, self.ae, self.graph_search, cfg)

        self.ae_opt = torch.optim.Adam(self.ae.parameters(), lr=cfg.ae_lr)
        self.landmark_opt = torch.optim.Adam(self.landmarks.parameters(), lr=cfg.landmark_lr)

        if cfg.use_value_contrastive:
            if cfg.n_value_negatives < 1:
                raise ValueError("n_value_negatives must be >= 1 when use_value_contrastive=True")
            if cfg.value_contrastive_temperature <= 0:
                raise ValueError("value_contrastive_temperature must be > 0")

        self.buffer = HERReplayBuffer(
            size_episodes=100_000, horizon=self.T, obs_dim=od, goal_dim=gd, act_dim=ad,
            compute_reward=vec_env.compute_reward, her_ratio=cfg.her_ratio,
            hindsight_range=cfg.hindsight_range,
            n_value_negatives=cfg.n_value_negatives if cfg.use_value_contrastive else 0,
            negative_sampling_strategy=cfg.negative_sampling_strategy)

        self.total_env_steps = 0
        self.episodes_collected = 0
        self.centroids_initialized = False
        self.grad_step_count = 0

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

    def collect_episode(self, env, use_planning: bool, random_actions: bool) -> Dict[str, np.ndarray]:
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

    def collect(self) -> None:
        for env in self.env.envs:
            random_actions = self.episodes_collected < self.cfg.initial_random_trajs * self.env.n
            use_planning = self.rng.random() < self.cfg.search_prob_train
            ep = self.collect_episode(env, use_planning, random_actions)
            self.buffer.store_episode(ep)
            self.agent.update_normalizers(ep["obs"][:-1], ep["g"])
            self.episodes_collected += 1
            self.total_env_steps += self.T

        # Initialize latent centroids via GLS once enough warm-up data is in.
        if not self.centroids_initialized and self.episodes_collected >= self.cfg.n_warmup_trajs:
            self._init_centroids()

    def _init_centroids(self) -> None:
        goals = self.buffer.sample_achieved_goals(self.cfg.gls_batch_size, self.rng)
        with torch.no_grad():
            z = self.ae.encode(self.agent.to_tensor(goals))
            idx = greedy_latent_sparsification(z, self.cfg.n_landmarks, self.rng)
            self.landmarks.centroids.data.copy_(z[idx])
        self.centroids_initialized = True

    # ------------------------------------------------------------------ optimization
    def _make_batch(self, raw: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        a = self.agent
        batch = dict(
            obs=a.norm_obs(raw["obs"]),
            next_obs=a.norm_obs(raw["next_obs"]),
            g=a.norm_goal(raw["g"]),
            act=a.to_tensor(raw["act"]),
            reward=a.to_tensor(raw["reward"]),
            next_ag=a.to_tensor(raw["next_ag"]),          # raw goal space for V
            future_ag=a.to_tensor(raw["future_ag"]),      # raw goal space for V
            future_ag_norm=a.norm_goal(raw["future_ag"]),  # normalized for the critic
        )
        if "neg_ag" in raw:
            batch["neg_ag"] = a.to_tensor(raw["neg_ag"])  # raw goal space for V
        return batch

    def update(self, n_steps: int) -> Dict[str, float]:
        logs = {"critic": 0.0, "value": 0.0, "actor": 0.0, "ae_rec": 0.0,
                "ae_latent": 0.0, "elbo": 0.0}
        for _ in range(n_steps):
            raw = self.buffer.sample(self.cfg.batch_size, self.rng)
            batch = self._make_batch(raw)

            logs["value"] += self.agent.update_value(batch)
            logs["critic"] += self.agent.update_critic(batch)
            logs["actor"] += self.agent.update_actor(batch)

            # Auto-encoder (Eq. 2) on a fresh batch of achieved goals.
            goals = self.buffer.sample_achieved_goals(self.cfg.batch_size, self.rng)
            goals_t = self.agent.to_tensor(goals)
            l_rec, l_latent, ae_total = ae_losses(self.ae, self.agent.value, goals_t,
                                                  self.cfg.ae_lambda)
            self.ae_opt.zero_grad()
            ae_total.backward()
            self.ae_opt.step()
            logs["ae_rec"] += float(l_rec.item())
            logs["ae_latent"] += float(l_latent.item())

            # Latent centroids (ELBO, Eq. 5) on a GLS-sparsified batch.
            if self.centroids_initialized:
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

        return {k: v / max(1, n_steps) for k, v in logs.items()}

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, n_episodes: int = 20, use_planning: bool = True) -> float:
        self.env.set_eval(True)
        env = self.env.envs[0]
        successes = 0
        for _ in range(n_episodes):
            obs_dict = env.reset()
            goal = obs_dict["desired_goal"].astype(np.float32)
            planning = use_planning and self.centroids_initialized
            if planning:
                self.planner.reset(goal)
            success = 0.0
            # Test-time horizon (paper uses a longer horizon than training, e.g.
            # 200 train / 500 test for the mazes — Section 5.2 / Figure 5).
            for _ in range(self.cfg.test_episode_steps):
                if planning:
                    a = self.planner.act(obs_dict["observation"])
                else:
                    a = self.agent.act(obs_dict["observation"], goal)
                obs_dict, reward, done, info = env.step(a)
                success = max(success, _episode_success(info, reward))
            successes += int(success > 0)
        self.env.set_eval(False)
        return successes / n_episodes

    # ------------------------------------------------------------------ main loop
    def train(self, total_steps: Optional[int] = None,
              checkpoint_path: Optional[str] = None, checkpoint_every: int = 0,
              save_training_state: bool = False,
              time_limit_seconds: Optional[float] = None) -> None:
        total_steps = total_steps or self.cfg.total_steps
        t0 = time.time()
        deadline = t0 + time_limit_seconds if time_limit_seconds is not None else None
        last_log = 0
        last_ckpt = 0
        while self.total_env_steps < total_steps:
            if deadline is not None and time.time() >= deadline:
                print(f"    >> time limit reached @ {self.total_env_steps} steps", flush=True)
                break
            self.collect()

            if self.total_env_steps >= self.cfg.train_after and len(self.buffer) > 1:
                collected = self.env.n * self.T * self.cfg.k_env
                n_updates = max(1, collected // self.cfg.env_steps_per_opt)
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

            if self.total_env_steps % self.cfg.eval_interval < self.env.n * self.T:
                sr = self.evaluate(self.cfg.eval_episodes)
                print(f"    >> eval success rate (long-horizon test): {sr:.2f}", flush=True)

            if checkpoint_path and checkpoint_every and \
                    self.total_env_steps - last_ckpt >= checkpoint_every:
                last_ckpt = self.total_env_steps
                self.save(checkpoint_path, include_training_state=save_training_state)
                print(f"    >> checkpoint saved @ {self.total_env_steps} steps", flush=True)

    def save(self, path: str, include_training_state: bool = False) -> None:
        d = dict(agent=self.agent.state_dict(), ae=self.ae.state_dict(),
                 landmarks=self.landmarks.state_dict(),
                 centroids_initialized=self.centroids_initialized)
        if include_training_state:
            d["training_state"] = dict(
                total_env_steps=self.total_env_steps,
                episodes_collected=self.episodes_collected,
                grad_step_count=self.grad_step_count,
                trainer_rng_state=self.rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(),
                buffer=self.buffer.state_dict(),
                actor_opt=self.agent.actor_opt.state_dict(),
                critic_opt=self.agent.critic_opt.state_dict(),
                value_opt=self.agent.value_opt.state_dict(),
                ae_opt=self.ae_opt.state_dict(),
                landmark_opt=self.landmark_opt.state_dict(),
            )
            if torch.cuda.is_available():
                d["training_state"]["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
        torch.save(d, path)

    def load(self, path: str, restore_training_state: bool = False) -> None:
        # weights_only=False: the checkpoint stores numpy normalizer statistics.
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(d["agent"])
        self.ae.load_state_dict(d["ae"])
        # Rebuild the landmark module if the checkpoint used a different N.
        ckpt_n = d["landmarks"]["centroids"].shape[0]
        if ckpt_n != self.landmarks.n_landmarks:
            self.landmarks = LatentLandmarks(ckpt_n, self.cfg.embedding_size).to(self.device)
            self.planner.landmarks = self.landmarks
        self.landmarks.load_state_dict(d["landmarks"])
        self.centroids_initialized = d["centroids_initialized"]
        if restore_training_state and "training_state" in d:
            ts = d["training_state"]
            self.total_env_steps = int(ts.get("total_env_steps", self.total_env_steps))
            self.episodes_collected = int(ts.get("episodes_collected", self.episodes_collected))
            self.grad_step_count = int(ts.get("grad_step_count", self.grad_step_count))
            if "trainer_rng_state" in ts:
                self.rng.bit_generator.state = ts["trainer_rng_state"]
            if "torch_rng_state" in ts:
                torch.set_rng_state(ts["torch_rng_state"].cpu())
            if "torch_cuda_rng_state" in ts and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ts["torch_cuda_rng_state"])
            if "buffer" in ts:
                self.buffer.load_state_dict(ts["buffer"])
            if "actor_opt" in ts:
                self.agent.actor_opt.load_state_dict(ts["actor_opt"])
            if "critic_opt" in ts:
                self.agent.critic_opt.load_state_dict(ts["critic_opt"])
            if "value_opt" in ts:
                self.agent.value_opt.load_state_dict(ts["value_opt"])
            if "ae_opt" in ts:
                self.ae_opt.load_state_dict(ts["ae_opt"])
            if "landmark_opt" in ts:
                self.landmark_opt.load_state_dict(ts["landmark_opt"])
