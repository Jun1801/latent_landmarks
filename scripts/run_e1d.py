#!/usr/bin/env python3
"""E1d: does MCTS still help when the execution itself is stochastic?

This is the "Noi 1,2,3" row from docs/SPEC_MCTS_Landmark_L3P.md:

  * Noi 1: the planner's landmark graph is built from a noisy distance estimate.
  * Noi 2: MCTS rollout/tree edges resample that noisy estimate on every use.
  * Noi 3: the executed macro-step duration is also noisy.

For feedback to have a consistent unit, this harness uses the critic-D distance
graph (env-step scale), like E1c. The user-facing `sigma` is the spec knob; it is
mapped to graph-D noise by `--graph-sigma-scale` and to env-step commitment noise
by `--exec-sigma-scale`.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config, list_envs
from l3p.envs import make_vec_env
from l3p.planning.graph_search import GraphSearch
from l3p.planning.mcts_planner import LandmarkMCTS, MCTSPlanner
from l3p.planning.noise import CriticEdgeFn, bootstrap_ci, dmax_candidates
from l3p.planning.planner import LatentPlanner
from l3p.trainer import L3PTrainer, _episode_success

BRANCHES = ["soft_floyd", "mcts_nofb", "mcts_fb"]
LABELS = {
    "soft_floyd": "Soft Floyd",
    "mcts_nofb": "MCTS no-feedback",
    "mcts_fb": "MCTS + recovery/feedback",
}


class IndexedNoisyCriticD:
    """Critic-D edge function plus additive Loai-2 noise, resampled per call."""

    def __init__(self, agent, nodes_t, sigma, rng):
        self.nodes = nodes_t
        self.nodes_np = nodes_t.detach().cpu().numpy()
        self._key_to_idx = {
            tuple(np.round(row, 6)): i for i, row in enumerate(self.nodes_np)
        }
        self.sigma = float(sigma)
        self.rng = rng
        clean = CriticEdgeFn(agent)
        m = nodes_t.shape[0]
        gi = nodes_t[:, None, :].expand(m, m, nodes_t.shape[1]).reshape(m * m, -1)
        gj = nodes_t[None, :, :].expand(m, m, nodes_t.shape[1]).reshape(m * m, -1)
        self.clean_matrix = clean(gi, gj).view(m, m).detach().cpu().numpy()

    def _idx(self, g):
        rows = g.detach().cpu().numpy()
        out = []
        for row in rows:
            key = tuple(np.round(row, 6))
            idx = self._key_to_idx.get(key)
            if idx is None:
                idx = int(np.argmin(np.linalg.norm(self.nodes_np - row[None, :], axis=1)))
            out.append(idx)
        return np.asarray(out, dtype=np.int64)

    def __call__(self, g1, g2):
        i = self._idx(g1)
        j = self._idx(g2)
        v = torch.as_tensor(self.clean_matrix[i, j], dtype=g1.dtype, device=g1.device)
        if self.sigma <= 0:
            return v
        eta = self.rng.normal(0.0, self.sigma, size=tuple(v.shape))
        noisy = v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)
        return noisy.clamp_min(0.0)


class CorrectedNoisyCriticD:
    """Overlay feedback EMA estimates, then keep sampling residual Loai-2 noise."""

    def __init__(self, base, v_exec):
        self.base = base
        self.nodes = base.nodes
        self.sigma = base.sigma
        self.rng = base.rng
        self.v_exec = v_exec

    def _idx(self, g):
        return self.base._idx(g)

    def __call__(self, g1, g2):
        i = self._idx(g1)
        j = self._idx(g2)
        v = torch.as_tensor(self.base.clean_matrix[i, j], dtype=g1.dtype, device=g1.device)
        if self.v_exec:
            v = v.clone()
            for (ii, jj), val in self.v_exec.items():
                sel = (i == ii) & (j == jj)
                if sel.any():
                    v[torch.as_tensor(sel)] = float(val)
        if self.sigma > 0:
            eta = self.rng.normal(0.0, self.sigma, size=tuple(v.shape))
            v = v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)
        return v.clamp_min(0.0)


def _perturb_commitment(k_pred, exec_sigma, rng):
    if exec_sigma <= 0:
        return max(1.0, float(round(k_pred)))
    return max(1.0, float(round(k_pred + rng.normal(0.0, exec_sigma))))


class SoftFloydE1d(LatentPlanner):
    """Static Soft Floyd on a noisy critic-D graph, plus noisy macro execution."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg,
                 graph_sigma, exec_sigma, noise_seed):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg)
        self.graph_sigma = graph_sigma
        self.exec_sigma = exec_sigma
        self.noise_seed = noise_seed
        self._exec_rng = np.random.default_rng(noise_seed + 17)

    @torch.no_grad()
    def reset(self, goal, extra_centroids=None):
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()
        nodes_t = torch.as_tensor(
            np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0),
            dtype=torch.float32, device=self.device)
        noisy = IndexedNoisyCriticD(self.agent, nodes_t, self.graph_sigma,
                                    np.random.default_rng(self.noise_seed))
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae, noisy).cpu().numpy()

    @torch.no_grad()
    def _replan(self, obs):
        super()._replan(obs)
        self.cnt = _perturb_commitment(self.cnt, self.exec_sigma, self._exec_rng)


class FeedbackMCTSE1d(MCTSPlanner):
    """MCTS under E1d with optional recovery/feedback/loop guard.

    PointMaze-only for now: obs == achieved_goal, so snapping and critic-D
    edge feedback can use the planner's observation directly.
    """

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg,
                 graph_sigma, exec_sigma, noise_seed, feedback=True,
                 rho=0.5, tau_reach=2.0, tau_progress=3.0, r_max=2, rng=None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg, rng=rng)
        self.graph_sigma = graph_sigma
        self.exec_sigma = exec_sigma
        self.noise_seed = noise_seed
        self.feedback = feedback
        self.rho = rho
        self.tau_reach = tau_reach
        self.tau_progress = tau_progress
        self.r_max = r_max
        self._exec_rng = np.random.default_rng(noise_seed + 17)

    @torch.no_grad()
    def reset(self, goal, extra_centroids=None):
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            self._admissible = self._noisy = None
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()
        self._nodes_t = torch.as_tensor(
            np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0),
            dtype=torch.float32, device=self.device)
        self._noisy = IndexedNoisyCriticD(self.agent, self._nodes_t, self.graph_sigma,
                                          np.random.default_rng(self.noise_seed))
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae, self._noisy).cpu().numpy()
        self._admissible = LandmarkMCTS.build_admissibility(self._nodes_t, self._noisy, self.cfg)

        self._v_exec = {}
        self._attempts = {}
        self._blacklist = set()
        self._macro_k = 0
        self._macro_start_i = None
        self._macro_start_z = None
        self._cur_j = None
        self.stats = dict(macros=0, reached=0, progressed=0, stuck=0,
                          blacklisted=0, corrections=0)

    def _snap(self, z):
        return int(np.argmin(np.linalg.norm(self.landmark_goals - z[None, :], axis=1)))

    def _critic_dist(self, z, target_goal):
        return float(self.agent.distance_after_action(z[None, :], target_goal[None, :])[0])

    def _observed_edge(self, i, j):
        return float(self._noisy(self._nodes_t[i:i + 1], self._nodes_t[j:j + 1]).item())

    def _corrected_fn(self):
        return CorrectedNoisyCriticD(self._noisy, self._v_exec)

    def _observe(self, z_end):
        i, j = self._macro_start_i, self._cur_j
        if j is None or j >= self.n_landmarks:
            return
        c_j = self.landmark_goals[j]
        dist_to_goal = self._critic_dist(z_end, c_j)
        dist_travelled = self._critic_dist(self._macro_start_z, z_end)
        realized = self._macro_k + max(0.0, dist_to_goal)

        prev = self._v_exec.get((i, j), self._observed_edge(i, j))
        self._v_exec[(i, j)] = (1 - self.rho) * prev + self.rho * realized
        self.stats["corrections"] += 1

        if dist_to_goal <= self.tau_reach:
            self._attempts.clear()
            self.stats["reached"] += 1
        elif dist_travelled >= self.tau_progress:
            self._attempts[j] = self._attempts.get(j, 0) + 1
            self.stats["progressed"] += 1
            if self._attempts[j] >= self.r_max:
                self._blacklist.add(j)
                self.stats["blacklisted"] += 1
        else:
            self._blacklist.add(j)
            self.stats["stuck"] += 1
            self.stats["blacklisted"] += 1

    def finalize(self, obs):
        if self.feedback and self._cur_j is not None:
            self._observe(obs)
            self._cur_j = None

    @torch.no_grad()
    def act(self, obs, noise_scale=0.0, random_prob=0.0):
        if self.n_landmarks == 0:
            return self.agent.act(obs, self.goal, noise_scale, random_prob)
        if self.cnt > 1.0:
            self.cnt -= 1.0
        else:
            if self.feedback and self._cur_j is not None:
                self._observe(obs)
            self._replan(obs)
        self._macro_k += 1
        return self.agent.act(obs, self._current_subgoal(), noise_scale, random_prob)

    @torch.no_grad()
    def _replan(self, obs):
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)
        d_s2c = -self.agent.distance_after_action(obs, candidates)
        nodes_t = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)

        if self.feedback:
            value_fn = self._corrected_fn()
            goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
            d_c2g = self.gs.distances_to_goal(self.centroids, goal_t, self.ae, value_fn).cpu().numpy()
            admissible = LandmarkMCTS.build_admissibility(nodes_t, value_fn, self.cfg)
        else:
            value_fn, d_c2g, admissible = self._noisy, self.d_c2g, self._admissible

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True
        for b in self._blacklist if self.feedback else ():
            if b < self.n_landmarks:
                mask[b] = True

        mcts = LandmarkMCTS(n_landmarks=self.n_landmarks, value_fn=value_fn,
                            nodes_t=nodes_t, d_c2g_heuristic=d_c2g, cfg=self.cfg,
                            rng=self.rng, admissible=admissible)
        best_idx, _ = mcts.search(d_s2c, mask=mask)
        if best_idx is None:
            best_idx = self.n_landmarks

        self.subg_idx = best_idx
        k_pred = max(1.0, float(round(-d_s2c[best_idx])))
        self.cnt = _perturb_commitment(k_pred, self.exec_sigma, self._exec_rng)
        self.prev_landmark = self.subg_idx
        self.stats["macros"] += 1
        if self.feedback:
            self._macro_k = 0
            self._macro_start_i = self._snap(obs)
            self._macro_start_z = np.asarray(obs, dtype=np.float32)
            self._cur_j = best_idx


def make_branch(name, trainer, cfg, sigma, noise_seed, args):
    graph_sigma = sigma * args.graph_sigma_scale
    exec_sigma = sigma * args.exec_sigma_scale
    a, lm, ae, gs = trainer.agent, trainer.landmarks, trainer.ae, trainer.graph_search
    if name == "soft_floyd":
        return SoftFloydE1d(a, lm, ae, gs, cfg, graph_sigma, exec_sigma, noise_seed)
    return FeedbackMCTSE1d(
        a, lm, ae, gs, cfg, graph_sigma, exec_sigma, noise_seed,
        feedback=(name == "mcts_fb"), rho=args.rho, tau_reach=args.tau_reach,
        tau_progress=args.tau_progress, r_max=args.r_max,
        rng=np.random.default_rng(noise_seed + 1))


def d_edge_matrix(trainer):
    with torch.no_grad():
        lg = trainer.ae.decode(trainer.landmarks.centroids.detach())
    edge = CriticEdgeFn(trainer.agent)
    m = lg.shape[0]
    gi = lg[:, None, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
    gj = lg[None, :, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
    return edge(gi, gj).view(m, m).cpu().numpy()


def evaluate_one(trainer, planner, episode_seed):
    trainer.env.set_eval(True)
    env = trainer.env.envs[0]
    env.rng = np.random.default_rng(episode_seed)
    obs_dict = env.reset()
    goal = obs_dict["desired_goal"].astype(np.float32)
    planner.reset(goal)
    success = 0.0
    for _ in range(trainer.cfg.test_episode_steps):
        a = planner.act(obs_dict["observation"])
        obs_dict, reward, _, info = env.step(a)
        success = max(success, _episode_success(info, reward))
    if hasattr(planner, "finalize"):
        planner.finalize(obs_dict["observation"])
    trainer.env.set_eval(False)
    stats = getattr(planner, "stats", {})
    return int(success > 0), {k: float(v) for k, v in stats.items()}


def calibrate_d_max(trainer, n_episodes, base_seed):
    candidates = dmax_candidates(d_edge_matrix(trainer))
    best, sweep = (candidates[0], -1.0), []
    for dmax in candidates:
        trainer.cfg.d_max = dmax
        succ = 0.0
        for ep in range(n_episodes):
            es = base_seed * 1_000_003 + ep
            pl = SoftFloydE1d(trainer.agent, trainer.landmarks, trainer.ae,
                              trainer.graph_search, trainer.cfg,
                              graph_sigma=0.0, exec_sigma=0.0, noise_seed=es)
            s, _ = evaluate_one(trainer, pl, es)
            succ += s
        sr = succ / n_episodes
        sweep.append((dmax, sr))
        if sr > best[1]:
            best = (dmax, sr)
    trainer.cfg.d_max = best[0]
    return best[0], sweep


def run_point(trainer, sigma, name, args, base_seed):
    outcomes, stats, t_tot = [], [], 0.0
    for ep in range(args.episodes):
        es = base_seed * 1_000_003 + ep
        pl = make_branch(name, trainer, trainer.cfg, sigma, es, args)
        t0 = time.time()
        s, st = evaluate_one(trainer, pl, es)
        t_tot += time.time() - t0
        outcomes.append(s)
        stats.append(st)
    return outcomes, stats, t_tot


def summarize_stats(items):
    if not items:
        return {}
    keys = sorted({k for item in items for k in item})
    return {k: float(np.mean([item.get(k, 0.0) for item in items])) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze", choices=list_envs())
    p.add_argument("--load", type=str, default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.3])
    p.add_argument("--graph-sigma-scale", type=float, default=1.0,
                   help="graph-D noise std = sigma * this scale")
    p.add_argument("--exec-sigma-scale", type=float, default=10.0,
                   help="macro K noise std in env steps = sigma * this scale")
    p.add_argument("--rho", type=float, default=0.5)
    p.add_argument("--tau-reach", type=float, default=2.0)
    p.add_argument("--tau-progress", type=float, default=3.0)
    p.add_argument("--r-max", type=int, default=2)
    p.add_argument("--sanity-tol", type=float, default=0.2)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--mcts-n-simulations", type=int, default=200)
    p.add_argument("--mcts-rollout-horizon", type=int, default=10)
    p.add_argument("--d-max", type=float, default=None)
    p.add_argument("--calibrate-episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="logs/e1d_results.json")
    p.add_argument("--plot", type=str, default="logs/e1d_curve.png")
    args = p.parse_args()

    if 0.0 not in args.sigmas:
        args.sigmas = [0.0] + list(args.sigmas)

    print(f"E1d on {args.env}: Loai-2 noise at Noi 1,2,3 with MCTS "
          "recovery/feedback/loop guard. Graph substrate = critic-D env-step scale.")
    cfg = get_config(args.env, seed=args.seeds[0],
                     mcts_n_simulations=args.mcts_n_simulations,
                     mcts_rollout_horizon=args.mcts_rollout_horizon)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)
    if trainer.env.obs_dim != trainer.env.goal_dim:
        print("ERROR: E1d feedback currently requires obs_dim == goal_dim because "
              "critic-D graph edges use goal coordinates as states. Use PointMaze, "
              "or add an env-specific goal->observation lifting adapter.", file=sys.stderr)
        sys.exit(2)
    if not trainer.centroids_initialized:
        print("ERROR: checkpoint has no initialized landmark centroids.", file=sys.stderr)
        sys.exit(1)

    if args.d_max is not None:
        trainer.cfg.d_max = args.d_max
        print(f"Using fixed d_max={args.d_max}.")
    else:
        best, sweep = calibrate_d_max(trainer, args.calibrate_episodes, args.seeds[0])
        print("d_max calibration (clean-D Soft Floyd, spec R3):")
        for dmax, sr in sweep:
            print(f"    d_max={dmax:7.3f} -> {sr:.2f}" + ("   <- chosen" if dmax == best else ""))

    print(f"Config: d_max={trainer.cfg.d_max:.3f}  sims={trainer.cfg.mcts_n_simulations}  "
          f"rollout={trainer.cfg.mcts_rollout_horizon}  graph_sigma_scale={args.graph_sigma_scale}  "
          f"exec_sigma_scale={args.exec_sigma_scale}  feedback=(rho={args.rho}, "
          f"tau_reach={args.tau_reach}, tau_progress={args.tau_progress}, r_max={args.r_max})  "
          f"seeds={args.seeds}  eps/seed={args.episodes}", flush=True)

    results, mcts_t, mcts_n = [], 0.0, 0
    for sigma in args.sigmas:
        pooled = {b: [] for b in BRANCHES}
        stats_pooled = {b: [] for b in BRANCHES}
        per_seed = {}
        for seed in args.seeds:
            per_seed[seed] = {}
            for b in BRANCHES:
                out, st, t = run_point(trainer, sigma, b, args, seed)
                pooled[b].extend(out)
                stats_pooled[b].extend(st)
                per_seed[seed][b] = float(np.mean(out))
                print(f"  done sigma={sigma:.2f} seed={seed} {b}: "
                      f"mean={per_seed[seed][b]:.2f} time={t:.1f}s", flush=True)
                if b != "soft_floyd":
                    mcts_t += t
                    mcts_n += args.episodes
        agg, rng = {}, np.random.default_rng(args.seeds[0])
        for b in BRANCHES:
            mean, lo, hi = bootstrap_ci(pooled[b], n_boot=args.n_boot, rng=rng)
            agg[b] = dict(mean=mean, ci_low=lo, ci_high=hi, n=len(pooled[b]),
                          stats=summarize_stats(stats_pooled[b]))
        results.append(dict(sigma=sigma, aggregate=agg, per_seed=per_seed))
        print(f"sigma={sigma:.2f}  " + "  ".join(
            f"{LABELS[b]}={agg[b]['mean']:.2f}[{agg[b]['ci_low']:.2f},{agg[b]['ci_high']:.2f}]"
            for b in BRANCHES), flush=True)
        if agg["mcts_fb"]["stats"]:
            s = agg["mcts_fb"]["stats"]
            print("  mcts_fb avg/episode: macros={macros:.1f} reached={reached:.1f} "
                  "progressed={progressed:.1f} stuck={stuck:.1f} blacklist={blacklisted:.1f} "
                  "corrections={corrections:.1f}".format(**s), flush=True)

        if sigma == 0.0:
            gap = abs(agg["mcts_fb"]["mean"] - agg["soft_floyd"]["mean"])
            status = "passed" if gap <= args.sanity_tol else "WARNING"
            print(f"  >> sigma=0 sanity: |mcts_fb - floyd| = {gap:.2f} ({status})")

    print(f"\nMCTS latency: {mcts_t / max(1, mcts_n):.2f} s/episode.", flush=True)
    _save(args.out, results, args, trainer)
    print(f"Saved results to {args.out}")
    _plot(args.plot, results)


def _save(path, results, args, trainer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(meta=dict(env=args.env, load=args.load, seeds=args.seeds, episodes=args.episodes,
                                 sigmas=args.sigmas,
                                 graph_sigma_scale=args.graph_sigma_scale,
                                 exec_sigma_scale=args.exec_sigma_scale,
                                 rho=args.rho, tau_reach=args.tau_reach,
                                 tau_progress=args.tau_progress, r_max=args.r_max,
                                 d_max=trainer.cfg.d_max,
                                 mcts_n_simulations=trainer.cfg.mcts_n_simulations,
                                 mcts_rollout_horizon=trainer.cfg.mcts_rollout_horizon),
                       results=results), f, indent=2)


def _plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    xs = [r["sigma"] for r in results]
    plt.figure(figsize=(6, 4))
    for b in BRANCHES:
        mean = [r["aggregate"][b]["mean"] for r in results]
        lo = [r["aggregate"][b]["ci_low"] for r in results]
        hi = [r["aggregate"][b]["ci_high"] for r in results]
        line, = plt.plot(xs, mean, "o-", label=LABELS[b])
        plt.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    plt.xlabel("sigma (Loai-2 noise knob)")
    plt.ylabel("success rate")
    plt.title("E1d: noisy graph + noisy execution (mean +/- 95% CI)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
