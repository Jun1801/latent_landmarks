#!/usr/bin/env python3
"""E1a: does MCTS-over-landmarks degrade slower than Soft Floyd as the
landmark-landmark distance estimate V gets noisier? (docs/SPEC_MCTS_Landmark_L3P.md,
Sections 5-8, 10 -- Phase 1 scope: Loai-2 noise, Noi 1&2 only, no beta*sigma_V
uncertainty bonus.)

Loads a trained PointMaze-Hard checkpoint and, for each sigma in the sweep,
evaluates THREE planners (spec Sec 7's mandatory baselines for E1a):
  * "soft_floyd"   -- the existing LatentPlanner (Algorithm 1), unmodified.
  * "naive_replan" -- Baseline 2 (L3P Fig. 8): re-plans via the same Algorithm-1
    formula but every step, no K-step commitment. Separates "MCTS wins because it
    re-plans more often" from "MCTS wins because of the tree-search machinery".
  * "mcts"         -- MCTSPlanner (UCT + sample-based rollout). Note: in Phase 1
    (no uncertainty bonus) this IS the "MCTS-beta=0" ablation of E1b.

FAIRNESS (the whole point of E1a):
  1. d_max is CALIBRATED ONCE on the noise-free Soft Floyd baseline (spec R3:
     d_max is very sensitive and must be tuned for the checkpoint's V-scale; the
     paper's fixed default assumes a different V magnitude and pins the baseline
     at ~0 here). The same tuned d_max is used by all planners and all seeds, so
     the comparison is never confounded by an untuned baseline or by "we tuned
     MCTS better". Calibration uses only the first seed; reporting seeds inherit
     the fixed value (no tuning on the reported data).
  2. Paired episodes: for a given (seed, sigma), all three planners are scored on
     the SAME start/goal pairs -- the env RNG is reseeded identically before each
     planner's turn on each episode (spec Sec 5.2).
  3. Same noisy V_obs seed per episode across planners (best-effort: each planner
     calls value_fn a different number of times, so only the STARTING draw is
     synchronized -- Soft Floyd/Naive build their graph from one batched draw;
     MCTS additionally samples per simulation).
  4. Fixed MCTS budget (mcts_n_simulations / rollout_horizon) across all sigma and
     seeds; reported below with per-episode MCTS latency (spec Sec 10 metric).

STATISTICS (spec Sec 10): runs multiple seeds and reports, per (sigma, planner),
the mean success rate with a percentile bootstrap CI over the pooled per-episode
outcomes. The sigma=0 sanity check (acceptance criterion #1: MCTS ~= Soft Floyd)
is evaluated on the aggregate and aborts early (before the costly sigma>0 sweep)
if it fails.

Caveat (spec's own risk R2): this runs on PointMaze-Hard, not AntMaze-Hard --
AntMaze needs MuJoCo (see docs/ENVIRONMENTS.md). The landmark graph here is
smaller/shorter-horizon than the paper's setting, and the flat policy already
solves the task ~0.9, so planning has thin headroom over "no planning".

Example (the ~2.5h "real" run):
    python scripts/run_e1a.py --load checkpoint/l3p_pointmaze_full.pt \
        --seeds 0 1 2 --episodes 50 --sigmas 0 0.1 0.3 0.5
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
from l3p.planning.baselines import NaiveReplanPlanner
from l3p.planning.mcts_planner import MCTSPlanner
from l3p.planning.noise import (NoisyValueFn, ValueOverrideAgent, dmax_candidates,
                                bootstrap_ci)
from l3p.planning.planner import LatentPlanner
from l3p.trainer import L3PTrainer

PLANNER_NAMES = ["soft_floyd", "naive_replan", "mcts"]
PLANNER_LABELS = {"soft_floyd": "Soft Floyd", "naive_replan": "Naive re-plan", "mcts": "MCTS"}


def make_planner(name, trainer, noisy_agent, episode_seed):
    if name == "soft_floyd":
        return LatentPlanner(noisy_agent, trainer.landmarks, trainer.ae,
                             trainer.graph_search, trainer.cfg)
    if name == "naive_replan":
        return NaiveReplanPlanner(noisy_agent, trainer.landmarks, trainer.ae,
                                  trainer.graph_search, trainer.cfg)
    return MCTSPlanner(noisy_agent, trainer.landmarks, trainer.ae,
                       trainer.graph_search, trainer.cfg,
                       rng=np.random.default_rng(episode_seed + 1))


def run_sigma_sweep(trainer, sigma, n_episodes, base_seed):
    """Paired per-episode evaluation for one seed. Returns
    {name: [0/1 per episode]} plus the mean MCTS per-episode latency (seconds)."""
    outcomes = {name: [] for name in PLANNER_NAMES}
    mcts_time, mcts_eps = 0.0, 0
    for ep in range(n_episodes):
        episode_seed = base_seed * 1_000_003 + ep
        for name in PLANNER_NAMES:
            noisy_value = NoisyValueFn(trainer.agent.value, sigma=sigma,
                                       rng=np.random.default_rng(episode_seed))
            noisy_agent = ValueOverrideAgent(trainer.agent, noisy_value)
            planner = make_planner(name, trainer, noisy_agent, episode_seed)
            trainer.env.envs[0].rng = np.random.default_rng(episode_seed)
            t0 = time.time()
            s = trainer.evaluate(1, planner=planner)
            if name == "mcts":
                mcts_time += time.time() - t0
                mcts_eps += 1
            outcomes[name].append(int(s > 0))
    return outcomes, (mcts_time / max(1, mcts_eps))


def _floyd_success(trainer, n_episodes, base_seed):
    """Noise-free Soft Floyd success (paired reseeding). For d_max calibration."""
    succ = 0.0
    for ep in range(n_episodes):
        es = base_seed * 1_000_003 + ep
        fp = LatentPlanner(trainer.agent, trainer.landmarks, trainer.ae,
                           trainer.graph_search, trainer.cfg)
        trainer.env.envs[0].rng = np.random.default_rng(es)
        succ += trainer.evaluate(1, planner=fp)
    return succ / n_episodes


def calibrate_d_max(trainer, n_episodes, base_seed):
    """SPEC R3: pick the d_max (from candidates derived from THIS checkpoint's own
    pairwise-V distribution) that maximizes the noise-free Soft Floyd baseline,
    then use it for every planner/seed. Returns (best_d_max, [(d_max, success)])."""
    with torch.no_grad():
        lg = trainer.ae.decode(trainer.landmarks.centroids.detach())
        m = lg.shape[0]
        gi = lg[:, None, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
        gj = lg[None, :, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
        V = trainer.agent.value(gi, gj).view(m, m).cpu().numpy()
    candidates = dmax_candidates(V)

    sweep, best = [], (candidates[0], -1.0)
    for dmax in candidates:
        trainer.cfg.d_max = dmax
        sr = _floyd_success(trainer, n_episodes, base_seed)
        sweep.append((dmax, sr))
        if sr > best[1]:
            best = (dmax, sr)
    trainer.cfg.d_max = best[0]
    return best[0], sweep


def aggregate(pooled, n_boot, alpha, seed):
    """pooled: {name: [0/1,...]} pooled across seeds -> {name: (mean, lo, hi, n)}."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in PLANNER_NAMES:
        mean, lo, hi = bootstrap_ci(pooled[name], n_boot=n_boot, alpha=alpha, rng=rng)
        out[name] = dict(mean=mean, ci_low=lo, ci_high=hi, n=len(pooled[name]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze", choices=list_envs())
    p.add_argument("--load", type=str, default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="reporting seeds; pooled for mean +/- bootstrap CI (spec Sec 10)")
    p.add_argument("--episodes", type=int, default=50, help="eval episodes per (planner, sigma, seed)")
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    p.add_argument("--sanity-tol", type=float, default=0.15,
                   help="max allowed |mean_mcts - mean_floyd| at sigma=0 before aborting")
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the CI")
    p.add_argument("--out", type=str, default="logs/e1a_results.json")
    p.add_argument("--plot", type=str, default="logs/e1a_curve.png")
    p.add_argument("--mcts-n-simulations", type=int, default=None,
                   help="override Config.mcts_n_simulations (fixed across all sigma/seeds)")
    p.add_argument("--mcts-rollout-horizon", type=int, default=None,
                   help="override Config.mcts_rollout_horizon")
    p.add_argument("--d-max", type=float, default=None,
                   help="fix d_max explicitly and SKIP calibration (default: auto-calibrate, spec R3)")
    p.add_argument("--calibrate-episodes", type=int, default=20,
                   help="episodes used to calibrate d_max on the noise-free Soft Floyd baseline")
    args = p.parse_args()

    if 0.0 not in args.sigmas:
        args.sigmas = [0.0] + list(args.sigmas)

    print(f"E1a on {args.env}. Caveat (spec risk R2): short/easy envs can have "
          "thin planning headroom; MuJoCo envs require gymnasium-robotics/mujoco.")

    cfg_overrides = {}
    if args.mcts_n_simulations is not None:
        cfg_overrides["mcts_n_simulations"] = args.mcts_n_simulations
    if args.mcts_rollout_horizon is not None:
        cfg_overrides["mcts_rollout_horizon"] = args.mcts_rollout_horizon
    cfg = get_config(args.env, seed=args.seeds[0], **cfg_overrides)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)
    if not trainer.centroids_initialized:
        print("ERROR: checkpoint has no initialized landmark centroids; "
              "train longer before running E1a.", file=sys.stderr)
        sys.exit(1)

    # ---- fairness step 1: calibrate d_max ONCE, fix for all seeds ----
    if args.d_max is not None:
        trainer.cfg.d_max = args.d_max
        print(f"Using fixed d_max={args.d_max} (calibration skipped).")
    else:
        best_dmax, sweep = calibrate_d_max(trainer, args.calibrate_episodes, base_seed=args.seeds[0])
        print("d_max calibration (noise-free Soft Floyd, spec R3):")
        for dmax, sr in sweep:
            print(f"    d_max={dmax:7.3f} -> floyd success {sr:.2f}"
                  + ("   <- chosen" if dmax == best_dmax else ""))
        if sweep and max(sr for _, sr in sweep) < 0.1:
            print("\n*** WARNING: even the best-tuned Soft Floyd baseline is near 0. E1a "
                  "results would be meaningless; retrain / pick a better checkpoint "
                  "(spec Sec 8 KB3).\n", file=sys.stderr)
    print(f"Config: d_max={trainer.cfg.d_max:.3f}  mcts_n_simulations={trainer.cfg.mcts_n_simulations}  "
          f"mcts_rollout_horizon={trainer.cfg.mcts_rollout_horizon}  "
          f"seeds={args.seeds}  episodes/seed={args.episodes}")

    # ---- run: sigma outer, seed inner (so sigma=0 finishes first for early sanity abort) ----
    results = []
    total_mcts_time, total_mcts_calls = 0.0, 0
    for sigma in args.sigmas:
        pooled = {name: [] for name in PLANNER_NAMES}
        per_seed = {}
        for seed in args.seeds:
            outcomes, mcts_lat = run_sigma_sweep(trainer, sigma, args.episodes, base_seed=seed)
            per_seed[seed] = {name: float(np.mean(outcomes[name])) for name in PLANNER_NAMES}
            for name in PLANNER_NAMES:
                pooled[name].extend(outcomes[name])
            total_mcts_time += mcts_lat * args.episodes
            total_mcts_calls += args.episodes
        agg = aggregate(pooled, args.n_boot, 0.05, args.seeds[0])
        results.append(dict(sigma=sigma, aggregate=agg, per_seed=per_seed))
        line = f"sigma={sigma:.2f}  " + "  ".join(
            f"{PLANNER_LABELS[n]}={agg[n]['mean']:.2f} [{agg[n]['ci_low']:.2f},{agg[n]['ci_high']:.2f}]"
            for n in PLANNER_NAMES)
        print(line, flush=True)

        if sigma == 0.0:
            gap = abs(agg["mcts"]["mean"] - agg["soft_floyd"]["mean"])
            if gap > args.sanity_tol:
                print(f"\n*** SANITY CHECK FAILED: |mcts-floyd| at sigma=0 = {gap:.2f} "
                      f"> tol {args.sanity_tol:.2f} (spec Sec 8/10 KB3). Aborting before the "
                      f"costly sigma>0 sweep -- debug the fairness/setup first. ***\n", file=sys.stderr)
                _save(args.out, results, args, trainer)
                sys.exit(1)
            print(f"  >> sigma=0 sanity check passed (|mcts-floyd|={gap:.2f} <= {args.sanity_tol:.2f})")

    print(f"\nMCTS latency: {total_mcts_time / max(1, total_mcts_calls):.2f} s/episode "
          f"(budget {trainer.cfg.mcts_n_simulations} sims).")
    _save(args.out, results, args, trainer)
    print(f"Saved results to {args.out}")
    _plot(args.plot, results)


def _save(path, results, args, trainer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(
            meta=dict(load=args.load, seeds=args.seeds, episodes=args.episodes,
                      env=args.env, sigmas=args.sigmas, d_max=trainer.cfg.d_max,
                      mcts_n_simulations=trainer.cfg.mcts_n_simulations,
                      mcts_rollout_horizon=trainer.cfg.mcts_rollout_horizon,
                      n_boot=args.n_boot),
            results=results), f, indent=2)


def _plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (results JSON still saved).")
        return
    sigmas = [r["sigma"] for r in results]
    plt.figure(figsize=(6, 4))
    for name in PLANNER_NAMES:
        mean = [r["aggregate"][name]["mean"] for r in results]
        lo = [r["aggregate"][name]["ci_low"] for r in results]
        hi = [r["aggregate"][name]["ci_high"] for r in results]
        line, = plt.plot(sigmas, mean, "o-", label=PLANNER_LABELS[name])
        plt.fill_between(sigmas, lo, hi, alpha=0.2, color=line.get_color())
    plt.xlabel("sigma (V noise std)")
    plt.ylabel("success rate")
    plt.title("E1a: PointMaze-Hard long-horizon test (mean +/- 95% bootstrap CI)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
