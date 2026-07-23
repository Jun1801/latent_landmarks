#!/usr/bin/env python3
"""E1b: does the UNCERTAINTY BONUS contribute on its own, beyond plain tree
search? (docs/SPEC_MCTS_Landmark_L3P.md Sec 3, Sec 8 row E1b.)

Compares three MCTS variants under the SAME heterogeneous per-edge oracle
uncertainty (Sec 3, Cach 1 -- some landmark-landmark edges are secretly noisier,
decorrelated from V; MCTS is told the true sigma_ij):
  * "none"  -- plain MCTS = the MCTS-beta=0 ablation (Phase 1 planner).
  * "alpha" -- risk penalty in the reward: edge cost -= lambda_risk * sigma
               (avoid uncertain edges).
  * "beta"  -- exploration bonus in UCT: score += beta_unc * sigma
               (probe uncertain edges to pin down their value).

The independent variable is sigma_hi (magnitude of the unreliable edges); the
SET of unreliable edges is fixed per episode (seeded) and shared across variants
and across sigma_hi levels, so only the magnitude changes. sigma_hi=0 is the
sanity point: the sigma matrix is all-zero, so all three variants must coincide.

FAIRNESS: d_max is calibrated ONCE on the clean (noise-free) Soft Floyd baseline
and shared by all variants (same as E1a, spec R3). For a given (sigma_hi, seed,
episode) all three variants get the same env start/goal and the same noise seed.

HONEST EXPECTATION (stated up front): Loai-2 noise is zero-mean, and MCTS with
many simulations already averages it out, so 'none' is fairly robust on its own.
'beta' may help slightly (better simulation allocation to noisy edges); 'alpha'
may even hurt (it avoids good-but-noisy edges). A small/null separation is a
legitimate outcome and means the uncertainty win, if any, needs BIASED noise
(Loai 1 = E1c) to show clearly -- report accordingly.

Example (~2.6h):
    python scripts/run_e1b.py --load l3p_pointmaze_full.pt --seeds 0 1 \
        --episodes 40 --sigma-hi 0 0.3 0.5 --mcts-n-simulations 120
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.planning.mcts_planner import UncertaintyMCTSPlanner
from l3p.planning.noise import dmax_candidates, bootstrap_ci
from l3p.planning.planner import LatentPlanner
from l3p.trainer import L3PTrainer

VARIANTS = ["none", "alpha", "beta"]
LABELS = {"none": "MCTS (β=0)", "alpha": "MCTS+α (risk penalty)", "beta": "MCTS+β (UCT bonus)"}


def _floyd_success(trainer, n_episodes, base_seed):
    succ = 0.0
    for ep in range(n_episodes):
        es = base_seed * 1_000_003 + ep
        fp = LatentPlanner(trainer.agent, trainer.landmarks, trainer.ae,
                           trainer.graph_search, trainer.cfg)
        trainer.env.envs[0].rng = np.random.default_rng(es)
        succ += trainer.evaluate(1, planner=fp)
    return succ / n_episodes


def calibrate_d_max(trainer, n_episodes, base_seed):
    """Same fair calibration as E1a: tune d_max on the clean Soft Floyd baseline
    (spec R3), reuse for every MCTS variant."""
    with torch.no_grad():
        lg = trainer.ae.decode(trainer.landmarks.centroids.detach())
        m = lg.shape[0]
        gi = lg[:, None, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
        gj = lg[None, :, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
        V = trainer.agent.value(gi, gj).view(m, m).cpu().numpy()
    best, sweep = (dmax_candidates(V)[0], -1.0), []
    for dmax in dmax_candidates(V):
        trainer.cfg.d_max = dmax
        sr = _floyd_success(trainer, n_episodes, base_seed)
        sweep.append((dmax, sr))
        if sr > best[1]:
            best = (dmax, sr)
    trainer.cfg.d_max = best[0]
    return best[0], sweep


def run_point(trainer, base_cfg, sigma_hi, variant, args, base_seed):
    """Per-episode paired evaluation of one variant at one sigma_hi/seed. Returns
    (per-episode 0/1 list, total mcts seconds)."""
    cfg = replace(base_cfg, mcts_uncertainty_mode=variant,
                  mcts_lambda_risk=args.lambda_risk, mcts_beta_uncertainty=args.beta_unc)
    outcomes, t_tot = [], 0.0
    for ep in range(args.episodes):
        es = base_seed * 1_000_003 + ep
        planner = UncertaintyMCTSPlanner(
            trainer.agent, trainer.landmarks, trainer.ae, trainer.graph_search, cfg,
            frac_high=args.frac_high, sigma_hi=sigma_hi, sigma_lo=args.sigma_lo,
            noise_seed=es, rng=np.random.default_rng(es + 1))
        trainer.env.envs[0].rng = np.random.default_rng(es)
        t0 = time.time()
        s = trainer.evaluate(1, planner=planner)
        t_tot += time.time() - t0
        outcomes.append(int(s > 0))
    return outcomes, t_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--load", type=str, default="l3p_pointmaze_full.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--sigma-hi", type=float, nargs="+", default=[0.0, 0.3, 0.5],
                   help="magnitude of the unreliable edges (independent variable)")
    p.add_argument("--frac-high", type=float, default=0.3,
                   help="fraction of landmark-landmark edges that are unreliable")
    p.add_argument("--sigma-lo", type=float, default=0.0, help="sigma of the reliable edges")
    p.add_argument("--lambda-risk", type=float, default=3.0, help="alpha reward-penalty weight")
    p.add_argument("--beta-unc", type=float, default=1.0, help="beta UCT-bonus weight")
    p.add_argument("--sanity-tol", type=float, default=0.15,
                   help="max spread across variants at sigma_hi=0 before aborting")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--mcts-n-simulations", type=int, default=120)
    p.add_argument("--mcts-rollout-horizon", type=int, default=10)
    p.add_argument("--d-max", type=float, default=None)
    p.add_argument("--calibrate-episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="logs/e1b_results.json")
    p.add_argument("--plot", type=str, default="logs/e1b_curve.png")
    args = p.parse_args()

    if 0.0 not in args.sigma_hi:
        args.sigma_hi = [0.0] + list(args.sigma_hi)

    print("E1b: uncertainty-bonus ablation (none / alpha / beta) under heterogeneous "
          "per-edge oracle uncertainty. Caveat R2: PointMaze-Hard, flat policy ~0.9.")

    cfg = get_config("PointMaze", seed=args.seeds[0],
                     mcts_n_simulations=args.mcts_n_simulations,
                     mcts_rollout_horizon=args.mcts_rollout_horizon)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)
    if not trainer.centroids_initialized:
        print("ERROR: checkpoint has no initialized landmark centroids.", file=sys.stderr)
        sys.exit(1)

    if args.d_max is not None:
        trainer.cfg.d_max = args.d_max
        print(f"Using fixed d_max={args.d_max}.")
    else:
        best, sweep = calibrate_d_max(trainer, args.calibrate_episodes, args.seeds[0])
        print("d_max calibration (clean Soft Floyd, spec R3):")
        for dmax, sr in sweep:
            print(f"    d_max={dmax:7.3f} -> {sr:.2f}" + ("   <- chosen" if dmax == best else ""))
    base_cfg = replace(trainer.cfg)   # carries the calibrated d_max + mcts budget
    print(f"Config: d_max={base_cfg.d_max:.3f}  sims={base_cfg.mcts_n_simulations}  "
          f"frac_high={args.frac_high}  lambda_risk={args.lambda_risk}  beta_unc={args.beta_unc}  "
          f"seeds={args.seeds}  eps/seed={args.episodes}")

    results, mcts_t, mcts_n = [], 0.0, 0
    for sigma_hi in args.sigma_hi:
        pooled = {v: [] for v in VARIANTS}
        per_seed = {}
        for seed in args.seeds:
            per_seed[seed] = {}
            for v in VARIANTS:
                out, t = run_point(trainer, base_cfg, sigma_hi, v, args, seed)
                pooled[v].extend(out)
                per_seed[seed][v] = float(np.mean(out))
                mcts_t += t
                mcts_n += args.episodes
        agg = {}
        rng = np.random.default_rng(args.seeds[0])
        for v in VARIANTS:
            mean, lo, hi = bootstrap_ci(pooled[v], n_boot=args.n_boot, rng=rng)
            agg[v] = dict(mean=mean, ci_low=lo, ci_high=hi, n=len(pooled[v]))
        results.append(dict(sigma_hi=sigma_hi, aggregate=agg, per_seed=per_seed))
        print(f"sigma_hi={sigma_hi:.2f}  " + "  ".join(
            f"{LABELS[v]}={agg[v]['mean']:.2f}[{agg[v]['ci_low']:.2f},{agg[v]['ci_high']:.2f}]"
            for v in VARIANTS), flush=True)

        if sigma_hi == 0.0:
            spread = max(agg[v]["mean"] for v in VARIANTS) - min(agg[v]["mean"] for v in VARIANTS)
            if spread > args.sanity_tol:
                print(f"\n*** SANITY FAILED: variant spread at sigma_hi=0 = {spread:.2f} "
                      f"> {args.sanity_tol}. With no uncertainty the bonuses must be inert "
                      f"and all variants coincide. Aborting. ***\n", file=sys.stderr)
                _save(args.out, results, args, base_cfg)
                sys.exit(1)
            print(f"  >> sigma_hi=0 sanity passed (variant spread={spread:.2f} <= {args.sanity_tol})")

    print(f"\nMCTS latency: {mcts_t / max(1, mcts_n):.2f} s/episode (budget {base_cfg.mcts_n_simulations}).")
    _save(args.out, results, args, base_cfg)
    print(f"Saved results to {args.out}")
    _plot(args.plot, results)


def _save(path, results, args, base_cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(meta=dict(load=args.load, seeds=args.seeds, episodes=args.episodes,
                                 sigma_hi=args.sigma_hi, frac_high=args.frac_high,
                                 sigma_lo=args.sigma_lo, lambda_risk=args.lambda_risk,
                                 beta_unc=args.beta_unc, d_max=base_cfg.d_max,
                                 mcts_n_simulations=base_cfg.mcts_n_simulations),
                       results=results), f, indent=2)


def _plot(path, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return
    xs = [r["sigma_hi"] for r in results]
    plt.figure(figsize=(6, 4))
    for v in VARIANTS:
        mean = [r["aggregate"][v]["mean"] for r in results]
        lo = [r["aggregate"][v]["ci_low"] for r in results]
        hi = [r["aggregate"][v]["ci_high"] for r in results]
        line, = plt.plot(xs, mean, "o-", label=LABELS[v])
        plt.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    plt.xlabel("sigma_hi (magnitude of unreliable edges)")
    plt.ylabel("success rate")
    plt.title("E1b: uncertainty-bonus ablation (mean +/- 95% bootstrap CI)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
