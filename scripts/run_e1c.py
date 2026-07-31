#!/usr/bin/env python3
"""E1c: can MCTS DETECT a systematic estimation bias through execution and beat
Soft Floyd, which trusts the biased estimate once and never corrects it?
(docs/SPEC_MCTS_Landmark_L3P.md Sec 4-6, Sec 8 row E1c.)

Noise = Loai-1 (estimation bias): the landmark graph is built from the critic's
CLEAN distance D (env-step scale, `CriticEdgeFn`) with a per-episode FIXED
multiplicative bias V_obs = V_true*(1+b_ij), b_ij~N(0,sigma^2). Because the bias
is fixed (not resampled), rollout averaging cannot remove it -- only execution
feedback can. The env executes on the clean dynamics, so a negatively-biased
edge is a "wormhole" that looks like a great shortcut but isn't.

Three branches (spec Sec 7):
  * "soft_floyd" -- SoftFloydE1c: plans once on the biased graph, static (L3P).
  * "mcts_nofb"  -- FeedbackMCTSPlanner(feedback=False): plain MCTS on the same
    static biased graph. Isolates "does lookahead alone help?" from feedback.
  * "mcts_fb"    -- FeedbackMCTSPlanner(feedback=True): MCTS that snaps to the
    nearest landmark, measures realized env-step cost of each executed edge,
    EMA-corrects V_exec[i][j], rebuilds d_c2g, and blacklists repeatedly-failing
    edges (Sec 4.1/4.3/4.4). Expected to detect and route around the wormhole.

Everything stays on the learned step-distance scale: PointMaze uses critic-D
between goal coordinates; higher-dimensional goal envs use Eq.4 goal-to-goal V.
`d_max` is calibrated on the matching clean substrate and shared by all branches.
sigma=0 (no bias) is the sanity point: all branches should coincide.

Example (~2-3h; tune sims/episodes/bias to taste):
    python scripts/run_e1c.py --load checkpoint/l3p_pointmaze_full.pt --seeds 0 1 \
        --episodes 40 --sigmas 0 0.1 0.3 --mcts-n-simulations 120
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config, list_envs
from l3p.envs import make_vec_env
from l3p.planning.mcts_planner import SoftFloydE1c, FeedbackMCTSPlanner
from l3p.planning.noise import (CriticEdgeFn, dmax_candidates,
                                hierarchical_bootstrap_ci, calibration_seed)
from l3p.trainer import L3PTrainer

BRANCHES = ["soft_floyd", "mcts_nofb", "mcts_fb"]
LABELS = {"soft_floyd": "Soft Floyd (static)", "mcts_nofb": "MCTS no-feedback",
          "mcts_fb": "MCTS + execution feedback"}


def make_branch(name, trainer, cfg, sigma, noise_seed, args):
    a, lm, ae, gs = trainer.agent, trainer.landmarks, trainer.ae, trainer.graph_search
    edge_fn = CriticEdgeFn(a) if trainer.env.obs_dim == trainer.env.goal_dim else a.value
    if name == "soft_floyd":
        return SoftFloydE1c(
            a, lm, ae, gs, cfg, sigma=sigma, noise_seed=noise_seed,
            edge_value_fn=edge_fn)
    # The sigma=0 point is a strict deterministic control: feedback would
    # otherwise change the graph from real executions and break equivalence
    # with the static Soft-Floyd baseline even though no bias was injected.
    fb = (name == "mcts_fb" and sigma > 0.0)
    return FeedbackMCTSPlanner(a, lm, ae, gs, cfg, sigma=sigma, noise_seed=noise_seed,
                              feedback=fb, rho=args.rho, tau_reach=args.tau_reach,
                              tau_progress=args.tau_progress, tau_snap=args.tau_snap,
                              r_max=args.r_max, edge_value_fn=edge_fn,
                              rng=np.random.default_rng(noise_seed + 1))


def d_edge_matrix(trainer):
    """Clean critic-D distance between every landmark pair + goal (E1c substrate)."""
    with torch.no_grad():
        lg = trainer.ae.decode(trainer.landmarks.centroids.detach())
    edge = (CriticEdgeFn(trainer.agent)
            if trainer.env.obs_dim == trainer.env.goal_dim
            else trainer.agent.value)
    m = lg.shape[0]
    gi = lg[:, None, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
    gj = lg[None, :, :].expand(m, m, lg.shape[1]).reshape(m * m, -1)
    with torch.no_grad():
        return edge(gi, gj).view(m, m).cpu().numpy()


def calibrate_d_max(trainer, cfg, n_episodes, base_seed):
    """Tune d_max on the CLEAN-D Soft Floyd baseline (sigma=0), spec R3. Candidates
    are percentiles of the critic-D edge distribution (env-step scale)."""
    cands = dmax_candidates(d_edge_matrix(trainer), fallback=trainer.cfg.d_max)
    best, sweep = (cands[0], -1.0), []
    for dmax in cands:
        trainer.cfg.d_max = dmax
        succ = 0.0
        edge_fn = (CriticEdgeFn(trainer.agent)
                   if trainer.env.obs_dim == trainer.env.goal_dim
                   else trainer.agent.value)
        for ep in range(n_episodes):
            es = base_seed * 1_000_003 + ep
            pl = SoftFloydE1c(trainer.agent, trainer.landmarks, trainer.ae,
                              trainer.graph_search, trainer.cfg, sigma=0.0,
                              noise_seed=es, edge_value_fn=edge_fn)
            trainer.env.envs[0].rng = np.random.default_rng(es)
            succ += trainer.evaluate(1, planner=pl)
        sweep.append((dmax, succ / n_episodes))
        if sweep[-1][1] > best[1]:
            best = (dmax, sweep[-1][1])
    trainer.cfg.d_max = best[0]
    return best[0], sweep


def run_point(trainer, sigma, name, args, base_seed):
    outcomes, stats, t_tot = [], [], 0.0
    for ep in range(args.episodes):
        es = base_seed * 1_000_003 + ep
        pl = make_branch(name, trainer, trainer.cfg, sigma, es, args)
        trainer.env.envs[0].rng = np.random.default_rng(es)
        outcomes.append(int(trainer.evaluate(1, planner=pl) > 0))
        t_tot += float(getattr(pl, "search_seconds", 0.0))
        stats.append({k: float(v) for k, v in getattr(pl, "stats", {}).items()})
    return outcomes, stats, t_tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PointMaze", choices=list_envs())
    p.add_argument("--load", type=str, default="checkpoint/l3p_pointmaze_full.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.3],
                   help="Loai-1 bias magnitude (std of multiplicative b_ij)")
    p.add_argument("--allow-missing-zero", action="store_true",
                   help="allow a point-only continuation without prepending sigma=0")
    p.add_argument("--resume", action="store_true",
                   help="resume completed noise points from --out")
    p.add_argument("--rho", type=float, default=None, help="EMA rate for V_exec")
    p.add_argument("--tau-reach", type=float, default=None,
                   help="D/V-scale fallback reached threshold (default: 0.25*d_max)")
    p.add_argument("--tau-progress", type=float, default=None,
                   help="D/V-scale progressed threshold (default: 0.5*d_max)")
    p.add_argument("--tau-snap", type=float, default=None,
                   help="D/V-scale maximum SNAP distance (default: d_max)")
    p.add_argument("--r-max", type=int, default=None, help="failed attempts before blacklisting an edge")
    p.add_argument("--sanity-tol", type=float, default=0.1,
                   help="max |mcts_fb - soft_floyd| at sigma=0 before aborting")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--mcts-n-simulations", type=int, default=120)
    p.add_argument("--mcts-rollout-horizon", type=int, default=10)
    p.add_argument("--uncertainty-mode", choices=["none", "alpha", "beta", "both"],
                   default="none")
    p.add_argument("--lambda-risk", type=float, default=1.0)
    p.add_argument("--beta-unc", type=float, default=1.0)
    p.add_argument("--d-max", type=float, default=None)
    p.add_argument("--calibrate-episodes", type=int, default=20)
    p.add_argument("--out", type=str, default="logs/e1c_results.json")
    p.add_argument("--plot", type=str, default="logs/e1c_curve.png")
    args = p.parse_args()

    if 0.0 not in args.sigmas and not args.allow_missing_zero:
        args.sigmas = [0.0] + list(args.sigmas)

    cfg = get_config(args.env, seed=args.seeds[0],
                     mcts_n_simulations=args.mcts_n_simulations,
                     mcts_rollout_horizon=args.mcts_rollout_horizon,
                     mcts_uncertainty_mode=args.uncertainty_mode,
                     mcts_lambda_risk=args.lambda_risk,
                     mcts_beta_uncertainty=args.beta_unc)
    env = make_vec_env(cfg, 1, cfg.seed)
    trainer = L3PTrainer(env, cfg)
    trainer.load(args.load)
    substrate = ("critic-D" if trainer.env.obs_dim == trainer.env.goal_dim
                 else "goal-to-goal V")
    print(f"E1c on {args.env}: bias-detection via execution feedback (Loai-1). "
          f"Graph substrate={substrate} + fixed multiplicative bias.")
    if not trainer.centroids_initialized:
        print("ERROR: checkpoint has no initialized landmark centroids.", file=sys.stderr)
        sys.exit(1)

    results, resume_dmax = [], None
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            previous = json.load(f)
        results = previous.get("results", [])
        resume_dmax = previous.get("meta", {}).get("d_max")
        print(f"Resuming {len(results)} completed noise points from {args.out}.")

    if resume_dmax is not None:
        trainer.cfg.d_max = float(resume_dmax)
        print(f"Reusing resumed d_max={trainer.cfg.d_max}.")
    elif args.d_max is not None:
        trainer.cfg.d_max = args.d_max
        print(f"Using fixed d_max={args.d_max}.")
    else:
        best, sweep = calibrate_d_max(
            trainer, trainer.cfg, args.calibrate_episodes,
            calibration_seed(args.seeds[0]))
        print(f"d_max calibration (clean {substrate} Soft Floyd, spec R3):")
        for dmax, sr in sweep:
            print(f"    d_max={dmax:7.3f} -> {sr:.2f}" + ("   <- chosen" if dmax == best else ""))
    args.rho = trainer.cfg.mcts_feedback_rho if args.rho is None else args.rho
    args.r_max = trainer.cfg.mcts_r_max if args.r_max is None else args.r_max
    if args.tau_reach is None:
        args.tau_reach = (trainer.cfg.mcts_tau_reach
                          if trainer.cfg.mcts_tau_reach is not None
                          else 0.25 * trainer.cfg.d_max)
    if args.tau_progress is None:
        args.tau_progress = (trainer.cfg.mcts_tau_progress
                             if trainer.cfg.mcts_tau_progress is not None
                             else 0.5 * trainer.cfg.d_max)
    if args.tau_snap is None:
        args.tau_snap = (trainer.cfg.mcts_tau_snap
                         if trainer.cfg.mcts_tau_snap is not None
                         else trainer.cfg.d_max)
    effective_floor = 2 * (trainer.landmarks.centroids.shape[0] + 1)
    print(f"Config: d_max={trainer.cfg.d_max:.3f}  requested_sims={trainer.cfg.mcts_n_simulations}  "
          f"effective_sims>=max(requested,{effective_floor})  "
          f"rho={args.rho}  tau_reach={args.tau_reach}  tau_snap={args.tau_snap}  "
          f"r_max={args.r_max}  "
          f"seeds={args.seeds}  eps/seed={args.episodes}", flush=True)

    mcts_t, mcts_n = 0.0, 0
    completed = {float(r["sigma"]) for r in results}
    for sigma in args.sigmas:
        if float(sigma) in completed:
            print(f"Skipping completed sigma={sigma:.2f}.", flush=True)
            continue
        pooled = {b: [] for b in BRANCHES}
        grouped = {b: [] for b in BRANCHES}
        stats_pooled = {b: [] for b in BRANCHES}
        per_seed = {}
        per_episode = {}
        for seed in args.seeds:
            per_seed[seed] = {}
            per_episode[seed] = {}
            for b in BRANCHES:
                out, stats, t = run_point(trainer, sigma, b, args, seed)
                pooled[b].extend(out)
                grouped[b].append(out)
                stats_pooled[b].extend(stats)
                per_seed[seed][b] = float(np.mean(out))
                per_episode[seed][b] = dict(outcomes=out, stats=stats)
                print(f"  done sigma={sigma:.2f} seed={seed} {LABELS[b]}: "
                      f"mean={per_seed[seed][b]:.2f} time={t:.1f}s", flush=True)
                if b != "soft_floyd":
                    mcts_t += t; mcts_n += args.episodes
        agg, rng = {}, np.random.default_rng(args.seeds[0])
        for b in BRANCHES:
            mean, lo, hi = hierarchical_bootstrap_ci(
                grouped[b], n_boot=args.n_boot, rng=rng)
            keys = sorted({k for row in stats_pooled[b] for k in row})
            stats_mean = {
                k: float(np.mean([row.get(k, 0.0) for row in stats_pooled[b]]))
                for k in keys
            }
            agg[b] = dict(mean=mean, ci_low=lo, ci_high=hi,
                          n=len(pooled[b]), stats=stats_mean)
        results.append(dict(sigma=sigma, aggregate=agg, per_seed=per_seed,
                            per_episode=per_episode))
        print(f"sigma={sigma:.2f}  " + "  ".join(
            f"{LABELS[b]}={agg[b]['mean']:.2f}[{agg[b]['ci_low']:.2f},{agg[b]['ci_high']:.2f}]"
            for b in BRANCHES), flush=True)

        if sigma == 0.0:
            gap = abs(agg["mcts_fb"]["mean"] - agg["soft_floyd"]["mean"])
            if gap > args.sanity_tol:
                print(f"\n*** SANITY FAILED: |mcts_fb-floyd|={gap:.2f} > "
                      f"{args.sanity_tol:.2f}; aborting before sigma>0. ***\n",
                      file=sys.stderr)
                _save(args.out, results, args, trainer)
                sys.exit(1)
            print(f"  >> sigma=0 sanity passed: |mcts_fb-floyd|={gap:.2f}")
        _save(args.out, results, args, trainer)
        _plot(args.plot, results)
        print(f"  checkpointed {len(results)}/{len(args.sigmas)} noise points", flush=True)

    print(f"\nMCTS latency: {mcts_t / max(1, mcts_n):.2f} s/episode.", flush=True)
    print(f"Saved results to {args.out}")
    if results:
        _plot(args.plot, results)


def _save(path, results, args, trainer):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(meta=dict(env=args.env, load=args.load, seeds=args.seeds, episodes=args.episodes,
                                 sigmas=args.sigmas, rho=args.rho, tau_reach=args.tau_reach,
                                 tau_progress=args.tau_progress, tau_snap=args.tau_snap,
                                 r_max=args.r_max,
                                 uncertainty_mode=args.uncertainty_mode,
                                 lambda_risk=args.lambda_risk,
                                 beta_unc=args.beta_unc,
                                 d_max=trainer.cfg.d_max,
                                 mcts_n_simulations=trainer.cfg.mcts_n_simulations,
                                 mcts_root_coverage_floor=2 * (
                                     trainer.landmarks.centroids.shape[0] + 1)),
                       results=results), f, indent=2)
    os.replace(tmp, path)


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
    plt.xlabel("sigma (Loai-1 bias std)")
    plt.ylabel("success rate")
    plt.title("E1c: bias detection via execution feedback (mean +/- 95% CI)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path)
    print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
