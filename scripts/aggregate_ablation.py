#!/usr/bin/env python3
"""Aggregate the local Track-L ablation JSONs (E1a/E1c across training seeds) into a
master success table (mean +/- hierarchical bootstrap CI over training seeds) and a
success-vs-sigma plot per regime.

Reads logs/exp_suite/local/{regime}_seed{S}_{suffix}.json (from run_e1a/run_e1c),
pools the per-episode 0/1 outcomes, and keeps training-seed as the bootstrap group
(preserves between-seed variation, spec Sec 10).

    python scripts/aggregate_ablation.py --dir logs/exp_suite/local --seeds 0 1 2 3
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from l3p.planning.noise import hierarchical_bootstrap_ci

# regime -> [(row label, file suffix, branch key in JSON)]
PLANNERS = {
    "e1a": [("soft_floyd", "base", "soft_floyd"),
            ("naive_replan", "base", "naive_replan"),
            ("MPC(fresh)", "base", "fresh_graph_replan"),
            ("mcts", "base", "mcts"),
            ("mcts+pw", "pw", "mcts"),
            ("mcts+bayes", "bayes", "mcts")],
    "e1c": [("soft_floyd", "base", "soft_floyd"),
            ("mcts_nofb", "base", "mcts_nofb"),
            ("mcts_fb", "base", "mcts_fb")],
}


def load(path):
    with open(path) as f:
        return json.load(f)


def per_seed_outcomes(d, sigma, branch):
    """Pooled 0/1 outcomes (over eval-seeds) for one training-seed file at one sigma."""
    for r in d["results"]:
        if abs(float(r["sigma"]) - sigma) < 1e-9:
            out = []
            for _es, branches in r.get("per_episode", {}).items():
                val = branches.get(branch, [])
                if isinstance(val, dict):        # run_e1c stores {"outcomes":[...], "stats":[...]}
                    val = val.get("outcomes", [])
                out.extend(val)
            return out
    return []


def collect(directory, regime, seeds):
    """{planner: {sigma: [per-training-seed outcome lists]}} + sorted sigmas."""
    files = {}  # (suffix) -> {seed: dict}
    sigmas = set()
    for suffix in {p[1] for p in PLANNERS[regime]}:
        files[suffix] = {}
        for s in seeds:
            path = os.path.join(directory, f"{regime}_seed{s}_{suffix}.json")
            if os.path.exists(path):
                d = load(path)
                files[suffix][s] = d
                sigmas.update(float(r["sigma"]) for r in d["results"])
    sigmas = sorted(sigmas)
    table = {}
    for label, suffix, branch in PLANNERS[regime]:
        table[label] = {}
        for sig in sigmas:
            groups = []
            for s in seeds:
                d = files.get(suffix, {}).get(s)
                if d is None:
                    continue
                o = per_seed_outcomes(d, sig, branch)
                if o:
                    groups.append(o)
            table[label][sig] = groups
    return table, sigmas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="logs/exp_suite/local")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--regimes", nargs="+", default=["e1a", "e1c"])
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--plot", default="logs/exp_suite/summary")
    args = p.parse_args()

    for regime in args.regimes:
        table, sigmas = collect(args.dir, regime, args.seeds)
        if not sigmas:
            print(f"[{regime}] no JSON found in {args.dir}"); continue
        rng = np.random.default_rng(0)
        means = {}
        print(f"\n===== {regime.upper()}  (mean [95% CI] over {len(args.seeds)} training seeds) =====")
        hdr = f"{'planner':14}" + "".join(f"  s={s:<11}" for s in sigmas)
        print(hdr)
        for label in table:
            cells, mrow = [], []
            for sig in sigmas:
                groups = table[label][sig]
                if not groups:
                    cells.append(f"{'-':<13}"); mrow.append(None); continue
                m, lo, hi = hierarchical_bootstrap_ci(groups, n_boot=args.n_boot, rng=rng)
                cells.append(f"{m:.2f}[{lo:.2f},{hi:.2f}]".ljust(13)); mrow.append(m)
            means[label] = mrow
            print(f"{label:14}" + "  ".join(cells))
        _plot(f"{args.plot}_{regime}.png", regime, sigmas, means)


def _plot(path, regime, sigmas, means):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(6, 4))
    for label, mrow in means.items():
        xs = [s for s, m in zip(sigmas, mrow) if m is not None]
        ys = [m for m in mrow if m is not None]
        if ys:
            plt.plot(xs, ys, "o-", label=label)
    plt.xlabel("sigma"); plt.ylabel("success"); plt.ylim(-0.02, 1.02)
    plt.title(f"{regime.upper()}: success vs noise (multi-seed mean)")
    plt.legend(fontsize=8); plt.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path); print(f"  saved plot {path}")


if __name__ == "__main__":
    main()
