#!/usr/bin/env python3
"""Overlay success-vs-sigma for every planner from paper-harness JSON(s).

Merges one or more `eval_ablation.py --out` files (e.g. a CLASSICAL phase + an
MCTS phase that share env/regime) onto one axis. Planners run with
`--noise-seeds` (a `per_seed` field) get a [min,max] band; single-seed planners
get a plain line. Also prints a markdown success table and a latency summary.

    python scripts/plot_planners.py \
        --json logs/exp_suite/antmaze_e1c_classical.json logs/exp_suite/antmaze_e1c_mcts.json \
        --out logs/exp_suite/antmaze_e1c_planners.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(paths):
    """Merge JSONs -> (env, regime, {planner: {sig, mean, lo, hi, lat}}). First
    occurrence of a planner name wins (soft_floyd appears in both phases)."""
    env = regime = None
    curves = {}
    for p in paths:
        d = json.load(open(p))
        env = env or d.get("env")
        regime = regime or d.get("regime")
        sig = [float(s) for s in d["sigmas"]]
        per_seed = d.get("per_seed", {}) or {}
        lat = d.get("latency", {}) or {}
        for name, means in d["variants"].items():
            if name in curves:
                continue
            ps = per_seed.get(name)
            lo = hi = None
            if ps:                                  # per_seed[name][i] = [vals over seeds]
                lo = [float(np.min(v)) for v in ps]
                hi = [float(np.max(v)) for v in ps]
            curves[name] = dict(sig=sig, mean=[float(m) for m in means],
                                lo=lo, hi=hi, lat=lat.get(name))
    return env, regime, curves


# stable color per planner family so plots read as one system across envs
_COLORS = {
    "soft_floyd": "#1f77b4", "dijkstra": "#d62728", "astar": "#ff7f0e",
    "greedy": "#2ca02c", "mcts": "#9467bd", "mcts_nofb": "#8c564b",
    "mcts_fb": "#e377c2", "mcts+suffix": "#7f7f7f", "mcts+pw": "#bcbd22",
    "mcts+bayes": "#17becf",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="one or more eval_ablation JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()

    env, regime, curves = load(a.json)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, c in curves.items():
        col = _COLORS.get(name)
        ax.plot(c["sig"], c["mean"], marker="o", ms=4, lw=1.8, color=col, label=name)
        if c["lo"] is not None:
            ax.fill_between(c["sig"], c["lo"], c["hi"], color=col, alpha=0.15, linewidth=0)
    ax.set_xlabel("noise σ")
    ax.set_ylabel("test-plan success")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title(a.title or f"{env} · {regime} · planner comparison")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"saved -> {a.out}")

    # markdown success table (rows = sigma, cols = planner)
    sigs = sorted({s for c in curves.values() for s in c["sig"]})
    names = list(curves)
    print("\n| σ | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for s in sigs:
        row = []
        for n in names:
            c = curves[n]
            row.append(f"{c['mean'][c['sig'].index(s)]:.2f}" if s in c["sig"] else "—")
        print(f"| {s:g} | " + " | ".join(row) + " |")

    lat = {n: c["lat"] for n, c in curves.items() if c["lat"]}
    if lat:
        print("\nlatency (ms/search, mean over σ):")
        for n, v in lat.items():
            vv = [x for x in v if x]
            if vv:
                print(f"  {n:14} {np.mean(vv):.1f}")


if __name__ == "__main__":
    main()
