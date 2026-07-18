#!/usr/bin/env python3
"""Plot a multi-seed averaged learning curve with a confidence band (cf. Fig. 4).

Parses several training logs (one per seed), aligns the eval success rates by
env step, and plots mean +/- std across seeds as a shaded band. This reproduces
the paper-style curve, whose smoothness / confidence interval comes precisely
from averaging over multiple independent runs.

Example:
    python scripts/plot_seeds.py --glob "logs/pointmaze_hard*.log" --out logs/seeds_curve.png
"""

import argparse
import glob as globlib
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP = re.compile(r"\[\s*(\d+)\s+steps")
EVAL = re.compile(r"eval success rate \(long-horizon test\):\s*([-\d.]+)")


def parse_evals(path):
    """Return {step: success_rate} for one log file."""
    out = {}
    last_step = 0
    with open(path) as f:
        for line in f:
            m = STEP.search(line)
            if m:
                last_step = int(m.group(1))
                continue
            e = EVAL.search(line)
            if e:
                out[last_step] = float(e.group(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="logs/pointmaze_hard*.log",
                   help="glob for per-seed log files")
    p.add_argument("--out", default="logs/seeds_curve.png")
    p.add_argument("--title", default="$L^3P$ on PointMaze-Hard (15x15, hindsight=20)")
    p.add_argument("--band", choices=["std", "minmax"], default="std")
    args = p.parse_args()

    files = sorted(globlib.glob(args.glob))
    if not files:
        raise SystemExit(f"No logs match {args.glob}")

    per_seed = {f: parse_evals(f) for f in files}
    per_seed = {f: d for f, d in per_seed.items() if d}    # drop empty
    if not per_seed:
        raise SystemExit("No eval points parsed yet — let training run longer.")

    # union of all steps that appear in any seed
    all_steps = sorted({s for d in per_seed.values() for s in d})
    xs, mean, lo, hi, nseed = [], [], [], [], []
    for s in all_steps:
        vals = [d[s] for d in per_seed.values() if s in d]
        if not vals:
            continue
        vals = np.array(vals)
        xs.append(s / 1e3)
        mean.append(vals.mean())
        nseed.append(len(vals))
        if args.band == "std":
            sd = vals.std()
            lo.append(max(0.0, vals.mean() - sd))
            hi.append(min(1.0, vals.mean() + sd))
        else:
            lo.append(vals.min()); hi.append(vals.max())

    xs = np.array(xs); mean = np.array(mean)
    lo = np.array(lo); hi = np.array(hi)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    # individual seeds (faint)
    for f, d in per_seed.items():
        sx = sorted(d)
        ax.plot([s / 1e3 for s in sx], [d[s] for s in sx],
                color="0.75", lw=0.9, alpha=0.6, zorder=1)
    # mean + band
    ax.fill_between(xs, lo, hi, color="#d62728", alpha=0.2, zorder=2,
                    label=f"mean $\\pm$ {'std' if args.band=='std' else 'min/max'} "
                          f"({len(per_seed)} seeds)")
    ax.plot(xs, mean, "-o", color="#d62728", lw=2.2, ms=4, zorder=3,
            label="mean success rate")

    ax.axvspan(0, 100, color="0.92", zorder=0, label="warm-up (no planning)")
    ax.axvline(100, color="0.5", ls="--", lw=1, zorder=0)
    ax.set_xlabel("Env steps (thousands)")
    ax.set_ylabel("Test success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(args.title + " — multi-seed learning curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")
    print(f"seeds: {len(per_seed)} ({', '.join(f.split('/')[-1] for f in per_seed)})")
    print(f"steps covered: up to {int(max(all_steps)/1e3)}k | "
          f"seeds-per-point range: {min(nseed)}-{max(nseed)}")


if __name__ == "__main__":
    main()
