#!/usr/bin/env python3
"""Plot the L3P learning curve from a training log file.

Parses the progress lines and eval lines written by `train_pointmaze.py` and
produces a figure with (top) the long-horizon test success rate vs. env steps
and (bottom) the key training losses. Saves a PNG.

Example:
    python scripts/plot_log.py --log logs/pointmaze_full.log --out logs/learning_curve.png
"""

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# "[  140000 steps | 700 eps | 547.0s] landmarks=on | critic=0.031 value=0.039 ..."
PROG = re.compile(
    r"\[\s*(\d+)\s+steps.*?landmarks=(on|off)\s*\|\s*"
    r"critic=([-\d.]+)\s+value=([-\d.]+)\s+actor=([-\d.]+)\s+"
    r"ae_rec=([-\d.]+)\s+ae_latent=([-\d.]+)\s+elbo=([-\d.]+)"
)
EVAL = re.compile(r"eval success rate \(long-horizon test\):\s*([-\d.]+)")


def parse(path):
    steps, critic, value, actor, ae_rec, elbo = [], [], [], [], [], []
    eval_steps, eval_sr = [], []
    landmark_on_step = None
    last_step = 0
    with open(path) as f:
        for line in f:
            m = PROG.search(line)
            if m:
                s = int(m.group(1))
                last_step = s
                if m.group(2) == "on" and landmark_on_step is None:
                    landmark_on_step = s
                steps.append(s)
                critic.append(float(m.group(3)))
                value.append(float(m.group(4)))
                actor.append(float(m.group(5)))
                ae_rec.append(float(m.group(6)))
                elbo.append(float(m.group(8)))
                continue
            e = EVAL.search(line)
            if e:
                eval_steps.append(last_step)
                eval_sr.append(float(e.group(1)))
    return dict(steps=steps, critic=critic, value=value, actor=actor,
                ae_rec=ae_rec, elbo=elbo, eval_steps=eval_steps, eval_sr=eval_sr,
                landmark_on=landmark_on_step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="logs/pointmaze_full.log")
    p.add_argument("--out", default="logs/learning_curve.png")
    p.add_argument("--title", default="L3P on PointMaze-Hard")
    args = p.parse_args()

    d = parse(args.log)
    if not d["eval_steps"]:
        raise SystemExit(f"No eval lines found in {args.log}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                   gridspec_kw={"height_ratios": [2, 1]})

    # ---- top: long-horizon test success rate ----
    xs = [s / 1e3 for s in d["eval_steps"]]
    ax1.plot(xs, d["eval_sr"], "-o", color="#d62728", lw=2, ms=5,
             label="$L^3P$ (long-horizon test)")
    ax1.set_ylabel("Test success rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title(args.title + " — learning curve")
    if d["landmark_on"] is not None:
        lo = d["landmark_on"] / 1e3
        ax1.axvspan(0, lo, color="0.9", label="warm-up (no planning)")
        ax1.axvline(lo, color="0.4", ls="--", lw=1)
        ax1.annotate("landmarks on\n(planning starts)", xy=(lo, 0.5),
                     xytext=(lo + max(xs) * 0.03, 0.45), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="0.4"))
    ax1.grid(alpha=0.3)
    ax1.legend(loc="center right")

    # ---- bottom: training losses ----
    sx = [s / 1e3 for s in d["steps"]]
    ax2.plot(sx, d["critic"], color="#1f77b4", lw=1.2, label="critic TD (Eq.1)")
    ax2.plot(sx, d["value"], color="#2ca02c", lw=1.2, label="value reg (Eq.4)")
    ax2.plot(sx, d["ae_rec"], color="#ff7f0e", lw=1.2, label="AE recon (Eq.2)")
    ax2.set_xlabel("Env steps (thousands)")
    ax2.set_ylabel("loss")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", ncol=3, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")
    print(f"eval points: {len(d['eval_steps'])}, "
          f"landmark-on at {d['landmark_on']} steps, "
          f"max success rate: {max(d['eval_sr']):.2f}")


if __name__ == "__main__":
    main()
