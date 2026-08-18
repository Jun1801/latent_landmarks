#!/usr/bin/env python3
"""Paper-grade planner comparison from paper-harness JSON(s).

Reads one or more `eval_ablation.py --out` files (e.g. a CLASSICAL phase + an
MCTS phase sharing env/regime). For JSONs carrying `per_episode` (paired 0/1
outcomes) it computes proper **paired bootstrap 95% CIs** — resampling the
shared rollout indices so every planner is resampled identically, which also
yields a CI on the pairwise gap. It then locates, per phase:
  * each planner's KNEE σ (success first drops below --knee)
  * each planner's COLLAPSE σ (success first drops below --collapse)
  * σ* — the σ maximizing the (best-MCTS − soft_floyd) gap whose paired-gap
    CI excludes 0 (a statistically significant advantage)
  * σ_all_fail — the smallest σ where EVERY planner is below --collapse

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

BOOT = 5000
_COLORS = {
    "soft_floyd": "#1f77b4", "dijkstra": "#d62728", "astar": "#ff7f0e",
    "greedy": "#2ca02c", "mcts": "#9467bd", "mcts_nofb": "#8c564b",
    "mcts_fb": "#e377c2", "mcts+suffix": "#7f7f7f", "mcts+pw": "#bcbd22",
    "mcts+bayes": "#17becf",
}


def _pool(per_ep_planner):
    """per_episode[name] = [sigma_idx][seed_idx] -> [0/1..]  ->  per-sigma 1-D arrays
    (seeds concatenated in order; index alignment across planners is preserved)."""
    return [np.array([o for seed in sig for o in seed], dtype=float) for sig in per_ep_planner]


def paired_bootstrap(arrays_by_planner, n_boot=BOOT, seed=0):
    """arrays_by_planner: {name: 1-D 0/1 array, all same length N (paired by index)}.
    Returns point means and 95% CIs; resamples the SAME indices for every planner."""
    names = list(arrays_by_planner)
    N = len(next(iter(arrays_by_planner.values())))
    rng = np.random.default_rng(seed)
    boot = {n: np.empty(n_boot) for n in names}
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        for n in names:
            boot[n][b] = arrays_by_planner[n][idx].mean()
    point = {n: float(arrays_by_planner[n].mean()) for n in names}
    ci = {n: (float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))) for n in names}
    return point, ci, boot


def gap_ci(boot, a, b):
    """95% CI of the paired difference mean(a) - mean(b) from bootstrap draws."""
    d = boot[a] - boot[b]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def load_phase(path):
    d = json.load(open(path))
    sig = [float(s) for s in d["sigmas"]]
    per_ep = d.get("per_episode", {}) or {}
    curves = {}
    for name, means in d["variants"].items():
        c = {"sig": sig, "mean": [float(m) for m in means], "lo": None, "hi": None,
             "lat": (d.get("latency", {}) or {}).get(name)}
        curves[name] = c
    pooled = {name: _pool(per_ep[name]) for name in per_ep} if per_ep else {}
    # per-sigma paired bootstrap (all planners in this JSON share indices)
    boots = []
    if pooled:
        for i in range(len(sig)):
            arrays = {n: pooled[n][i] for n in pooled if len(pooled[n][i]) > 0}
            if not arrays:
                boots.append(None); continue
            pt, ci, bt = paired_bootstrap(arrays, seed=1000 + i)
            for n in arrays:
                curves[n]["mean"][i] = pt[n]
                curves[n].setdefault("lo_list", [None] * len(sig))
                curves[n].setdefault("hi_list", [None] * len(sig))
                curves[n]["lo_list"][i] = ci[n][0]
                curves[n]["hi_list"][i] = ci[n][1]
            boots.append(bt)
        for n in curves:
            if "lo_list" in curves[n]:
                curves[n]["lo"] = curves[n]["lo_list"]
                curves[n]["hi"] = curves[n]["hi_list"]
    return d.get("env"), d.get("regime"), sig, curves, pooled, boots


def crossing(sig, mean, thresh):
    """smallest sigma at which success first goes < thresh (or None)."""
    for s, m in zip(sig, mean):
        if m < thresh:
            return s
    return None


def analyse_phase(tag, sig, curves, boots, knee, collapse):
    print(f"\n===== {tag} =====")
    print("knee (success<%.2f) / collapse (<%.2f) per planner:" % (knee, collapse))
    for n, c in curves.items():
        print(f"  {n:12} knee σ={str(crossing(c['sig'], c['mean'], knee)):>6}   "
              f"collapse σ={str(crossing(c['sig'], c['mean'], collapse)):>6}")
    # sigma_all_fail: every planner below collapse
    all_fail = None
    for i, s in enumerate(sig):
        if all(curves[n]["mean"][i] < collapse for n in curves):
            all_fail = s; break
    print(f"  -> σ_all_fail (every planner < {collapse}): {all_fail}")
    # sigma* : max significant gap best-MCTS vs soft_floyd (needs paired bootstrap)
    mcts_names = [n for n in curves if n.startswith("mcts")]
    if boots and "soft_floyd" in curves and mcts_names:
        best = None
        for i, s in enumerate(sig):
            if boots[i] is None or "soft_floyd" not in boots[i]:
                continue
            for mn in mcts_names:
                if mn not in boots[i]:
                    continue
                g, lo, hi = gap_ci(boots[i], mn, "soft_floyd")
                if lo > 0 and (best is None or g > best[1]):     # significant + largest
                    best = (s, g, lo, hi, mn)
        if best:
            s, g, lo, hi, mn = best
            print(f"  -> σ* (max significant gap): σ={s}  {mn}−soft_floyd = {g:+.3f} [{lo:+.3f},{hi:+.3f}]")
        else:
            print("  -> σ*: no σ with a significant MCTS>soft_floyd gap (CI excludes 0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--knee", type=float, default=0.5)
    ap.add_argument("--collapse", type=float, default=0.25)
    a = ap.parse_args()

    env = regime = None
    merged = {}                                   # planner -> curve (first phase wins for dedup)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for path in a.json:
        e, r, sig, curves, pooled, boots = load_phase(path)
        env = env or e; regime = regime or r
        analyse_phase(f"{e}·{r}·{os.path.basename(path)}", sig, curves, boots, a.knee, a.collapse)
        for n, c in curves.items():
            if n not in merged:
                merged[n] = c

    for n, c in merged.items():
        col = _COLORS.get(n)
        ax.plot(c["sig"], c["mean"], marker="o", ms=4, lw=1.8, color=col, label=n)
        if c.get("lo") is not None:
            lo = [x if x is not None else np.nan for x in c["lo"]]
            hi = [x if x is not None else np.nan for x in c["hi"]]
            ax.fill_between(c["sig"], lo, hi, color=col, alpha=0.15, linewidth=0)
    ax.axhline(a.collapse, ls=":", c="gray", lw=1)
    ax.set_xlabel("noise σ"); ax.set_ylabel("test-plan success (mean, 95% CI)")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.set_title(a.title or f"{env} · {regime} · planner comparison (paired, bootstrap CI)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print(f"\nsaved -> {a.out}")

    # markdown table: mean [lo,hi]
    sigs = sorted({s for c in merged.values() for s in c["sig"]})
    names = list(merged)
    print("\n| σ | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for s in sigs:
        cells = []
        for n in names:
            c = merged[n]
            if s in c["sig"]:
                i = c["sig"].index(s)
                if c.get("lo") is not None and c["lo"][i] is not None:
                    cells.append(f"{c['mean'][i]:.2f} [{c['lo'][i]:.2f},{c['hi'][i]:.2f}]")
                else:
                    cells.append(f"{c['mean'][i]:.2f}")
            else:
                cells.append("—")
        print(f"| {s:g} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
