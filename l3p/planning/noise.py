"""Noise injection for the MCTS-over-landmarks experiments
(docs/SPEC_MCTS_Landmark_L3P.md, Section 5).

Only "Loại 2" (execution-stochasticity) noise is implemented here:

    V_actual(ci, cj) = V_true(ci, cj) + eta,   eta ~ N(0, sigma^2)

resampled on every call. `sigma <= 0` is an exact passthrough (no allocation,
no RNG draw), which is what makes the sigma=0 sanity check exact.

`GraphSearch.build_weight_matrix` and `LatentPlanner`/`MCTSPlanner` already take
their goal-to-goal distance function as a plain callable (dependency injection),
so wrapping `agent.value` with `NoisyValueFn` is enough to inject noise at
"Noi 1" (Soft Floyd calls it once per episode reset) and "Noi 2" (MCTS calls it
many times across simulations) without touching graph_search.py or planner.py.
`ValueOverrideAgent` lets a script hand a noisy value function to the existing
planners without constructing a new DDPGAgent: it forwards every attribute to
the real agent except `.value`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class NoisyValueFn:
    def __init__(self, value_fn, sigma: float, rng: np.random.Generator):
        self.value_fn = value_fn
        self.sigma = sigma
        self.rng = rng

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        v = self.value_fn(g1, g2)
        if self.sigma <= 0:
            return v
        eta = self.rng.normal(0.0, self.sigma, size=tuple(v.shape))
        noisy = v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)
        return noisy.clamp_min(0.0)


def dmax_candidates(value_matrix, percentiles=(5, 8, 11, 15, 20, 25, 30)) -> list:
    """Candidate `d_max` edge-cutoff values derived from the distribution of the
    graph's own pairwise V distances (docs/SPEC_MCTS_Landmark_L3P.md R3: d_max is
    very sensitive and must be tuned for the checkpoint's V-scale, not left at the
    paper's fixed default which assumes a different V magnitude).

    Returns the given percentiles of the off-diagonal V values, deduplicated and
    sorted. `value_matrix` is a square [m, m] array of V(i, j) distances.
    """
    v = np.asarray(value_matrix, dtype=np.float64)
    m = v.shape[0]
    off = v[~np.eye(m, dtype=bool)]
    cands = np.percentile(off, list(percentiles))
    return sorted(set(round(float(c), 6) for c in cands))


def bootstrap_ci(outcomes, n_boot: int = 2000, alpha: float = 0.05, rng=None):
    """Percentile bootstrap CI for the mean of binary success outcomes
    (docs/SPEC_MCTS_Landmark_L3P.md Sec 10: report mean +/- CI). Pools all
    per-episode 0/1 results and resamples them with replacement `n_boot` times.
    Returns (mean, ci_low, ci_high). Empty input -> (0, 0, 0)."""
    x = np.asarray(outcomes, dtype=np.float64)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def build_sigma_matrix(m: int, frac_high: float, sigma_lo: float, sigma_hi: float,
                       rng: np.random.Generator, goal_idx: Optional[int] = None) -> np.ndarray:
    """Heterogeneous per-edge oracle uncertainty for E1b (docs/... Sec 3, Cach 1):
    a symmetric [m, m] matrix where a random `frac_high` of the unordered
    landmark-landmark edges get sigma_hi and the rest sigma_lo. The high-sigma
    edges are assigned INDEPENDENTLY of V (a short-looking edge can be secretly
    unreliable) -- that decorrelation is what makes knowing sigma useful. The
    goal node's edges (if `goal_idx` given) are kept at sigma_lo (goal
    reachability is treated as reliably estimated). Diagonal is 0."""
    sig = np.full((m, m), sigma_lo, dtype=np.float64)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)
             if goal_idx is None or (i != goal_idx and j != goal_idx)]
    k = int(round(frac_high * len(pairs)))
    if k > 0 and pairs:
        chosen = rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)
        for c in chosen:
            i, j = pairs[c]
            sig[i, j] = sig[j, i] = sigma_hi
    np.fill_diagonal(sig, 0.0)
    return sig


class HeterogeneousNoise:
    """Loai-2 noise with a PER-EDGE sigma (E1b). Wraps a base value_fn and, given
    a fixed node set, adds N(0, sigma_ij^2) to V(i, j) -- resampled every call --
    where sigma_ij comes from `sigma_matrix`. Node indices are recovered from the
    query coordinates by nearest-node lookup (every planner/graph-search query is
    on the fixed episode node set, so the match is exact). `sigma_of(i, j)` gives
    the oracle sigma an index-based caller (the MCTS bonus) can read directly."""

    def __init__(self, value_fn, nodes_t: torch.Tensor, sigma_matrix: np.ndarray,
                 rng: np.random.Generator):
        self.value_fn = value_fn
        self.nodes = nodes_t                       # [m, gdim]
        self.sigma = np.asarray(sigma_matrix, dtype=np.float64)
        self.rng = rng

    def _idx(self, g: torch.Tensor) -> np.ndarray:
        return torch.cdist(g, self.nodes).argmin(dim=1).cpu().numpy()

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        v = self.value_fn(g1, g2)
        s = self.sigma[self._idx(g1), self._idx(g2)]          # [k] per-edge std
        if not np.any(s > 0):
            return v
        eta = self.rng.normal(0.0, s)                          # N(0, s_k)
        noisy = v + torch.as_tensor(eta.reshape(v.shape), dtype=v.dtype, device=v.device)
        return noisy.clamp_min(0.0)

    def sigma_of(self, i: int, j: int) -> float:
        return float(self.sigma[i, j])


class ValueOverrideAgent:
    """Forwards every attribute to `agent` except `.value`, which is fixed to
    `value_fn`. Lets callers build a "noisy V_obs" variant of an existing
    agent without touching DDPGAgent or re-instantiating one."""

    def __init__(self, agent, value_fn):
        self._agent = agent
        self.value = value_fn

    def __getattr__(self, name):
        return getattr(self._agent, name)
