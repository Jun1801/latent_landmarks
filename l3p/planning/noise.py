"""Noise injection for the MCTS-over-landmarks experiments
(docs/SPEC_MCTS_Landmark_L3P.md, Section 5).

The module implements both Loại 2 stochastic noise and Loại 1 fixed edge bias.
Loại 2 follows:

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


CALIBRATION_SEED_OFFSET = 10_000_019


def calibration_seed(report_seed: int) -> int:
    """Return a deterministic seed namespace disjoint from reporting episodes."""
    return int(report_seed) + CALIBRATION_SEED_OFFSET


class NoisyValueFn:
    def __init__(self, value_fn, sigma: float, rng: np.random.Generator):
        self.value_fn = value_fn
        self.sigma = sigma
        self.rng = rng
        self._base_matrix = None

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        v = self.value_fn(g1, g2)
        if self.sigma <= 0:
            return v
        eta = self.rng.normal(0.0, self.sigma, size=tuple(v.shape))
        return v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)

    @torch.no_grad()
    def prepare_nodes(self, nodes: torch.Tensor) -> None:
        """Cache deterministic V_true for a fixed MCTS landmark node set."""
        m, dim = nodes.shape
        g1 = nodes[:, None, :].expand(m, m, dim).reshape(m * m, dim)
        g2 = nodes[None, :, :].expand(m, m, dim).reshape(m * m, dim)
        self._base_matrix = self.value_fn(g1, g2).reshape(m, m).detach()

    def sample_indexed(self, i: int, js) -> torch.Tensor:
        """Read cached V_true edges and resample Loai-2 noise per traversal."""
        if self._base_matrix is None:
            raise RuntimeError("prepare_nodes must be called before sample_indexed")
        v = self._base_matrix[i, js]
        if self.sigma <= 0:
            return v
        eta = self.rng.normal(0.0, self.sigma, size=tuple(v.shape))
        return v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)


def dmax_candidates(value_matrix, percentiles=(5, 8, 11, 15, 20, 25, 30),
                    fallback: Optional[float] = None) -> list:
    """Candidate `d_max` edge-cutoff values derived from the distribution of the
    graph's own pairwise V distances (docs/SPEC_MCTS_Landmark_L3P.md R3: d_max is
    very sensitive and must be tuned for the checkpoint's V-scale, not left at the
    paper's fixed default which assumes a different V magnitude).

    Returns the given percentiles of positive off-diagonal V values,
    deduplicated and sorted. `value_matrix` is a square [m, m] array of V(i, j)
    distances. `fallback` should be the environment's configured d_max; it keeps
    calibration from collapsing when a checkpoint has many near-zero V wormholes
    (observed on Fetch).
    """
    v = np.asarray(value_matrix, dtype=np.float64)
    m = v.shape[0]
    off = v[~np.eye(m, dtype=bool)]
    off = off[np.isfinite(off)]
    positive = off[off > 1e-8]
    source = positive if positive.size else off
    vals = []
    if source.size:
        vals.extend(float(c) for c in np.percentile(source, list(percentiles)))
    if fallback is not None and np.isfinite(fallback) and fallback > 0:
        vals.append(float(fallback))
    vals = [round(float(c), 6) for c in vals if np.isfinite(c) and c >= 0]
    return sorted(set(vals)) or [0.0]


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


def hierarchical_bootstrap_ci(groups, n_boot: int = 2000, alpha: float = 0.05,
                              rng=None):
    """Bootstrap a mean while preserving between-seed variation.

    `groups` is one binary-outcome sequence per reporting seed. Each replicate
    samples seeds with replacement, then episodes within each selected seed.
    """
    arrays = [np.asarray(g, dtype=np.float64) for g in groups if len(g)]
    if not arrays:
        return 0.0, 0.0, 0.0
    rng = rng if rng is not None else np.random.default_rng(0)
    observed = float(np.concatenate(arrays).mean())
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, len(arrays), size=len(arrays))
        samples = []
        for idx in chosen:
            group = arrays[int(idx)]
            samples.append(group[rng.integers(0, group.size, size=group.size)])
        means[b] = np.concatenate(samples).mean()
    lo, hi = np.percentile(
        means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return observed, float(lo), float(hi)


def build_sigma_matrix(m: int, frac_high: float, sigma_lo: float, sigma_hi: float,
                       rng: np.random.Generator, goal_idx: Optional[int] = None) -> np.ndarray:
    """Heterogeneous per-edge oracle uncertainty for E1b (docs/... Sec 3, Cach 1):
    a directed [m, m] matrix where a random `frac_high` of the ordered graph
    edges get sigma_hi and the rest sigma_lo. The high-sigma
    edges are assigned INDEPENDENTLY of V (a short-looking edge can be secretly
    unreliable) -- that decorrelation is what makes knowing sigma useful.
    Goal edges participate like every other graph edge. Diagonal is 0."""
    sig = np.full((m, m), sigma_lo, dtype=np.float64)
    pairs = [(i, j) for i in range(m) for j in range(m) if i != j]
    k = int(round(frac_high * len(pairs)))
    if k > 0 and pairs:
        chosen = rng.choice(len(pairs), size=min(k, len(pairs)), replace=False)
        for c in chosen:
            i, j = pairs[c]
            sig[i, j] = sigma_hi
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
        self._base_matrix = None

    def _idx(self, g: torch.Tensor) -> np.ndarray:
        return torch.cdist(g, self.nodes).argmin(dim=1).cpu().numpy()

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        v = self.value_fn(g1, g2)
        s = self.sigma[self._idx(g1), self._idx(g2)]          # [k] per-edge std
        if not np.any(s > 0):
            return v
        eta = self.rng.normal(0.0, s)                          # N(0, s_k)
        return v + torch.as_tensor(eta.reshape(v.shape), dtype=v.dtype, device=v.device)

    def sigma_of(self, i: int, j: int) -> float:
        return float(self.sigma[i, j])

    @torch.no_grad()
    def prepare_nodes(self, nodes: torch.Tensor) -> None:
        m, dim = nodes.shape
        g1 = nodes[:, None, :].expand(m, m, dim).reshape(m * m, dim)
        g2 = nodes[None, :, :].expand(m, m, dim).reshape(m * m, dim)
        self._base_matrix = self.value_fn(g1, g2).reshape(m, m).detach()

    def sample_indexed(self, i: int, js) -> torch.Tensor:
        if self._base_matrix is None:
            raise RuntimeError("prepare_nodes must be called before sample_indexed")
        v = self._base_matrix[i, js]
        s = np.asarray(self.sigma[i, js])
        if not np.any(s > 0):
            return v
        eta = self.rng.normal(0.0, s, size=tuple(v.shape))
        return v + torch.as_tensor(eta, dtype=v.dtype, device=v.device)


def build_bias_matrix(m: int, sigma: float, rng: np.random.Generator,
                      goal_idx: Optional[int] = None) -> np.ndarray:
    """Loai-1 estimation bias for E1c (docs/... Sec 5.1): a directed [m, m]
    matrix of multiplicative biases b_ij ~ N(0, sigma^2), one per directed
    landmark edge, sampled ONCE (fixed for the episode). Edges with b_ij < 0
    look shorter than they are -- the "wormhole" traps. Goal-node edges are
    biased by the same process; only the diagonal stays zero."""
    b = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            if i != j:
                b[i, j] = rng.normal(0.0, sigma)
    return b


class CriticEdgeFn:
    """A `value_fn`-interface adapter returning the critic's CLEAN state->goal
    distance D(g1, pi(g1, g2), g2) on env-step scale. This is the E1c graph
    substrate: unlike `agent.value` (compressed ~10x and only weakly correlated
    with D on this checkpoint), the critic D is accurate and on the same scale
    as the realized env-step cost used by execution feedback -- so a Loai-1 bias
    injected on top is the ONLY error, and step-count feedback can correct it
    without a unit mismatch. PointMaze-only: a goal coordinate doubles as a state
    (obs == achieved_goal there)."""

    def __init__(self, agent):
        self.agent = agent

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        d = self.agent.distance_after_action(g1.detach().cpu().numpy(),
                                             g2.detach().cpu().numpy())
        return torch.as_tensor(np.atleast_1d(d), dtype=torch.float32)


class BiasedValueFn:
    """Loai-1 noise (E1c): V_obs(i,j) = V_true(i,j) * (1 + b_ij) with a per-episode
    FIXED bias matrix (NOT resampled) -- a systematic mis-estimate that averaging
    over rollout samples cannot remove, so only execution feedback can correct it.
    Nearest-node index lookup follows `HeterogeneousNoise`; the exact
    multiplicative formula is preserved without boundary clipping."""

    def __init__(self, value_fn, nodes_t: torch.Tensor, bias_matrix: np.ndarray):
        self.value_fn = value_fn
        self.nodes = nodes_t
        self.bias = np.asarray(bias_matrix, dtype=np.float64)
        self._base_matrix = None

    def _idx(self, g: torch.Tensor) -> np.ndarray:
        return torch.cdist(g, self.nodes).argmin(dim=1).cpu().numpy()

    def __call__(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        v = self.value_fn(g1, g2)
        b = self.bias[self._idx(g1), self._idx(g2)]
        factor = torch.as_tensor((1.0 + b).reshape(v.shape), dtype=v.dtype, device=v.device)
        return v * factor

    @torch.no_grad()
    def prepare_nodes(self, nodes: torch.Tensor) -> None:
        m, dim = nodes.shape
        g1 = nodes[:, None, :].expand(m, m, dim).reshape(m * m, dim)
        g2 = nodes[None, :, :].expand(m, m, dim).reshape(m * m, dim)
        self._base_matrix = self.value_fn(g1, g2).reshape(m, m).detach()

    def sample_indexed(self, i: int, js) -> torch.Tensor:
        if self._base_matrix is None:
            raise RuntimeError("prepare_nodes must be called before sample_indexed")
        v = self._base_matrix[i, js]
        factor = torch.as_tensor(
            1.0 + np.asarray(self.bias[i, js]), dtype=v.dtype, device=v.device)
        return v * factor


class ValueOverrideAgent:
    """Forwards every attribute to `agent` except `.value`, which is fixed to
    `value_fn`. Lets callers build a "noisy V_obs" variant of an existing
    agent without touching DDPGAgent or re-instantiating one."""

    def __init__(self, agent, value_fn):
        self._agent = agent
        self.value = value_fn

    def __getattr__(self, name):
        return getattr(self._agent, name)
