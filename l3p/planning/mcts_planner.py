"""MCTS-over-landmarks (docs/SPEC_MCTS_Landmark_L3P.md, Sections 2-4).

Implements UCT selection, sample-based rollout, alpha/beta uncertainty handling,
execution feedback, recovery, and loop guards used by E1a-E1d.

`LandmarkMCTS` is a standalone, directly-testable search engine: it knows
nothing about the agent/env, only about the discrete landmark+goal graph. Its
root represents the *real current state* (not a landmark): root edges are the
caller-supplied `d_s2c` (fixed for the whole search, exactly like Algorithm 1's
d_{s->c} — this is D(s, pi(s,c), c) from the critic, not V, so it is never
resampled). Edges between landmarks/goal use the injected `value_fn` (typically
a `NoisyValueFn`), resampled on every use — this is what makes the tree
"stochastic transitions" the SPEC calls for. Rollouts are Soft-Floyd-greedy:
one-step lookahead against the precomputed static `d_c2g_heuristic` (the same
quantity `LatentPlanner.reset()` already computes via `GraphSearch`), bootstrapped
with that heuristic if the rollout horizon runs out before reaching the goal.

`MCTSPlanner` subclasses `LatentPlanner` and overrides only `_replan`: `reset()`,
`act()`, `_current_subgoal()`, the commit-for-K-steps counter and the
previous-landmark bookkeeping are all inherited unchanged.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from l3p.planning.planner import LatentPlanner
from l3p.planning.noise import (build_sigma_matrix, HeterogeneousNoise,
                                build_bias_matrix, CriticEdgeFn, BiasedValueFn)


def update_running_variance(stats, key, sample):
    """Welford update for per-edge execution residuals."""
    n, mean, m2 = stats.get(key, (0, 0.0, 0.0))
    n += 1
    delta = float(sample) - mean
    mean += delta / n
    m2 += delta * (float(sample) - mean)
    stats[key] = (n, mean, m2)


def running_sigma_matrix(stats, size):
    sigma = np.zeros((size, size), dtype=np.float64)
    for (i, j), (n, _, m2) in stats.items():
        if n > 1:
            sigma[i, j] = math.sqrt(max(0.0, m2 / (n - 1)))
    return sigma


class _Node:
    __slots__ = ("idx", "untried", "ranked", "pw", "children", "visits",
                 "child_visits", "child_total", "child_total_sq")

    def __init__(self, idx: Optional[int], candidates: List[int],
                 ranked: Optional[List[int]] = None, pw=None):
        self.idx = idx                      # None for the (virtual) root
        self.pw = pw                        # (c, alpha) for progressive widening, else None
        if pw is None:                      # legacy: expand every candidate, random order
            self.untried = list(candidates)
            self.ranked = None
        else:                               # PW: reveal top-k(visits) candidates in priority order
            self.untried = None
            self.ranked = list(ranked if ranked is not None else candidates)
        self.children: Dict[int, "_Node"] = {}
        self.visits = 0
        self.child_visits: Dict[int, int] = defaultdict(int)
        self.child_total: Dict[int, float] = defaultdict(float)
        self.child_total_sq: Dict[int, float] = defaultdict(float)  # sum of squared backups (bayes/thompson)

    def _pw_allowed(self) -> int:
        c, alpha = self.pw
        return max(1, math.ceil(c * (self.visits ** alpha)))

    def fully_expanded(self) -> bool:
        if self.pw is None:
            return len(self.untried) == 0
        return len(self.children) >= min(self._pw_allowed(), len(self.ranked))

    def next_expand(self, rng) -> Optional[int]:
        """Candidate to expand next: a random untried one (legacy) or the
        highest-prior not-yet-expanded one within the PW visit budget."""
        if self.pw is None:
            if not self.untried:
                return None
            return self.untried.pop(int(rng.integers(len(self.untried))))
        if len(self.children) >= self._pw_allowed():
            return None
        for j in self.ranked:
            if j not in self.children:
                return j
        return None

    def best_uct_child(self, c_uct: float, bonus=None) -> Optional[int]:
        """bonus(child_j) -> extra additive score (E1b beta uncertainty bonus)."""
        best_j, best_score = None, -float("inf")
        for j, visits in self.child_visits.items():
            if visits == 0:
                continue
            q = self.child_total[j] / visits
            score = q + c_uct * math.sqrt(math.log(max(self.visits, 1)) / visits)
            if bonus is not None:
                score += bonus(j)
            if score > best_score:
                best_score, best_j = score, j
        return best_j


class LandmarkMCTS:
    def __init__(self, n_landmarks: int, value_fn, nodes_t: torch.Tensor,
                d_c2g_heuristic: np.ndarray, cfg, rng: np.random.Generator,
                admissible: Optional[np.ndarray] = None,
                sigma_matrix: Optional[np.ndarray] = None):
        self.n = n_landmarks
        self.goal_idx = n_landmarks
        self.value_fn = value_fn
        self.nodes_t = nodes_t              # [N+1, gdim]: landmarks then goal
        self.heuristic = d_c2g_heuristic    # [N+1], static soft-Floyd d_{c->g}
        self.cfg = cfg
        self.rng = rng
        # Progressive widening: deep nodes reveal only top-k(visits) candidates in
        # prior order (root stays full, so the returned action is never pruned).
        self.pw = ((getattr(cfg, "mcts_pw_c", 1.0), getattr(cfg, "mcts_pw_alpha", 0.5))
                   if getattr(cfg, "mcts_progressive_widening", False) else None)
        # E1b oracle uncertainty (docs/... Sec 3): per-edge sigma_ij + a mode.
        # "alpha" subtracts lambda_risk*variance from landmark->landmark edge costs
        # (risk-averse: avoid uncertain edges); "beta" adds beta_unc*sigma to the
        # UCT selection score (probe uncertain edges). "none" => plain MCTS
        # (the MCTS-beta=0 ablation). sigma is defined only over landmark/goal
        # edges; root (state->landmark) edges use the clean critic D, so they
        # carry no uncertainty term.
        self.sigma = sigma_matrix
        self.unc_mode = getattr(cfg, "mcts_uncertainty_mode", "none")
        self.lambda_risk = getattr(cfg, "mcts_lambda_risk", 0.0)
        self.lambda_goal = getattr(cfg, "mcts_lambda_goal", 0.0)
        self.beta_unc = getattr(cfg, "mcts_beta_uncertainty", 0.0)
        # bayes/thompson estimate uncertainty from observed returns (no oracle),
        # so only the oracle-sigma modes are disabled when no sigma is supplied.
        self.bayes_sigma0 = getattr(cfg, "mcts_bayes_sigma0", 1.0)
        self.bayes_n0 = getattr(cfg, "mcts_bayes_n0", 1.0)
        if self.sigma is None and self.unc_mode in ("alpha", "beta", "both"):
            self.unc_mode = "none"
        prepare_nodes = getattr(self.value_fn, "prepare_nodes", None)
        if callable(prepare_nodes):
            prepare_nodes(self.nodes_t)
        # d_max masking is a STRUCTURAL property of the graph (which edges
        # exist at all). It must not be re-decided per traversal: unlike Soft
        # Floyd's softmax (which lets a hugely-negative masked logit vanish to
        # ~0 weight), MCTS backs up a plain arithmetic mean, so ever letting a
        # masked edge contribute a `neg_inf`-scale cost to that mean would
        # swamp it. Only the *cost* of an already-admissible edge is resampled
        # on every use (Noi 2). Callers that run many searches within one
        # episode (MCTSPlanner) should compute this ONCE per episode (matching
        # Soft Floyd's own once-per-episode graph build) and pass it in here --
        # left to build fresh only as a convenience default for standalone use
        # (e.g. tests), since rebuilding it every macro-step would let it drift
        # out of sync with a heuristic/d_c2g that IS only built once per episode.
        self._admissible = (admissible if admissible is not None
                           else self.build_admissibility(nodes_t, value_fn, cfg))

    @staticmethod
    def build_admissibility(nodes_t: torch.Tensor, value_fn, cfg) -> np.ndarray:
        """Single batched value_fn call over all (i,j) pairs (mirrors
        GraphSearch.build_weight_matrix), instead of m*(m-1) individual calls --
        with N~50 landmarks that's the difference between one forward pass and
        thousands of them."""
        m = nodes_t.shape[0]
        goal_idx = m - 1
        gi = nodes_t[:, None, :].expand(m, m, nodes_t.shape[1]).reshape(m * m, -1)
        gj = nodes_t[None, :, :].expand(m, m, nodes_t.shape[1]).reshape(m * m, -1)
        dist = value_fn(gi, gj).view(m, m).detach().cpu().numpy()
        adm = dist <= cfg.d_max
        np.fill_diagonal(adm, False)
        adm[goal_idx, :] = False             # goal is an absorbing sink
        return adm

    def _candidates_of(self, i: int) -> List[int]:
        return [j for j in range(self.n + 1) if j != i and self._admissible[i, j]]

    def _rank(self, i: int, cands: List[int]) -> List[int]:
        """Order candidates out of node i by the greedy value-to-go prior
        (edge value + heuristic), best first -- the same score the rollout uses.
        Only consulted for progressive widening's expansion order."""
        if not cands:
            return []
        scores = self._batch_edge_costs(i, cands) + self.heuristic[cands]
        return [cands[k] for k in np.argsort(-scores)]

    def _child_candidates(self, j: int):
        """(candidates, ranked) for a new child j; `ranked` is the PW priority
        order (None in legacy mode)."""
        if j == self.goal_idx:
            return [], []
        cands = self._candidates_of(j)
        return cands, (self._rank(j, cands) if self.pw is not None else None)

    def _edge_cost(self, i: int, j: int) -> float:
        sample_indexed = getattr(self.value_fn, "sample_indexed", None)
        if callable(sample_indexed):
            v = sample_indexed(i, [j]).item()
        else:
            v = self.value_fn(self.nodes_t[i:i + 1], self.nodes_t[j:j + 1]).item()
        cost = -v
        if self.unc_mode in ("alpha", "both"):
            cost -= self.lambda_risk * float(self.sigma[i, j]) ** 2
        if j == self.goal_idx:
            cost += self.lambda_goal
        return cost

    def _batch_edge_costs(self, i: int, js: List[int]) -> np.ndarray:
        """Costs for many candidate edges out of `i` in one value_fn call."""
        sample_indexed = getattr(self.value_fn, "sample_indexed", None)
        if callable(sample_indexed):
            values = sample_indexed(i, js)
        else:
            gi = self.nodes_t[i:i + 1].expand(len(js), -1)
            gj = self.nodes_t[js]
            values = self.value_fn(gi, gj)
        costs = -values.detach().cpu().numpy()
        if self.unc_mode in ("alpha", "both"):
            costs = costs - self.lambda_risk * np.square(self.sigma[i, js])
        if self.lambda_goal:
            costs = costs + self.lambda_goal * (np.asarray(js) == self.goal_idx)
        return costs

    def _posterior(self, node: "_Node", j: int) -> Tuple[float, float]:
        """Normal-normal posterior (mean, variance) of edge (node->j)'s value from
        the returns observed in this search. sigma_lik^2 is the sample variance of
        the backed-up returns (the prior variance until 2 samples exist); the
        posterior variance of the mean is sigma_lik^2/(n+n0). A deterministic edge
        (zero observed variance) collapses to posterior variance 0 -> no bonus,
        recovering greedy behaviour exactly where there is nothing to probe."""
        n = node.child_visits[j]
        mean = node.child_total[j] / n if n else 0.0
        if n >= 2:
            var = max(0.0, (node.child_total_sq[j] - node.child_total[j] ** 2 / n) / (n - 1))
        else:
            var = self.bayes_sigma0 ** 2
        post_var = var / (n + self.bayes_n0)
        return mean, post_var

    def _thompson_child(self, node: "_Node") -> Optional[int]:
        """Sample each visited child's value from its posterior and take the
        argmax (Thompson sampling over the landmark graph)."""
        best_j, best = None, -float("inf")
        for j, n in node.child_visits.items():
            if n == 0:
                continue
            mean, post_var = self._posterior(node, j)
            theta = mean + math.sqrt(max(0.0, post_var)) * float(self.rng.normal())
            if theta > best:
                best, best_j = theta, j
        return best_j

    def _uct_bonus(self, node: "_Node"):
        """Exploration-bonus fn for UCT selection at `node`, or None.
        - beta/both (E1b): oracle per-edge sigma, sigma/sqrt(n) (root edges have
          no oracle uncertainty -> None at the root).
        - bayes: data-driven posterior std of the edge value (works at the root
          too, since downstream return variance is observed there)."""
        if self.unc_mode == "bayes":
            return lambda j: self.beta_unc * math.sqrt(max(0.0, self._posterior(node, j)[1]))
        if self.unc_mode not in ("beta", "both") or node.idx is None:
            return None
        pi = node.idx
        # Known aleatoric sigma implies standard error sigma/sqrt(n) for the
        # estimated edge mean. The information bonus therefore decays as the
        # simulation gathers samples instead of permanently favoring noisy edges.
        return lambda j: (
            self.beta_unc * float(self.sigma[pi, j])
            / math.sqrt(max(1, node.child_visits[j]))
        )

    def _rollout(self, start_idx: int, budget: int) -> float:
        start_heuristic = float(self.heuristic[start_idx])
        cur = start_idx
        total = 0.0
        depth = 0
        while cur != self.goal_idx and depth < budget:
            cands = self._candidates_of(cur)
            if not cands:
                break                        # dead end: no admissible edge out
            costs = self._batch_edge_costs(cur, cands)
            scores = costs + self.heuristic[cands]
            best_local = int(np.argmax(scores))
            total += float(costs[best_local])
            cur = cands[best_local]
            depth += 1
        if cur != self.goal_idx:
            total += float(self.heuristic[cur])   # bootstrap: heuristic-to-go
        if getattr(self.cfg, "mcts_cap_rollout_by_heuristic", False):
            # Soft Floyd's value-to-go is the baseline heuristic used for
            # sigma=0 fairness. On some high-dimensional goal envs, the raw V
            # has near-zero landmark->goal wormholes that a hard rollout would
            # exploit even though Soft Floyd assigns the node a much lower
            # value. Cap optimistic rollouts so MCTS cannot rate a node better
            # than the planner heuristic it is bootstrapping from.
            total = min(total, start_heuristic)
        return total

    @staticmethod
    def _edge_backups(edge_costs, rollout_return, suffix):
        """Per-edge backup value for one simulation. `suffix=False` (legacy):
        every edge on the path gets the full root->leaf return. `suffix=True`:
        edge k gets its return-to-go (sum of costs from k onward + rollout) --
        the standard MCTS credit assignment. The root edge (k=0) is identical
        either way, so the planner's returned action is unaffected; only deeper
        edges shed the shared-but-noisy prefix, which sharpens deep selection
        under resampled (stochastic) edge costs."""
        n = len(edge_costs)
        if not suffix:
            total = float(sum(edge_costs)) + rollout_return
            return [total] * n
        out = [0.0] * n
        running = rollout_return
        for k in range(n - 1, -1, -1):
            running += edge_costs[k]
            out[k] = running
        return out

    def search(self, d_s2c: np.ndarray, mask: Optional[np.ndarray] = None
              ) -> Tuple[Optional[int], Dict[int, Tuple[int, float]]]:
        """Run `cfg.mcts_n_simulations` simulations rooted at the real state
        (edges = `d_s2c`) and return (best_root_child, {child: (visits, mean_q)}).
        `mask[j]=True` excludes candidate j from the ROOT only (mirrors
        Algorithm 1's previous-landmark removal)."""
        if mask is None:
            mask = np.zeros(self.n + 1, dtype=bool)
        root = _Node(idx=None, candidates=[j for j in range(self.n + 1) if not mask[j]])
        started = time.perf_counter()

        # UCT cannot compare an action it has never expanded. Reserve one pass to
        # cover every root action and one pass for UCT to allocate repeat samples;
        # otherwise max-visit selection degenerates to a tie of one noisy rollout.
        root_width = len(root.untried)
        n_simulations = max(
            int(self.cfg.mcts_n_simulations),
            2 * root_width if root_width else 0)
        if root_width == 0:
            self.last_search_seconds = time.perf_counter() - started
            self.actual_simulations = 0
            return None, {}
        for _ in range(n_simulations):
            path: List[Tuple[_Node, int]] = []
            node = root
            depth = 0

            # SELECTION + EXPANSION (unified to support progressive widening;
            # with PW disabled this is equivalent to the classic "descend through
            # fully-expanded nodes via UCT, then expand one random child").
            while depth < self.cfg.mcts_rollout_horizon:
                if node.idx == self.goal_idx:
                    break
                if not node.fully_expanded():
                    j = node.next_expand(self.rng)
                    if j is None:
                        break
                    candidates, ranked = self._child_candidates(j)
                    child = node.children.setdefault(
                        j, _Node(j, candidates, ranked=ranked, pw=self.pw))
                    path.append((node, j))
                    node = child
                    depth += 1
                    break                       # expand one node per simulation
                if not node.children:
                    break
                if self.unc_mode == "thompson":
                    j = self._thompson_child(node)
                else:
                    j = node.best_uct_child(self.cfg.mcts_c_uct, bonus=self._uct_bonus(node))
                if j is None:
                    break
                path.append((node, j))
                node = node.children[j]
                depth += 1
                if j == self.goal_idx:
                    break

            # tree-portion per-edge costs: root edges are the fixed d_s2c;
            # landmark/goal edges resample the (possibly noisy) value_fn on use.
            edge_costs = []
            for parent, child_idx in path:
                if parent.idx is None:
                    c = d_s2c[child_idx]
                    if child_idx == self.goal_idx:
                        c += self.lambda_goal
                else:
                    c = self._edge_cost(parent.idx, child_idx)
                edge_costs.append(c)

            # ROLLOUT (Soft-Floyd-greedy) from the reached node
            if node.idx == self.goal_idx:
                rollout_return = 0.0
            else:
                rollout_return = self._rollout(node.idx, self.cfg.mcts_rollout_horizon - depth)

            # BACKPROPAGATION: credit each edge with its backup value (full
            # root->leaf return by default; return-to-go suffix when enabled).
            backups = self._edge_backups(
                edge_costs, rollout_return,
                getattr(self.cfg, "mcts_suffix_backup", False))
            for (parent, child_idx), r in zip(path, backups):
                parent.visits += 1
                parent.child_visits[child_idx] += 1
                parent.child_total[child_idx] += r
                parent.child_total_sq[child_idx] += r * r

        stats: Dict[int, Tuple[int, float]] = {}
        best_idx, best_key = None, (-1, -float("inf"))
        for j, visits in root.child_visits.items():
            q = root.child_total[j] / visits if visits > 0 else -float("inf")
            stats[j] = (visits, q)
            key = (visits, q)              # robust child: max visits, tie-break by Q
            if key > best_key:
                best_key, best_idx = key, j
        self.last_search_seconds = time.perf_counter() - started
        self.actual_simulations = n_simulations
        return best_idx, stats


class MCTSPlanner(LatentPlanner):
    """Drop-in replacement for `LatentPlanner` that picks sub-goals via
    `LandmarkMCTS` instead of the direct `argmax(d_s2c + d_c2g)` formula.
    Everything except `_replan` is inherited unchanged."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg,
                rng: Optional[np.random.Generator] = None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg)
        self.rng = rng if rng is not None else np.random.default_rng(cfg.mcts_seed)
        self._admissible: Optional[np.ndarray] = None
        self.search_seconds = 0.0
        self.search_calls = 0

    @staticmethod
    def _soft_floyd_action(d_s2c, d_c2g, mask):
        combined = np.asarray(d_s2c) + np.asarray(d_c2g)
        combined = combined.copy()
        combined[np.asarray(mask, dtype=bool)] = -np.inf
        return None if np.all(np.isneginf(combined)) else int(np.argmax(combined))

    def _is_deterministic_limit(self):
        sigma = getattr(self.agent.value, "sigma", None)
        return sigma is not None and float(sigma) <= 0.0

    @torch.no_grad()
    def reset(self, goal: np.ndarray, extra_centroids: Optional[torch.Tensor] = None) -> None:
        """Same as LatentPlanner.reset(), plus caching the d_max admissibility
        mask ONCE for the whole episode -- matching self.d_c2g's cadence, so
        MCTS's own view of "which edges exist" never drifts from the heuristic
        it scores rollouts against (see LandmarkMCTS.build_admissibility)."""
        self.reset_state()
        self.search_seconds = 0.0
        self.search_calls = 0
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            self._admissible = None
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        d_c2g, admissible = self.gs.distances_to_goal_with_admissibility(
            centroids, goal_t, self.ae, self.agent.value)
        self.d_c2g = d_c2g.cpu().numpy()
        self._admissible = admissible.cpu().numpy()

    @torch.no_grad()
    def _replan(self, obs: np.ndarray) -> None:
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)  # [N+1, gdim]
        d = self.agent.distance_after_action(obs, candidates)      # D(s, pi(s,c), c), real/un-noised
        d_s2c = -d

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True

        if self._is_deterministic_limit():
            best_idx = self._soft_floyd_action(d_s2c, self.d_c2g, mask)
        else:
            nodes_t = torch.as_tensor(
                candidates, dtype=torch.float32, device=self.device)
            mcts = LandmarkMCTS(
                n_landmarks=self.n_landmarks, value_fn=self.agent.value,
                nodes_t=nodes_t, d_c2g_heuristic=self.d_c2g, cfg=self.cfg,
                rng=self.rng, admissible=self._admissible)
            best_idx, _ = mcts.search(d_s2c, mask=mask)
            self.search_seconds += mcts.last_search_seconds
            self.search_calls += 1
        if best_idx is None:
            best_idx = self.n_landmarks

        self.subg_idx = best_idx
        self.cnt = max(1.0, float(round(-d_s2c[best_idx])))
        self.prev_landmark = self.subg_idx


# --------------------------------------------------------------------------- E1c
# Loai-1 (estimation bias): the landmark graph is built from the critic's CLEAN
# distance D (env-step scale, `CriticEdgeFn`) with a per-episode FIXED multiplicative
# bias on top (`BiasedValueFn`). Everything (d_c2g, d_s2c, realized cost) is then on
# one env-step scale, so execution feedback can correct the bias without a unit
# mismatch. See docs/SPEC_MCTS_Landmark_L3P.md Sec 4-6.

def _build_biased_d_graph(planner, sigma, noise_seed):
    """Shared per-episode setup for the E1c planners: decode landmarks, build the
    Loai-1 biased critic-D value fn over this node set, and return
    (nodes_t, biased_fn, bias_matrix). Assumes planner.goal / landmark_goals set."""
    nodes_t = torch.as_tensor(
        np.concatenate([planner.landmark_goals, planner.goal[None, :]], axis=0),
        dtype=torch.float32, device=planner.device)
    m = nodes_t.shape[0]
    bias = build_bias_matrix(m, sigma, np.random.default_rng(noise_seed), goal_idx=m - 1)
    base_edge = getattr(planner, "edge_value_fn", None) or CriticEdgeFn(planner.agent)
    biased = BiasedValueFn(base_edge, nodes_t, bias)
    return nodes_t, biased, bias


class SoftFloydE1c(LatentPlanner):
    """E1c baseline: L3P Soft Floyd on a Loai-1-biased edge graph, STATIC
    (plans once per episode, never corrects) -- the "trusts the wormhole and never
    fixes it" behaviour E1c contrasts against."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg, sigma,
                 noise_seed, edge_value_fn=None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg)
        self.sigma = sigma
        self.noise_seed = noise_seed
        self.edge_value_fn = edge_value_fn

    @torch.no_grad()
    def reset(self, goal, extra_centroids=None):
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()
        _, biased, _ = _build_biased_d_graph(self, self.sigma, self.noise_seed)
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae, biased).cpu().numpy()
    # inherits _replan (biased d_c2g + clean critic d_s2c) and act -> static Algorithm 1


class FeedbackMCTSPlanner(MCTSPlanner):
    """E1c MCTS on the Loai-1-biased edge graph, with optional EXECUTION
    FEEDBACK (docs/... Sec 4.1/4.3/4.4). When `feedback=True`, after each executed
    macro-step it snaps the current state to the nearest landmark i, measures the
    realized cost of the traversed edge (i -> chosen subgoal j) in env steps, blends
    it into an EMA estimate V_exec[i][j], rebuilds d_c2g with the corrected estimates,
    and blacklists edges that repeatedly fail (loop guard). Soft Floyd cannot do this
    (static), so MCTS+feedback should detect and route around a biased "wormhole"
    that Soft Floyd keeps trusting. `feedback=False` is the MCTS-no-feedback control
    (plain MCTS on the static biased graph)."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg, sigma,
                noise_seed, feedback=True, rho=0.5, tau_reach=2.0, tau_progress=3.0,
                tau_snap=4.0, r_max=2, edge_value_fn=None, rng=None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg, rng=rng)
        self.sigma = sigma
        self.noise_seed = noise_seed
        self.feedback = feedback
        self.rho = rho
        self.tau_reach = tau_reach
        self.tau_progress = tau_progress
        self.tau_snap = tau_snap
        self.r_max = r_max
        self.edge_value_fn = edge_value_fn
        # Execution feedback was designed for the critic-D substrate (env-step
        # scale, obs == goal): realized cost mixes the env-step count macro_k
        # with a critic-D remaining distance, which is coherent only there. On a
        # V substrate (obs != goal, edge_value_fn is the ~10x-compressed value
        # fn) that mix is a unit error -- so realized cost and snap thresholds
        # are measured on the substrate (`_sub_goal_dist`) instead. A missing /
        # CriticEdgeFn edge fn IS the critic-D substrate -> keep the original path.
        self._step_scale = (edge_value_fn is None
                            or isinstance(edge_value_fn, CriticEdgeFn))

    @torch.no_grad()
    def reset(self, goal, extra_centroids=None):
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            self._admissible = self._biased = None
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()
        nodes_t, biased, bias = _build_biased_d_graph(self, self.sigma, self.noise_seed)
        self._nodes_t = nodes_t
        self._biased = biased
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        d_c2g, admissible = self.gs.distances_to_goal_with_admissibility(
            centroids, goal_t, self.ae, biased)
        self.d_c2g = d_c2g.cpu().numpy()
        self._admissible = admissible.cpu().numpy()
        # execution-feedback episode state
        self._v_exec = {}                 # (i, j) -> corrected env-step cost
        self._residual_stats = {}          # (i, j) -> Welford(n, mean, M2)
        self._attempts = {}               # subgoal j -> failed attempts
        self._blacklist = set()
        self._macro_k = 0
        self._macro_start_i = None
        self._macro_start_z = None
        self._macro_start_ag = None
        self._cur_ag = None
        self._cur_j = None
        self.search_seconds = 0.0
        self.search_calls = 0
        self.stats = dict(macros=0, reached=0, progressed=0, stuck=0,
                          blacklisted=0, corrections=0, snaps=0,
                          virtual_roots=0, uncertain_edges=0)

    # ---- feedback helpers ----
    def _sub_goal_dist(self, ga, gb):
        """Distance between GOAL-space point(s) on the SAME substrate as the
        graph edges (edge_value_fn). On a critic-D substrate this equals the
        env-step distance; on a compressed V substrate it keeps realized cost
        and snap thresholds in the graph's own units (rather than mixing them
        with the env-step count macro_k)."""
        fn = self.edge_value_fn if self.edge_value_fn is not None else CriticEdgeFn(self.agent)
        ta = torch.as_tensor(np.atleast_2d(ga), dtype=torch.float32, device=self.device)
        tb = torch.as_tensor(np.atleast_2d(gb), dtype=torch.float32, device=self.device)
        return np.atleast_1d(fn(ta, tb).detach().cpu().numpy())

    def _snap(self, obs, achieved_goal=None):
        if (not self._step_scale) and achieved_goal is not None:
            ag = np.asarray(achieved_goal, dtype=np.float32)
            distances = self._sub_goal_dist(
                np.repeat(ag[None, :], len(self.landmark_goals), axis=0),
                self.landmark_goals)
        else:
            distances = np.asarray(
                self.agent.distance_after_action(obs, self.landmark_goals))
        idx = int(np.argmin(distances))
        nearest = float(distances[idx])
        return (idx, nearest) if nearest <= self.tau_snap else (None, nearest)

    def _critic_dist(self, z, target_goal):
        return float(self.agent.distance_after_action(z[None, :], target_goal[None, :])[0])

    def _biased_edge(self, i, j):
        return float(self._biased(self._nodes_t[i:i + 1], self._nodes_t[j:j + 1]).item())

    def _corrected_fn(self):
        """value_fn that overlays the learned V_exec corrections on the biased graph."""
        biased, vexec = self._biased, self._v_exec
        def fn(g1, g2):
            v = biased(g1, g2).clone()
            if vexec:
                i = biased._idx(g1); j = biased._idx(g2)
                for (ii, jj), val in vexec.items():
                    sel = (i == ii) & (j == jj)
                    if sel.any():
                        v[torch.as_tensor(sel)] = val
            return v
        return fn

    def _observe(self, obs_end, achieved_goal=None):
        """Finalize the just-executed macro-step (i -> j): update V_exec (EMA),
        classify the outcome, and apply the loop guard (Sec 4.1/4.3/4.4)."""
        i, j = self._macro_start_i, self._cur_j
        if j is None:
            return
        c_j = self.goal if j >= self.n_landmarks else self.landmark_goals[j]
        z_end = (np.asarray(achieved_goal, dtype=np.float32)
                 if achieved_goal is not None else np.asarray(obs_end, dtype=np.float32))
        if (not self._step_scale) and achieved_goal is not None:
            # V substrate: realized cost stays in the graph's V units (both the
            # traversed and the remaining leg via edge_value_fn), so the EMA
            # blends like-with-like instead of adding the env-step count macro_k
            # to a ~10x-compressed V.
            start_ag = getattr(self, "_macro_start_ag", None)
            if start_ag is None:
                start_ag = z_end
            dist_to_goal = float(self._sub_goal_dist(z_end, c_j)[0])
            dist_travelled = float(self._sub_goal_dist(start_ag, z_end)[0])
            realized = dist_travelled + max(0.0, dist_to_goal)
            reached = float(np.linalg.norm(z_end - c_j)) <= self.cfg.goal_threshold
        else:
            dist_to_goal = self._critic_dist(obs_end, c_j)
            dist_travelled = self._critic_dist(self._macro_start_z, z_end)
            realized = self._macro_k + max(0.0, dist_to_goal)     # env-step scale
            reached = (
                float(np.linalg.norm(z_end - c_j)) <= self.cfg.goal_threshold
                if achieved_goal is not None else dist_to_goal <= self.tau_reach
            )

        if i is not None:
            prev = self._v_exec.get((i, j), self._biased_edge(i, j))
            update_running_variance(
                self._residual_stats, (i, j), realized - prev)
            self.stats["uncertain_edges"] = sum(
                n > 1 and m2 > 0 for n, _, m2 in self._residual_stats.values())
            self._v_exec[(i, j)] = (1 - self.rho) * prev + self.rho * realized
            self.stats["corrections"] += 1

        if reached:                                            # REACHED
            self._attempts.clear()
            self.stats["reached"] += 1
        elif dist_travelled >= self.tau_progress:              # PROGRESSED_ELSEWHERE
            self._attempts[j] = self._attempts.get(j, 0) + 1
            self.stats["progressed"] += 1
            if self._attempts[j] >= self.r_max and j not in self._blacklist:
                self._blacklist.add(j)
                self.stats["blacklisted"] += 1
        else:                                                  # STUCK -> blacklist now
            self.stats["stuck"] += 1
            if j not in self._blacklist:
                self._blacklist.add(j)
                self.stats["blacklisted"] += 1
        self._cur_j = None

    def _candidate_mask(self):
        """Root candidate mask (Algorithm 1 prev-landmark removal + loop-guard
        blacklist). The GOAL node (index n_landmarks) is never masked: it is the
        target, not a routing landmark, so blacklisting a failed direct-to-goal
        attempt must not forbid targeting the goal later in the episode."""
        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True
        if self.feedback:
            for b in self._blacklist:
                if b < self.n_landmarks:          # never the goal node
                    mask[b] = True
        return mask

    def _reached_current_subgoal(self, obs, achieved_goal=None):
        if self._cur_j is None:
            return False
        target = (self.goal if self._cur_j >= self.n_landmarks
                  else self.landmark_goals[self._cur_j])
        if achieved_goal is not None:
            return float(np.linalg.norm(
                np.asarray(achieved_goal) - target)) <= self.cfg.goal_threshold
        return self._critic_dist(obs, target) <= self.tau_reach

    def finalize(self, obs, achieved_goal=None):
        if self.feedback and self._cur_j is not None:
            self._observe(obs, achieved_goal)

    @torch.no_grad()
    def act(self, obs, noise_scale=0.0, random_prob=0.0, achieved_goal=None):
        if self.n_landmarks == 0:
            return self.agent.act(obs, self.goal, noise_scale, random_prob)
        self._cur_ag = achieved_goal          # goal-space pose for substrate snap
        if self.feedback and self._reached_current_subgoal(obs, achieved_goal):
            self._observe(obs, achieved_goal)
            self.cnt = 0.0
        if self.cnt > 1.0:
            self.cnt -= 1.0
        else:
            if self.feedback and self._cur_j is not None:
                self._observe(obs, achieved_goal)              # finalize previous macro-step
            self._replan(obs)
        self._macro_k += 1
        subgoal = self._current_subgoal()
        return self.agent.act(obs, subgoal, noise_scale, random_prob)

    @torch.no_grad()
    def _replan(self, obs):
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)
        d_s2c = -self.agent.distance_after_action(obs, candidates)   # clean critic D
        nodes_t = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)

        if self.feedback:
            value_fn = self._corrected_fn()
            goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
            d_c2g_t, admissible_t = self.gs.distances_to_goal_with_admissibility(
                self.centroids, goal_t, self.ae, value_fn)
            d_c2g = d_c2g_t.cpu().numpy()
            admissible = admissible_t.cpu().numpy()
        else:
            value_fn, d_c2g, admissible = self._biased, self.d_c2g, self._admissible

        mask = self._candidate_mask()

        if self.sigma <= 0:
            best_idx = self._soft_floyd_action(d_s2c, d_c2g, mask)
        else:
            sigma_matrix = running_sigma_matrix(
                self._residual_stats, self.n_landmarks + 1)
            mcts = LandmarkMCTS(
                n_landmarks=self.n_landmarks, value_fn=value_fn,
                nodes_t=nodes_t, d_c2g_heuristic=d_c2g, cfg=self.cfg,
                rng=self.rng, admissible=admissible,
                sigma_matrix=sigma_matrix)
            best_idx, _ = mcts.search(d_s2c, mask=mask)
            self.search_seconds += mcts.last_search_seconds
            self.search_calls += 1
        if best_idx is None:                        # everything masked -> head to goal
            best_idx = self.n_landmarks

        self.subg_idx = best_idx
        self.cnt = max(1.0, float(round(-d_s2c[best_idx])))
        self.prev_landmark = self.subg_idx
        self.stats["macros"] += 1
        if self.feedback:                           # open a new macro-step
            ag = getattr(self, "_cur_ag", None)
            self._macro_k = 0
            self._macro_start_i, _ = self._snap(obs, ag)
            if self._macro_start_i is None:
                self.stats["virtual_roots"] += 1
            else:
                self.stats["snaps"] += 1
            self._macro_start_z = np.asarray(obs, dtype=np.float32)
            self._macro_start_ag = None if ag is None else np.asarray(ag, dtype=np.float32)
            self._cur_j = best_idx


class UncertaintyMCTSPlanner(MCTSPlanner):
    """E1b planner: like MCTSPlanner but with HETEROGENEOUS per-edge oracle
    uncertainty (docs/... Sec 3, Cach 1). It owns the noise (no external
    NoisyValueFn): per episode it assigns a sigma_ij matrix over its own node
    set (a random `frac_high` of landmark-landmark edges get sigma_hi, decorrelated
    from V), wraps the CLEAN agent value with HeterogeneousNoise, and uses that
    SAME noisy V for the soft-Floyd d_c2g, the admissibility mask, and the MCTS
    rollout/tree edges -- while handing the oracle sigma matrix to LandmarkMCTS so
    the alpha/beta bonus (set via cfg.mcts_uncertainty_mode) can act on it.

    Root (state->landmark) edges still use the clean critic D (d_s2c), so they
    carry no uncertainty -- only the landmark graph is noisy, matching E1a."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg,
                frac_high: float, sigma_hi: float, sigma_lo: float = 0.0,
                noise_seed: int = 0, rng: Optional[np.random.Generator] = None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg, rng=rng)
        self.frac_high = frac_high
        self.sigma_hi = sigma_hi
        self.sigma_lo = sigma_lo
        self.noise_seed = noise_seed
        self._base_value = agent.value      # CLEAN value fn; noise is added here
        self._hetero = None
        self._sigma = None

    @torch.no_grad()
    def reset(self, goal: np.ndarray, extra_centroids: Optional[torch.Tensor] = None) -> None:
        # Reimplements LatentPlanner.reset so the heterogeneous noisy V (built over
        # THIS episode's node set) is the value fn used for d_c2g + admissibility.
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)
        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]
        if self.n_landmarks == 0:
            self._admissible = self._hetero = self._sigma = None
            return
        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()

        nodes_t = torch.as_tensor(
            np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0),
            dtype=torch.float32, device=self.device)
        m = nodes_t.shape[0]
        self._sigma = build_sigma_matrix(m, self.frac_high, self.sigma_lo, self.sigma_hi,
                                         np.random.default_rng(self.noise_seed), goal_idx=m - 1)
        self._hetero = HeterogeneousNoise(self._base_value, nodes_t, self._sigma,
                                          np.random.default_rng(self.noise_seed + 1))

        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        d_c2g, admissible = self.gs.distances_to_goal_with_admissibility(
            centroids, goal_t, self.ae, self._hetero)
        self.d_c2g = d_c2g.cpu().numpy()
        self._admissible = admissible.cpu().numpy()
        self.search_seconds = 0.0
        self.search_calls = 0

    @torch.no_grad()
    def _replan(self, obs: np.ndarray) -> None:
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)
        d = self.agent.distance_after_action(obs, candidates)      # clean critic D
        d_s2c = -d

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True

        if not np.any(self._sigma > 0):
            best_idx = self._soft_floyd_action(d_s2c, self.d_c2g, mask)
        else:
            nodes_t = torch.as_tensor(
                candidates, dtype=torch.float32, device=self.device)
            mcts = LandmarkMCTS(
                n_landmarks=self.n_landmarks, value_fn=self._hetero,
                nodes_t=nodes_t, d_c2g_heuristic=self.d_c2g, cfg=self.cfg,
                rng=self.rng, admissible=self._admissible,
                sigma_matrix=self._sigma)
            best_idx, _ = mcts.search(d_s2c, mask=mask)
            self.search_seconds += mcts.last_search_seconds
            self.search_calls += 1
        if best_idx is None:
            best_idx = self.n_landmarks

        self.subg_idx = best_idx
        self.cnt = max(1.0, float(round(-d_s2c[best_idx])))
        self.prev_landmark = self.subg_idx
