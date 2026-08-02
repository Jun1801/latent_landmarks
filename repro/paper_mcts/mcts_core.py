"""Self-contained LandmarkMCTS engine for the paper repo (Route A).

Copied verbatim from l3p/planning/mcts_planner.py (helpers + _Node + LandmarkMCTS,
which are pure numpy/torch and carry the 3 upgrades: suffix backup, progressive
widening, bayes/thompson). Kept dependency-free so it imports cleanly under the
paper's old torch 1.5.1 env. Plus `MatrixNoisyValueFn`: a value_fn over a cached
clean edge matrix with optional Loai-2 additive noise (E1a) or Loai-1 fixed bias
(E1c), matching the LandmarkMCTS value_fn interface (prepare_nodes/sample_indexed).
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


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




# --------------------------------------------------------------------------- noise
class MatrixNoisyValueFn:
    """value_fn over a CACHED clean edge matrix M[(N+1)x(N+1)] (row=from, col=to).
    Optional Loai-2 additive noise (resampled per use) or Loai-1 fixed
    multiplicative bias. Implements the LandmarkMCTS value_fn interface."""

    def __init__(self, matrix, sigma=0.0, bias=None, rng=None):
        self.M = np.asarray(matrix, dtype=np.float64)
        self.sigma = float(sigma)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float64)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._nodes = None

    def prepare_nodes(self, nodes):
        self._nodes = nodes                      # only used by the __call__ fallback

    def _apply(self, vals, i, js):
        vals = np.asarray(vals, dtype=np.float64).copy()
        if self.bias is not None:
            vals = vals * (1.0 + self.bias[i, js])
        if self.sigma > 0:
            vals = vals + self.rng.normal(0.0, self.sigma, size=vals.shape)
        return vals

    def sample_indexed(self, i, js):
        js = list(js)
        return torch.as_tensor(self._apply(self.M[i, js], i, js), dtype=torch.float32)

    def __call__(self, g1, g2):
        # fallback (admissibility is precomputed in the port, so rarely hit):
        # map each row to its nearest cached node, then look up the matrix.
        i = torch.cdist(g1, self._nodes).argmin(dim=1).cpu().numpy()
        j = torch.cdist(g2, self._nodes).argmin(dim=1).cpu().numpy()
        return torch.as_tensor(self._apply(self.M[i, j], i, j), dtype=torch.float32)
