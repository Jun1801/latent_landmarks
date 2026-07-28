"""MCTS-over-landmarks (docs/SPEC_MCTS_Landmark_L3P.md, Section 2) — Phase 1.

Plain UCT selection + sample-based rollout, NO beta*sigma_V uncertainty bonus
(that, plus reward-shaping / case-2-3 recovery / execution feedback, is Phase 2
and lives in a follow-up plan; see the SPEC's Section 4/§3 and the "Explicitly
deferred" note in the approved plan).

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
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from l3p.planning.planner import LatentPlanner
from l3p.planning.noise import (build_sigma_matrix, HeterogeneousNoise,
                                build_bias_matrix, CriticEdgeFn, BiasedValueFn)


class _Node:
    __slots__ = ("idx", "untried", "children", "visits", "child_visits", "child_total")

    def __init__(self, idx: Optional[int], candidates: List[int]):
        self.idx = idx                      # None for the (virtual) root
        self.untried = list(candidates)
        self.children: Dict[int, "_Node"] = {}
        self.visits = 0
        self.child_visits: Dict[int, int] = defaultdict(int)
        self.child_total: Dict[int, float] = defaultdict(float)

    def fully_expanded(self) -> bool:
        return len(self.untried) == 0

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
        # E1b oracle uncertainty (docs/... Sec 3): per-edge sigma_ij + a mode.
        # "alpha" subtracts lambda_risk*sigma from landmark->landmark edge costs
        # (risk-averse: avoid uncertain edges); "beta" adds beta_unc*sigma to the
        # UCT selection score (probe uncertain edges). "none" => plain MCTS
        # (the MCTS-beta=0 ablation). sigma is defined only over landmark/goal
        # edges; root (state->landmark) edges use the clean critic D, so they
        # carry no uncertainty term.
        self.sigma = sigma_matrix
        self.unc_mode = getattr(cfg, "mcts_uncertainty_mode", "none")
        self.lambda_risk = getattr(cfg, "mcts_lambda_risk", 0.0)
        self.beta_unc = getattr(cfg, "mcts_beta_uncertainty", 0.0)
        if self.sigma is None:
            self.unc_mode = "none"
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

    def _edge_cost(self, i: int, j: int) -> float:
        v = self.value_fn(self.nodes_t[i:i + 1], self.nodes_t[j:j + 1]).item()
        cost = -v
        if self.unc_mode == "alpha":            # risk penalty on the reward (E1b)
            cost -= self.lambda_risk * float(self.sigma[i, j])
        return cost

    def _batch_edge_costs(self, i: int, js: List[int]) -> np.ndarray:
        """Costs for many candidate edges out of `i` in one value_fn call."""
        gi = self.nodes_t[i:i + 1].expand(len(js), -1)
        gj = self.nodes_t[js]
        costs = -self.value_fn(gi, gj).detach().cpu().numpy()
        if self.unc_mode == "alpha":
            costs = costs - self.lambda_risk * self.sigma[i, js]
        return costs

    def _uct_bonus(self, node: "_Node"):
        """beta exploration bonus fn for UCT selection at `node` (E1b), or None.
        Root (state->landmark) edges carry no oracle uncertainty."""
        if self.unc_mode != "beta" or node.idx is None:
            return None
        pi = node.idx
        return lambda j: self.beta_unc * float(self.sigma[pi, j])

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

    def search(self, d_s2c: np.ndarray, mask: Optional[np.ndarray] = None
              ) -> Tuple[int, Dict[int, Tuple[int, float]]]:
        """Run `cfg.mcts_n_simulations` simulations rooted at the real state
        (edges = `d_s2c`) and return (best_root_child, {child: (visits, mean_q)}).
        `mask[j]=True` excludes candidate j from the ROOT only (mirrors
        Algorithm 1's previous-landmark removal)."""
        if mask is None:
            mask = np.zeros(self.n + 1, dtype=bool)
        root = _Node(idx=None, candidates=[j for j in range(self.n + 1) if not mask[j]])

        for _ in range(self.cfg.mcts_n_simulations):
            path: List[Tuple[_Node, int]] = []
            node = root
            depth = 0

            # SELECTION
            while node.fully_expanded() and node.children and depth < self.cfg.mcts_rollout_horizon:
                j = node.best_uct_child(self.cfg.mcts_c_uct, bonus=self._uct_bonus(node))
                if j is None:
                    break
                path.append((node, j))
                node = node.children[j]
                depth += 1
                if j == self.goal_idx:
                    break

            # EXPANSION
            if node.idx != self.goal_idx and node.untried and depth < self.cfg.mcts_rollout_horizon:
                pick = int(self.rng.integers(len(node.untried)))
                j = node.untried.pop(pick)
                candidates = [] if j == self.goal_idx else self._candidates_of(j)
                child = node.children.setdefault(j, _Node(j, candidates))
                path.append((node, j))
                node = child
                depth += 1

            # tree-portion cost: root edges are the fixed d_s2c; landmark/goal
            # edges resample the (possibly noisy) value_fn on every use.
            tree_return = 0.0
            for parent, child_idx in path:
                tree_return += (d_s2c[child_idx] if parent.idx is None
                               else self._edge_cost(parent.idx, child_idx))

            # ROLLOUT (Soft-Floyd-greedy) from the reached node
            if node.idx == self.goal_idx:
                rollout_return = 0.0
            else:
                rollout_return = self._rollout(node.idx, self.cfg.mcts_rollout_horizon - depth)

            total_return = tree_return + rollout_return

            # BACKPROPAGATION
            for parent, child_idx in path:
                parent.visits += 1
                parent.child_visits[child_idx] += 1
                parent.child_total[child_idx] += total_return

        stats: Dict[int, Tuple[int, float]] = {}
        best_idx, best_key = None, (-1, -float("inf"))
        for j, visits in root.child_visits.items():
            q = root.child_total[j] / visits if visits > 0 else -float("inf")
            stats[j] = (visits, q)
            key = (visits, q)              # robust child: max visits, tie-break by Q
            if key > best_key:
                best_key, best_idx = key, j
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

    @torch.no_grad()
    def reset(self, goal: np.ndarray, extra_centroids: Optional[torch.Tensor] = None) -> None:
        """Same as LatentPlanner.reset(), plus caching the d_max admissibility
        mask ONCE for the whole episode -- matching self.d_c2g's cadence, so
        MCTS's own view of "which edges exist" never drifts from the heuristic
        it scores rollouts against (see LandmarkMCTS.build_admissibility)."""
        super().reset(goal, extra_centroids)
        if self.n_landmarks == 0:
            self._admissible = None
            return
        nodes_t = torch.as_tensor(
            np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0),
            dtype=torch.float32, device=self.device)
        self._admissible = LandmarkMCTS.build_admissibility(nodes_t, self.agent.value, self.cfg)

    @torch.no_grad()
    def _replan(self, obs: np.ndarray) -> None:
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)  # [N+1, gdim]
        d = self.agent.distance_after_action(obs, candidates)      # D(s, pi(s,c), c), real/un-noised
        d_s2c = -d

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True

        nodes_t = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
        mcts = LandmarkMCTS(n_landmarks=self.n_landmarks, value_fn=self.agent.value,
                            nodes_t=nodes_t, d_c2g_heuristic=self.d_c2g, cfg=self.cfg,
                            rng=self.rng, admissible=self._admissible)
        best_idx, _ = mcts.search(d_s2c, mask=mask)

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
    biased = BiasedValueFn(CriticEdgeFn(planner.agent), nodes_t, bias)
    return nodes_t, biased, bias


class SoftFloydE1c(LatentPlanner):
    """E1c baseline: L3P Soft Floyd on the Loai-1-biased critic-D graph, STATIC
    (plans once per episode, never corrects) -- the "trusts the wormhole and never
    fixes it" behaviour E1c contrasts against."""

    def __init__(self, agent, landmarks, autoencoder, graph_search, cfg, sigma, noise_seed):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg)
        self.sigma = sigma
        self.noise_seed = noise_seed

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
    """E1c MCTS on the Loai-1-biased critic-D graph, with optional EXECUTION
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
                r_max=2, rng=None):
        super().__init__(agent, landmarks, autoencoder, graph_search, cfg, rng=rng)
        self.sigma = sigma
        self.noise_seed = noise_seed
        self.feedback = feedback
        self.rho = rho
        self.tau_reach = tau_reach
        self.tau_progress = tau_progress
        self.r_max = r_max

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
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae, biased).cpu().numpy()
        self._admissible = LandmarkMCTS.build_admissibility(nodes_t, biased, self.cfg)
        # execution-feedback episode state
        self._v_exec = {}                 # (i, j) -> corrected env-step cost
        self._attempts = {}               # subgoal j -> failed attempts
        self._blacklist = set()
        self._macro_k = 0
        self._macro_start_i = None
        self._macro_start_z = None
        self._cur_j = None

    # ---- feedback helpers ----
    def _snap(self, z):
        return int(np.argmin(np.linalg.norm(self.landmark_goals - z[None, :], axis=1)))

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

    def _observe(self, z_end):
        """Finalize the just-executed macro-step (i -> j): update V_exec (EMA),
        classify the outcome, and apply the loop guard (Sec 4.1/4.3/4.4)."""
        i, j = self._macro_start_i, self._cur_j
        if j is None or j >= self.n_landmarks:      # subgoal was the goal node
            return
        c_j = self.landmark_goals[j]
        dist_to_goal = self._critic_dist(z_end, c_j)
        dist_travelled = self._critic_dist(self._macro_start_z, z_end)
        realized = self._macro_k + max(0.0, dist_to_goal)     # env-step scale

        prev = self._v_exec.get((i, j), self._biased_edge(i, j))
        self._v_exec[(i, j)] = (1 - self.rho) * prev + self.rho * realized

        if dist_to_goal <= self.tau_reach:                     # REACHED
            self._attempts.clear()
        elif dist_travelled >= self.tau_progress:              # PROGRESSED_ELSEWHERE
            self._attempts[j] = self._attempts.get(j, 0) + 1
            if self._attempts[j] >= self.r_max:
                self._blacklist.add(j)
        else:                                                  # STUCK -> blacklist now
            self._blacklist.add(j)

    @torch.no_grad()
    def act(self, obs, noise_scale=0.0, random_prob=0.0):
        if self.n_landmarks == 0:
            return self.agent.act(obs, self.goal, noise_scale, random_prob)
        if self.cnt > 1.0:
            self.cnt -= 1.0
        else:
            if self.feedback and self._cur_j is not None:
                self._observe(obs)                             # finalize previous macro-step
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
            d_c2g = self.gs.distances_to_goal(self.centroids, goal_t, self.ae, value_fn).cpu().numpy()
            admissible = LandmarkMCTS.build_admissibility(nodes_t, value_fn, self.cfg)
        else:
            value_fn, d_c2g, admissible = self._biased, self.d_c2g, self._admissible

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True
        for b in self._blacklist if self.feedback else ():
            if b < self.n_landmarks:
                mask[b] = True

        mcts = LandmarkMCTS(n_landmarks=self.n_landmarks, value_fn=value_fn,
                            nodes_t=nodes_t, d_c2g_heuristic=d_c2g, cfg=self.cfg,
                            rng=self.rng, admissible=admissible)
        best_idx, _ = mcts.search(d_s2c, mask=mask)
        if best_idx is None:                        # everything masked -> head to goal
            best_idx = self.n_landmarks

        self.subg_idx = best_idx
        self.cnt = max(1.0, float(round(-d_s2c[best_idx])))
        self.prev_landmark = self.subg_idx
        if self.feedback:                           # open a new macro-step
            self._macro_k = 0
            self._macro_start_i = self._snap(obs)
            self._macro_start_z = np.asarray(obs, dtype=np.float32)
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
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae, self._hetero).cpu().numpy()
        self._admissible = LandmarkMCTS.build_admissibility(nodes_t, self._hetero, self.cfg)

    @torch.no_grad()
    def _replan(self, obs: np.ndarray) -> None:
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)
        d = self.agent.distance_after_action(obs, candidates)      # clean critic D
        d_s2c = -d

        mask = np.zeros(self.n_landmarks + 1, dtype=bool)
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            mask[self.prev_landmark] = True

        nodes_t = torch.as_tensor(candidates, dtype=torch.float32, device=self.device)
        mcts = LandmarkMCTS(n_landmarks=self.n_landmarks, value_fn=self._hetero,
                            nodes_t=nodes_t, d_c2g_heuristic=self.d_c2g, cfg=self.cfg,
                            rng=self.rng, admissible=self._admissible, sigma_matrix=self._sigma)
        best_idx, _ = mcts.search(d_s2c, mask=mask)

        self.subg_idx = best_idx
        self.cnt = max(1.0, float(round(-d_s2c[best_idx])))
        self.prev_landmark = self.subg_idx
