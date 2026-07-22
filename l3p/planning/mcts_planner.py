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

    def best_uct_child(self, c_uct: float) -> Optional[int]:
        best_j, best_score = None, -float("inf")
        for j, visits in self.child_visits.items():
            if visits == 0:
                continue
            q = self.child_total[j] / visits
            score = q + c_uct * math.sqrt(math.log(max(self.visits, 1)) / visits)
            if score > best_score:
                best_score, best_j = score, j
        return best_j


class LandmarkMCTS:
    def __init__(self, n_landmarks: int, value_fn, nodes_t: torch.Tensor,
                d_c2g_heuristic: np.ndarray, cfg, rng: np.random.Generator,
                admissible: Optional[np.ndarray] = None):
        self.n = n_landmarks
        self.goal_idx = n_landmarks
        self.value_fn = value_fn
        self.nodes_t = nodes_t              # [N+1, gdim]: landmarks then goal
        self.heuristic = d_c2g_heuristic    # [N+1], static soft-Floyd d_{c->g}
        self.cfg = cfg
        self.rng = rng
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
        return -v

    def _batch_edge_costs(self, i: int, js: List[int]) -> np.ndarray:
        """Costs for many candidate edges out of `i` in one value_fn call."""
        gi = self.nodes_t[i:i + 1].expand(len(js), -1)
        gj = self.nodes_t[js]
        return -self.value_fn(gi, gj).detach().cpu().numpy()

    def _rollout(self, start_idx: int, budget: int) -> float:
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
                j = node.best_uct_child(self.cfg.mcts_c_uct)
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
