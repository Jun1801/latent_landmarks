"""Extra baselines required by the E1 experiment matrix
(docs/SPEC_MCTS_Landmark_L3P.md, Section 7 "Baselines bat buoc so sanh").

`NaiveReplanPlanner` is Baseline 2 (L3P Fig. 8): it re-plans via Algorithm 1's
own selection formula (argmax(d_s2c + d_c2g)) but on EVERY step, without the
K-step commitment `LatentPlanner` uses. The spec requires this baseline
specifically to separate "MCTS wins because it re-plans more often than
Algorithm 1 commits for K steps" from "MCTS wins because of the tree-search /
uncertainty-aware machinery itself" -- without it, an E1a result showing MCTS
beating Soft Floyd is not attributable to either cause.
"""

from __future__ import annotations

import numpy as np
import torch

from l3p.planning.planner import LatentPlanner


class NaiveReplanPlanner(LatentPlanner):
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0,
            random_prob: float = 0.0, achieved_goal=None) -> np.ndarray:
        if self.n_landmarks == 0:            # no graph yet -> direct goal reaching
            return self.agent.act(obs, self.goal, noise_scale, random_prob)
        self._replan(obs)                    # every step, no commitment
        subgoal = self._current_subgoal()
        return self.agent.act(obs, subgoal, noise_scale, random_prob)


class FreshGraphReplanPlanner(LatentPlanner):
    """SPEC baseline 3: replan every step and refresh noisy V_obs every step."""

    @torch.no_grad()
    def reset(self, goal: np.ndarray, extra_centroids=None) -> None:
        super().reset(goal, extra_centroids)
        self._first_act = True

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0,
            random_prob: float = 0.0, achieved_goal=None) -> np.ndarray:
        if self.n_landmarks == 0:
            return self.agent.act(obs, self.goal, noise_scale, random_prob)
        # reset() already made the first paired V_obs draw. Every later decision
        # rebuilds the graph, which is the variable this baseline isolates.
        if self._first_act:
            self._first_act = False
        else:
            goal_t = torch.as_tensor(
                self.goal, dtype=torch.float32, device=self.device)
            self.d_c2g = self.gs.distances_to_goal(
                self.centroids, goal_t, self.ae, self.agent.value).cpu().numpy()
        self._replan(obs)
        return self.agent.act(
            obs, self._current_subgoal(), noise_scale, random_prob)
