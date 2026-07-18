"""Online planning with latent landmarks (paper Algorithm 1, Eq. 7).

At the start of an episode we run graph search once to obtain d_{c->g}, the
(negated) best graph distance from every landmark to the goal. Then, at each
planning step, we compute d_{s->c} (Eq. 7): the negated distance from the
current state to each landmark under the current policy. The next sub-goal is

    subgoal = argmax( d_{s->c} + d_{c->g} )

Crucially the planner does *not* re-plan every step. Having chosen a sub-goal it
commits to it for K = -d_{s->c}[subgoal] steps (how many steps it thinks it
needs to reach it) via the counter `Cnt`. This temporal abstraction is central
to L3P's robustness. When it does re-plan, the immediate previous landmark is
removed from the candidate set (masked to -inf) so the agent tries something new
if it failed to reach its last sub-goal, avoiding getting stuck.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from l3p.planning.graph_search import GraphSearch


class LatentPlanner:
    def __init__(self, agent, landmarks, autoencoder, graph_search: GraphSearch, cfg):
        self.agent = agent
        self.landmarks = landmarks
        self.ae = autoencoder
        self.gs = graph_search
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.reset_state()

    def reset_state(self) -> None:
        self.cnt = 0.0
        self.subg_idx: Optional[int] = None
        self.prev_landmark: Optional[int] = None
        self.centroids = None
        self.landmark_goals = None
        self.d_c2g = None
        self.goal = None

    @torch.no_grad()
    def reset(self, goal: np.ndarray, extra_centroids: Optional[torch.Tensor] = None) -> None:
        """Begin a new episode toward `goal`. Runs graph search once.

        `extra_centroids` (optional) adds temporary landmarks for this episode
        (e.g. the random GLS landmarks used during training exploration).
        """
        self.reset_state()
        self.goal = np.asarray(goal, dtype=np.float32)

        centroids = self.landmarks.centroids.detach()
        if extra_centroids is not None and extra_centroids.numel() > 0:
            centroids = torch.cat([centroids, extra_centroids.to(self.device)], dim=0)
        self.n_landmarks = centroids.shape[0]

        if self.n_landmarks == 0:
            return  # no graph yet -> fall back to direct policy

        self.centroids = centroids
        self.landmark_goals = self.ae.decode(centroids).cpu().numpy()  # [N, gdim]
        goal_t = torch.as_tensor(self.goal, dtype=torch.float32, device=self.device)
        self.d_c2g = self.gs.distances_to_goal(centroids, goal_t, self.ae,
                                               self.agent.value).cpu().numpy()  # [N+1]

    @torch.no_grad()
    def _replan(self, obs: np.ndarray) -> None:
        """Recompute d_{s->c}, pick a new sub-goal, and set the commitment count."""
        # Candidate goals: decoded landmarks followed by the true goal.
        candidates = np.concatenate([self.landmark_goals, self.goal[None, :]], axis=0)  # [N+1, gdim]
        d = self.agent.distance_after_action(obs, candidates)      # D(s, pi(s,c), c), [N+1]
        d_s2c = -d                                                 # Eq. 7 (negated)

        combined = d_s2c + self.d_c2g                              # Alg 1, line 8
        # Remove the immediate previous landmark (only landmark nodes, not the goal).
        if self.prev_landmark is not None and self.prev_landmark < self.n_landmarks:
            combined[self.prev_landmark] = -np.inf

        self.subg_idx = int(np.argmax(combined))
        # K = -d_{s->c}[subgoal]: how many steps we think we need to reach it.
        self.cnt = max(1.0, float(round(-d_s2c[self.subg_idx])))
        self.prev_landmark = self.subg_idx

    def _current_subgoal(self) -> np.ndarray:
        if self.subg_idx is None or self.subg_idx >= self.n_landmarks:
            return self.goal
        return self.landmark_goals[self.subg_idx]

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0, random_prob: float = 0.0) -> np.ndarray:
        """Return the low-level action for the current state under the plan."""
        if self.n_landmarks == 0:            # no graph yet -> direct goal reaching
            return self.agent.act(obs, self.goal, noise_scale, random_prob)

        if self.cnt > 1.0:                   # commit: do not re-plan (Alg 1, lines 4-5)
            self.cnt -= 1.0
        else:                                # re-plan (Alg 1, lines 6-13)
            self._replan(obs)

        subgoal = self._current_subgoal()
        return self.agent.act(obs, subgoal, noise_scale, random_prob)
