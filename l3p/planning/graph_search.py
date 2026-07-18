"""Graph search over the latent-landmark world model (paper Section 4.3 + Appendix B).

Nodes are the decoded latent landmarks f_D(c_i) plus the episode goal g. Edge
weights are *negated* reachability distances w_{i,j} = -V(f_D(c_i), f_D(c_j))
(Eq. 6), so shortest-path search becomes a max-plus / longest-path problem and
relaxation uses (soft) max.

We use a *soft* Floyd algorithm (Eq. 8): each relaxation step replaces the hard
`max_k (w_{i,k} + w_{k,j})` with a softmax-weighted average over the
intermediate node k, controlled by temperature beta. This is empirically more
robust to inaccurate neural distance estimates. Setting beta -> 0 recovers the
hard Floyd algorithm.

Edges longer than `d_max` are masked with a large negative penalty (Appendix B):
we only trust the local neural distance estimates and let graph search stitch
together the long-horizon, global distances.
"""

from __future__ import annotations

import torch

from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.networks import ValueFunction


class GraphSearch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    @torch.no_grad()
    def build_weight_matrix(self, landmark_goals: torch.Tensor, goal: torch.Tensor,
                            value_fn: ValueFunction) -> torch.Tensor:
        """Construct W (Eq. 6). `landmark_goals` is [N, goal_dim] (decoded
        centroids), `goal` is [goal_dim]. Returns W [(N+1), (N+1)] with the goal
        as the last node.
        """
        n = landmark_goals.shape[0]
        nodes = torch.cat([landmark_goals, goal.view(1, -1)], dim=0)  # [N+1, gdim]
        m = n + 1

        # Pairwise reachability distances V(node_i, node_j).
        gi = nodes[:, None, :].expand(m, m, nodes.shape[1]).reshape(m * m, -1)
        gj = nodes[None, :, :].expand(m, m, nodes.shape[1]).reshape(m * m, -1)
        dist = value_fn(gi, gj).view(m, m)          # V(i -> j), >= 0

        W = -dist                                    # negate: Eq. 6
        # Zero diagonal (staying put costs nothing).
        idx = torch.arange(m, device=W.device)
        W[idx, idx] = 0.0
        # d_max masking (Appendix B): drop edges whose distance exceeds d_max.
        far = dist > self.cfg.d_max
        W = W + far.float() * self.cfg.neg_inf
        W[idx, idx] = 0.0                            # keep diagonal clean after masking
        # Goal is an absorbing sink: no outgoing edges (last row = -inf), except self.
        W[n, :] = self.cfg.neg_inf
        W[n, n] = 0.0
        return W

    @torch.no_grad()
    def soft_floyd(self, W: torch.Tensor) -> torch.Tensor:
        """Soft Floyd relaxation (Eq. 8), repeated for `S` iterations."""
        beta = self.cfg.beta
        m = W.shape[0]
        for _ in range(self.cfg.soft_iters):
            # via[i, j, k] = W[i, k] + W[k, j]
            via = W[:, None, :] + W.transpose(0, 1)[None, :, :]   # [i, j, k]
            # softmax over k (numerically stabilized), then weighted sum.
            logits = via / beta
            logits = logits - logits.max(dim=-1, keepdim=True).values
            weights = torch.softmax(logits, dim=-1)
            W = (weights * via).sum(dim=-1)
            # keep the sink/diagonal structure stable across iterations
            idx = torch.arange(m, device=W.device)
            W[idx, idx] = 0.0
            W[m - 1, :] = self.cfg.neg_inf
            W[m - 1, m - 1] = 0.0
        return W

    @torch.no_grad()
    def distances_to_goal(self, centroids: torch.Tensor, goal: torch.Tensor,
                          ae: ReachabilityAutoEncoder, value_fn: ValueFunction) -> torch.Tensor:
        """Full pipeline: decode centroids -> build W -> soft Floyd -> return
        d_{c->g}, the (negated) best distance from every node to the goal.

        Returns a 1-D tensor of length N+1 (N landmarks followed by the goal
        node, whose self-distance is 0).
        """
        landmark_goals = ae.decode(centroids)
        W = self.build_weight_matrix(landmark_goals, goal, value_fn)
        W = self.soft_floyd(W)
        # last column: value of the best path from each node to the goal node.
        return W[:, -1].clone()
