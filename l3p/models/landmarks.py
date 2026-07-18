"""Latent landmarks via clustering (paper Section 4.2 + Appendix A).

We fit a mixture of Gaussians in the (reachability-constrained) latent space
with ``n_landmarks`` trainable centroids and a shared trainable variance vector,
under a uniform prior over components. Maximizing the ELBO (Eq. 5) with the
optimal variational posterior is equivalent to maximizing the marginal
log-likelihood, which we implement with a numerically stable log-sum-exp.

The decoded centroids  f_D(c_i)  are the *latent landmarks* used as graph nodes.

Greedy Latent Sparsification (GLS, Algorithm 2) sub-samples a diverse batch of
encoded goals by farthest-point selection. It is used both to build training
batches for the clustering objective and to propose extra random landmarks
during training.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class LatentLandmarks(nn.Module):
    def __init__(self, n_landmarks: int, embedding_size: int, init_sigma: float = 1.0):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.embedding_size = embedding_size
        # Centroids in latent space; initialized small, re-initialized by GLS after warm-up.
        self.centroids = nn.Parameter(0.1 * torch.randn(n_landmarks, embedding_size))
        # Shared trainable (diagonal) variance vector, parameterized via log-sigma.
        self.log_sigma = nn.Parameter(math.log(init_sigma) * torch.ones(embedding_size))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def _log_gaussian(self, z: torch.Tensor) -> torch.Tensor:
        """Per-component log N(z; c_i, diag(sigma^2)).  Returns [B, N]."""
        var = self.sigma ** 2 + 1e-8                       # [E]
        diff = z[:, None, :] - self.centroids[None, :, :]  # [B, N, E]
        log_norm = -0.5 * (torch.log(2 * math.pi * var)).sum()          # scalar (shared over comps)
        quad = -0.5 * (diff ** 2 / var).sum(dim=-1)                     # [B, N]
        return quad + log_norm

    def responsibilities(self, z: torch.Tensor) -> torch.Tensor:
        """Variational posterior q(c | z) with uniform prior. Returns [B, N]."""
        log_prior = -math.log(self.n_landmarks)
        logits = self._log_gaussian(z) + log_prior
        return torch.softmax(logits, dim=-1)

    def elbo_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Negative ELBO (= negative marginal log-likelihood under uniform prior).

        Minimizing this maximizes Eq. 5 and scatters the centroids to cover the
        latent distribution of achieved goals.
        """
        log_prior = -math.log(self.n_landmarks)
        log_joint = self._log_gaussian(z) + log_prior       # [B, N]
        marginal = torch.logsumexp(log_joint, dim=-1)        # [B]
        return -marginal.mean()

    def assign(self, z: torch.Tensor) -> torch.Tensor:
        """Hard cluster assignment (argmax responsibility)."""
        return self.responsibilities(z).argmax(dim=-1)


@torch.no_grad()
def greedy_latent_sparsification(embeddings: torch.Tensor, m: int,
                                 rng: Optional[np.random.Generator] = None) -> torch.Tensor:
    """Greedy Latent Sparsification (Algorithm 2).

    Given a batch of encoded goals ``embeddings`` [K, E], greedily select ``m``
    indices that are maximally spread out (farthest-point sampling) in latent
    space. Returns a LongTensor of the selected indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    k = embeddings.shape[0]
    m = min(m, k)
    device = embeddings.device

    # Start from a random point (Algorithm 2, line 2: sample k ~ {0..K-1}).
    start = int(rng.integers(0, k))
    selected = [start]
    # dist[j] = min distance from point j to the currently selected set.
    dist = torch.sum((embeddings - embeddings[start]) ** 2, dim=-1)  # [K]

    for _ in range(1, m):
        nxt = int(torch.argmax(dist).item())   # farthest remaining point
        selected.append(nxt)
        new_dist = torch.sum((embeddings - embeddings[nxt]) ** 2, dim=-1)
        dist = torch.minimum(dist, new_dist)    # ElementwiseMin (Algorithm 2, line 8)

    return torch.tensor(selected, dtype=torch.long, device=device)
