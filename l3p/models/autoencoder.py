"""Reachability-constrained auto-encoder (paper Section 4.1).

The encoder f_E maps goals into a latent space; the decoder f_D maps back.
Beyond the usual reconstruction loss, the latent space is constrained so that
squared L2 distance between two latent codes matches the (symmetrized)
reachability distance between the corresponding goals:

    L_rec(g)        = || f_D(f_E(g)) - g ||^2                                 (recon)
    L_latent(g1,g2) = ( ||f_E(g1) - f_E(g2)||^2 - 0.5*(V(g1,g2)+V(g2,g1)) )^2  (Eq. 2)

The joint objective L_rec + lambda * L_latent gives the latent space structure:
goals that are close in reachability cluster together, which is what makes the
downstream latent clustering (landmarks) meaningful.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from l3p.models.networks import mlp


class ReachabilityAutoEncoder(nn.Module):
    def __init__(self, goal_dim: int, embedding_size: int = 16,
                 hidden_units: int = 128, hidden_layers: int = 2):
        super().__init__()
        enc_sizes = [goal_dim] + [hidden_units] * hidden_layers + [embedding_size]
        dec_sizes = [embedding_size] + [hidden_units] * hidden_layers + [goal_dim]
        self.encoder = mlp(enc_sizes, activation=nn.ReLU, output_activation=nn.Identity)
        self.decoder = mlp(dec_sizes, activation=nn.ReLU, output_activation=nn.Identity)
        self.goal_dim = goal_dim
        self.embedding_size = embedding_size
        # Optional goal normalizer: the encoder sees normalized goals so the
        # latent space (and its reachability clustering) is well-scaled even for
        # large-coordinate goal spaces (e.g. AntMaze). Set externally.
        self.normalizer = None

    def encode(self, g: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            g = self.normalizer.normalize_torch(g)
        return self.encoder(g)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(g))
