"""Auxiliary loss functions for the auto-encoder (paper Eq. 2).

The critic (Eq. 1), value (Eq. 4) and actor losses live on the DDPG agent; the
landmark ELBO (Eq. 5) lives on the LatentLandmarks module. This module holds the
reachability-constrained auto-encoder losses that tie the two together.
"""

from __future__ import annotations

from typing import Tuple

import torch

from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.networks import ValueFunction


def ae_losses(ae: ReachabilityAutoEncoder, value_fn: ValueFunction,
              goals: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruction + reachability-constraint losses.

        L_rec        = || f_D(f_E(g)) - g ||^2                              (recon)
        L_latent     = ( ||z1 - z2||^2 - 0.5*(V(g1,g2)+V(g2,g1)) )^2        (Eq. 2)
        total        = L_rec + lambda * L_latent

    Pairs (g1, g2) are formed by a random permutation of the batch. The
    reachability target uses V with gradients stopped (V is trained separately).
    Returns (l_rec, l_latent, total).
    """
    z = ae.encode(goals)
    recon = ae.decode(z)
    l_rec = ((recon - goals) ** 2).sum(dim=-1).mean()

    perm = torch.randperm(goals.shape[0], device=goals.device)
    g1, g2 = goals, goals[perm]
    z1, z2 = z, z[perm]
    latent_dist = ((z1 - z2) ** 2).sum(dim=-1)
    with torch.no_grad():
        reach = 0.5 * (value_fn(g1, g2) + value_fn(g2, g1))
    l_latent = ((latent_dist - reach) ** 2).mean()

    return l_rec, l_latent, l_rec + lam * l_latent


def ae_contrastive_loss(ae: ReachabilityAutoEncoder, anchor: torch.Tensor,
                        positive: torch.Tensor, negative: torch.Tensor,
                        margin: float) -> torch.Tensor:
    """Triplet-style auxiliary loss over AE latent goals.

    The positive is a temporally nearby future achieved goal from the same
    episode. Negatives are unrelated achieved goals. The loss is zero when each
    negative is at least `margin` farther from the anchor than the positive.
    """
    if margin < 0:
        raise ValueError("ae_contrastive_margin must be >= 0")
    if anchor.shape != positive.shape:
        raise ValueError("anchor and positive must have the same shape")
    if negative.ndim not in {2, 3}:
        raise ValueError("negative must have shape [batch, goal_dim] or [batch, K, goal_dim]")
    if negative.shape[0] != anchor.shape[0] or negative.shape[-1] != anchor.shape[-1]:
        raise ValueError("anchor, positive, and negative have incompatible shapes")

    z_a = ae.encode(anchor)
    z_p = ae.encode(positive)
    d_pos = ((z_a - z_p) ** 2).sum(dim=-1)

    if negative.ndim == 2:
        z_n = ae.encode(negative)
        d_neg = ((z_a - z_n) ** 2).sum(dim=-1)
        return torch.relu(d_pos - d_neg + margin).mean()

    B, K, G = negative.shape
    z_n = ae.encode(negative.reshape(B * K, G)).view(B, K, -1)
    d_neg = ((z_a[:, None, :] - z_n) ** 2).sum(dim=-1)
    return torch.relu(d_pos[:, None] - d_neg + margin).mean()
