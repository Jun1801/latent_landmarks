"""Unit tests for the L3P components (paper equations / algorithms).

Runnable with `pytest tests/` or directly: `python tests/test_modules.py`.
All tests use synthetic data and only require torch + numpy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.agent.ddpg import DDPGAgent, value_contrastive_loss
from l3p.config import get_config
from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.landmarks import LatentLandmarks, greedy_latent_sparsification
from l3p.models.networks import Critic, ValueFunction
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.replay.her_buffer import HERReplayBuffer
from l3p.losses import ae_losses


def test_q_from_distance():
    """Eq. 3: Q = -(1 - gamma^D)/(1 - gamma); D >= 0; Q <= 0."""
    gamma = 0.98
    D = torch.rand(1000) * 50
    q = Critic.q_from_distance(D, gamma)
    expected = -(1 - gamma ** D) / (1 - gamma)
    assert torch.allclose(q, expected, atol=1e-5)
    assert (q <= 1e-6).all()
    assert (q >= -1.0 / (1 - gamma) - 1e-3).all()
    # D = 0 (already at goal) -> Q = 0
    assert abs(Critic.q_from_distance(torch.zeros(1), gamma).item()) < 1e-6
    print("ok  test_q_from_distance")


def test_value_regression():
    """V should fit a synthetic distance target; loss must decrease."""
    torch.manual_seed(0)
    V = ValueFunction(goal_dim=2, hidden_units=64, hidden_layers=2)
    opt = torch.optim.Adam(V.parameters(), lr=1e-2)
    g1 = torch.rand(512, 2)
    g2 = torch.rand(512, 2)
    target = (g1 - g2).norm(dim=-1) * 10.0     # arbitrary "steps" target >= 0
    first = None
    for i in range(300):
        pred = V(g1, g2)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < 0.3 * first, (first, loss.item())
    print("ok  test_value_regression")


def test_replay_samples_negative_goals():
    """Replay can attach K contrastive negative achieved goals per transition."""
    T, obs_dim, goal_dim, act_dim = 5, 3, 2, 1

    def reward(ag, g):
        return -(np.linalg.norm(ag - g, axis=-1) > 0.1).astype(np.float32)

    buf = HERReplayBuffer(
        size_episodes=4, horizon=T, obs_dim=obs_dim, goal_dim=goal_dim,
        act_dim=act_dim, compute_reward=reward, n_value_negatives=3,
        negative_sampling_strategy="cross_episode",
    )
    for ep_id in range(4):
        obs = np.zeros((T + 1, obs_dim), dtype=np.float32)
        ag = np.zeros((T + 1, goal_dim), dtype=np.float32)
        ag[:, 0] = ep_id
        ag[:, 1] = np.arange(T + 1)
        g = np.zeros((T, goal_dim), dtype=np.float32)
        act = np.zeros((T, act_dim), dtype=np.float32)
        buf.store_episode(dict(obs=obs, ag=ag, g=g, act=act))

    sample = buf.sample(batch_size=8, rng=np.random.default_rng(0))
    assert sample["neg_ag"].shape == (8, 3, goal_dim)
    assert sample["neg_ag"].dtype == np.float32
    # The cross-episode strategy should not sample negatives from the anchor's episode.
    assert (sample["neg_ag"][:, :, 0] != sample["ag"][:, None, 0]).all()
    print("ok  test_replay_samples_negative_goals")


def test_value_contrastive_loss_orders_goals():
    """InfoNCE teaches V(anchor, positive) < V(anchor, negative) on a line."""
    torch.manual_seed(0)
    V = ValueFunction(goal_dim=1, hidden_units=64, hidden_layers=2)
    opt = torch.optim.Adam(V.parameters(), lr=3e-3)

    B, K = 128, 4
    anchor = torch.linspace(-1.0, 1.0, B).unsqueeze(1)
    positive = anchor + 0.02 * torch.randn(B, 1)
    offsets = torch.linspace(2.0, 5.0, K).view(1, K, 1)
    negatives = anchor[:, None, :] + offsets

    first = None
    for i in range(400):
        loss = value_contrastive_loss(V, anchor, positive, negatives, temperature=0.5)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()

    with torch.no_grad():
        pos = V(anchor, positive).mean().item()
        neg = V(anchor[:, None, :].expand(B, K, 1).reshape(B * K, 1),
                negatives.reshape(B * K, 1)).mean().item()
    assert loss.item() < 0.3 * first, (first, loss.item())
    assert pos < neg, (pos, neg)
    print(f"ok  test_value_contrastive_loss_orders_goals (pos={pos:.2f}, neg={neg:.2f})")


def test_update_value_without_negatives_when_disabled():
    """The original value-regression path must not require neg_ag."""
    torch.manual_seed(0)
    cfg = get_config("PointMaze", hidden_units=16, hidden_layers=1,
                     use_value_contrastive=False)
    agent = DDPGAgent(obs_dim=3, goal_dim=2, act_dim=1, max_action=1.0, cfg=cfg)
    batch = dict(
        obs=torch.randn(8, 3),
        act=torch.randn(8, 1),
        next_ag=torch.randn(8, 2),
        future_ag=torch.randn(8, 2),
        future_ag_norm=torch.randn(8, 2),
    )
    loss = agent.update_value(batch)
    assert isinstance(loss, float)
    assert np.isfinite(loss)
    print("ok  test_update_value_without_negatives_when_disabled")


class _ScaledV:
    """Deterministic reachability metric: scale * Euclidean distance."""
    def __init__(self, scale=5.0):
        self.scale = scale

    def __call__(self, g1, g2):
        return self.scale * torch.norm(g1 - g2, dim=-1)


def test_autoencoder_reachability():
    """L_rec decreases and latent L2 distance correlates with the reachability
    target (Eq. 2)."""
    torch.manual_seed(0)
    goal_dim = 2
    ae = ReachabilityAutoEncoder(goal_dim, embedding_size=8, hidden_units=64, hidden_layers=2)
    # A deterministic reachability metric (scaled Euclidean) with real variance.
    Vnet = _ScaledV(scale=5.0)
    opt = torch.optim.Adam(ae.parameters(), lr=3e-3)
    goals = torch.rand(1024, goal_dim)
    first_rec = None
    for i in range(600):
        l_rec, l_latent, total = ae_losses(ae, Vnet, goals, lam=1.0)
        opt.zero_grad(); total.backward(); opt.step()
        if i == 0:
            first_rec = l_rec.item()
    assert l_rec.item() < 0.5 * first_rec
    # correlation between latent dist^2 and symmetric reachability
    with torch.no_grad():
        z = ae.encode(goals)
        perm = torch.randperm(goals.shape[0])
        ld = ((z - z[perm]) ** 2).sum(-1).numpy()
        reach = (0.5 * (Vnet(goals, goals[perm]) + Vnet(goals[perm], goals))).numpy()
    corr = np.corrcoef(ld, reach)[0, 1]
    assert corr > 0.5, corr
    print(f"ok  test_autoencoder_reachability (corr={corr:.2f})")


def test_gls_and_elbo():
    """GLS returns m distinct spread-out indices; centroid ELBO improves."""
    torch.manual_seed(0)
    # three well-separated clusters in 2-D latent space
    centers = torch.tensor([[-5., -5.], [5., 5.], [-5., 5.]])
    z = torch.cat([c + 0.3 * torch.randn(200, 2) for c in centers], dim=0)

    idx = greedy_latent_sparsification(z, m=10, rng=np.random.default_rng(0))
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10                 # distinct
    # farthest-point picks should span all three clusters
    picked = z[idx]
    assert picked[:, 0].max() - picked[:, 0].min() > 6

    lm = LatentLandmarks(n_landmarks=3, embedding_size=2, init_sigma=1.0)
    opt = torch.optim.Adam(lm.parameters(), lr=0.1)
    first = lm.elbo_loss(z).item()
    for _ in range(300):
        loss = lm.elbo_loss(z)
        opt.zero_grad(); loss.backward(); opt.step()
    assert lm.elbo_loss(z).item() < first
    # each cluster should be captured by some centroid
    with torch.no_grad():
        d = torch.cdist(centers, lm.centroids)
    assert (d.min(dim=1).values < 1.5).all(), d
    print("ok  test_gls_and_elbo")


class _FakeV:
    """Value function = Euclidean distance between goal vectors."""
    def __call__(self, g1, g2):
        return torch.norm(g1 - g2, dim=-1)


class _IdentityAE:
    def decode(self, z):
        return z


def test_soft_floyd_and_dmax():
    """Soft Floyd (near-hard, small beta) recovers the multi-hop shortest path;
    d_max masks out edges that are too long."""
    cfg = get_config("PointMaze", beta=0.05, soft_iters=40, d_max=1.5, neg_inf=-1e6)
    gs = GraphSearch(cfg)
    # landmarks on a line at x=0,1,2 ; goal at x=3.  Only unit-length hops survive
    # the d_max=1.5 cutoff, so the only route 0->goal is 0->1->2->goal (cost 3).
    landmarks = torch.tensor([[0.], [1.], [2.]])
    goal = torch.tensor([3.])
    W = gs.build_weight_matrix(landmarks, goal, _FakeV())
    # direct 0->goal (dist 3 > 1.5) must be masked
    assert W[0, 3].item() < -1e5
    # unit edge 0->1 survives
    assert abs(W[0, 1].item() + 1.0) < 1e-4
    Wr = gs.soft_floyd(W)
    # best path value from node 0 to goal ~ -(1+1+1) = -3, and finite (recovered)
    assert Wr[0, 3].item() > -100.0
    assert abs(Wr[0, 3].item() + 3.0) < 0.6, Wr[0, 3].item()
    print(f"ok  test_soft_floyd_and_dmax (d_c->g[0]={Wr[0,3].item():.2f})")


class _FakeAgent:
    """Minimal agent exposing what the planner's _replan / act need."""
    def __init__(self, distances):
        # distances[i] = D(s, pi(s, candidate_i), candidate_i)
        self._d = np.asarray(distances, dtype=np.float32)
        self.calls = 0

    def distance_after_action(self, obs, goals):
        self.calls += 1
        return self._d.copy()

    def act(self, obs, goal, noise=0.0, rp=0.0):
        return np.zeros(2, dtype=np.float32)


def test_planner_commitment_and_masking():
    """Planner commits for K steps (no re-plan) and never immediately reselects
    the previous landmark (Algorithm 1)."""
    cfg = get_config("PointMaze")
    # 3 landmark candidates + goal. Distances-to-candidate (D): landmark1 nearest.
    # d_c2g will make landmark index 1 the argmax on the first plan.
    agent = _FakeAgent(distances=[2.0, 3.0, 8.0, 20.0])   # length N+1 = 4 (N=3)
    lm = LatentLandmarks(n_landmarks=3, embedding_size=2)
    planner = LatentPlanner(agent, lm, _IdentityAE(), GraphSearch(cfg), cfg)

    # inject a fixed graph result and decoded landmarks
    planner.n_landmarks = 3
    planner.landmark_goals = np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32)
    planner.goal = np.array([3., 3.], dtype=np.float32)
    planner.d_c2g = np.array([-1.0, -0.5, -6.0, 0.0], dtype=np.float32)
    planner.cnt = 0.0
    planner.subg_idx = None
    planner.prev_landmark = None

    obs = np.zeros(2, dtype=np.float32)
    # first plan: combined = -D + d_c2g
    planner.act(obs)
    first_calls = agent.calls
    first_idx = planner.subg_idx
    K = planner.cnt
    assert K >= 1
    # commit: for the next round(K)-1 steps it must NOT re-plan (calls unchanged)
    for _ in range(int(K) - 1):
        planner.act(obs)
        assert planner.subg_idx == first_idx
    assert agent.calls == first_calls           # never re-planned while committed

    # force a re-plan and check the previous landmark is masked out
    planner.cnt = 0.0
    planner.act(obs)
    assert planner.subg_idx != first_idx, "must not reselect the previous landmark"
    print("ok  test_planner_commitment_and_masking")


ALL_TESTS = [
    test_q_from_distance,
    test_value_regression,
    test_replay_samples_negative_goals,
    test_value_contrastive_loss_orders_goals,
    test_update_value_without_negatives_when_disabled,
    test_autoencoder_reachability,
    test_gls_and_elbo,
    test_soft_floyd_and_dmax,
    test_planner_commitment_and_masking,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print("\nAll tests passed.")
