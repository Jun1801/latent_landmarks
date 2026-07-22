"""Unit tests for the L3P components (paper equations / algorithms).

Runnable with `pytest tests/` or directly: `python tests/test_modules.py`.
All tests use synthetic data and only require torch + numpy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from l3p.config import get_config
from l3p.models.autoencoder import ReachabilityAutoEncoder
from l3p.models.landmarks import LatentLandmarks, greedy_latent_sparsification
from l3p.models.networks import Critic, ValueFunction
from l3p.planning.graph_search import GraphSearch
from l3p.planning.planner import LatentPlanner
from l3p.planning.noise import NoisyValueFn, ValueOverrideAgent, dmax_candidates, bootstrap_ci
from l3p.planning.mcts_planner import LandmarkMCTS, MCTSPlanner
from l3p.planning.baselines import NaiveReplanPlanner
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


def test_noisy_value_zero_sigma_passthrough():
    """sigma=0 must be an exact passthrough (bitwise) -- this is what makes the
    MCTS-vs-Soft-Floyd sigma=0 sanity check exact, not just approximate."""
    rng = np.random.default_rng(0)
    base = _FakeV()
    noisy = NoisyValueFn(base, sigma=0.0, rng=rng)
    g1 = torch.rand(10, 2)
    g2 = torch.rand(10, 2)
    assert torch.equal(noisy(g1, g2), base(g1, g2))
    print("ok  test_noisy_value_zero_sigma_passthrough")


def test_noisy_value_matches_distribution():
    """Loi 2 noise: V_actual = V_true + eta, eta~N(0,sigma^2), resampled every call."""
    rng = np.random.default_rng(0)
    noisy = NoisyValueFn(_FakeV(), sigma=0.5, rng=rng)
    g1 = torch.zeros(1, 2)
    g2 = torch.tensor([[3.0, 4.0]])   # true distance = 5.0
    samples = torch.stack([noisy(g1, g2) for _ in range(2000)]).numpy().flatten()
    assert abs(samples.mean() - 5.0) < 0.1, samples.mean()
    assert abs(samples.std() - 0.5) < 0.1, samples.std()
    print(f"ok  test_noisy_value_matches_distribution (mean={samples.mean():.2f}, std={samples.std():.2f})")


def test_mcts_matches_soft_floyd_at_sigma0():
    """Mandatory sanity check (SPEC Sec 8/10, acceptance criterion #1): at
    sigma=0, MCTS must recover the same distance Soft Floyd computes on the
    same line-graph fixture as test_soft_floyd_and_dmax."""
    cfg = get_config("PointMaze", beta=0.05, soft_iters=40, d_max=1.5, neg_inf=-1e6,
                     mcts_n_simulations=500, mcts_c_uct=1.4, mcts_rollout_horizon=10)
    gs = GraphSearch(cfg)
    landmarks = torch.tensor([[0.], [1.], [2.]])
    goal = torch.tensor([3.])
    value_fn = NoisyValueFn(_FakeV(), sigma=0.0, rng=np.random.default_rng(0))

    W = gs.build_weight_matrix(landmarks, goal, value_fn)
    d_c2g = gs.soft_floyd(W)[:, -1].numpy()      # same as LatentPlanner.reset()'s self.d_c2g
    nodes_t = torch.cat([landmarks, goal.view(1, -1)], dim=0)

    mcts = LandmarkMCTS(n_landmarks=3, value_fn=value_fn, nodes_t=nodes_t,
                        d_c2g_heuristic=d_c2g, cfg=cfg, rng=np.random.default_rng(1))
    # only landmark 0 is reachable from the "current state" (root); everything
    # else must be routed through the (d_max-masked) graph, exactly like
    # test_soft_floyd_and_dmax's 0->1->2->goal cost-3 path.
    d_s2c = np.array([0.0, -1e6, -1e6, -1e6], dtype=np.float32)
    best_idx, stats = mcts.search(d_s2c)
    assert best_idx == 0
    _, mean_q = stats[0]
    assert abs(mean_q + 3.0) < 0.5, mean_q
    print(f"ok  test_mcts_matches_soft_floyd_at_sigma0 (mean_q={mean_q:.2f})")


def test_mcts_respects_prev_landmark_mask():
    """The root must never select a masked (previous) landmark, even when it
    would otherwise trivially win."""
    cfg = get_config("PointMaze", d_max=20.0, neg_inf=-1e6, mcts_n_simulations=100)
    landmarks = torch.tensor([[0.], [5.], [10.]])
    goal = torch.tensor([15.])
    value_fn = NoisyValueFn(_FakeV(), sigma=0.0, rng=np.random.default_rng(0))
    gs = GraphSearch(cfg)
    d_c2g = gs.soft_floyd(gs.build_weight_matrix(landmarks, goal, value_fn))[:, -1].numpy()
    nodes_t = torch.cat([landmarks, goal.view(1, -1)], dim=0)

    mcts = LandmarkMCTS(n_landmarks=3, value_fn=value_fn, nodes_t=nodes_t,
                        d_c2g_heuristic=d_c2g, cfg=cfg, rng=np.random.default_rng(0))
    d_s2c = np.array([-1.0, -0.1, -5.0, -5.0], dtype=np.float32)   # index 1 would trivially win
    mask = np.array([False, True, False, False])
    best_idx, _ = mcts.search(d_s2c, mask=mask)
    assert best_idx != 1
    print("ok  test_mcts_respects_prev_landmark_mask")


def test_mcts_planner_interface_compatible():
    """MCTSPlanner subclasses LatentPlanner and overrides only _replan; confirm
    reset()/act(), commit-for-K-steps, and previous-landmark masking still hold."""
    cfg = get_config("PointMaze", d_max=20.0, neg_inf=-1e6, mcts_n_simulations=50)
    agent = _FakeAgent(distances=[2.0, 3.0, 8.0, 20.0])
    agent.value = _FakeV()
    lm = LatentLandmarks(n_landmarks=3, embedding_size=2)
    planner = MCTSPlanner(agent, lm, _IdentityAE(), GraphSearch(cfg), cfg,
                          rng=np.random.default_rng(0))

    planner.n_landmarks = 3
    planner.landmark_goals = np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32)
    planner.goal = np.array([3., 3.], dtype=np.float32)
    planner.d_c2g = np.array([-1.0, -0.5, -6.0, 0.0], dtype=np.float32)
    planner.cnt = 0.0
    planner.subg_idx = None
    planner.prev_landmark = None

    obs = np.zeros(2, dtype=np.float32)
    planner.act(obs)
    first_calls = agent.calls
    first_idx = planner.subg_idx
    assert first_idx is not None and 0 <= first_idx <= 3
    K = planner.cnt
    assert K >= 1
    for _ in range(int(K) - 1):
        planner.act(obs)
        assert planner.subg_idx == first_idx
    assert agent.calls == first_calls           # never re-planned while committed

    planner.cnt = 0.0
    planner.act(obs)
    assert planner.subg_idx != first_idx, "must not reselect the previous landmark"
    print("ok  test_mcts_planner_interface_compatible")


def test_naive_replan_every_step():
    """Baseline 2 (SPEC Sec 7): NaiveReplanPlanner must re-plan on every act()
    call, unlike LatentPlanner's commit-for-K-steps behavior."""
    cfg = get_config("PointMaze")
    agent = _FakeAgent(distances=[2.0, 3.0, 8.0, 20.0])
    lm = LatentLandmarks(n_landmarks=3, embedding_size=2)
    planner = NaiveReplanPlanner(agent, lm, _IdentityAE(), GraphSearch(cfg), cfg)

    planner.n_landmarks = 3
    planner.landmark_goals = np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32)
    planner.goal = np.array([3., 3.], dtype=np.float32)
    planner.d_c2g = np.array([-1.0, -0.5, -6.0, 0.0], dtype=np.float32)
    planner.cnt = 0.0
    planner.subg_idx = None
    planner.prev_landmark = None

    obs = np.zeros(2, dtype=np.float32)
    planner.act(obs)
    assert agent.calls == 1
    planner.act(obs)
    assert agent.calls == 2          # re-planned again -- no K-step commitment
    planner.act(obs)
    assert agent.calls == 3
    print("ok  test_naive_replan_every_step")


def test_mcts_admissibility_cached_once_per_episode():
    """Regression for the admissibility-cadence bug found during spec review:
    MCTSPlanner.reset() must build the d_max admissibility mask ONCE per
    episode (matching self.d_c2g's cadence), and _replan() must reuse that
    same mask rather than resampling a fresh one from the noisy value_fn on
    every macro-step."""
    cfg = get_config("PointMaze", d_max=20.0, neg_inf=-1e6, mcts_n_simulations=20)
    agent = _FakeAgent(distances=[2.0, 3.0, 8.0, 20.0])
    agent.value = NoisyValueFn(_FakeV(), sigma=0.3, rng=np.random.default_rng(0))
    lm = LatentLandmarks(n_landmarks=3, embedding_size=2)
    planner = MCTSPlanner(agent, lm, _IdentityAE(), GraphSearch(cfg), cfg,
                          rng=np.random.default_rng(1))

    planner.reset(np.array([3., 3.], dtype=np.float32))
    first_admissible = planner._admissible
    assert first_admissible is not None

    obs = np.zeros(2, dtype=np.float32)
    planner._replan(obs)
    planner.prev_landmark = None    # don't let masking exhaust every candidate
    planner._replan(obs)

    assert planner._admissible is first_admissible, "admissibility must not be rebuilt per macro-step"
    print("ok  test_mcts_admissibility_cached_once_per_episode")


def test_dmax_candidates_track_v_scale():
    """d_max candidates must scale with the graph's own V magnitude (SPEC R3),
    ignore the diagonal, and be sorted/deduplicated."""
    # a V matrix whose off-diagonal values live in [1, 10]; diagonal is 0
    m = 6
    v = np.zeros((m, m))
    vals = np.linspace(1.0, 10.0, m * m).reshape(m, m)
    off_mask = ~np.eye(m, dtype=bool)
    v[off_mask] = vals[off_mask]
    cands = dmax_candidates(v, percentiles=(10, 50, 90))
    assert cands == sorted(cands) and len(cands) == len(set(cands))
    # all candidates lie within the off-diagonal value range (diagonal 0 ignored)
    off = v[off_mask]
    assert min(cands) >= off.min() - 1e-6 and max(cands) <= off.max() + 1e-6
    # scaling V by 10x scales the candidates by ~10x (relative cutoff, not absolute)
    cands_scaled = dmax_candidates(v * 10.0, percentiles=(10, 50, 90))
    assert abs(cands_scaled[1] / cands[1] - 10.0) < 1e-3
    print("ok  test_dmax_candidates_track_v_scale")


def test_bootstrap_ci():
    """Bootstrap CI must bracket the sample mean, collapse to a point for
    all-equal outcomes, and narrow as sample size grows (SPEC Sec 10)."""
    rng = np.random.default_rng(0)
    # all successes -> mean 1, degenerate CI
    m, lo, hi = bootstrap_ci([1, 1, 1, 1], rng=rng)
    assert m == 1.0 and lo == 1.0 and hi == 1.0
    # empty -> zeros
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    # mixed: mean ~0.5, CI brackets it, and wider for small n than large n
    small = np.concatenate([np.ones(5), np.zeros(5)])
    big = np.concatenate([np.ones(500), np.zeros(500)])
    ms, los, his = bootstrap_ci(small, rng=np.random.default_rng(1))
    mb, lob, hib = bootstrap_ci(big, rng=np.random.default_rng(1))
    assert abs(ms - 0.5) < 1e-9 and los <= ms <= his
    assert (his - los) > (hib - lob), "CI should narrow with more data"
    print("ok  test_bootstrap_ci")


ALL_TESTS = [
    test_q_from_distance,
    test_value_regression,
    test_autoencoder_reachability,
    test_gls_and_elbo,
    test_soft_floyd_and_dmax,
    test_planner_commitment_and_masking,
    test_noisy_value_zero_sigma_passthrough,
    test_noisy_value_matches_distribution,
    test_mcts_matches_soft_floyd_at_sigma0,
    test_mcts_respects_prev_landmark_mask,
    test_mcts_planner_interface_compatible,
    test_naive_replan_every_step,
    test_mcts_admissibility_cached_once_per_episode,
    test_dmax_candidates_track_v_scale,
    test_bootstrap_ci,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print("\nAll tests passed.")
