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
from l3p.planning.noise import (NoisyValueFn, ValueOverrideAgent, dmax_candidates,
                                 bootstrap_ci, build_sigma_matrix, HeterogeneousNoise,
                                 build_bias_matrix, CriticEdgeFn, BiasedValueFn,
                                 calibration_seed, hierarchical_bootstrap_ci)
from l3p.planning.mcts_planner import (LandmarkMCTS, MCTSPlanner, _Node,
                                        FeedbackMCTSPlanner, SoftFloydE1c,
                                        update_running_variance,
                                        running_sigma_matrix)
from l3p.planning.baselines import NaiveReplanPlanner, FreshGraphReplanPlanner
from scripts.run_e1c import make_branch as make_e1c_branch
from scripts.run_e1d import (apply_execution_noise,
                             evaluate_one as evaluate_e1d_one,
                             make_branch as make_e1d_branch)
from l3p.losses import ae_losses
from l3p.trainer import L3PTrainer


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


def test_evaluate_stops_after_success():
    class Env:
        def __init__(self):
            self.steps = 0
        def reset(self):
            self.steps = 0
            return {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(2, dtype=np.float32),
                "desired_goal": np.ones(2, dtype=np.float32),
            }
        def step(self, action):
            self.steps += 1
            obs = {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(2, dtype=np.float32),
                "desired_goal": np.ones(2, dtype=np.float32),
            }
            return obs, -1.0, False, {"is_success": float(self.steps == 3)}

    class VecEnv:
        def __init__(self):
            self.envs = [Env()]
        def set_eval(self, enabled):
            self.eval_enabled = enabled

    class Planner:
        def reset(self, goal):
            self.finalized = False
        def act(self, obs, achieved_goal=None):
            return np.zeros(1, dtype=np.float32)
        def finalize(self, obs, achieved_goal):
            self.finalized = True

    trainer = L3PTrainer.__new__(L3PTrainer)
    trainer.env = VecEnv()
    trainer.cfg = get_config("PointMaze", test_episode_steps=20)
    trainer.centroids_initialized = True
    planner = Planner()
    success = trainer.evaluate(1, planner=planner)
    assert success == 1.0
    assert trainer.env.envs[0].steps == 3
    assert planner.finalized
    assert trainer.env.eval_enabled is False
    print("ok  test_evaluate_stops_after_success")


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


def test_noisy_value_remains_zero_mean_near_zero():
    """Gaussian injection must not become positively biased near distance zero."""
    class SmallV:
        def __call__(self, g1, g2):
            return torch.full((g1.shape[0],), 0.1, device=g1.device)

    noisy = NoisyValueFn(SmallV(), sigma=0.5, rng=np.random.default_rng(0))
    g = torch.zeros(1, 1)
    samples = torch.stack([noisy(g, g) for _ in range(10000)]).numpy().ravel()
    assert abs(samples.mean() - 0.1) < 0.02, samples.mean()
    assert (samples < 0).any(), "an unclipped Gaussian must retain its negative tail"
    print("ok  test_noisy_value_remains_zero_mean_near_zero")


def test_mcts_indexed_noise_caches_clean_values_but_resamples_noise():
    """MCTS may cache deterministic V_true, but each edge traversal must still
    receive an independent Loai-2 sample."""
    class CountingV:
        def __init__(self):
            self.calls = 0
        def __call__(self, g1, g2):
            self.calls += 1
            return torch.norm(g1 - g2, dim=-1)

    base = CountingV()
    noisy = NoisyValueFn(base, sigma=0.5, rng=np.random.default_rng(0))
    nodes = torch.tensor([[0.0], [2.0], [5.0]])
    noisy.prepare_nodes(nodes)
    first = noisy.sample_indexed(0, [1, 2])
    second = noisy.sample_indexed(0, [1, 2])
    assert base.calls == 1
    assert not torch.equal(first, second)
    samples = torch.stack(
        [noisy.sample_indexed(0, [2])[0] for _ in range(3000)]).numpy()
    assert abs(samples.mean() - 5.0) < 0.1
    assert abs(samples.std() - 0.5) < 0.1
    assert base.calls == 1
    print("ok  test_mcts_indexed_noise_caches_clean_values_but_resamples_noise")


def test_mcts_uses_indexed_value_cache():
    class CountingV:
        def __init__(self):
            self.calls = 0
        def __call__(self, g1, g2):
            self.calls += 1
            return torch.norm(g1 - g2, dim=-1)

    base = CountingV()
    noisy = NoisyValueFn(base, sigma=0.2, rng=np.random.default_rng(0))
    nodes = torch.tensor([[0.0], [1.0], [2.0]])
    cfg = get_config(
        "PointMaze", d_max=10.0, mcts_n_simulations=30,
        mcts_rollout_horizon=4)
    mcts = LandmarkMCTS(
        n_landmarks=2, value_fn=noisy, nodes_t=nodes,
        d_c2g_heuristic=np.array([-2.0, -1.0, 0.0]), cfg=cfg,
        rng=np.random.default_rng(1))
    calls_after_prepare = base.calls
    mcts.search(np.array([-1.0, -1.0, -2.0]))
    assert calls_after_prepare == 2  # admissibility observation + clean cache
    assert base.calls == calls_after_prepare
    print("ok  test_mcts_uses_indexed_value_cache")


def test_graph_heuristic_and_admissibility_share_one_observation():
    """A stochastic graph must be queried once for both Floyd and MCTS masking."""
    class CountingV:
        def __init__(self):
            self.calls = 0
        def __call__(self, g1, g2):
            self.calls += 1
            return torch.norm(g1 - g2, dim=-1)

    cfg = get_config("PointMaze", d_max=1.5)
    gs, value = GraphSearch(cfg), CountingV()
    centroids = torch.tensor([[0.], [1.], [2.]])
    d_c2g, admissible = gs.distances_to_goal_with_admissibility(
        centroids, torch.tensor([3.]), _IdentityAE(), value)
    assert value.calls == 1
    assert d_c2g.shape == (4,) and admissible.shape == (4, 4)
    assert admissible[0, 1] and not admissible[0, 2]
    print("ok  test_graph_heuristic_and_admissibility_share_one_observation")


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


def test_mcts_covers_every_root_action():
    """Cheap presets must still evaluate every landmark once at the root."""
    n = 50
    cfg = get_config("PointMaze", d_max=100.0, mcts_n_simulations=20)
    nodes = torch.arange(n + 1, dtype=torch.float32).view(-1, 1)
    mcts = LandmarkMCTS(
        n, _FakeV(), nodes, np.r_[-np.ones(n), 0.0], cfg,
        np.random.default_rng(0))
    _, stats = mcts.search(np.r_[-np.ones(n), -100.0])
    assert len(stats) == n + 1
    assert mcts.actual_simulations >= 2 * (n + 1)
    print("ok  test_mcts_covers_every_root_action")


def test_mcts_returns_none_when_all_root_actions_masked():
    cfg = get_config("PointMaze", mcts_n_simulations=20)
    mcts = LandmarkMCTS(
        2, _FakeV(), torch.tensor([[0.], [1.], [2.]]),
        np.array([-2.0, -1.0, 0.0]), cfg, np.random.default_rng(0))
    best, stats = mcts.search(
        np.array([-1.0, -1.0, -1.0]),
        mask=np.ones(3, dtype=bool))
    assert best is None and stats == {} and mcts.actual_simulations == 0
    print("ok  test_mcts_returns_none_when_all_root_actions_masked")


def test_mcts_terminal_goal_reward_applies_at_root():
    nodes = torch.tensor([[0.0], [2.0]])
    heuristic = np.array([-10.0, 0.0])
    d_s2c = np.array([0.0, -20.0])

    def choose(goal_reward):
        cfg = get_config(
            "PointMaze", d_max=1.0, mcts_n_simulations=100,
            mcts_lambda_goal=goal_reward)
        return LandmarkMCTS(
            1, _FakeV(), nodes, heuristic, cfg,
            np.random.default_rng(0)).search(d_s2c)[0]

    assert choose(0.0) == 0
    assert choose(15.0) == 1
    print("ok  test_mcts_terminal_goal_reward_applies_at_root")


class _WormholeV:
    """Raw V has an over-optimistic landmark->goal edge."""
    def __call__(self, g1, g2):
        out = torch.norm(g1 - g2, dim=-1)
        # Edge 0 -> goal coordinate 10 looks free, like Fetch's V wormholes.
        edge = (torch.isclose(g1[:, 0], torch.tensor(0., device=g1.device)) &
                torch.isclose(g2[:, 0], torch.tensor(10., device=g2.device)))
        return torch.where(edge, torch.zeros_like(out), out)


def test_mcts_rollout_capped_by_soft_floyd_heuristic():
    """Regression for Fetch: hard MCTS rollout must not rate a node better than
    the Soft-Floyd heuristic it bootstraps from when raw V has near-zero
    landmark->goal wormholes."""
    cfg = get_config("PointMaze", d_max=20.0, mcts_n_simulations=20,
                     mcts_cap_rollout_by_heuristic=True)
    nodes_t = torch.tensor([[0.], [10.]])       # one landmark plus goal
    d_c2g = np.array([-8.0, 0.0], dtype=np.float32)
    mcts = LandmarkMCTS(n_landmarks=1, value_fn=_WormholeV(), nodes_t=nodes_t,
                        d_c2g_heuristic=d_c2g, cfg=cfg, rng=np.random.default_rng(0))

    assert mcts._rollout(0, budget=1) <= -8.0
    d_s2c = np.array([0.0, -5.0], dtype=np.float32)
    best_idx, stats = mcts.search(d_s2c)
    assert best_idx == 1, stats
    print("ok  test_mcts_rollout_capped_by_soft_floyd_heuristic")


def test_mcts_rollout_cap_is_opt_in():
    """The conservative rollout cap must not silently change MCTS by env."""
    assert get_config("PointMaze").mcts_cap_rollout_by_heuristic is False
    assert get_config("PointMazeMuJoCo").mcts_cap_rollout_by_heuristic is False
    assert get_config("FetchPickAndPlace").mcts_cap_rollout_by_heuristic is False
    print("ok  test_mcts_rollout_cap_is_opt_in")


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


def test_mcts_deterministic_limit_matches_soft_floyd_action():
    d_s2c = np.array([-2.0, -1.0, -10.0])
    d_c2g = np.array([-1.0, -4.0, 0.0])
    mask = np.array([False, False, False])
    assert MCTSPlanner._soft_floyd_action(d_s2c, d_c2g, mask) == 0
    mask[0] = True
    assert MCTSPlanner._soft_floyd_action(d_s2c, d_c2g, mask) == 1
    assert MCTSPlanner._soft_floyd_action(
        d_s2c, d_c2g, np.ones(3, dtype=bool)) is None
    print("ok  test_mcts_deterministic_limit_matches_soft_floyd_action")


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


def test_fresh_graph_baseline_refreshes_value_observation():
    """SPEC baseline 3 keeps the first paired graph then refreshes V_obs per act."""
    class CountingV(_FakeV):
        def __init__(self):
            self.calls = 0
        def __call__(self, g1, g2):
            self.calls += 1
            return super().__call__(g1, g2)

    cfg = get_config("PointMaze", d_max=20.0)
    agent = _FakeAgent(distances=[2.0, 3.0, 8.0, 20.0])
    agent.value = CountingV()
    planner = FreshGraphReplanPlanner(
        agent, LatentLandmarks(3, 2), _IdentityAE(), GraphSearch(cfg), cfg)
    planner.reset(np.array([3.0, 3.0], dtype=np.float32))
    assert agent.value.calls == 1
    planner.act(np.zeros(2, dtype=np.float32))
    assert agent.value.calls == 1
    planner.act(np.zeros(2, dtype=np.float32))
    assert agent.value.calls == 2
    print("ok  test_fresh_graph_baseline_refreshes_value_observation")


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


def test_dmax_candidates_keep_fallback_for_tiny_v():
    """Fetch-style checkpoints can contain many near-zero V wormholes; calibration
    must still test the environment default cutoff instead of only tiny values."""
    v = np.array([[0.0, 0.0, 1e-9],
                  [0.0, 0.0, 2e-9],
                  [1e-9, 2e-9, 0.0]])
    cands = dmax_candidates(v, percentiles=(50,), fallback=15.0)
    assert 15.0 in cands, cands
    assert cands == sorted(cands) and len(cands) == len(set(cands))
    print("ok  test_dmax_candidates_keep_fallback_for_tiny_v")


def test_calibration_seed_is_disjoint_and_deterministic():
    assert calibration_seed(7) == calibration_seed(7)
    assert calibration_seed(7) != 7
    assert calibration_seed(7) != calibration_seed(8)
    print("ok  test_calibration_seed_is_disjoint_and_deterministic")


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


def test_hierarchical_bootstrap_preserves_seed_variation():
    groups = [np.zeros(100), np.ones(100)]
    mean, lo, hi = hierarchical_bootstrap_ci(
        groups, n_boot=4000, rng=np.random.default_rng(0))
    assert mean == 0.5
    assert lo <= 0.05 and hi >= 0.95, (lo, hi)
    print("ok  test_hierarchical_bootstrap_preserves_seed_variation")


def test_build_sigma_matrix():
    """Directed, zero diagonal, and ~frac_high of ordered graph edges are high."""
    m, goal_idx = 6, 5
    sig = build_sigma_matrix(m, frac_high=0.5, sigma_lo=0.1, sigma_hi=0.9,
                             rng=np.random.default_rng(0), goal_idx=goal_idx)
    assert np.allclose(np.diag(sig), 0.0)                 # zero diagonal
    pairs = [(i, j) for i in range(m) for j in range(m) if i != j]
    n_high = sum(np.isclose(sig[i, j], 0.9) for i, j in pairs)
    assert n_high == round(0.5 * len(pairs)), (n_high, len(pairs))
    print("ok  test_build_sigma_matrix")


def test_heterogeneous_noise():
    """Zero-sigma edges pass through exactly; a high-sigma edge gets the right
    empirical std; sigma_of exposes the oracle by index."""
    nodes = torch.tensor([[0.], [3.], [7.]])              # 3 nodes on a line
    base = _FakeV()                                        # V = |g1-g2|
    sig = np.array([[0.0, 0.0, 0.6],
                    [0.0, 0.0, 0.0],
                    [0.6, 0.0, 0.0]])
    hz = HeterogeneousNoise(base, nodes, sig, np.random.default_rng(0))
    # edge (0,1): sigma 0 -> exact passthrough (V=3)
    assert torch.allclose(hz(nodes[0:1], nodes[1:2]), torch.tensor([3.0]))
    # edge (0,2): sigma 0.6 around true V=7
    samples = torch.stack([hz(nodes[0:1], nodes[2:3]) for _ in range(3000)]).numpy().ravel()
    assert abs(samples.mean() - 7.0) < 0.1 and abs(samples.std() - 0.6) < 0.1
    assert hz.sigma_of(0, 2) == 0.6 and hz.sigma_of(0, 1) == 0.0
    print(f"ok  test_heterogeneous_noise (mean={samples.mean():.2f}, std={samples.std():.2f})")


def test_mcts_alpha_avoids_uncertain_edge():
    """E1b alpha: with a risk penalty, MCTS routes AWAY from a landmark whose
    onward edge to the goal is high-sigma, even though that landmark is marginally
    preferred with no uncertainty term."""
    nodes = torch.tensor([[1.0], [3.0], [2.0]])   # L0, L1, goal (V=euclidean)
    heuristic = np.array([-1.0, -1.0, 0.0])
    d_s2c = np.array([0.0, -0.1, -10.0])           # root slightly prefers L0; goal far
    sigma = np.array([[0.0, 0.0, 0.9],             # L0->goal is unreliable
                      [0.0, 0.0, 0.0],             # L1->goal is reliable
                      [0.9, 0.0, 0.0]])

    def run(mode):
        cfg = get_config("PointMaze", d_max=5.0, mcts_n_simulations=300,
                         mcts_rollout_horizon=6, mcts_uncertainty_mode=mode,
                         mcts_lambda_risk=5.0)
        mcts = LandmarkMCTS(n_landmarks=2, value_fn=_FakeV(), nodes_t=nodes,
                            d_c2g_heuristic=heuristic, cfg=cfg,
                            rng=np.random.default_rng(0), sigma_matrix=sigma)
        return mcts.search(d_s2c)[0]

    assert run("none") == 0, "no uncertainty term -> take the marginally-closer L0"
    assert run("alpha") == 1, "risk penalty -> avoid L0's unreliable onward edge, take L1"
    print("ok  test_mcts_alpha_avoids_uncertain_edge")


def test_uct_beta_bonus_shifts_selection():
    """E1b beta: the UCT exploration bonus steers selection toward the
    higher-uncertainty child when Q and visit counts are otherwise tied."""
    node = _Node(idx=0, candidates=[1, 2])
    node.visits = 10
    for j in (1, 2):
        node.child_visits[j] = 5
        node.child_total[j] = -5.0                 # equal Q = -1 for both
    # no bonus: tie -> deterministic first-max; bonus favoring child 2 -> child 2
    assert node.best_uct_child(1.4, bonus=lambda j: 1.0 if j == 2 else 0.0) == 2
    assert node.best_uct_child(1.4, bonus=lambda j: 1.0 if j == 1 else 0.0) == 1
    print("ok  test_uct_beta_bonus_shifts_selection")


def test_uct_beta_information_bonus_decays_with_samples():
    cfg = get_config(
        "PointMaze", mcts_uncertainty_mode="beta",
        mcts_beta_uncertainty=2.0)
    sigma = np.ones((3, 3), dtype=np.float64)
    mcts = LandmarkMCTS(
        2, _FakeV(), torch.tensor([[0.], [1.], [2.]]),
        np.array([-2.0, -1.0, 0.0]), cfg, np.random.default_rng(0),
        sigma_matrix=sigma)
    node = _Node(idx=0, candidates=[1])
    node.child_visits[1] = 1
    first = mcts._uct_bonus(node)(1)
    node.child_visits[1] = 16
    later = mcts._uct_bonus(node)(1)
    assert later == first / 4
    print("ok  test_uct_beta_information_bonus_decays_with_samples")


def test_build_bias_matrix():
    """E1c Loai-1 bias: directed, zero diagonal, all edges ~N(0,sigma^2)."""
    m, goal_idx, sigma = 5, 4, 0.3
    b = build_bias_matrix(m, sigma, np.random.default_rng(0), goal_idx=goal_idx)
    assert np.allclose(np.diag(b), 0.0)
    assert not np.allclose(b, b.T)
    assert np.any(np.abs(b[goal_idx, :goal_idx]) > 0)
    # empirical std of all directed edge biases ~ sigma (large sample)
    big = build_bias_matrix(80, sigma, np.random.default_rng(1), goal_idx=79)
    off = big[~np.eye(80, dtype=bool)]
    assert abs(off.std() - sigma) < 0.05, off.std()
    print("ok  test_build_bias_matrix")


class _StubAgent:
    """Minimal agent exposing distance_after_action = scale * euclidean(obs, goal)."""
    def __init__(self, scale=3.0):
        self.scale = scale
    def distance_after_action(self, obs, goals):
        obs = np.atleast_2d(obs); goals = np.atleast_2d(goals)
        return self.scale * np.linalg.norm(obs - goals, axis=1)


def test_critic_edge_fn():
    """CriticEdgeFn exposes the critic's matched-row distance as a value_fn."""
    fn = CriticEdgeFn(_StubAgent(scale=2.0))
    g1 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    g2 = torch.tensor([[3.0, 4.0], [1.0, 0.0]])   # dists 5 and 0 -> *2 = 10, 0
    out = fn(g1, g2)
    assert torch.allclose(out, torch.tensor([10.0, 0.0]))
    print("ok  test_critic_edge_fn")


def test_biased_value_fn():
    """Loai-1: multiplicative fixed bias, deterministic (no resampling), and
    nearest-node index lookup picks the right per-edge bias."""
    nodes = torch.tensor([[0.0], [2.0], [5.0]])
    base = _FakeV()                                # V = |g1 - g2|
    bias = np.array([[0.0, 0.5, 0.0],              # edge (0,1) inflated +50%
                     [0.5, 0.0, -0.4],             # edge (1,2) deflated -40% (wormhole)
                     [0.0, -0.4, 0.0]])
    bv = BiasedValueFn(base, nodes, bias)
    # (0,1): V=2 -> 2*1.5 = 3.0 ; (1,2): V=3 -> 3*0.6 = 1.8
    assert torch.allclose(bv(nodes[0:1], nodes[1:2]), torch.tensor([3.0]))
    assert torch.allclose(bv(nodes[1:2], nodes[2:3]), torch.tensor([1.8]))
    # deterministic: repeated calls identical (fixed bias, unlike Loai-2)
    assert torch.equal(bv(nodes[1:2], nodes[2:3]), bv(nodes[1:2], nodes[2:3]))
    print("ok  test_biased_value_fn")


def _make_feedback_planner():
    cfg = get_config("PointMaze", d_max=50.0, mcts_n_simulations=20)
    pl = FeedbackMCTSPlanner(_StubAgent(scale=1.0), LatentLandmarks(3, 2), _IdentityAE(),
                             GraphSearch(cfg), cfg, sigma=0.0, noise_seed=0,
                             rho=0.5, tau_reach=1.0, tau_progress=3.0, r_max=2,
                             rng=np.random.default_rng(0))
    # manually set episode state (bypass reset, like test_planner_commitment_and_masking)
    pl.n_landmarks = 3
    pl.landmark_goals = np.array([[0., 0.], [10., 0.], [0., 10.]], dtype=np.float32)
    pl.goal = np.array([10., 10.], dtype=np.float32)
    pl._nodes_t = torch.tensor([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])
    bias = np.zeros((4, 4)); bias[0, 1] = bias[1, 0] = 0.5      # edge (0,1) inflated +50%
    pl._biased = BiasedValueFn(CriticEdgeFn(_StubAgent(scale=1.0)), pl._nodes_t, bias)
    pl._v_exec, pl._residual_stats, pl._attempts, pl._blacklist = {}, {}, {}, set()
    pl.stats = dict(macros=0, reached=0, progressed=0, stuck=0,
                    blacklisted=0, corrections=0, snaps=0, virtual_roots=0)
    return pl


def test_feedback_ema_correction_and_reached():
    """Sec 4.3: a REACHED macro-step blends realized env-step cost into V_exec[i][j]
    via EMA (rho), and clears the failure counter."""
    pl = _make_feedback_planner()
    pl._macro_start_i, pl._cur_j = 0, 1
    pl._macro_start_z = np.array([0., 0.], dtype=np.float32)
    pl._macro_k = 8
    pl._attempts = {1: 1}
    pl._observe(np.array([10., 0.], dtype=np.float32))          # reached c_1 (dist 0)
    # biased edge (0,1) = |(0,0)-(10,0)| * 1.5 = 15 ; realized = 8 + 0 ; rho=0.5
    assert abs(pl._v_exec[(0, 1)] - 0.5 * 15 - 0.5 * 8) < 1e-6, pl._v_exec[(0, 1)]
    assert pl._attempts == {}                                   # cleared on REACHED
    assert 1 not in pl._blacklist
    print("ok  test_feedback_ema_correction_and_reached")


def test_feedback_never_blacklists_goal():
    """Regression: the GOAL node must never be masked out of the candidate set,
    even after a failed direct-to-goal macro-step blacklists it (the fix for the
    `<= n_landmarks` bug). A blacklisted landmark IS masked; the goal is not."""
    pl = _make_feedback_planner()
    goal_idx = pl.n_landmarks                       # 3
    pl._blacklist = {goal_idx, 1}                   # goal + landmark 1 blacklisted
    pl.prev_landmark = None
    mask = pl._candidate_mask()
    assert mask[goal_idx] == False, "goal node must never be masked"
    assert mask[1] == True, "a blacklisted landmark must be masked"
    print("ok  test_feedback_never_blacklists_goal")


def test_feedback_stuck_blacklists_immediately():
    """Sec 4.1/4.4: a STUCK macro-step (barely moved) blacklists the subgoal at once."""
    pl = _make_feedback_planner()
    pl._macro_start_i, pl._cur_j = 0, 1
    pl._macro_start_z = np.array([0., 0.], dtype=np.float32)
    pl._macro_k = 5
    pl._observe(np.array([0.5, 0.], dtype=np.float32))          # moved 0.5 < tau_progress
    assert 1 in pl._blacklist
    print("ok  test_feedback_stuck_blacklists_immediately")


def test_feedback_progressed_blacklists_after_rmax():
    """Sec 4.4: a PROGRESSED-but-not-reached edge is blacklisted only after r_max tries."""
    pl = _make_feedback_planner()
    for attempt in range(2):                                    # r_max = 2
        pl._macro_start_i, pl._cur_j = 0, 1
        pl._macro_start_z = np.array([0., 0.], dtype=np.float32)
        pl._macro_k = 5
        # end at (5,0): travelled 5 >= tau_progress, dist to c_1 (10,0) = 5 > tau_reach
        blacklisted_before = 1 in pl._blacklist
        pl._observe(np.array([5., 0.], dtype=np.float32))
        if attempt == 0:
            assert not blacklisted_before and 1 not in pl._blacklist, "not yet at attempt 1"
    assert 1 in pl._blacklist, "blacklisted after reaching r_max"
    print("ok  test_feedback_progressed_blacklists_after_rmax")


def test_feedback_virtual_root_and_direct_goal_outcome():
    """Off-graph states stay virtual; failed direct-goal attempts are not ignored."""
    pl = _make_feedback_planner()
    pl.tau_snap = 0.1
    idx, nearest = pl._snap(np.array([5.0, 5.0], dtype=np.float32))
    assert idx is None and nearest > pl.tau_snap

    pl.stats = dict(macros=0, reached=0, progressed=0, stuck=0,
                    blacklisted=0, corrections=0, snaps=0, virtual_roots=0)
    pl._macro_start_i = None
    pl._cur_j = pl.n_landmarks
    pl._macro_start_z = np.array([0.0, 0.0], dtype=np.float32)
    pl._macro_k = 2
    pl._observe(np.array([0.0, 0.0], dtype=np.float32))
    assert pl.n_landmarks in pl._blacklist
    assert pl._cur_j is None
    print("ok  test_feedback_virtual_root_and_direct_goal_outcome")


def test_feedback_accepts_separate_observation_and_achieved_goal():
    """Ant/Fetch feedback must not assume obs_dim == goal_dim."""
    class SplitObsAgent(_StubAgent):
        def distance_after_action(self, obs, goals):
            obs = np.atleast_2d(obs)
            goals = np.atleast_2d(goals)
            xy = obs[:, :2]
            if xy.shape[0] == 1 and goals.shape[0] > 1:
                xy = np.repeat(xy, goals.shape[0], axis=0)
            return np.linalg.norm(xy - goals, axis=1)

    cfg = get_config("PointMaze", d_max=50.0)
    pl = FeedbackMCTSPlanner(
        SplitObsAgent(), LatentLandmarks(2, 2), _IdentityAE(), GraphSearch(cfg),
        cfg, sigma=0.0, noise_seed=0, edge_value_fn=_FakeV())
    pl.n_landmarks = 2
    pl.landmark_goals = np.array([[0., 0.], [10., 0.]], dtype=np.float32)
    pl.goal = np.array([10., 10.], dtype=np.float32)
    pl._nodes_t = torch.tensor([[0., 0.], [10., 0.], [10., 10.]])
    pl._biased = BiasedValueFn(
        _FakeV(), pl._nodes_t, np.zeros((3, 3), dtype=np.float64))
    pl._v_exec, pl._residual_stats, pl._attempts, pl._blacklist = {}, {}, {}, set()
    pl.stats = dict(macros=0, reached=0, progressed=0, stuck=0,
                    blacklisted=0, corrections=0, snaps=0, virtual_roots=0)
    pl._macro_start_i, pl._cur_j = 0, 1
    pl._macro_start_z = np.array([0., 0., 7., 8.], dtype=np.float32)
    pl._macro_k = 5
    pl._observe(
        np.array([10., 0., 9., 9.], dtype=np.float32),
        achieved_goal=np.array([10., 0.], dtype=np.float32))
    assert pl.stats["reached"] == 1 and (0, 1) in pl._v_exec
    print("ok  test_feedback_accepts_separate_observation_and_achieved_goal")


def test_execution_noise_is_applied_at_env_action_boundary():
    action = np.array([0.25, -0.25], dtype=np.float32)
    assert np.array_equal(
        apply_execution_noise(action, 0.0, np.random.default_rng(0), 1.0),
        action)
    noisy = apply_execution_noise(
        action, 0.5, np.random.default_rng(0), 1.0)
    assert not np.array_equal(noisy, action)
    assert np.all(noisy <= 1.0) and np.all(noisy >= -1.0)
    print("ok  test_execution_noise_is_applied_at_env_action_boundary")


def test_e1d_evaluate_stops_after_success():
    class Env:
        def __init__(self):
            self.steps = 0
            self.rng = None
        def reset(self):
            self.steps = 0
            return {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(2, dtype=np.float32),
                "desired_goal": np.ones(2, dtype=np.float32),
            }
        def step(self, action):
            self.steps += 1
            obs = {
                "observation": np.zeros(2, dtype=np.float32),
                "achieved_goal": np.zeros(2, dtype=np.float32),
                "desired_goal": np.ones(2, dtype=np.float32),
            }
            return obs, -1.0, False, {"is_success": float(self.steps == 3)}

    class VecEnv:
        max_action = 1.0
        def __init__(self):
            self.envs = [Env()]
        def set_eval(self, enabled):
            self.eval_enabled = enabled

    class Planner:
        exec_sigma = 0.0
        search_seconds = 0.0
        search_calls = 0
        stats = {}
        def reset(self, goal):
            self.finalized = False
        def act(self, obs, achieved_goal=None):
            return np.zeros(1, dtype=np.float32)
        def finalize(self, obs, achieved_goal):
            self.finalized = True

    class Trainer:
        env = VecEnv()
        cfg = get_config("PointMaze", test_episode_steps=20)

    planner = Planner()
    success, _ = evaluate_e1d_one(Trainer(), planner, episode_seed=0)
    assert success == 1
    assert Trainer.env.envs[0].steps == 3
    assert planner.finalized
    assert Trainer.env.eval_enabled is False
    print("ok  test_e1d_evaluate_stops_after_success")


def test_execution_residuals_produce_directed_edge_uncertainty():
    stats = {}
    update_running_variance(stats, (0, 1), -2.0)
    update_running_variance(stats, (0, 1), 2.0)
    update_running_variance(stats, (1, 0), 0.0)
    sigma = running_sigma_matrix(stats, 3)
    assert abs(sigma[0, 1] - np.sqrt(8.0)) < 1e-9
    assert sigma[1, 0] == 0.0
    print("ok  test_execution_residuals_produce_directed_edge_uncertainty")


def test_sigma0_disables_feedback_treatment():
    """E1c/E1d no-noise branches must remain the static Floyd control."""
    class EnvShape:
        obs_dim = 2
        goal_dim = 2

    class TrainerStub:
        agent = _StubAgent()
        landmarks = LatentLandmarks(2, 2)
        ae = _IdentityAE()
        graph_search = GraphSearch(get_config("PointMaze"))
        env = EnvShape()

    class Args:
        rho = 0.5
        tau_reach = 1.0
        tau_progress = 2.0
        tau_snap = 3.0
        r_max = 2
        graph_sigma_scale = 1.0
        exec_sigma_scale = 1.0

    cfg = get_config("PointMaze")
    clean_c = make_e1c_branch(
        "mcts_fb", TrainerStub(), cfg, sigma=0.0, noise_seed=0, args=Args())
    noisy_c = make_e1c_branch(
        "mcts_fb", TrainerStub(), cfg, sigma=0.1, noise_seed=0, args=Args())
    clean_d = make_e1d_branch(
        "mcts_fb", TrainerStub(), cfg, sigma=0.0, noise_seed=0, args=Args())
    noisy_d = make_e1d_branch(
        "mcts_fb", TrainerStub(), cfg, sigma=0.1, noise_seed=0, args=Args())
    assert clean_c.feedback is False and clean_d.feedback is False
    assert noisy_c.feedback is True and noisy_d.feedback is True
    print("ok  test_sigma0_disables_feedback_treatment")


ALL_TESTS = [
    test_q_from_distance,
    test_value_regression,
    test_autoencoder_reachability,
    test_gls_and_elbo,
    test_soft_floyd_and_dmax,
    test_planner_commitment_and_masking,
    test_evaluate_stops_after_success,
    test_noisy_value_zero_sigma_passthrough,
    test_noisy_value_matches_distribution,
    test_noisy_value_remains_zero_mean_near_zero,
    test_mcts_indexed_noise_caches_clean_values_but_resamples_noise,
    test_mcts_uses_indexed_value_cache,
    test_graph_heuristic_and_admissibility_share_one_observation,
    test_mcts_matches_soft_floyd_at_sigma0,
    test_mcts_covers_every_root_action,
    test_mcts_returns_none_when_all_root_actions_masked,
    test_mcts_terminal_goal_reward_applies_at_root,
    test_mcts_rollout_capped_by_soft_floyd_heuristic,
    test_mcts_rollout_cap_is_opt_in,
    test_mcts_respects_prev_landmark_mask,
    test_mcts_planner_interface_compatible,
    test_mcts_deterministic_limit_matches_soft_floyd_action,
    test_naive_replan_every_step,
    test_fresh_graph_baseline_refreshes_value_observation,
    test_mcts_admissibility_cached_once_per_episode,
    test_dmax_candidates_track_v_scale,
    test_dmax_candidates_keep_fallback_for_tiny_v,
    test_calibration_seed_is_disjoint_and_deterministic,
    test_bootstrap_ci,
    test_hierarchical_bootstrap_preserves_seed_variation,
    test_build_sigma_matrix,
    test_heterogeneous_noise,
    test_mcts_alpha_avoids_uncertain_edge,
    test_uct_beta_bonus_shifts_selection,
    test_uct_beta_information_bonus_decays_with_samples,
    test_build_bias_matrix,
    test_critic_edge_fn,
    test_biased_value_fn,
    test_feedback_ema_correction_and_reached,
    test_feedback_never_blacklists_goal,
    test_feedback_stuck_blacklists_immediately,
    test_feedback_progressed_blacklists_after_rmax,
    test_feedback_virtual_root_and_direct_goal_outcome,
    test_feedback_accepts_separate_observation_and_achieved_goal,
    test_execution_noise_is_applied_at_env_action_boundary,
    test_e1d_evaluate_stops_after_success,
    test_execution_residuals_produce_directed_edge_uncertainty,
    test_sigma0_disables_feedback_treatment,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
    print("\nAll tests passed.")
