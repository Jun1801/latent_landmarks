"""Standalone tests for pathfind.py (pure numpy, no wmag/MuJoCo).

Run locally:  python repro/paper_mcts/test_pathfind.py
Validates the classical planners against an independent Floyd-Warshall +
Dijkstra reference. Tie-safe: asserts the returned first hop lies on an
OPTIMAL path (compares path cost, not the index).
"""
import numpy as np

import pathfind as pf

INF = float("inf")


# ---- independent reference implementations -------------------------------
def ref_floyd_cost_to_goal(edge_sub, goal_col, dist_clip):
    """Floyd-Warshall over landmarks+goal; return cost_i->goal (inf if none)."""
    n = edge_sub.shape[0]
    N = n + 1  # goal = index n
    D = np.full((N, N), INF)
    np.fill_diagonal(D, 0.0)
    for i in range(n):
        for j in range(n):
            if i != j and edge_sub[i, j] >= dist_clip:
                D[i, j] = -edge_sub[i, j]
        if goal_col[i] >= dist_clip:
            D[i, n] = -goal_col[i]
    for k in range(N):
        for i in range(N):
            if D[i, k] == INF:
                continue
            for j in range(N):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D[:n, n]


def ref_optimal_s2goal(d_s2c_full, edge_sub, goal_col, dist_clip, mask):
    """Optimal state->goal cost, honoring the first-hop mask + dist_clip."""
    n = edge_sub.shape[0]
    ctg = ref_floyd_cost_to_goal(edge_sub, goal_col, dist_clip)
    best = INF
    d = np.asarray(d_s2c_full, float)
    for i in range(n):                       # via landmark i (first hop)
        if mask is not None and mask[i]:
            continue
        if d[i] >= dist_clip and ctg[i] < INF:
            best = min(best, -d[i] + ctg[i])
    if d[n] >= dist_clip:                     # go direct
        best = min(best, -d[n])
    return best, ctg


def first_hop_cost(idx, d_s2c_full, ctg, dist_clip):
    n = len(d_s2c_full) - 1
    d = np.asarray(d_s2c_full, float)
    if idx == n:
        return -d[n] if d[n] >= dist_clip else INF
    if d[idx] < dist_clip or ctg[idx] == INF:
        return INF
    return -d[idx] + ctg[idx]


# ---- randomized graph generator ------------------------------------------
def rand_graph(rng, n, dist_clip):
    edge = rng.uniform(1.4 * dist_clip, 0.0, size=(n, n))   # some < dist_clip (dead)
    np.fill_diagonal(edge, 0.0)
    goal_col = rng.uniform(1.4 * dist_clip, 0.0, size=n)
    d_s2c = rng.uniform(1.4 * dist_clip, 0.0, size=n + 1)
    heur = np.concatenate([ref_floyd_cost_to_goal(edge, goal_col, dist_clip), [0.0]])
    heur = -heur                                            # value units (<=0), -inf if unreach
    heur[np.isinf(heur)] = -1e9
    return d_s2c, edge, goal_col, heur


# ---- tests ----------------------------------------------------------------
def test_hard_cost_to_goal_matches_floyd():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(2, 9)); dc = -20.0
        d, edge, gcol, _ = rand_graph(rng, n, dc)
        got = pf.hard_cost_to_goal(edge, gcol, dc)           # value (<=0)
        ref = ref_floyd_cost_to_goal(edge, gcol, dc)          # cost (>=0)
        for i in range(n):
            if ref[i] == INF:
                assert got[i] == -INF, (i, got[i])
            else:
                assert abs((-got[i]) - ref[i]) < 1e-9, (i, -got[i], ref[i])
    print("ok  hard_cost_to_goal == Floyd-Warshall (300 graphs)")


def test_dijkstra_first_hop_is_optimal():
    rng = np.random.default_rng(1)
    for _ in range(500):
        n = int(rng.integers(2, 9)); dc = -20.0
        d, edge, gcol, _ = rand_graph(rng, n, dc)
        mask = rng.random(n) < 0.2
        idx = pf.dijkstra_first_hop(d, edge, gcol, dc, mask=mask)
        opt, ctg = ref_optimal_s2goal(d, edge, gcol, dc, mask)
        cost = first_hop_cost(idx, d, ctg, dc)
        if opt == INF:                                       # nothing reachable -> go direct
            assert idx == n
        else:
            assert abs(cost - opt) < 1e-9, (idx, cost, opt)
            assert not (mask is not None and idx < n and mask[idx])
    print("ok  dijkstra_first_hop lies on an optimal path (500 graphs, w/ mask)")


def test_astar_matches_dijkstra_and_expands_less():
    rng = np.random.default_rng(2)
    worse = 0
    for _ in range(500):
        n = int(rng.integers(2, 9)); dc = -20.0
        d, edge, gcol, heur = rand_graph(rng, n, dc)         # heur = admissible soft=hard here
        mask = rng.random(n) < 0.2
        a_idx, expanded = pf.astar_first_hop(d, edge, gcol, dc, heur, mask=mask)
        opt, ctg = ref_optimal_s2goal(d, edge, gcol, dc, mask)
        cost = first_hop_cost(a_idx, d, ctg, dc)
        if opt == INF:
            assert a_idx == n
        else:
            assert abs(cost - opt) < 1e-9, (a_idx, cost, opt)  # A* returns an optimal first hop
        assert expanded <= n + 1
    print("ok  astar_first_hop optimal (== dijkstra cost) & expanded <= n+1 (500 graphs)")


def test_greedy_picks_nearest_reachable():
    rng = np.random.default_rng(3)
    for _ in range(500):
        n = int(rng.integers(2, 9)); dc = -20.0
        d, edge, gcol, heur = rand_graph(rng, n, dc)
        mask = rng.random(n) < 0.2
        idx = pf.greedy_first_hop(d, heur, dc, mask=mask)
        cand = [i for i in range(n) if d[i] >= dc and not mask[i]]
        if not cand:
            assert idx == n
        else:
            best = max(cand, key=lambda i: heur[i])
            assert abs(heur[idx] - heur[best]) < 1e-9           # nearest-to-goal reachable
            assert d[idx] >= dc and not mask[idx]
    print("ok  greedy_first_hop = nearest-to-goal reachable, honors mask+cutoff (500 graphs)")


def test_edge_cases():
    dc = -20.0
    # unreachable goal (no admissible edge to goal, no landmark path) -> go direct index n
    edge = np.full((3, 3), -100.0); np.fill_diagonal(edge, 0.0)   # all edges dead (< dc)
    gcol = np.full(3, -100.0)                                      # goal unreachable
    d = np.array([-1.0, -1.0, -1.0, -1.0])                        # direct also dead? d>=dc so ok
    idx = pf.dijkstra_first_hop(d, edge, gcol, dc)
    assert idx == 3, idx                                          # only the direct option survives
    # prev-goal mask forbids that landmark as first hop
    edge = np.array([[0., -1.], [-1., 0.]]); gcol = np.array([-1., -50.])
    d = np.array([-0.5, -50., -50.])                              # only landmark 0 reachable/cheap
    free = pf.dijkstra_first_hop(d, edge, gcol, dc)
    masked = pf.dijkstra_first_hop(d, edge, gcol, dc, mask=np.array([True, False]))
    assert free == 0 and masked != 0, (free, masked)
    print("ok  edge cases: unreachable->direct, mask forbids first hop")


ALL = [test_hard_cost_to_goal_matches_floyd, test_dijkstra_first_hop_is_optimal,
       test_astar_matches_dijkstra_and_expands_less, test_greedy_picks_nearest_reachable,
       test_edge_cases]

if __name__ == "__main__":
    for t in ALL:
        t()
    print(f"\nALL {len(ALL)} pathfind tests passed.")
