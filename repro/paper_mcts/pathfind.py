"""Pure-numpy classical planners over a noisy latent-landmark graph.

Deliberately free of any `rl.*` (wmag) import so it is unit-testable locally,
without the MuJoCo paper stack. `planner_mcts.PaperMCTSPlanner` wraps these in
thin `_dijkstra_select` / `_astar_select` / `_greedy_select` methods.

Conventions (matching planner_mcts / the L3P paper harness):
  * All quantities are VALUES <= 0; higher (closer to 0) = closer / cheaper.
  * cost = -value >= 0. An edge is admissible iff value >= `dist_clip`
    (dist_clip < 0); a non-admissible edge does not exist (cost = +inf).
  * Node indices: 0..n-1 are landmarks; the goal is a separate sink.
  * `d_s2c_full` has length n+1: [state->landmark_0 .. state->landmark_{n-1},
    state->goal]. `heur_full` likewise (heur_full[n] == 0 at the goal).
  * `edge_sub` is the (n, n) landmark->landmark value block; `goal_col` is the
    (n,) landmark->goal value column.
  * `mask` is a bool array of length n: masked landmarks may not be chosen as
    the immediate first hop (previous sub-goal + feedback blacklist).
"""
from __future__ import annotations

import heapq

import numpy as np

INF = float("inf")


def _edge_cost(edge_sub, dist_clip):
    """(n,n) landmark->landmark cost matrix; non-admissible edges -> +inf."""
    e = np.asarray(edge_sub, dtype=np.float64)
    C = -e
    C[e < dist_clip] = INF
    return C


def _to_goal_cost(goal_col, dist_clip):
    """(n,) landmark->goal cost; non-admissible -> +inf."""
    g = np.asarray(goal_col, dtype=np.float64)
    c = -g
    c[g < dist_clip] = INF
    return c


def hard_cost_to_goal(edge_sub, goal_col, dist_clip):
    """Exact hard shortest cost from every landmark to the goal (reverse Dijkstra).

    Returns a length-n VALUE vector (-cost, i.e. <= 0), with -inf where the goal
    is unreachable. This is the hard (min-plus) analogue of the soft value-iter
    `d_c2g`, and equals hard Floyd-Warshall's goal column.
    """
    C = _edge_cost(edge_sub, dist_clip)          # (n,n) i->j
    cg = _to_goal_cost(goal_col, dist_clip)       # (n,)  i->goal
    n = C.shape[0]
    dist = np.full(n, INF, dtype=np.float64)      # cost landmark_i -> goal
    pq = []
    for i in range(n):                            # seed: direct edges to goal
        if cg[i] < INF:
            dist[i] = cg[i]
            heapq.heappush(pq, (cg[i], i))
    while pq:                                     # relax predecessors u -> v -> goal
        d, v = heapq.heappop(pq)
        if d > dist[v]:
            continue
        preds = np.nonzero(C[:, v] < INF)[0]
        for u in preds:
            nd = C[u, v] + d
            if nd < dist[u]:
                dist[u] = nd
                heapq.heappush(pq, (nd, u))
    val = -dist
    val[dist == INF] = -INF
    return val


def _argmax_first_hop(score, mask):
    """argmax over [n landmarks + goal] with masked landmarks removed; the goal
    (last index) is never masked. Returns n (=go direct) if nothing is finite."""
    n = len(score) - 1
    s = np.array(score, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        s[:n][m] = -INF
    if not np.any(np.isfinite(s)):
        return n
    idx = int(np.argmax(s))
    return idx


def dijkstra_first_hop(d_s2c_full, edge_sub, goal_col, dist_clip, mask=None):
    """Hard shortest path state->goal; return the first landmark hop (n = direct).

    First hop = argmax_i [ d_s2c(state->i) + d_c2g_hard(i->goal) ], with the goal
    itself (going direct) as candidate n. Identical to hard-Floyd's greedy pick.
    """
    n = edge_sub.shape[0]
    dtg = hard_cost_to_goal(edge_sub, goal_col, dist_clip)     # (n,) value
    d = np.asarray(d_s2c_full, dtype=np.float64)
    score = np.empty(n + 1, dtype=np.float64)
    score[:n] = d[:n] + dtg                                    # via landmark i
    score[:n][d[:n] < dist_clip] = -INF                        # state can't reach i in one hop
    score[n] = d[n] if d[n] >= dist_clip else -INF             # go direct to goal (if admissible)
    return _argmax_first_hop(score, mask)


def astar_first_hop(d_s2c_full, edge_sub, goal_col, dist_clip, heur_full, mask=None):
    """Forward A* from the state to the goal; return (first_hop, n_expanded).

    Heuristic h(i) = max(0, -heur_full[i]) (soft d_c2g is a lower bound on the
    hard cost since soft-min <= min, so h is admissible -> A* returns the SAME
    optimal first hop as `dijkstra_first_hop`, expanding fewer nodes). n_expanded
    is a latency proxy.
    """
    n = edge_sub.shape[0]
    C = _edge_cost(edge_sub, dist_clip)
    cg = _to_goal_cost(goal_col, dist_clip)
    d = np.asarray(d_s2c_full, dtype=np.float64)
    cs = -d[:n]                                                # state->landmark cost
    cs[d[:n] < dist_clip] = INF
    if mask is not None:
        cs = cs.copy()
        cs[np.asarray(mask, dtype=bool)] = INF                 # forbid masked first hop
    cs_goal = -d[n] if d[n] >= dist_clip else INF              # state->goal direct
    GOAL = n
    h = np.zeros(n + 1, dtype=np.float64)
    if heur_full is not None:
        h[:n] = np.maximum(0.0, -np.asarray(heur_full, dtype=np.float64)[:n])
    g = np.full(n + 1, INF, dtype=np.float64)
    first = np.full(n + 1, -1, dtype=int)                      # first landmark on the path
    settled = np.zeros(n + 1, dtype=bool)
    pq = []
    for i in range(n):                                         # state -> landmark i
        if cs[i] < INF:
            g[i] = cs[i]
            first[i] = i
            heapq.heappush(pq, (g[i] + h[i], i))
    if cs_goal < INF:                                          # state -> goal direct
        g[GOAL] = cs_goal
        first[GOAL] = GOAL
        heapq.heappush(pq, (g[GOAL], GOAL))
    expanded = 0
    while pq:
        _, u = heapq.heappop(pq)
        if settled[u]:
            continue
        settled[u] = True
        expanded += 1
        if u == GOAL:
            break
        for v in range(n):                                     # u -> landmark v
            if C[u, v] < INF and g[u] + C[u, v] < g[v]:
                g[v] = g[u] + C[u, v]
                first[v] = first[u]
                heapq.heappush(pq, (g[v] + h[v], v))
        if cg[u] < INF and g[u] + cg[u] < g[GOAL]:             # u -> goal
            g[GOAL] = g[u] + cg[u]
            first[GOAL] = first[u]
            heapq.heappush(pq, (g[GOAL], GOAL))
    fh = int(first[GOAL])
    if fh < 0:
        return n, expanded                                     # unreachable -> go direct
    return (n if fh == GOAL else fh), expanded


def greedy_first_hop(d_s2c_full, heur_full, dist_clip, mask=None):
    """Myopic best-first: among landmarks reachable one hop from the state, pick
    the one NEAREST the goal (max heur), ignoring multi-hop cost. n = go direct."""
    n = len(d_s2c_full) - 1
    d = np.asarray(d_s2c_full, dtype=np.float64)
    heur = np.asarray(heur_full, dtype=np.float64)
    score = np.array(heur[:n], dtype=np.float64)
    score[d[:n] < dist_clip] = -INF                            # unreachable in one hop
    if mask is not None:
        score[np.asarray(mask, dtype=bool)] = -INF
    if not np.any(np.isfinite(score)):
        return n
    return int(np.argmax(score))
