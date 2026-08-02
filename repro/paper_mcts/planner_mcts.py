"""Route A, Step 2: MCTS-over-landmarks as a drop-in test-time planner for the
paper repo. `PaperMCTSPlanner` subclasses the paper's `Planner` and overrides only
the subgoal SELECTION inside `get_subgoals` (the commit-for-K-steps and
mask-previous-landmark bookkeeping are kept identical). Everything else -- the
landmark set, the soft-Floyd value-to-go (`dists_to_goals`), the state->landmark
distances (`agent.pairwise_value`) -- is reused from the trained model.

Sign convention bridge (paper vs our engine):
  * paper value V is a NEGATIVE value (higher = closer); admissible if V >= dist_clip.
  * our LandmarkMCTS expects a POSITIVE distance D (edge cost = -D) and d_s2c /
    heuristic as NEGATIVE values-to-go.
  So we feed the engine `-M` (V negated -> positive distance) as the edge value_fn,
  and pass `dists_to_landmarks` / `dists_to_goals` unchanged (already negative).
The sigma=0 sanity (MCTS ~ Soft Floyd ~ 1.0) catches any sign mistake immediately.
"""
import numpy as np
import torch

from rl.search.latent_planner import Planner, clip_dist, v_pairwise_dists

from mcts_core import LandmarkMCTS, MatrixNoisyValueFn


class MctsCfg:
    """Minimal config object carrying the mcts_* fields LandmarkMCTS reads."""
    def __init__(self, **kw):
        self.mcts_n_simulations = kw.get("n_simulations", 120)
        self.mcts_c_uct = kw.get("c_uct", 1.4)
        self.mcts_rollout_horizon = kw.get("rollout_horizon", 8)
        self.mcts_cap_rollout_by_heuristic = kw.get("cap_rollout", True)   # V substrate -> cap on
        self.mcts_suffix_backup = kw.get("suffix_backup", False)
        self.mcts_progressive_widening = kw.get("progressive_widening", False)
        self.mcts_pw_c = kw.get("pw_c", 1.0)
        self.mcts_pw_alpha = kw.get("pw_alpha", 0.5)
        self.mcts_uncertainty_mode = kw.get("uncertainty_mode", "none")
        self.mcts_beta_uncertainty = kw.get("beta_uncertainty", 1.0)
        self.mcts_bayes_sigma0 = kw.get("bayes_sigma0", 1.0)
        self.mcts_bayes_n0 = kw.get("bayes_n0", 1.0)
        self.mcts_lambda_risk = kw.get("lambda_risk", 0.0)
        self.mcts_lambda_goal = kw.get("lambda_goal", 0.0)


class PaperMCTSPlanner(Planner):
    """Planner that selects sub-goals with LandmarkMCTS instead of the softmax
    over `dists_to_landmarks + dists_to_goals`. Noise (Loai-2 additive `sigma`,
    Loai-1 fixed `bias`) is injected on the cached clean edge matrix per episode
    to mimic a mis-estimated world model (E1a / E1c)."""

    def configure_mcts(self, mcts_cfg, sigma=0.0, bias_sigma=0.0, noise_seed=0):
        self._mcts_cfg = mcts_cfg
        self._sigma = float(sigma)              # Loai-2 additive noise std (on distance scale)
        self._bias_sigma = float(bias_sigma)    # Loai-1 fixed multiplicative bias std
        self._noise_seed = int(noise_seed)
        self._rng = np.random.default_rng(noise_seed + 1)
        self._edge_clean = None
        self._bias = None

    def update(self, goals, test_time=False):
        super().update(goals, test_time=test_time)
        # Cache the CLEAN landmark -> (landmark+goal) V matrix (paper's pre-value_iter
        # graph, clamped to <= 0 exactly like update()). self.landmarks == [lm; goals].
        lm_only = self.landmarks[:self.n_landmarks]
        with torch.no_grad():
            M = v_pairwise_dists(lm_only, self.landmarks, agent=self.agent)
        M = torch.min(M, M * 0.0).detach().cpu().numpy()            # (n_landmark, n_landmark+K)
        self._edge_clean = M
        # Per-episode fixed Loai-1 bias over the (N+1)-node graph (goal is last node).
        if self._bias_sigma > 0:
            n = self.n_landmarks + 1
            rng = np.random.default_rng(self._noise_seed)
            b = rng.normal(0.0, self._bias_sigma, size=(n, n))
            b[:, -1] = b[-1, :] = 0.0                               # no bias on goal edges
            self._bias = b
        else:
            self._bias = None

    def _mcts_select(self, env_id, d_s2c_row, heur_row):
        """Run MCTS over [landmarks; this env's goal]; return the chosen node idx
        (0..n_landmark-1 = landmark, n_landmark = goal)."""
        n = self.n_landmarks
        cols = list(range(n)) + [n + env_id]                       # N+1 node columns
        d_s2c = d_s2c_row[cols].detach().cpu().numpy().astype(np.float64)      # negative values
        heur = heur_row[cols].detach().cpu().numpy().astype(np.float64)        # negative values-to-go
        # clean edge submatrix over these N+1 nodes, negated to a positive distance
        sub = np.full((n + 1, n + 1), -1.0, dtype=np.float64) * 0.0
        sub[:n, :n] = self._edge_clean[:n, :n]
        sub[:n, n] = self._edge_clean[:n, n + env_id]
        Dpos = -sub                                                # positive distance for the engine
        adm = (sub >= self.args.dist_clip)                         # V >= clip  <=>  admissible
        np.fill_diagonal(adm, False)
        adm[n, :] = False                                          # goal is absorbing
        value_fn = MatrixNoisyValueFn(Dpos, sigma=self._sigma, bias=self._bias, rng=self._rng)
        nodes_t = self.landmarks[cols].detach().float()
        value_fn.prepare_nodes(nodes_t)
        mcts = LandmarkMCTS(n_landmarks=n, value_fn=value_fn, nodes_t=nodes_t,
                            d_c2g_heuristic=heur, cfg=self._mcts_cfg,
                            rng=self._rng, admissible=adm)
        mask = np.zeros(n + 1, dtype=bool)
        prev = self.past_goal[env_id]
        if prev != -1 and prev < n:
            mask[prev] = True
        idx, _ = mcts.search(d_s2c, mask=mask)
        return n if idx is None else int(idx)

    def get_subgoals(self, obs, goals):
        obs = self.to_2d_array(obs)
        goals = self.to_2d_array(goals).copy()
        landmarks = self.landmarks
        obs_t = self.to_tensor(obs)[:, None, :].repeat(1, landmarks.size(0), 1)
        lm_rep = landmarks[None, :, :].repeat(obs_t.size(0), 1, 1)
        with torch.no_grad():
            dists_to_landmarks = self.agent.pairwise_value(obs_t, lm_rep)
        n_goals = goals.shape[0]
        dists_to_landmarks = dists_to_landmarks.reshape(n_goals, -1)
        dists_to_landmarks = clip_dist(
            dists_to_landmarks, clip=self.args.dist_clip, inf_value=self.args.inf_value)

        n = self.n_landmarks
        extra_steps = 1.0
        for env_id in range(obs_t.size(0)):
            env_goal_idx = n + env_id
            prev_idx = self.past_goal[env_id]
            if self.subgoal_cnt[env_id] > 1.0:                     # mid-commitment: hold subgoal
                goals[env_id] = self.landmarks[prev_idx].detach().cpu().numpy()
                self.subgoal_cnt[env_id] -= 1.0
                continue
            if dists_to_landmarks[env_id, env_goal_idx] < -self.args.local_horizon:
                idx = self._mcts_select(env_id, dists_to_landmarks[env_id], self.dists_to_goals[env_id])
                if idx < n:                                        # a landmark sub-goal
                    steps = float(-dists_to_landmarks[env_id, idx].cpu().numpy())
                    goals[env_id] = self.landmarks[idx].detach().cpu().numpy()
                    self.past_goal[env_id] = idx
                    self.subgoal_cnt[env_id] = steps + extra_steps
                else:                                              # head straight to the goal
                    steps = float(-dists_to_landmarks[env_id, env_goal_idx].cpu().numpy())
                    self.past_goal[env_id] = env_goal_idx
                    self.subgoal_cnt[env_id] = max(1.0, steps) + extra_steps
            # else: goal is directly reachable -> keep goals[env_id] as the real goal
        assert goals.ndim == 2
        return goals
