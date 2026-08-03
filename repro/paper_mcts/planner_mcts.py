"""Route A: MCTS-over-landmarks as a test-time planner for the paper repo, with
two noise regimes matched to the reimpl (l3p/planning/mcts_planner.py):

  * E1a (stochastic, Loai-2): additive zero-mean noise on the landmark edge matrix,
    one draw per episode builds a NOISY value-to-go d_c2g shared by both planners;
    MCTS additionally RESAMPLES noise on rollout edges (sample-averaging).
  * E1c (bias, Loai-1): a FIXED per-episode multiplicative bias (a systematic
    "wormhole") that averaging cannot remove. `mcts_fb` adds EXECUTION FEEDBACK
    (snap-to-nearest-landmark, EMA-correct the realized edge cost, blacklist
    repeatedly-failing edges) and re-plans on the corrected graph -- the mechanism
    that detects and routes around the wormhole that soft-Floyd trusts forever.

Sign bridge (paper V<=0 value vs engine D>=0 distance): the engine's edge value_fn
gets `-M`; d_s2c / d_c2g are passed as-is (already negative). sigma=0 & bias=0
reduce to the clean case -> match soft-Floyd (~checkpoint success) = the port sanity.

Feedback needs `achieved_goal`, which the paper `get_subgoals(ob, bg)` does not
receive -> the E1c eval loop (eval_ablation.py) passes it via the extra kwarg;
E1a keeps using the paper's own run_test_env_plan_eval (achieved_goal=None).
"""
import numpy as np
import torch

from rl.search.latent_planner import (Planner, clip_dist, v_pairwise_dists,
                                       value_iter, adaptive_clip_dist)

from mcts_core import LandmarkMCTS, MatrixNoisyValueFn


class MctsCfg:
    def __init__(self, **kw):
        self.mcts_n_simulations = kw.get("n_simulations", 120)
        self.mcts_c_uct = kw.get("c_uct", 1.4)
        self.mcts_rollout_horizon = kw.get("rollout_horizon", 8)
        self.mcts_cap_rollout_by_heuristic = kw.get("cap_rollout", True)
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
    def configure_mcts(self, mcts_cfg, sigma=0.0, select_mode="mcts", regime="e1a",
                       feedback=False, rho=0.5, r_max=2, noise_seed=0):
        self._mcts_cfg = mcts_cfg
        self._sigma = float(sigma)           # E1a additive std / E1c bias std
        self._select_mode = select_mode      # "mcts" or "softfloyd"
        self._regime = regime                # "e1a" or "e1c"
        self._feedback = bool(feedback) and sigma > 0.0   # sigma=0 -> deterministic control
        self._rho = rho
        self._r_max = r_max
        self._noise_seed = int(noise_seed)
        self._rng = np.random.default_rng(noise_seed + 1)

    # ---- graph build helpers ----
    def _clamp_dtg(self, M_land):
        """value_iter on a landmark->node matrix M_land (n, n+K), <=0 -> d_c2g (K, n+K)."""
        n, K = self.n_landmarks, self.n_goals
        full = np.full((n + K, n + K), -self.args.inf_value, dtype=np.float64)
        full[:n, :] = np.minimum(M_land, 0.0)          # paper value_iter asserts <= 0
        ft = torch.as_tensor(full, dtype=torch.float32)
        ft = adaptive_clip_dist(ft, clip=self.args.dist_clip, inf_value=self.args.inf_value)
        ft = value_iter(ft, temp=self.args.temp, n_iter=self.args.vi_iter)
        return ft[:, -K:].permute(1, 0).detach().cpu().numpy()

    def update(self, goals, test_time=False):
        super().update(goals, test_time=test_time)
        lm_only = self.landmarks[:self.n_landmarks]
        with torch.no_grad():
            M = v_pairwise_dists(lm_only, self.landmarks, agent=self.agent)
        self._edge_clean = torch.min(M, M * 0.0).detach().cpu().numpy()   # (n, n+K) <= 0
        n = self.n_landmarks
        # per-episode noise on the landmark edges
        if self._sigma <= 0:
            self._edge_noisy = self._edge_clean
        elif self._regime == "e1c":                    # fixed multiplicative bias
            b = self._rng.normal(0.0, self._sigma, size=self._edge_clean.shape)
            self._edge_noisy = np.minimum(self._edge_clean * (1.0 + b), 0.0)
        else:                                          # e1a additive
            self._edge_noisy = np.minimum(
                self._edge_clean + self._rng.normal(0.0, self._sigma, self._edge_clean.shape), 0.0)
        self._noisy_dtg = self._clamp_dtg(self._edge_noisy)
        # feedback episode state
        self._v_exec = {}
        self._blacklist = set()
        self._attempts = {}
        self._macro_start_ag = None
        self._cur_j = None
        self._macro_start_i = None

    # ---- substrate distance (goal space, edge_value_fn) ----
    def _sub_dist(self, ga, gb):
        ta = torch.as_tensor(np.atleast_2d(ga), dtype=torch.float32)
        tb = torch.as_tensor(np.atleast_2d(gb), dtype=torch.float32)
        with torch.no_grad():
            return float(self.agent.get_vf_value(ta, tb).detach().cpu().numpy().reshape(-1)[0])

    def _lm_xy(self):
        return self.landmarks[:self.n_landmarks].detach().cpu().numpy()

    def _snap(self, ag):
        lm = self._lm_xy()
        d = np.array([self._sub_dist(ag, lm[i]) for i in range(lm.shape[0])])
        return int(np.argmax(d))                       # V<=0, higher=closer

    # ---- E1c execution feedback ----
    def _tau(self, frac):
        return frac * abs(self.args.dist_clip)         # substrate-scale threshold

    def _observe(self, env_id, ag_now):
        i, j = self._macro_start_i, self._cur_j
        if j is None:
            return
        n = self.n_landmarks
        c_j = (self.landmarks[n + env_id] if j >= n else self.landmarks[j]).detach().cpu().numpy()
        v_reach = -self._sub_dist(ag_now, c_j)         # >=0 distance to subgoal (V negated)
        start = self._macro_start_ag if self._macro_start_ag is not None else ag_now
        travelled = -self._sub_dist(start, ag_now)
        realized = -(travelled + max(0.0, v_reach))    # back to <=0 value scale
        if i is not None and j < n:
            prev = self._v_exec.get((i, j), float(self._edge_noisy[i, j]))
            self._v_exec[(i, j)] = (1 - self._rho) * prev + self._rho * realized
        reached = v_reach <= self._tau(0.25)
        if reached:
            self._attempts.clear()
        elif travelled >= self._tau(0.5):              # progressed elsewhere
            self._attempts[j] = self._attempts.get(j, 0) + 1
            if self._attempts[j] >= self._r_max and j < n:
                self._blacklist.add(j)
        elif j < n:                                    # stuck -> blacklist
            self._blacklist.add(j)
        self._cur_j = None

    def _corrected_edges(self):
        """biased landmark edge matrix with EMA V_exec corrections overlaid."""
        M = self._edge_noisy.copy()
        for (i, j), v in self._v_exec.items():
            if i < M.shape[0] and j < M.shape[1]:
                M[i, j] = min(v, 0.0)
        return M

    # ---- selection ----
    def _submatrix(self, edge_land, env_id):
        n = self.n_landmarks
        sub = np.zeros((n + 1, n + 1), dtype=np.float64)
        sub[:n, :n] = edge_land[:n, :n]
        sub[:n, n] = edge_land[:n, n + env_id]
        return sub

    def _mask(self, env_id):
        n = self.n_landmarks
        m = np.zeros(n + 1, dtype=bool)
        prev = self.past_goal[env_id]
        if prev != -1 and prev < n:
            m[prev] = True
        for b in self._blacklist:                      # never mask the goal node
            if b < n:
                m[b] = True
        return m

    def _mcts_select(self, env_id, d_s2c_row, heur_row, edge_land):
        n = self.n_landmarks
        cols = list(range(n)) + [n + env_id]
        d_s2c = d_s2c_row[cols].detach().cpu().numpy().astype(np.float64)
        heur = np.asarray(heur_row)[cols].astype(np.float64)
        sub = self._submatrix(edge_land, env_id)
        adm = (sub >= self.args.dist_clip)
        np.fill_diagonal(adm, False)
        adm[n, :] = False
        value_fn = MatrixNoisyValueFn(-sub, sigma=(self._sigma if self._regime == "e1a" else 0.0),
                                      rng=self._rng)
        nodes_t = self.landmarks[cols].detach().float()
        value_fn.prepare_nodes(nodes_t)
        mcts = LandmarkMCTS(n_landmarks=n, value_fn=value_fn, nodes_t=nodes_t,
                            d_c2g_heuristic=heur, cfg=self._mcts_cfg, rng=self._rng,
                            admissible=adm)
        idx, _ = mcts.search(d_s2c, mask=self._mask(env_id))
        return n if idx is None else int(idx)

    def _softfloyd_select(self, env_id, d_s2c_row, heur_row):
        n = self.n_landmarks
        cols = list(range(n)) + [n + env_id]
        score = d_s2c_row[cols].detach().cpu().numpy() + np.asarray(heur_row)[cols]
        prev = self.past_goal[env_id]
        if prev != -1 and prev < n:
            score[prev] = -self.args.inf_value
        return int(np.argmax(score))

    def get_subgoals(self, obs, goals, achieved_goal=None):
        obs = self.to_2d_array(obs)
        goals = self.to_2d_array(goals).copy()
        landmarks = self.landmarks
        obs_t = self.to_tensor(obs)[:, None, :].repeat(1, landmarks.size(0), 1)
        lm_rep = landmarks[None, :, :].repeat(obs_t.size(0), 1, 1)
        with torch.no_grad():
            d2l = self.agent.pairwise_value(obs_t, lm_rep).reshape(goals.shape[0], -1)
        d2l = clip_dist(d2l, clip=self.args.dist_clip, inf_value=self.args.inf_value)
        n = self.n_landmarks
        extra = 1.0
        ag = None if achieved_goal is None else self.to_2d_array(achieved_goal)
        for env_id in range(obs_t.size(0)):
            gidx = n + env_id
            if self.subgoal_cnt[env_id] > 1.0:                 # mid-commitment
                goals[env_id] = self.landmarks[self.past_goal[env_id]].detach().cpu().numpy()
                self.subgoal_cnt[env_id] -= 1.0
                continue
            if self._feedback and ag is not None and self._cur_j is not None:
                self._observe(env_id, ag[env_id])              # finalize previous macro
            if d2l[env_id, gidx] < -self.args.local_horizon:
                if self._feedback:
                    edge_land = self._corrected_edges()
                    heur = self._clamp_dtg(edge_land)[env_id]
                else:
                    edge_land, heur = self._edge_noisy, self._noisy_dtg[env_id]
                if self._select_mode == "softfloyd":
                    idx = self._softfloyd_select(env_id, d2l[env_id], heur)
                else:
                    idx = self._mcts_select(env_id, d2l[env_id], heur, edge_land)
                if idx < n:
                    steps = float(-d2l[env_id, idx].cpu().numpy())
                    goals[env_id] = self.landmarks[idx].detach().cpu().numpy()
                    self.past_goal[env_id] = idx
                    self.subgoal_cnt[env_id] = steps + extra
                else:
                    steps = float(-d2l[env_id, gidx].cpu().numpy())
                    self.past_goal[env_id] = gidx
                    self.subgoal_cnt[env_id] = max(1.0, steps) + extra
                if self._feedback and ag is not None:          # open new macro
                    self._cur_j = idx
                    self._macro_start_ag = ag[env_id].copy()
                    self._macro_start_i = self._snap(ag[env_id])
        return goals
