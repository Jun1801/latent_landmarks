"""Route A, Step 2: MCTS-over-landmarks + a matched noisy soft-Floyd baseline as
test-time planners for the paper repo. `PaperMCTSPlanner` subclasses the paper's
`Planner` and overrides only the subgoal SELECTION in `get_subgoals`; the
commit-for-K-steps and mask-previous-landmark bookkeeping are kept identical.

Fair noise model (Loai-2, E1a-style additive on the landmark graph):
  * Per episode, one noise draw perturbs the clean landmark edge matrix, and the
    paper's own value_iter is run on it -> a NOISY value-to-go `d_c2g` that BOTH
    planners use. So the world model is mis-estimated identically for both.
  * soft_floyd (noisy): argmax(d_s2c + noisy d_c2g)      -- static, trusts it once.
  * mcts (noisy): heuristic = the SAME noisy d_c2g, but rollout/tree edges RESAMPLE
    the noise -> sample-averaging that static Soft Floyd cannot do.
  * d_s2c (state->landmark) stays clean, matching the reimpl E1a convention.

Sign bridge: paper V is a NEGATIVE value (higher=closer, admissible if V>=dist_clip);
our engine wants a POSITIVE distance D (edge cost=-D) with d_s2c/heuristic negative.
So the engine's edge value_fn gets `-M` (V negated) and d_s2c/d_c2g are passed as-is.
The sigma=0 run reduces to the clean case (noisy d_c2g == clean), so it still
matches Soft Floyd ~1.0 -- the port sanity.
"""
import numpy as np
import torch

from rl.search.latent_planner import (Planner, clip_dist, v_pairwise_dists,
                                       value_iter, adaptive_clip_dist)

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
    def configure_mcts(self, mcts_cfg, sigma=0.0, select_mode="mcts", noise_seed=0):
        self._mcts_cfg = mcts_cfg
        self._sigma = float(sigma)             # Loai-2 additive noise std (distance scale)
        self._select_mode = select_mode        # "mcts" or "softfloyd" (both on the noisy graph)
        self._rng = np.random.default_rng(noise_seed + 1)
        self._edge_clean = None
        self._noisy_dtg = None

    def update(self, goals, test_time=False):
        super().update(goals, test_time=test_time)
        lm_only = self.landmarks[:self.n_landmarks]
        with torch.no_grad():
            M = v_pairwise_dists(lm_only, self.landmarks, agent=self.agent)
        self._edge_clean = torch.min(M, M * 0.0).detach().cpu().numpy()   # (n, n+K), <= 0
        self._noisy_dtg = self._build_noisy_dtg()                          # (K, n+K)

    def _build_noisy_dtg(self):
        """Value-to-go from the paper's value_iter on a per-episode noisy graph.
        sigma=0 -> the clean d_c2g (so the port sanity is unchanged)."""
        if self._sigma <= 0:
            return self.dists_to_goals.detach().cpu().numpy()
        n, K = self.n_landmarks, self.n_goals
        Mn = self._edge_clean + self._rng.normal(0.0, self._sigma, size=self._edge_clean.shape)
        full = np.full((n + K, n + K), -self.args.inf_value, dtype=np.float64)
        full[:n, :] = Mn                                                   # goal rows stay -inf
        ft = torch.as_tensor(full, dtype=torch.float32)
        ft = adaptive_clip_dist(ft, clip=self.args.dist_clip, inf_value=self.args.inf_value)
        ft = value_iter(ft, temp=self.args.temp, n_iter=self.args.vi_iter)
        return ft[:, -K:].permute(1, 0).detach().cpu().numpy()            # (K, n+K)

    def _mcts_select(self, env_id, d_s2c_row, heur_row):
        n = self.n_landmarks
        cols = list(range(n)) + [n + env_id]
        d_s2c = d_s2c_row[cols].detach().cpu().numpy().astype(np.float64)
        heur = np.asarray(heur_row)[cols].astype(np.float64)
        sub = np.zeros((n + 1, n + 1), dtype=np.float64)
        sub[:n, :n] = self._edge_clean[:n, :n]
        sub[:n, n] = self._edge_clean[:n, n + env_id]
        adm = (sub >= self.args.dist_clip)                               # structural (clean) mask
        np.fill_diagonal(adm, False)
        adm[n, :] = False
        value_fn = MatrixNoisyValueFn(-sub, sigma=self._sigma, rng=self._rng)  # resampled rollouts
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

    def _softfloyd_select(self, env_id, d_s2c_row, heur_row):
        n = self.n_landmarks
        cols = list(range(n)) + [n + env_id]
        d_s2c = d_s2c_row[cols].detach().cpu().numpy()
        heur = np.asarray(heur_row)[cols]
        score = d_s2c + heur                                             # argmax = paper selection
        prev = self.past_goal[env_id]
        if prev != -1 and prev < n:
            score[prev] = -self.args.inf_value
        return int(np.argmax(score))

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
            if self.subgoal_cnt[env_id] > 1.0:
                goals[env_id] = self.landmarks[prev_idx].detach().cpu().numpy()
                self.subgoal_cnt[env_id] -= 1.0
                continue
            if dists_to_landmarks[env_id, env_goal_idx] < -self.args.local_horizon:
                heur_row = self._noisy_dtg[env_id]
                if self._select_mode == "softfloyd":
                    idx = self._softfloyd_select(env_id, dists_to_landmarks[env_id], heur_row)
                else:
                    idx = self._mcts_select(env_id, dists_to_landmarks[env_id], heur_row)
                if idx < n:
                    steps = float(-dists_to_landmarks[env_id, idx].cpu().numpy())
                    goals[env_id] = self.landmarks[idx].detach().cpu().numpy()
                    self.past_goal[env_id] = idx
                    self.subgoal_cnt[env_id] = steps + extra_steps
                else:
                    steps = float(-dists_to_landmarks[env_id, env_goal_idx].cpu().numpy())
                    self.past_goal[env_id] = env_goal_idx
                    self.subgoal_cnt[env_id] = max(1.0, steps) + extra_steps
        assert goals.ndim == 2
        return goals
