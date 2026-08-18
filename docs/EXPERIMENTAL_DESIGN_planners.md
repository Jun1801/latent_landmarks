# Experimental design — planner comparison under a noisy world-model

Protocol for the multi-planner ablation on the three L³P paper environments, written to be
publication-credible. It states exactly what is controlled, how significance is established,
and what the honest limitations are.

## 1. Research question
On a learned latent-landmark graph whose edge estimates `V` are noisy, **how does the routing
algorithm affect robustness, and at what noise level does every planner fail?** We isolate the
*routing algorithm* as the single independent variable: all planners consume the **same** noisy
graph; only how they turn it into a sub-goal differs.

## 2. Environments (paper stack, Kaggle)
| Env | Type | Checkpoint | Planner load-bearing? |
|---|---|---|---|
| AntMaze-v1 | long-horizon navigation | `antmaze_s221_paper` | yes (test-plan ≫ HER) |
| BoxDistractor (Box-aside-v0) | long-horizon manipulation | `boxdistractor_s829` | yes |
| FetchPickAndPlace-v1 | short-horizon control | `fetch_s967` | no (negative control) |

## 3. Planners (all on the same noisy graph)
- **soft_floyd** — L³P: greedy `argmax(d_s2c + d_c2g)`, `d_c2g` from *soft* value-iteration (averages over paths).
- **dijkstra** — exact hard shortest path (first hop); commits to a single min-cost path (= hard-Floyd).
- **astar** — Dijkstra + admissible V-heuristic; **same path as Dijkstra** → identical success, fewer node expansions (latency axis; also a cross-check).
- **greedy** — myopic best-first (nearest-to-goal reachable landmark, ignores multi-hop).
- **mcts** (+`suffix`/`pw`/`bayes`) — UCT + sampled rollouts; **mcts_nofb / mcts_fb** = without / with execution feedback (E1c).

Pure-numpy classical planners: `repro/paper_mcts/pathfind.py` (unit-tested vs Floyd-Warshall/brute-force on 1800+ random graphs, `test_pathfind.py`).

## 4. Noise model (`l3p`/paper harness)
Two bases, testing different capabilities; injected **only where the planner looks** (edge estimate + rollout), the **real env is kept clean** so success is measured on true dynamics — a fair comparison in which every planner is misled by the *same* estimate.
- **E1a (stochastic):** `V_obs = V_true + η`, `η∼N(0,σ²)`, resampled — cured by sample-averaging.
- **E1c (bias):** `V_obs = V_true·(1+b)`, `b∼N(0,σ²)` **fixed per episode** — a systematic "wormhole"; cured by execution feedback. σ is a *fractional* distortion, comparable across envs.

## 5. Evaluation protocol
- **Paired.** Rollout *i* uses env-reset seed `pair_seed+i` for **every** planner, so all planners face the **identical start/goal** each episode; the bias sequence is already shared (seeded by `noise_seed`). → planners differ only in algorithm, on the same episodes and same noise. (`eval_ablation.py::eval_loop`; `--pair-seed`, default on.)
- **Per-episode logging.** Every rollout's 0/1 success is stored (`per_episode` in the JSON), enabling proper interval estimates rather than a point mean.
- **Replication.** E1c MCTS uses `--noise-seeds 0 1 2` (three independent bias draws), pooled.
- **σ grids.** E1a additive `{0,5,10,20[,40]}`; E1c bias `{0,0.05,…,0.5}` (AntMaze) / `{0,0.1,…,0.5}` (Fetch/Box) — pushed high to reach the collapse floor.
- **Sample size.** Classical (cheap): `episodes 3 × n_test_rollouts 60` = 180 paired rollouts/(planner,σ). MCTS (expensive): `3 × 30` = 90, ×3 noise seeds pooled = 270 for the E1c band. (Success≈0.5 ⇒ 95% CI half-width ≈ ±0.07 at N=180, ±0.06 at N=270.)
- **Shared graph geometry.** `dist_clip` / `d_max` admissibility is identical across planners (no per-planner tuning), so no planner wins by a better cutoff.

## 6. Statistics (`scripts/plot_planners.py`)
- **Paired bootstrap 95% CI** — resample the shared rollout indices `B=5000×`; every planner is resampled on the *same* indices, giving each planner's mean CI **and** the CI of the pairwise gap (mcts − soft_floyd). Report **mean [lo, hi]**.
- **σ\*** — the σ maximizing the (best-MCTS − soft_floyd) gap whose paired-gap CI **excludes 0** (a statistically significant advantage). This is "how far MCTS beats classical."
- **σ_all_fail** — the smallest σ at which **every** planner's mean success falls below the collapse threshold (default 0.25). This answers "at what noise does everything fail."
- Per-planner **knee** (<0.5) and **collapse** (<0.25) σ are reported for each.

## 7. Correctness gates (must pass, else stop)
1. **σ=0 tie** — with no noise every planner must ≈ soft_floyd(clean); a mismatch means a port/harness bug.
2. **astar == dijkstra success** at every σ — A* with an admissible heuristic returns Dijkstra's path; a divergence flags a heuristic-admissibility or implementation bug.
3. `test_pathfind.py` green (Dijkstra == brute-force optimal; A* optimal & fewer expansions; greedy/mask/cutoff correct).

## 8. Metrics
Primary: **test-plan success vs σ** (paired mean, 95% CI). Secondary: **planning latency** (ms/search; Dijkstra vs A* node expansions), sub-goal counts on the dumped plan-comparison episode.

## 9. Reproducibility
Seeds (`pair_seed`, `noise_seeds`), σ grid, `sims`, sample sizes, and the paired flag are all recorded in each output JSON; the pure-numpy planners are deterministic and unit-tested; notebooks pin the `retrain` commit via `git pull`.

## 10. Limitations (stated, not hidden)
- **Single training seed per env** (s221 / s829 / s967). The paired design + noise-seed band control the *noise* variance, **not** training-seed variance. A fully firm claim needs 2–3 training seeds per env (new training runs); until then results are reported as single-training-seed with that caveat explicit. This is the primary threat to external validity.
- **Synthetic noise.** σ is injected, not the natural error of a specific learned world-model; it is a controlled stand-in whose two bases (additive / bias) bracket the real failure modes.
- **Fetch is a negative control** (short-horizon, ~0 sub-goals) — expected to saturate, not to separate planners; it validates that noise cannot break planning where there is no planning to break.
- **A\* ≡ Dijkstra on success** by construction; it contributes only the latency axis.

## 11. How to run & report
1. `antmaze_classical.ipynb` + `antmaze_mcts.ipynb`, `fetch_ablation.ipynb`, `boxdistractor_ablation.ipynb` on Kaggle → per-phase JSON.
2. Locally: `python scripts/plot_planners.py --json <env>_<regime>_classical.json <env>_<regime>_mcts.json --out logs/exp_suite/<env>_<regime>_planners.png` → success-vs-σ overlay with CI bands, the markdown mean[CI] table, and σ\* / σ_all_fail / per-planner knee & collapse.
3. Paper table = mean [95% CI] per (planner, σ); headline numbers = σ\* (with gap CI) and σ_all_fail.
