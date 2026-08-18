#!/usr/bin/env python3
"""Route A, Step 2: MCTS-vs-softFloyd ablation on a TRAINED paper agent, under a
shared per-episode noisy world model. Works for Fetch (rl.main_latent_fetch),
AntMaze (rl.main_latent), and BoxDistractor (rl.main_latent_robot) -- pass --env.
AntMaze (nav) and BoxDistractor (manip) are the long-horizon envs where landmark
planning is load-bearing (test-plan >> HER), so they are the meaningful tests.

Loads networks only (no replay), swaps algo.planner.__class__ in place, and runs
run_test_env_plan_eval for soft_floyd + MCTS variants over a sigma sweep.

    cd /kaggle/working/wmag
    conda run -n l3p python .../eval_ablation.py --env antmaze \
        --resume_ckpt antmaze_s221_paper --episodes 3 --n_test_rollouts 30 --sigmas 0 2 5 10 20
"""
import argparse
import importlib
import json
import os
import sys

ENV_CFG = {
    "fetch": dict(
        get_args="rl.main_latent_fetch", launch="rl.launcher_latent_fetch",
        env_name="FetchPickAndPlace-v1", ckpt="fetch_s967",
        flags=["--env_name", "FetchPickAndPlace-v1", "--test_env_name", "FetchPickAndPlace-v1",
               "--seed", "967", "--n_cycles", "10", "--clip_inputs", "--normalize_inputs",
               "--gamma", "0.99", "--n_initial_rollouts", "0", "--plan_eps", "0.5",
               "--n_latent_landmarks", "80", "--latent_batch_size", "150", "--n_extra_landmark", "20",
               "--dist_clip", "-15.0", "--start_planning_n_traj", "6000", "--use_forward_empty_step",
               "--n_workers", "1", "--play"]),
    "antmaze": dict(
        get_args="rl.main_latent", launch="rl.launcher_latent",
        env_name="AntMaze-v1", ckpt="antmaze_s221_paper",
        flags=["--env_name", "AntMaze-v1", "--test_env_name", "AntMazeTest-v1",
               "--seed", "221", "--gamma", "0.98", "--clip_return", "100", "--future_step", "100",
               "--n_extra_landmark", "150", "--dist_clip", "-20.0", "--n_latent_landmarks", "50",
               "--latent_batch_size", "256", "--batch_size", "1000",
               "--grad_value_clipping", "-1.0", "--grad_norm_clipping", "15.0",
               "--action_l2", "0.05", "--optimize_every", "2",
               "--n_workers", "1", "--play"]),
    "boxdistractor": dict(
        get_args="rl.main_latent_robot", launch="rl.launcher_latent_robot",
        env_name="Box-aside-v0", ckpt="boxdistractor_s829",
        flags=["--env_name", "Box-aside-v0", "--test_env_name", "Box-aside-v0",
               "--seed", "829", "--n_cycles", "15", "--clip_inputs", "--normalize_inputs",
               "--gamma", "0.99", "--n_initial_rollouts", "0", "--plan_eps", "0.5",
               "--n_latent_landmarks", "80", "--latent_batch_size", "150", "--n_extra_landmark", "20",
               "--dist_clip", "-15.0", "--start_planning_n_traj", "6000", "--use_forward_empty_step",
               "--n_workers", "1", "--play"]),
}


# name -> (select_mode, feedback, kw). Classical planners are static (no feedback).
PLANNER_REGISTRY = {
    "soft_floyd":  ("softfloyd", False, {}),
    "dijkstra":    ("dijkstra",  False, {}),
    "astar":       ("astar",     False, {}),
    "greedy":      ("greedy",    False, {}),
    "mcts":        ("mcts",      False, {}),
    "mcts+suffix": ("mcts",      False, {"suffix_backup": True}),
    "mcts+pw":     ("mcts",      False, {"progressive_widening": True}),
    "mcts+bayes":  ("mcts",      False, {"uncertainty_mode": "bayes"}),
    "mcts_nofb":   ("mcts",      False, {}),
    "mcts_fb":     ("mcts",      True,  {}),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(ENV_CFG), default="antmaze")
    p.add_argument("--regime", choices=["e1a", "e1c"], default="e1a",
                   help="e1a=stochastic (paper eval loop); e1c=fixed bias + execution feedback")
    p.add_argument("--repo", default="/kaggle/working/wmag")
    p.add_argument("--paper_mcts_dir", default="/kaggle/working/latent_landmarks/repro/paper_mcts")
    p.add_argument("--resume_ckpt", default=None, help="default = env's standard ckpt name")
    p.add_argument("--save_dir", default="/kaggle/working/experiments")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--n_test_rollouts", type=int, default=30)
    p.add_argument("--sims", type=int, default=120)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 2.0, 5.0, 10.0, 20.0])
    p.add_argument("--noise-seeds", type=int, nargs="+", default=[0],
                   help="noise seeds to average per (variant, sigma); >1 gives a spread band "
                        "(was hardcoded to a single seed 0)")
    p.add_argument("--planners", nargs="+", default=None,
                   help=f"explicit planner set (subset of {sorted(PLANNER_REGISTRY)}); "
                        "default = the regime's built-in list")
    p.add_argument("--pair-seed", type=int, default=20260818,
                   help="env-reset seed base: rollout i uses pair_seed+i for EVERY planner, so "
                        "all planners see identical start/goal per episode (paired comparison)")
    p.add_argument("--no-pairing", action="store_true",
                   help="disable paired env seeding (each planner re-randomizes start/goal)")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--latency", action="store_true",
                   help="also report MCTS planning latency (ms/search) per variant")
    p.add_argument("--out", default=None, help="save the sweep table to JSON (for aggregate/plot)")
    p.add_argument("--dump", default=None,
                   help="save one episode's graph (clean/noisy edges, landmark xy, soft-Floyd "
                        "trajectory + subgoals) to JSON for the clean-vs-noisy figure")
    p.add_argument("--dump-sigma", type=float, default=None, help="sigma for --dump (default: max)")
    p.add_argument("--dump-plans", default=None,
                   help="dump ALL 3 planners' routes (soft_floyd/mcts_nofb/mcts_fb) on the same "
                        "episode+graph to JSON for the plan-comparison figure (E1c)")
    a = p.parse_args()
    ec = ENV_CFG[a.env]
    ckpt = a.resume_ckpt or ec["ckpt"]

    sys.path.insert(0, a.repo)
    sys.path.insert(0, a.paper_mcts_dir)
    os.chdir(a.repo)

    argv = ["eval"] + ec["flags"] + [
        "--ckpt_name", "eval_tmp", "--save_dir", a.save_dir,
        "--n_test_rollouts", str(a.n_test_rollouts),
    ]
    if not a.no_cuda:
        argv.append("--cuda")
    sys.argv = argv

    get_args = importlib.import_module(ec["get_args"]).get_args
    launch = importlib.import_module(ec["launch"]).launch
    from rl.search.latent_planner import Planner
    from planner_mcts import PaperMCTSPlanner, MctsCfg
    import numpy as np

    args = get_args()
    print(f"[ablation:{a.env}] reconstructing + loading trained networks (no replay) ...", flush=True)
    algo = launch(args)
    state_path = os.path.join(a.save_dir, ec["env_name"], ckpt, "state")
    try:
        algo.load(state_path)
    except Exception as e:
        print(f"  (algo.pt skipped: {e})", flush=True)
    algo.agent.load(state_path)
    if hasattr(algo, "_clusters_initialized"):
        algo._clusters_initialized = True

    def eval_paper(passes):                    # paper's own plan-eval loop (clean-sanity print only)
        return float(np.mean([algo.run_test_env_plan_eval() for _ in range(passes)]))

    def eval_loop(passes, pair_seed):          # unified controlled loop; returns (mean, per-episode 0/1)
        # Paired: rollout index i uses env seed pair_seed+i for EVERY planner, so all planners
        # face the SAME start/goal per episode (the bias sequence is already paired via noise_seed).
        env = algo.test_env if hasattr(algo, "test_env") else algo.env
        outcomes, idx = [], 0
        for _ in range(passes):
            for _r in range(a.n_test_rollouts):
                if pair_seed is not None:
                    try:
                        env.seed(pair_seed + idx)
                    except Exception:
                        pass
                o = env.reset(); ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
                algo.planner.reset(); algo.planner.update(goals=bg.copy(), test_time=True)
                info = None
                for _t in range(env._max_episode_steps):
                    sub = algo.planner.get_subgoals(ob, bg.copy(), achieved_goal=ag.copy())
                    act = algo.agent.get_actions(ob, sub)
                    o, _, _, info = env.step(act)
                    ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
                if getattr(algo, "num_envs", 1) > 1:
                    outcomes.extend(int(pe['is_success'] == 1.0) for pe in info)
                else:
                    outcomes.append(int(info['is_success'] == 1.0))
                idx += 1
        return (float(np.mean(outcomes)) if outcomes else 0.0), outcomes

    runner = eval_loop                         # both regimes use the paired per-episode loop

    algo.planner.__class__ = Planner
    print(f"\nsoft_floyd (clean, paper loop): {eval_paper(a.episodes):.3f}", flush=True)

    if a.regime == "e1a":                      # (name, select_mode, feedback, kw)
        variants = [("soft_floyd", "softfloyd", False, dict()),
                    ("mcts",        "mcts",      False, dict()),
                    ("mcts+suffix", "mcts",      False, dict(suffix_backup=True)),
                    ("mcts+pw",     "mcts",      False, dict(progressive_widening=True)),
                    ("mcts+bayes",  "mcts",      False, dict(uncertainty_mode="bayes"))]
    else:                                      # e1c: bias + execution feedback
        variants = [("soft_floyd", "softfloyd", False, dict()),
                    ("mcts_nofb",  "mcts",      False, dict()),
                    ("mcts_fb",    "mcts",      True,  dict())]

    if a.planners:                             # explicit override from the registry
        bad = [x for x in a.planners if x not in PLANNER_REGISTRY]
        if bad:
            p.error(f"unknown --planners {bad}; choose from {sorted(PLANNER_REGISTRY)}")
        variants = [(name, *PLANNER_REGISTRY[name]) for name in a.planners]

    pair_seed = None if a.no_pairing else int(a.pair_seed)
    results = {"env": a.env, "regime": a.regime, "sigmas": list(a.sigmas),
               "sims": a.sims, "noise_seeds": list(a.noise_seeds),
               "paired": bool(pair_seed is not None), "pair_seed": a.pair_seed,
               "n_rollouts": a.episodes * a.n_test_rollouts,
               "variants": {}, "per_seed": {}, "per_episode": {}, "latency": {}}
    print(f"\n[{a.regime}] {'variant':14}" + "".join(f"  s={s}".ljust(9) for s in a.sigmas), flush=True)
    for name, mode, fb, kw in variants:
        row, lat, per_seed, per_ep = [], [], [], []
        for s in a.sigmas:
            seed_vals, seed_outs = [], []
            for ns in a.noise_seeds:
                algo.planner.__class__ = PaperMCTSPlanner
                algo.planner.configure_mcts(MctsCfg(n_simulations=a.sims, **kw), sigma=float(s),
                                            select_mode=mode, regime=a.regime, feedback=fb,
                                            noise_seed=int(ns))
                m, outs = runner(a.episodes, pair_seed)
                seed_vals.append(m); seed_outs.append(outs)
            row.append(float(np.mean(seed_vals)))
            per_seed.append(seed_vals)
            per_ep.append(seed_outs)          # [sigma_idx][seed_idx] -> list of 0/1 (paired by index)
            calls = getattr(algo.planner, "search_calls", 0)
            lat.append(1000.0 * algo.planner.search_seconds / calls if calls else 0.0)
        results["variants"][name] = row
        results["per_seed"][name] = per_seed
        results["per_episode"][name] = per_ep
        results["latency"][name] = lat
        print(f"     {name:14}" + "".join(f"  {v:.3f}".ljust(9) for v in row), flush=True)
        if len(a.noise_seeds) > 1:
            spread = ["[%.2f,%.2f]" % (min(v), max(v)) for v in per_seed]
            print(f"     {'  ^ [min,max]':14}" + "".join(f"  {v}".ljust(13) for v in spread), flush=True)
        if a.latency:
            print(f"     {'  ^ ms/search':14}" + "".join(f"  {v:.0f}".ljust(9) for v in lat), flush=True)

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"\nsaved sweep -> {a.out}", flush=True)

    if a.dump:
        _dump_episode(algo, a, np, PaperMCTSPlanner, MctsCfg,
                      a.dump_sigma if a.dump_sigma is not None else max(a.sigmas))

    if a.dump_plans:
        _dump_plans(algo, a, np, PaperMCTSPlanner, MctsCfg,
                    a.dump_sigma if a.dump_sigma is not None else max(a.sigmas))

    print(f"\nSANITY (s=0): all ~match soft_floyd(clean). Under s>0: "
          + ("MCTS holds vs static soft_floyd?" if a.regime == "e1a"
             else "mcts_fb detects+routes around the bias vs soft_floyd/mcts_nofb?"), flush=True)


def _dump_episode(algo, a, np, PaperMCTSPlanner, MctsCfg, sigma):
    """Record one episode's graph (clean/noisy edges, landmark xy) + soft-Floyd-style
    achieved trajectory + chosen sub-goals for the clean-vs-noisy figure."""
    algo.planner.__class__ = PaperMCTSPlanner
    algo.planner.configure_mcts(MctsCfg(n_simulations=a.sims), sigma=float(sigma),
                                select_mode="mcts", regime=a.regime, feedback=False, noise_seed=0)
    env = algo.test_env if hasattr(algo, "test_env") else algo.env
    o = env.reset(); ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
    algo.planner.reset(); algo.planner.update(goals=bg.copy(), test_time=True)
    xy = lambda v: np.asarray(v).reshape(-1)[:2].tolist()
    traj, subs, info = [xy(ag)], [], None
    for _t in range(env._max_episode_steps):
        sub = algo.planner.get_subgoals(ob, bg.copy(), achieved_goal=ag.copy())
        pg = algo.planner.past_goal.get(0, -1)
        if pg != -1 and (not subs or subs[-1] != pg):
            subs.append(int(pg))
        o, r, d, info = env.step(algo.agent.get_actions(ob, sub))
        ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
        traj.append(xy(ag))
        i0 = info[0] if isinstance(info, (list, tuple)) else info
        if i0.get('is_success') == 1.0:
            break
    n = algo.planner.n_landmarks
    lm = algo.planner.landmarks[:n].detach().cpu().numpy()[:, :2]
    dump = dict(env=a.env, regime=a.regime, sigma=float(sigma), n=int(n),
                landmark_xy=lm.tolist(), edge_clean=algo.planner._edge_clean.tolist(),
                edge_noisy=algo.planner._edge_noisy.tolist(),
                traj=traj, subgoals=subs, goal=xy(bg), start=traj[0])
    os.makedirs(os.path.dirname(a.dump) or ".", exist_ok=True)
    json.dump(dump, open(a.dump, "w"))
    print(f"dumped graph+trajectory -> {a.dump}  (n={n}, steps={len(traj)}, subgoals={len(subs)})")


def _dump_plans(algo, a, np, PaperMCTSPlanner, MctsCfg, sigma):
    """Record all THREE planners' routes on the SAME episode + graph (E1c) for the
    plan-comparison figure. Each planner runs from the same seeded reset (same
    start/goal) and the same fixed bias (noise_seed=0)."""
    env = algo.test_env if hasattr(algo, "test_env") else algo.env
    specs = [("soft_floyd", "softfloyd", False), ("mcts_nofb", "mcts", False),
             ("mcts_fb", "mcts", True)]
    out = {"env": a.env, "regime": a.regime, "sigma": float(sigma), "planners": {}}
    xy = lambda v: np.asarray(v).reshape(-1)[:2].tolist()
    for name, mode, fb in specs:
        algo.planner.__class__ = PaperMCTSPlanner
        algo.planner.configure_mcts(MctsCfg(n_simulations=a.sims), sigma=float(sigma),
                                    select_mode=mode, regime=a.regime, feedback=fb, noise_seed=0)
        try:
            env.seed(1234)                                 # same start/goal across planners
        except Exception:
            pass
        o = env.reset(); ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
        algo.planner.reset(); algo.planner.update(goals=bg.copy(), test_time=True)
        traj, subs, aims, succ, info = [xy(ag)], [], [], 0, None
        for _t in range(env._max_episode_steps):
            pos = xy(ag)
            sub = algo.planner.get_subgoals(ob, bg.copy(), achieved_goal=ag.copy())
            pg = algo.planner.past_goal.get(0, -1)
            if pg != -1 and (not subs or subs[-1] != pg):
                subs.append(int(pg)); aims.append([pos, int(pg)])
            o, r, d, info = env.step(algo.agent.get_actions(ob, sub))
            ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
            traj.append(xy(ag))
            i0 = info[0] if isinstance(info, (list, tuple)) else info
            if i0.get('is_success') == 1.0:
                succ = 1; break
        if "landmark_xy" not in out:                       # graph is shared (same episode/bias)
            n = algo.planner.n_landmarks
            out.update(n=int(n),
                       landmark_xy=algo.planner.landmarks[:n].detach().cpu().numpy()[:, :2].tolist(),
                       edge_clean=algo.planner._edge_clean.tolist(),
                       edge_noisy=algo.planner._edge_noisy.tolist(),
                       goal=xy(bg), start=traj[0])
        out["planners"][name] = dict(traj=traj, subgoals=subs, aims=aims, success=succ)
        print(f"  {name}: {'REACHED' if succ else 'FAILED'} ({len(subs)} subgoals)", flush=True)
    os.makedirs(os.path.dirname(a.dump_plans) or ".", exist_ok=True)
    json.dump(out, open(a.dump_plans, "w"))
    print(f"dumped 3-planner plans -> {a.dump_plans}")


if __name__ == "__main__":
    main()
