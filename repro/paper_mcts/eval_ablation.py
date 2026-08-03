#!/usr/bin/env python3
"""Route A, Step 2: MCTS-vs-softFloyd ablation on a TRAINED paper agent, under a
shared per-episode noisy world model. Works for both Fetch (rl.main_latent_fetch)
and AntMaze (rl.main_latent) -- pass --env. AntMaze is the long-horizon env where
landmark planning is load-bearing (test-plan >> HER), so it is the meaningful test.

Loads networks only (no replay), swaps algo.planner.__class__ in place, and runs
run_test_env_plan_eval for soft_floyd + MCTS variants over a sigma sweep.

    cd /kaggle/working/wmag
    conda run -n l3p python .../eval_ablation.py --env antmaze \
        --resume_ckpt antmaze_s221_paper --episodes 3 --n_test_rollouts 30 --sigmas 0 2 5 10 20
"""
import argparse
import importlib
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
    p.add_argument("--no_cuda", action="store_true")
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

    def eval_paper(passes):                    # E1a: paper's own plan-eval loop
        return float(np.mean([algo.run_test_env_plan_eval() for _ in range(passes)]))

    def eval_feedback(passes):                 # E1c: custom loop passing achieved_goal
        env = algo.test_env if hasattr(algo, "test_env") else algo.env
        means = []
        for _ in range(passes):
            succ = trials = 0
            for _r in range(a.n_test_rollouts):
                o = env.reset(); ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
                algo.planner.reset(); algo.planner.update(goals=bg.copy(), test_time=True)
                info = None
                for _t in range(env._max_episode_steps):
                    sub = algo.planner.get_subgoals(ob, bg.copy(), achieved_goal=ag.copy())
                    act = algo.agent.get_actions(ob, sub)
                    o, _, _, info = env.step(act)
                    ob, bg, ag = o['observation'], o['desired_goal'], o['achieved_goal']
                if getattr(algo, "num_envs", 1) > 1:
                    for pe in info:
                        trials += 1; succ += int(pe['is_success'] == 1.0)
                else:
                    trials += 1; succ += int(info['is_success'] == 1.0)
            means.append(succ / max(1, trials))
        return float(np.mean(means))

    runner = eval_feedback if a.regime == "e1c" else eval_paper

    algo.planner.__class__ = Planner
    print(f"\nsoft_floyd (clean):            {eval_paper(a.episodes):.3f}", flush=True)

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

    print(f"\n[{a.regime}] {'variant':14}" + "".join(f"  s={s}".ljust(9) for s in a.sigmas), flush=True)
    for name, mode, fb, kw in variants:
        row = []
        for s in a.sigmas:
            algo.planner.__class__ = PaperMCTSPlanner
            algo.planner.configure_mcts(MctsCfg(n_simulations=a.sims, **kw), sigma=float(s),
                                        select_mode=mode, regime=a.regime, feedback=fb, noise_seed=0)
            row.append(runner(a.episodes))
        print(f"     {name:14}" + "".join(f"  {v:.3f}".ljust(9) for v in row), flush=True)

    print(f"\nSANITY (s=0): all ~match soft_floyd(clean). Under s>0: "
          + ("MCTS holds vs static soft_floyd?" if a.regime == "e1a"
             else "mcts_fb detects+routes around the bias vs soft_floyd/mcts_nofb?"), flush=True)


if __name__ == "__main__":
    main()
