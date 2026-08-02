#!/usr/bin/env python3
"""Route A, Step 2: MCTS-vs-softFloyd ablation on the TRAINED paper Fetch agent.

Loads the trained model (networks only, no replay), then evaluates the paper's own
soft-Floyd planner and our LandmarkMCTS planner via the same
`Algo.run_test_env_plan_eval()` entry. The planner is swapped in place by
reassigning `algo.planner.__class__` (so all of the Planner's constructed state --
agent / replay / monitor / args -- is reused untouched).

First it runs the sigma=0 SANITY: MCTS with no injected noise must match soft-Floyd
(~1.0). If that holds, the sign/graph plumbing is correct; then it sweeps additive
noise to see whether MCTS lookahead is more robust than the static plan.

    cd /kaggle/working/wmag
    conda run -n l3p python /kaggle/working/latent_landmarks/repro/paper_mcts/eval_ablation.py \
        --resume_ckpt fetch_s967 --episodes 3 --n_test_rollouts 30
"""
import argparse
import os
import sys

PICK_FLAGS = [
    "--env_name", "FetchPickAndPlace-v1", "--test_env_name", "FetchPickAndPlace-v1",
    "--seed", "967", "--n_cycles", "10", "--clip_inputs", "--normalize_inputs",
    "--gamma", "0.99", "--n_initial_rollouts", "0", "--plan_eps", "0.5",
    "--n_latent_landmarks", "80", "--latent_batch_size", "150", "--n_extra_landmark", "20",
    "--dist_clip", "-15.0", "--start_planning_n_traj", "6000", "--use_forward_empty_step",
    "--n_workers", "1", "--play",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="/kaggle/working/wmag")
    p.add_argument("--paper_mcts_dir", default="/kaggle/working/latent_landmarks/repro/paper_mcts")
    p.add_argument("--resume_ckpt", default="fetch_s967")
    p.add_argument("--save_dir", default="/kaggle/working/experiments")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--n_test_rollouts", type=int, default=30)
    p.add_argument("--sims", type=int, default=120)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--no_cuda", action="store_true")
    a = p.parse_args()

    sys.path.insert(0, a.repo)
    sys.path.insert(0, a.paper_mcts_dir)      # so `import mcts_core` / `planner_mcts` resolve
    os.chdir(a.repo)

    argv = ["eval"] + PICK_FLAGS + [
        "--ckpt_name", "eval_tmp", "--save_dir", a.save_dir,
        "--n_test_rollouts", str(a.n_test_rollouts),
    ]
    if not a.no_cuda:
        argv.append("--cuda")
    sys.argv = argv

    from rl.main_latent_fetch import get_args
    from rl.launcher_latent_fetch import launch
    from rl.search.latent_planner import Planner
    from planner_mcts import PaperMCTSPlanner, MctsCfg
    import numpy as np

    args = get_args()
    print("[ablation] reconstructing + loading trained networks (no replay) ...", flush=True)
    algo = launch(args)
    state_path = os.path.join(a.save_dir, args.env_name, a.resume_ckpt, "state")
    try:
        algo.load(state_path)
    except Exception as e:
        print(f"  (algo.pt skipped: {e})", flush=True)
    algo.agent.load(state_path)
    if hasattr(algo, "_clusters_initialized"):
        algo._clusters_initialized = True

    def eval_mean(passes):
        return float(np.mean([algo.run_test_env_plan_eval() for _ in range(passes)]))

    # ---- reference: paper soft-Floyd (clean) ----
    algo.planner.__class__ = Planner
    ref = eval_mean(a.episodes)
    print(f"\nsoft_floyd (clean):            {ref:.3f}", flush=True)

    # ---- planners under the SAME noisy world model (soft_floyd vs MCTS) ----
    # (name, select_mode, mcts kwargs)
    variants = [
        ("soft_floyd",   "softfloyd", dict()),
        ("mcts",         "mcts",      dict()),
        ("mcts+suffix",  "mcts",      dict(suffix_backup=True)),
        ("mcts+pw",      "mcts",      dict(progressive_widening=True)),
        ("mcts+bayes",   "mcts",      dict(uncertainty_mode="bayes")),
    ]
    print(f"\n{'variant':16}" + "".join(f"  s={s}".ljust(9) for s in a.sigmas), flush=True)
    for name, mode, kw in variants:
        row = []
        for s in a.sigmas:
            algo.planner.__class__ = PaperMCTSPlanner
            algo.planner.configure_mcts(MctsCfg(n_simulations=a.sims, **kw),
                                        sigma=float(s), select_mode=mode, noise_seed=0)
            row.append(eval_mean(a.episodes))
        print(f"{name:16}" + "".join(f"  {v:.3f}".ljust(9) for v in row), flush=True)

    print("\nSANITY: at s=0, soft_floyd/mcts must ~match soft_floyd(clean). "
          "Under s>0: does MCTS hold success better than static soft_floyd?", flush=True)


if __name__ == "__main__":
    main()
