#!/usr/bin/env python3
"""Route A, Step 1 (sanity): load the TRAINED paper Fetch agent and run the paper's
OWN soft-Floyd landmark planner eval, to confirm we can reconstruct + load the model
and reproduce ~1.0 plan success BEFORE plugging MCTS into `Planner.get_subgoals`.

Runs INSIDE the paper repo (`wmag/`), in the `l3p` conda env, on Kaggle (the Fetch
MuJoCo env needs the old stack). Reuses the repo's own `get_args()` + `launch()`, then
calls `algo.run_test_env_plan_eval()` (the exact call that prints
Test_TestEnv_PlanSuccessRate during training) a few times.

    cd /kaggle/working/wmag
    conda run -n l3p python /kaggle/working/latent_landmarks/repro/paper_mcts/eval_baseline.py \
        --resume_ckpt fetch_s967 --episodes 5

Flags mirror scripts/pick.sh so the reconstructed graph matches the trained model.
"""
import argparse
import os
import sys


# pick.sh hyper-parameters that define the landmark graph / planner (must match training)
PICK_FLAGS = [
    "--env_name", "FetchPickAndPlace-v1", "--test_env_name", "FetchPickAndPlace-v1",
    "--seed", "967", "--n_cycles", "10", "--clip_inputs", "--normalize_inputs",
    "--gamma", "0.99", "--n_initial_rollouts", "0", "--plan_eps", "0.5",
    "--n_latent_landmarks", "80", "--latent_batch_size", "150", "--n_extra_landmark", "20",
    "--dist_clip", "-15.0", "--start_planning_n_traj", "6000", "--use_forward_empty_step",
    "--play",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="/kaggle/working/wmag",
                   help="paper repo root (added to sys.path so `import rl` works)")
    p.add_argument("--resume_ckpt", default="fetch_s967")
    p.add_argument("--save_dir", default="/kaggle/working/experiments")
    p.add_argument("--episodes", type=int, default=5, help="plan-eval passes (each = n_test_rollouts)")
    p.add_argument("--n_test_rollouts", type=int, default=30)
    p.add_argument("--no_cuda", action="store_true")
    a = p.parse_args()

    # Running `python /abs/path/eval_baseline.py` puts THIS file's dir on sys.path,
    # not the paper repo -- so `import rl` fails even after `cd wmag`. Put the repo
    # first and chdir into it (paper code reads some assets relative to cwd).
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)

    argv = ["eval"] + PICK_FLAGS + [
        "--ckpt_name", "eval_tmp",       # NOT --resume_ckpt: the paper's resume loads the
        "--save_dir", a.save_dir,        # 835MB replay + learner too; eval needs neither.
        "--n_test_rollouts", str(a.n_test_rollouts),
    ]
    if not a.no_cuda:
        argv.append("--cuda")
    sys.argv = argv

    from rl.main_latent_fetch import get_args
    from rl.launcher_latent_fetch import launch
    import numpy as np

    args = get_args()
    print("[eval] reconstructing agent (no training, no resume) ...", flush=True)
    algo = launch(args)                      # builds env/agent/planner; no resume -> no replay load

    # Load ONLY the trained networks: agent.pt (actor/critic/vf/ae/cluster) is
    # required; algo.pt (total_timesteps/normalizer) is best-effort. This skips
    # the 835MB replay_0.pt + learner.pt that the paper's load_all also pulls in,
    # so a replay-free 19MB checkpoint is enough for eval.
    state_path = os.path.join(a.save_dir, args.env_name, a.resume_ckpt, "state")
    print(f"[eval] loading trained networks (no replay) from {state_path} ...", flush=True)
    try:
        algo.load(state_path)                # algo.pt
    except Exception as e:
        print(f"  (algo.pt state skipped: {e})", flush=True)
    algo.agent.load(state_path)              # agent.pt -- the trained model

    # `run_test_env_plan_eval` needs the trained clusters; the loaded agent already
    # carries them, so we do NOT re-initialize. If the repo gates it on a runtime
    # flag, set it so the planning path (not the flat path) is used.
    if hasattr(algo, "_clusters_initialized"):
        algo._clusters_initialized = True

    print(f"[eval] running paper soft-Floyd plan eval x{a.episodes} "
          f"({a.n_test_rollouts} rollouts each) ...", flush=True)
    succ = []
    for i in range(a.episodes):
        s = algo.run_test_env_plan_eval()
        succ.append(float(s))
        print(f"  pass {i}: plan success = {s:.3f}", flush=True)
    print(f"\nBASELINE soft-Floyd plan success: mean={np.mean(succ):.3f}  runs={succ}", flush=True)
    print("If this is ~1.0, load+eval works -> proceed to plug in MCTS (Step 2).", flush=True)


if __name__ == "__main__":
    main()
