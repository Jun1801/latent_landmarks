#!/usr/bin/env python3
"""Resume-aware launcher for the ORIGINAL L³P repo (rl.main_latent) on Kaggle/cloud.

Auto-detects an existing checkpoint for `--ckpt-name` and adds `--resume_ckpt`
so the SAME cell can be re-run each session to continue training across Kaggle's
~12h limit. Checkpoints are saved every epoch by the repo to
    {save_dir}/{env_name}/{ckpt_name}/state/   (algo.pt, agent.pt, replay, learner)
and resume restores weights + optimizer + replay + total_timesteps (NOT the epoch
counter — the loop restarts at 0 but continues from the restored state; kill when
eval-success plateaus).

Run INSIDE the cloned paper repo (`wmag/`), inside the `l3p` conda env, e.g.:
    conda run -n l3p python /path/to/repro/kaggle_train.py \
        --env-name AntMaze-v1 --test-env-name AntMazeTest-v1 \
        --ckpt-name antmaze_s221 --n-workers 3 --seed 221 \
        --save-dir /kaggle/working/experiments --n-epochs 20000

NOTE: not verified on this macOS repo — sanity-check on the target Linux/GPU box
with a short smoke run first (see --n-epochs 20).
"""
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-name", default="AntMaze-v1")
    p.add_argument("--test-env-name", default="AntMazeTest-v1")
    p.add_argument("--ckpt-name", default="antmaze_s221",
                   help="stable checkpoint dir name (enables resume across sessions)")
    p.add_argument("--save-dir", default="/kaggle/working/experiments",
                   help="persist this dir across sessions (Kaggle Dataset / cloud bucket)")
    p.add_argument("--n-workers", type=int, default=3)
    p.add_argument("--n-epochs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=221)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--no-cuda", action="store_true", help="disable GPU (default: --cuda ON)")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="any extra flags passed straight to rl.main_latent")
    args = p.parse_args()

    state_dir = os.path.join(args.save_dir, args.env_name, args.ckpt_name, "state")
    resume = os.path.exists(os.path.join(state_dir, "algo.pt"))

    cmd = [sys.executable, "-m", "rl.main_latent",
           "--env_name", args.env_name,
           "--test_env_name", args.test_env_name,
           "--n_workers", str(args.n_workers),
           "--n_epochs", str(args.n_epochs),
           "--gamma", str(args.gamma),
           "--batch_size", str(args.batch_size),
           "--seed", str(args.seed),
           "--save_dir", args.save_dir,
           "--ckpt_name", args.ckpt_name]
    if not args.no_cuda:
        cmd.append("--cuda")
    if resume:
        cmd += ["--resume_ckpt", args.ckpt_name]
        print(f"[resume] checkpoint found at {state_dir} -> continuing (weights/replay restored)")
    else:
        print(f"[fresh]  no checkpoint at {state_dir} -> starting new run")
    cmd += args.extra

    print("RUN:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
