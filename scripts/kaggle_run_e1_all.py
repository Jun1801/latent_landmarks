#!/usr/bin/env python3
"""Run E1a/E1b/E1c/E1d batches on Kaggle or a local machine.

The runner is intentionally conservative:
  * It looks for checkpoints in the repo and under /kaggle/input.
  * It runs E1a/E1b on any configured env/checkpoint.
  * It runs E1c/E1d only for envs where goal coordinates can be used as states
    (currently PointMaze). Other envs are skipped unless --force-feedback is set.
  * It continues after a failed task and writes a summary JSON.

Example Kaggle cell:
    !python scripts/kaggle_run_e1_all.py --preset smoke

Fuller PointMaze-style run:
    !python scripts/kaggle_run_e1_all.py --preset report --only pointmaze_numpy

Custom manifest:
    !python scripts/kaggle_run_e1_all.py --manifest /kaggle/input/myset/e1_tasks.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = Path("/kaggle/input")
DEFAULT_OUTPUT_ROOT = (Path("/kaggle/working/e1_runs")
                       if Path("/kaggle/working").exists()
                       else REPO / "logs" / "kaggle_e1_runs")

FEEDBACK_SAFE_ENVS = {"PointMaze"}

DEFAULT_TASKS = [
    dict(name="pointmaze_numpy", env="PointMaze", checkpoint="checkpoint/l3p_pointmaze_full.pt",
         experiments=["e1a", "e1b", "e1c", "e1d"]),
    dict(name="pointmaze_mujoco", env="PointMazeMuJoCo", checkpoint="checkpoint/l3p_pointmaze_mujoco.pt",
         experiments=["e1a", "e1b"]),
    dict(name="fetch_pick_and_place", env="FetchPickAndPlace", checkpoint="checkpoint/l3p_fetch.pt",
         experiments=["e1a", "e1b"]),
    dict(name="antmaze", env="AntMaze", checkpoint="checkpoint/l3p_AntMaze.pt",
         experiments=["e1a", "e1b"]),
]

PRESETS = {
    # Fast sanity for Kaggle notebooks.
    "smoke": dict(
        seeds=["0"], episodes="5", calibrate_episodes="5",
        sanity_tol="1.0",
        e1a_sims="20", e1b_sims="20", e1c_sims="20", e1d_sims="10",
        e1a_sigmas=["0", "0.3"], e1b_sigma_hi=["0", "0.5"],
        e1c_sigmas=["0", "0.3"], e1d_sigmas=["0", "0.3"],
    ),
    # Close to current report settings while still reasonable for hosted notebooks.
    "report": dict(
        seeds=["0", "1"], episodes="30", calibrate_episodes="20",
        sanity_tol="0.2",
        e1a_sims="200", e1b_sims="100", e1c_sims="80", e1d_sims="20",
        e1a_sigmas=["0", "0.1", "0.3", "0.5"], e1b_sigma_hi=["0", "0.5", "1.0"],
        e1c_sigmas=["0", "0.1", "0.3"], e1d_sigmas=["0", "0.3"],
    ),
    # Longer confirmation; use this when Kaggle runtime budget is enough.
    "full": dict(
        seeds=["0", "1", "2"], episodes="50", calibrate_episodes="20",
        sanity_tol="0.15",
        e1a_sims="200", e1b_sims="120", e1c_sims="120", e1d_sims="80",
        e1a_sigmas=["0", "0.1", "0.3", "0.5"], e1b_sigma_hi=["0", "0.3", "0.5", "1.0"],
        e1c_sigmas=["0", "0.1", "0.3"], e1d_sigmas=["0", "0.3"],
    ),
}

SCRIPT = {
    "e1a": "scripts/run_e1a.py",
    "e1b": "scripts/run_e1b.py",
    "e1c": "scripts/run_e1c.py",
    "e1d": "scripts/run_e1d.py",
}
READINESS_SCRIPT = "scripts/check_checkpoint_readiness.py"


def load_manifest(path: str | None) -> list[dict]:
    if path is None:
        return DEFAULT_TASKS
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    if not isinstance(data, list):
        raise ValueError("manifest must be a list or an object with a 'tasks' list")
    return data


def find_checkpoint(name: str, input_root: Path) -> Path | None:
    p = Path(name)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([REPO / p, Path.cwd() / p])
        candidates.extend(REPO.rglob(p.name))
        if input_root.exists():
            candidates.extend(input_root.rglob(p.name))
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def should_run_exp(task: dict, exp: str, force_feedback: bool) -> tuple[bool, str]:
    if exp not in {"e1c", "e1d"}:
        return True, ""
    env = task["env"]
    if force_feedback or env in FEEDBACK_SAFE_ENVS:
        return True, ""
    return False, (
        f"{exp} skipped for {env}: feedback/recovery currently needs obs_dim == goal_dim "
        "or an env-specific goal->observation lifting adapter."
    )


def exp_args(exp: str, task: dict, ckpt: Path, out_dir: Path, cfg: dict) -> list[str]:
    base = [
        sys.executable, str(REPO / SCRIPT[exp]),
        "--env", task["env"],
        "--load", str(ckpt),
        "--seeds", *cfg["seeds"],
        "--episodes", cfg["episodes"],
        "--calibrate-episodes", cfg["calibrate_episodes"],
        "--out", str(out_dir / f"{exp}_results.json"),
        "--plot", str(out_dir / f"{exp}_curve.png"),
        "--sanity-tol", cfg["sanity_tol"],
    ]
    if exp == "e1a":
        return base + [
            "--sigmas", *cfg["e1a_sigmas"],
            "--mcts-n-simulations", cfg["e1a_sims"],
        ]
    if exp == "e1b":
        return base + [
            "--sigma-hi", *cfg["e1b_sigma_hi"],
            "--mcts-n-simulations", cfg["e1b_sims"],
        ]
    if exp == "e1c":
        return base + [
            "--sigmas", *cfg["e1c_sigmas"],
            "--mcts-n-simulations", cfg["e1c_sims"],
        ]
    if exp == "e1d":
        return base + [
            "--sigmas", *cfg["e1d_sigmas"],
            "--mcts-n-simulations", cfg["e1d_sims"],
        ]
    raise ValueError(exp)


def readiness_args(task: dict, ckpt: Path, out_dir: Path, args) -> list[str]:
    return [
        sys.executable, str(REPO / READINESS_SCRIPT),
        "--env", task["env"],
        "--load", str(ckpt),
        "--episodes", str(args.preflight_episodes),
        "--calibrate-episodes", str(args.preflight_calibrate_episodes),
        "--min-success", str(args.min_baseline_success),
        "--out", str(out_dir / "readiness.json"),
    ]


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(log_path.parent / "mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(log_path.parent / ".cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", buffering=1) as log:
        log.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        return p.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, default=None,
                   help="JSON list of {name, env, checkpoint, experiments}.")
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--only", nargs="+", default=None,
                   help="Run only task names listed here.")
    p.add_argument("--experiments", nargs="+", choices=["e1a", "e1b", "e1c", "e1d"],
                   default=None, help="Override experiments for every task.")
    p.add_argument("--force-feedback", action="store_true",
                   help="Attempt E1c/E1d even on envs without built-in feedback support.")
    p.add_argument("--require-ready", action="store_true",
                   help="Run checkpoint readiness preflight before E1 and skip tasks below threshold.")
    p.add_argument("--min-baseline-success", type=float, default=0.30,
                   help="minimum calibrated clean Soft Floyd success for --require-ready")
    p.add_argument("--preflight-episodes", type=int, default=10,
                   help="final readiness eval episodes")
    p.add_argument("--preflight-calibrate-episodes", type=int, default=5,
                   help="episodes per d_max candidate during readiness calibration")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    all_tasks = load_manifest(args.manifest)
    tasks = all_tasks
    if args.only:
        allowed = set(args.only)
        tasks = [t for t in all_tasks if t["name"] in allowed]
        missing = sorted(allowed - {t["name"] for t in tasks})
        if missing:
            available = ", ".join(t["name"] for t in all_tasks)
            print(f"WARNING: --only ignored unknown task name(s): {missing}", file=sys.stderr)
            print(f"Available task names: {available}", file=sys.stderr)
        if not tasks:
            print("ERROR: --only matched no tasks; nothing to run.", file=sys.stderr)
            sys.exit(2)
    cfg = PRESETS[args.preset]
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = dict(preset=args.preset, tasks=[], started_at=time.time())
    for task in tasks:
        name, env_name = task["name"], task["env"]
        out_dir = args.output_root / name
        task_sum = dict(name=name, env=env_name, checkpoint=task["checkpoint"], runs=[])
        ckpt = find_checkpoint(task["checkpoint"], args.input_root)
        if ckpt is None:
            task_sum["status"] = "skipped"
            task_sum["reason"] = f"checkpoint not found: {task['checkpoint']}"
            summary["tasks"].append(task_sum)
            print(f"\nSKIP {name}: {task_sum['reason']}", flush=True)
            continue
        task_sum["resolved_checkpoint"] = str(ckpt)
        if args.require_ready:
            cmd = readiness_args(task, ckpt, out_dir, args)
            task_sum["readiness_cmd"] = cmd
            if args.dry_run:
                print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
            else:
                rc = run_command(cmd, out_dir / "readiness.log")
                task_sum["readiness_returncode"] = rc
                task_sum["readiness_log"] = str(out_dir / "readiness.log")
                task_sum["readiness_json"] = str(out_dir / "readiness.json")
                if rc != 0:
                    task_sum["status"] = "skipped"
                    task_sum["reason"] = (
                        f"readiness preflight failed: calibrated clean baseline below "
                        f"{args.min_baseline_success:.2f} or env/checkpoint invalid"
                    )
                    summary["tasks"].append(task_sum)
                    print(f"SKIP {name}: {task_sum['reason']}", flush=True)
                    continue

        experiments = args.experiments or task.get("experiments", ["e1a", "e1b", "e1c", "e1d"])
        print(f"\n=== {name} | env={env_name} | checkpoint={ckpt} | exps={experiments} ===", flush=True)
        for exp in experiments:
            ok, reason = should_run_exp(task, exp, args.force_feedback)
            run_sum = dict(exp=exp)
            if not ok:
                run_sum.update(status="skipped", reason=reason)
                task_sum["runs"].append(run_sum)
                print("SKIP " + reason, flush=True)
                continue
            cmd = exp_args(exp, task, ckpt, out_dir, cfg)
            run_sum["cmd"] = cmd
            if args.dry_run:
                run_sum["status"] = "dry-run"
                print("$ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
            else:
                t0 = time.time()
                rc = run_command(cmd, out_dir / f"{exp}.log")
                run_sum.update(status="ok" if rc == 0 else "failed",
                               returncode=rc, seconds=time.time() - t0,
                               log=str(out_dir / f"{exp}.log"))
            task_sum["runs"].append(run_sum)
        task_sum["status"] = "done"
        summary["tasks"].append(task_sum)

    summary["finished_at"] = time.time()
    summary_path = args.output_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
