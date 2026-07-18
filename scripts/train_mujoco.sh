#!/usr/bin/env bash
# Convenience launchers for the MuJoCo benchmarks from the paper (Section 5).
# These REQUIRE gym + mujoco-py (see README.md); they are not verified in this
# repo's default (PointMaze-only) setup.
#
# Usage: bash scripts/train_mujoco.sh <antmaze|fetch|distractor|inside_box>
set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${1:-antmaze}"
case "$TASK" in
  antmaze)     ENV="AntMaze" ;;
  fetch)       ENV="FetchPickAndPlace" ;;
  distractor)  ENV="BoxDistractorPickAndPlace" ;;
  inside_box)  ENV="PlaceInsideBox" ;;
  *) echo "unknown task: $TASK"; exit 1 ;;
esac

python - "$ENV" <<'PY'
import sys
from l3p.config import get_config
from l3p.envs import make_vec_env
from l3p.trainer import L3PTrainer

env_name = sys.argv[1]
cfg = get_config(env_name)
venv = make_vec_env(cfg, cfg.n_workers, cfg.seed)
trainer = L3PTrainer(venv, cfg)
print(f"Training L3P on {env_name} (MuJoCo)")
trainer.train()
trainer.save(f"l3p_{env_name}.pt")
PY
