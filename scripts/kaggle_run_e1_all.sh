#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for Kaggle notebooks.
#
# Examples:
#   bash scripts/kaggle_run_e1_all.sh smoke
#   bash scripts/kaggle_run_e1_all.sh report --only pointmaze_numpy
#   bash scripts/kaggle_run_e1_all.sh smoke --manifest /kaggle/input/myset/e1_tasks.json

PRESET="${1:-smoke}"
shift || true

python scripts/kaggle_run_e1_all.py --preset "${PRESET}" "$@"
