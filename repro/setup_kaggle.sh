#!/usr/bin/env bash
# Set up the ORIGINAL L³P paper stack on Kaggle (or a fresh Ubuntu GPU box).
# Kaggle uses a fixed image, so we create a py3.7 conda env with the 2019 stack.
# Re-run each session (~10-15 min). Checkpoint training outputs to a persistent
# location (Kaggle Dataset) — the working dir is wiped between sessions.
#
# Usage (Kaggle first cell):
#   !bash repro/setup_kaggle.sh
#   !cd /kaggle/working/wmag && conda run -n l3p bash scripts/train_pointmaze.sh
#
# Requires a GPU accelerator that is cu101-compatible (T4 or P100 on Kaggle).
# NOT verified on this macOS repo — best-effort; iterate on the target box.
set -euo pipefail

echo "== [1/6] system deps =="
apt-get update -qq
apt-get install -y -qq wget unzip git build-essential gcc patchelf \
    libgl1-mesa-dev libglew-dev libosmesa6-dev libglfw3 \
    libopenmpi-dev openmpi-bin >/dev/null

echo "== [2/6] miniconda + python 3.7.4 env =="
if [ ! -x /opt/conda/bin/conda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/conda
fi
export PATH=/opt/conda/bin:$PATH
# Newer conda refuses defaults/anaconda channels until the ToS is accepted --
# this is the usual reason `conda create` fails silently on modern Kaggle images.
conda tos accept --override-channels --channel defaults >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
# Create the env ONLY if it is not already a VALID conda env. A bare directory
# (left by an interrupted create) is not valid -> conda-meta/history must exist.
if conda env list | awk '{print $1}' | grep -qx l3p \
        && [ -f /opt/conda/envs/l3p/conda-meta/history ]; then
    echo "  env l3p already valid -- reusing"
else
    echo "  (re)creating env l3p (python 3.7.4)"
    rm -rf /opt/conda/envs/l3p                 # clear any partial/broken dir
    conda create -y -n l3p python=3.7.4        # errors now surface (no silent ||)
fi

echo "== [3/6] MuJoCo 2.0 binaries + free key =="
mkdir -p "$HOME/.mujoco"
if [ ! -d "$HOME/.mujoco/mujoco200" ]; then
    wget -q https://www.roboti.us/download/mujoco200_linux.zip -O /tmp/mj.zip
    unzip -q /tmp/mj.zip -d "$HOME/.mujoco"
    mv "$HOME/.mujoco/mujoco200_linux" "$HOME/.mujoco/mujoco200"
fi
[ -f "$HOME/.mujoco/mjkey.txt" ] || wget -q https://www.roboti.us/file/mjkey.txt -O "$HOME/.mujoco/mjkey.txt"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco200/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"

echo "== [4/6] python deps (paper pins) =="
conda run -n l3p pip install -q --no-cache-dir numpy==1.19.5 pandas==1.1.1
conda run -n l3p pip install -q --no-cache-dir torch==1.5.1+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html
conda run -n l3p pip install -q --no-cache-dir tensorflow==1.13.1
conda run -n l3p pip install -q --no-cache-dir "cython<3" gym==0.13.1 mpi4py==3.0.3
# paper repo (goal_env/plane.py) imports cv2; headless build avoids libGL on the
# server and stays compatible with the pinned numpy 1.19 on py3.7.
conda run -n l3p pip install -q --no-cache-dir opencv-python-headless==4.5.5.64
# mujoco_py 2.0.2.13 is a pre-PEP517 sdist: modern pip aborts its wheel build
# with "cannot fall back to setuptools without 'wheel'". Use a 2019-era pip that
# builds via legacy setup.py, against the env's own setuptools/wheel/Cython.
conda run -n l3p pip install -q --no-cache-dir "pip<21" "setuptools<66" wheel
conda run -n l3p pip install -q --no-cache-dir --no-build-isolation --no-use-pep517 mujoco_py==2.0.2.13

echo "== [5/6] clone paper repo =="
cd /kaggle/working 2>/dev/null || cd "$HOME"
[ -d wmag ] || git clone -q https://github.com/LunjunZhang/world-model-as-a-graph wmag

echo "== [6/6] smoke check =="
conda run -n l3p python -c "import mujoco_py, torch; print('mujoco_py', mujoco_py.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo
echo "DONE. Next:"
echo "  cd wmag && conda run -n l3p bash scripts/train_pointmaze.sh   # validate pipeline (1 env) first"
echo "  (checkpoint outputs to a Kaggle Dataset to survive the ~12h session limit)"
