# Reproduce the ORIGINAL L³P paper (LunjunZhang/world-model-as-a-graph) on its
# exact MuJoCo envs (AntMaze / Fetch / PointMaze). Old 2019 stack — pinned here.
#
# GPU compatibility: torch 1.5.1+cu101 supports NVIDIA arch sm_37..sm_75 →
# works on K80/P100/T4/V100. DOES NOT work on Ampere+ (A100 sm_80 / H100) — use
# a T4/P100/V100 instance (Kaggle T4x2 or P100 are fine).
#
# Build:  docker build -t l3p-repro repro/
# Run:    docker run --gpus all -it -v $PWD/wmag:/workspace/wmag l3p-repro bash
#   (mount the cloned paper repo at ./wmag on the host)
#
# NOTE: NOT tested on this macOS machine (mujoco-py 2.0 cannot build on macOS ARM).
# This is a best-effort, iterate-on-target-Linux-GPU starting point.

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
# --- system deps: mujoco-py build + MPI + rendering ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip git build-essential gcc patchelf \
        libgl1-mesa-dev libglew-dev libosmesa6-dev libglfw3 \
        libopenmpi-dev openmpi-bin \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- Miniconda + Python 3.7.4 (paper pin) ---
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda create -y -n l3p python=3.7.4 && conda clean -a -y
SHELL ["conda", "run", "-n", "l3p", "/bin/bash", "-c"]

# --- MuJoCo 2.0 binaries + free license key (Roboti made MuJoCo free) ---
RUN mkdir -p /root/.mujoco && \
    wget -q https://www.roboti.us/download/mujoco200_linux.zip -O /tmp/mj.zip && \
    unzip -q /tmp/mj.zip -d /root/.mujoco && \
    mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 && \
    wget -q https://www.roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt && \
    rm /tmp/mj.zip
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco200/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco200

# --- Python deps (paper pins) ---
# torch cu101 from the legacy wheel index; tensorflow 1.13.1 is CPU-only here
# (used for logging; core training is torch → no CUDA-10.0 conflict).
RUN pip install --no-cache-dir numpy==1.19.5 pandas==1.1.1 && \
    pip install --no-cache-dir torch==1.5.1+cu101 \
        -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir tensorflow==1.13.1 && \
    pip install --no-cache-dir "cython<3" gym==0.13.1 mpi4py==3.0.3 && \
    pip install --no-cache-dir mujoco_py==2.0.2.13

# Pre-compile mujoco_py C extension at build time (first import is slow).
RUN python -c "import mujoco_py; print('mujoco_py OK', mujoco_py.__version__)" || \
    echo "WARN: mujoco_py precompile failed — will compile on first import"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
CMD ["conda", "run", "--no-capture-output", "-n", "l3p", "bash"]
