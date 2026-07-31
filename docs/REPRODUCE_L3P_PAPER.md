# Reproduce the ORIGINAL L³P paper on its exact envs (Kaggle / cloud)

Advisor requirement: run on the **paper's own MuJoCo environments** (AntMaze /
Fetch / PointMaze), not the gymnasium-robotics substitutes in this reimplementation.
That means using the **original repo** `LunjunZhang/world-model-as-a-graph` with its
custom `goal_env/` code and the 2019 software stack.

> ⚠️ **Not runnable on this macOS machine** (`mujoco_py 2.0` won't build on macOS ARM;
> `torch 1.5.1+cu101` is Linux-CUDA). Must run on **Linux x86_64 + NVIDIA GPU**.
> The setup below is **best-effort, untested here** — iterate on the target Linux/GPU box.

## 0. The exact stack (paper `README`)
```
python==3.7.4  numpy==1.19.5  torch==1.5.1+cu101  tensorflow==1.13.1
gym==0.13.1    mpi4py==3.0.3  mujoco_py==2.0.2.13  pandas==1.1.1
```
Entry point: `python -m rl.main_latent ...`. Envs self-register on import via
`goal_env/` (`gym.register` → `AntMaze-v1`, `AntMazeTest-v1`, …). Custom MuJoCo env
code needs `mujoco_py 2.0` + MuJoCo 2.0 binaries.

## 1. ⚠️ GPU compatibility (the #1 gotcha)
`torch 1.5.1+cu101` supports NVIDIA arch **sm_37…sm_75** →
- ✅ **K80 / P100 / T4 / V100** (Kaggle offers **T4×2** and **P100** — both work)
- ❌ **A100 (sm_80) / H100** — cu101 will NOT run; pick an older GPU or you must upgrade the whole stack.

## 2. MuJoCo 2.0 (now free)
```bash
mkdir -p ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip -O /tmp/mj.zip
unzip /tmp/mj.zip -d ~/.mujoco && mv ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200
wget https://www.roboti.us/file/mjkey.txt -O ~/.mujoco/mjkey.txt   # free key
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
```
System build deps (Ubuntu): `build-essential gcc patchelf libgl1-mesa-dev
libglew-dev libosmesa6-dev libglfw3 libopenmpi-dev openmpi-bin unzip wget`.

## 3. Two paths

### Path A — Docker on a cloud GPU VM (recommended, most reproducible)
`repro/Dockerfile` pins the whole stack. On a Linux VM with an **older GPU** + Docker + nvidia-container-toolkit:
```bash
git clone https://github.com/LunjunZhang/world-model-as-a-graph wmag
docker build -t l3p-repro repro/
docker run --gpus all -it -v $PWD/wmag:/workspace/wmag l3p-repro bash
# inside:
cd /workspace/wmag && bash scripts/train_pointmaze.sh   # or train_antmaze.sh / pick.sh
```

### Path B — Kaggle notebook (no custom Docker; install into the session)
Kaggle uses a fixed image, so install the old stack into a conda env each session
(≈10–15 min; checkpoint outputs to a **Kaggle Dataset** to survive the ~12h limit).
Use `repro/setup_kaggle.sh` in the first cell:
```python
!bash /kaggle/working/latent-lanmarks/repro/setup_kaggle.sh
!cd /kaggle/working/wmag && conda run -n l3p bash scripts/train_pointmaze.sh
```
Pick a **GPU (T4 or P100)** accelerator in Kaggle settings.

## 4. Exact training commands (from the repo's `scripts/`)
Run the repo's own scripts — they encode the paper's per-env hyper-parameters:
| Env | Script | Notes (from script) |
|---|---|---|
| PointMaze | `scripts/train_pointmaze.sh` | — |
| AntMaze | `scripts/train_antmaze.sh` | `--n_workers 3 --env_name AntMaze-v1 --test_env_name AntMazeTest-v1 --n_epochs 20000 --gamma 0.98 --batch_size 1000`, 4 seeds (221/246/732/391) |
| Fetch pick | `scripts/pick.sh` | more workers (Appendix E: Fetch ~12) |
| Box-Distractor | `scripts/distractor.sh` | — |
| Place-Inside-Box | `scripts/inside_box.sh` | — |

## 4.5 Checkpoint / resume (verified from `rl/algo/core.py`)
- Saves **every epoch, unconditionally** to `{save_dir}/{env_name}/{ckpt_name}/state/`
  (`algo.pt`, `agent.pt`, replay, learner). Safe against session kills.
- `--resume_ckpt <name>` loads from `{save_dir}/{env_name}/<name>/state/` and restores
  **weights + optimizer + replay buffer + total_timesteps** (full learning state).
- ⚠️ **Epoch counter is NOT restored — the loop restarts at epoch 0.** So on resume the
  learned state continues, but it will run another full `--n_epochs`; just let it run and
  **kill when eval-success plateaus** (each epoch is saved).
- Resumable pattern: always pass a stable `--ckpt_name`; on continuation add
  `--resume_ckpt <same name>` (load + keep saving to the same dir).
- Helper: `repro/kaggle_train.py` auto-detects an existing checkpoint and adds
  `--resume_ckpt` for you; `repro/kaggle_notebook.md` is the cell-by-cell Kaggle flow.

## 5. Compute reality
`n_epochs 20000` × multi-worker × **4 seeds** = **large** (this is why the paper used a
cluster). On a single T4/P100:
- Budget **many hours–days** per env; AntMaze is the heaviest.
- **Checkpoint frequently** and resume across sessions (Kaggle 12h cap). Store
  checkpoints in a persistent location (Kaggle Dataset / cloud bucket).
- Consider starting with **1 seed + PointMaze** to validate the pipeline end-to-end
  before committing all 4 seeds × all envs.

## 6. Where OUR contribution plugs in (after base reproduction)
Once base L³P reproduces on the paper's envs, port our extension into their `rl/`:
- `LandmarkMCTS` + execution-feedback (this repo `l3p/planning/mcts_planner.py`) and the
  noise/bias harness (`l3p/planning/noise.py`) as a **test-time planner** over their
  landmark graph. Keep the paper's training; swap only the planner at eval.
- The strongest carry-over (per P1, `MCTS.md §7.5`): **execution-feedback under systematic
  bias** — re-run E1c-style on the paper's harder envs where planning genuinely matters.

## 7. Honest caveats
- Old-stack reproduction is a known devops slog: `mujoco_py 2.0` C-extension build, CUDA
  compat, `tensorflow 1.13.1`. Expect iteration.
- `www.roboti.us` hosts MuJoCo 2.0 + the free key; if unreachable, mirror the files.
- This guide/Dockerfile were written from the repo's requirements, **not verified on a GPU
  box here**. First goal on the target machine: `python -c "import mujoco_py, torch; print(torch.cuda.is_available())"` → then one PointMaze smoke run.
