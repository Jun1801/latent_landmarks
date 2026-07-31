# Kaggle notebook — train the ORIGINAL L³P paper envs (resume-loop)

Cell-by-cell flow to train the paper's MuJoCo envs on Kaggle across the ~12h
session limit. **GPU accelerator must be T4 or P100** (Settings → Accelerator).
Files referenced live in this repo's `repro/` (upload it as a Kaggle Dataset, or
`git clone` your fork in Cell 1).

> Not verified here (no Linux GPU on the dev machine). Do Cell 2 + a SMOKE run
> (Cell 4) FIRST; only launch the long AntMaze run once the smoke works.

---

### Cell 1 — get the code (this repo's repro/ + the paper repo)
```python
# clone the paper repo
!git clone -q https://github.com/LunjunZhang/world-model-as-a-graph /kaggle/working/wmag
# get repro/ helpers: either clone your fork of THIS repo, or upload repro/ as a Dataset
!git clone -q <YOUR_FORK_OF_THIS_REPO> /kaggle/working/latent-lanmarks  # contains repro/
```

### Cell 2 — one-time setup (≈10–15 min): stack + MuJoCo + smoke import
```python
!bash /kaggle/working/latent-lanmarks/repro/setup_kaggle.sh
# ^ installs py3.7 conda env `l3p`, torch1.5.1+cu101, mujoco_py2.0, gym0.13.1, mpi4py, MuJoCo 2.0
```

### Cell 3 — restore previous checkpoints (skip on the very first session)
```python
# Attach the PREVIOUS session's output (or a Dataset holding experiments/) as an
# input under /kaggle/input/<name>, then copy it back so training can resume:
import os, shutil
src = "/kaggle/input/l3p-experiments/experiments"   # adjust to your attached dataset path
dst = "/kaggle/working/experiments"
if os.path.isdir(src):
    shutil.copytree(src, dst, dirs_exist_ok=True); print("restored", dst)
else:
    print("no prior experiments/ attached — starting fresh")
```

### Cell 4 — SMOKE test (do this before any long run!)
```python
%cd /kaggle/working/wmag
!conda run -n l3p python /kaggle/working/latent-lanmarks/repro/kaggle_train.py \
    --env-name PointMaze-v1 --test-env-name PointMazeTest-v1 \
    --ckpt-name smoke_pm --seed 123 --n-epochs 20 \
    --save-dir /kaggle/working/experiments
# expect: trains a few epochs, uses GPU, writes experiments/PointMaze-v1/smoke_pm/state/
```

### Cell 5 — real training (resume-aware; re-run each session)
```python
%cd /kaggle/working/wmag
# AntMaze (heaviest, the scientifically important env). Same cell re-runs to resume.
!conda run -n l3p python /kaggle/working/latent-lanmarks/repro/kaggle_train.py \
    --env-name AntMaze-v1 --test-env-name AntMazeTest-v1 \
    --ckpt-name antmaze_s221 --n-workers 3 --seed 221 \
    --gamma 0.98 --batch-size 1000 --n-epochs 20000 \
    --save-dir /kaggle/working/experiments
# kaggle_train.py auto-adds --resume_ckpt if experiments/AntMaze-v1/antmaze_s221/state/ exists.
# It runs until the session ends; each epoch is checkpointed. Kill/re-run to continue.
```

### Cell 6 — persist checkpoints for next session
```python
# /kaggle/working is saved as the notebook OUTPUT automatically on commit.
# Next session: publish this output as a Dataset (or attach the output directly)
# and point Cell 3's `src` at it. Verify the checkpoint is there:
!ls -la /kaggle/working/experiments/AntMaze-v1/antmaze_s221/state/
```

---

## Operating notes
- **Start small:** Cell 4 smoke → full **PointMaze** (cheap, reproduce one paper number)
  → then **AntMaze 1 seed**. Don't launch all 4 seeds × all envs at once.
- **Resume caveat:** epoch counter restarts at 0 on resume (weights/replay/total_timesteps
  ARE restored). So it keeps learning; watch `eval` success and **kill when it plateaus**.
  Verify resume works by resuming the smoke run once (Cell 4 twice) before trusting AntMaze.
- **AntMaze reality:** locomotion bootstraps slowly; expect success to stay ~0 for a while
  then climb. If it's still flat 0 after millions of steps, stop and diagnose (dense-reward /
  UMaze may be needed) rather than burning more GPU.
- **GPU:** T4/P100 only (torch cu101). `--cuda` is ON by default in `kaggle_train.py`.
