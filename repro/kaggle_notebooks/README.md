# Kaggle ablation notebooks (Route A — MCTS-vs-softFloyd trên env paper)

Chạy thí nghiệm robustness trên checkpoint paper-faithful đã train.

| Notebook | Env | Checkpoint | Từ output |
|---|---|---|---|
| `antmaze_ablation.ipynb` | AntMaze-v1 (long-horizon nav, PRIMARY) | `antmaze_s221_paper` | notebook train AntMaze |
| `fetch_ablation.ipynb` | FetchPickAndPlace-v1 (short-horizon manip, control) | `fetch_s967` | notebook train Fetch |
| `boxdistractor_ablation.ipynb` | Box-aside-v0 (long-horizon manip) | `boxdistractor_s829` | `boxdistractor_train.ipynb` |

Train BoxDistractor first with `boxdistractor_train.ipynb` (entry `rl.main_latent_robot`).

**Mỗi notebook:** setup env cũ → restore checkpoint (agent.pt+algo.pt) → E1a (σ=0 sanity +
sweep) → E1c (σ=0 sanity + sweep). Sửa `SLUG` ở cell 2 cho khớp `/kaggle/input/`.

**Cần:** GPU T4 + Internet ON + Add Data (output notebook chứa checkpoint tương ứng).
Import notebook: kaggle.com → Create → Notebook → File → Import → Link tới file .ipynb trên
GitHub (branch `retrain`), hoặc upload.
