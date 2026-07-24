# Checkpoints

This directory is the local mount point for model checkpoints. The repository
ignores `*.pt`, so checkpoint binaries are not committed to GitHub.

Use a Hugging Face Dataset repo to move checkpoints between local development
and Kaggle:

```bash
pip install huggingface_hub

# Upload local checkpoint/*.pt files.
HF_TOKEN=... python scripts/hf_sync_checkpoints.py upload \
  --repo-id Jun1801/mcts_vla \
  --create-repo

# Download them again before running experiments.
python scripts/hf_sync_checkpoints.py download \
  --repo-id Jun1801/mcts_vla
```

Expected default files:

```text
checkpoint/l3p_pointmaze_full.pt
checkpoint/l3p_pointmaze_mujoco.pt
checkpoint/l3p_fetch.pt
checkpoint/l3p_AntMaze.pt
```
