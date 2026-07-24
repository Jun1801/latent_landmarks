# Repository Guidelines

## Project Structure & Module Organization

This repository is a PyTorch reimplementation of L3P. Core package code lives in `l3p/`: configuration in `config.py`, agent logic in `agent/`, replay buffers in `replay/`, model components in `models/`, planners in `planning/`, environments in `envs/`, and the training loop in `trainer.py`. Command-line workflows live in `scripts/`, with `train_pointmaze.py`, `train.py`, evaluation, plotting, and experiment launchers. Tests are concentrated in `tests/test_modules.py` and use synthetic data. Documentation and experiment notes live in `docs/`, `README.md`, `MCTS.md`, and report files. Environment XML assets are under `l3p/envs/assets/`.

## Build, Test, and Development Commands

- `pip install -r requirements.txt`: install the required runtime dependencies, currently `torch` and `numpy`.
- `python tests/test_modules.py`: run the fast unit tests without relying on extra test-runner dependencies.
- `pytest tests/`: optional equivalent if `pytest` is installed locally.
- `python scripts/train_pointmaze.py --short`: run a short end-to-end PointMaze smoke test.
- `python scripts/train_pointmaze.py --steps 500000`: launch a longer PointMaze training run.
- `python scripts/eval.py --load l3p_pointmaze.pt --episodes 50 --show-landmarks`: evaluate a checkpoint and inspect landmarks.

## Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, descriptive snake_case functions and variables, and PascalCase classes. Keep module boundaries aligned with the existing package layout: planners belong in `l3p/planning/`, environment wrappers in `l3p/envs/`, and model definitions in `l3p/models/`. Prefer small, equation-referenced docstrings or comments when implementing paper logic. No formatter or linter is configured, so keep imports tidy and follow the surrounding file style.

## Testing Guidelines

Add or update tests in `tests/test_modules.py` for planner, model, replay, and loss behavior. Name tests `test_<behavior>` and use deterministic seeds for stochastic logic. Keep default tests synthetic and CPU-friendly; MuJoCo-dependent behavior should be guarded or documented separately. Run `python tests/test_modules.py` before handing off changes.

## Commit & Pull Request Guidelines

Recent commits use short lowercase labels such as `mcts-planner` and `e1a_e1b`. Continue with concise subjects, ideally adding a scope when useful, for example `planning: cache admissibility mask`. Pull requests should describe the behavioral change, list validation commands, call out changed experiment artifacts, and link any relevant issue or spec. Avoid committing generated checkpoints (`*.pt`), caches, or large logs unless they are intentional report artifacts.

## Security & Configuration Tips

Keep MuJoCo and Fetch dependencies optional; the default setup should continue to run with only `torch` and `numpy`. Store generated checkpoints outside version control, and avoid hard-coding local paths in scripts or configs.
