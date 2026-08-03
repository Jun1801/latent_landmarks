# Kaggle Training and W&B Logging Design

## Goal

Provide a single Python entrypoint that trains L3P or PN-LMCGS on Kaggle across
the supported environments, writes portable run artifacts under Kaggle's
working directory, and logs structured metrics to Weights & Biases (W&B).

## Scope

- Add `scripts/train_kaggle.py` as the Kaggle-oriented general training
  launcher.
- Add an optional metrics callback to `L3PTrainer`.
- Preserve all existing trainer and CLI behavior whenever no callback is
  supplied.
- Add W&B as an optional dependency and make `--wandb-mode disabled` work
  without importing the package.

This does not add a resume workflow, modify the training algorithm, or make
W&B a dependency of the normal local training scripts.

## Entry Point

`scripts/train_kaggle.py` accepts the general training options:

- `--env`, `--steps`, `--seed`, `--workers`, and `--eval-episodes`.
- `--pn-lmcgs` and `--hazardous-pointmaze`; hazardous PointMaze remains valid
  only for the PointMaze environment.
- `--output-dir`, defaulting to `/kaggle/working/l3p_runs` when it exists and
  `l3p_runs` otherwise.
- `--run-name`; if omitted, generate a deterministic name from environment,
  seed, and UTC timestamp.
- `--save-every` for interval checkpoints.
- `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, and
  `--wandb-mode` (`online`, `offline`, or `disabled`). The default mode is
  `offline`, so a Kaggle notebook with Internet disabled remains runnable.

Every run writes `config.json`, `train.log`, interval checkpoints, and the
final model into `<output-dir>/<run-name>/`. Online mode uses `WANDB_API_KEY`
from the Kaggle Secrets environment. W&B errors must explain how to select
offline or disabled mode rather than obscuring training errors.

## Metrics Callback

`L3PTrainer` receives an optional callback with an event name, global
environment step, and dictionary of scalar metrics. A missing callback is a
no-op.

The trainer emits these events:

- `train`: existing periodic train diagnostics and counters.
- `evaluation`: long-horizon success rate and, for PN-LMCGS, all values from
  `last_eval_metrics`, including safe-success rate and safety-cost metrics.
- `checkpoint`: a numeric checkpoint-saved flag at the saved step.
- `final`: final evaluation metrics and success rate.

The Kaggle launcher maps these events to stable W&B metric names, logs only
numeric metrics, and uses the global step as W&B's step. PN-only diagnostics
are included only when PN-LMCGS is enabled. The run config records all
resolved L3P configuration fields plus launcher arguments. Artifact paths are
owned by the launcher rather than the scalar callback payload.

## W&B Lifecycle

The launcher lazily imports and initializes W&B only when its mode is not
`disabled`. It finishes the run in a `finally` block. In online or offline
mode, it creates a final model artifact after the model is saved. Artifact
upload failures are reported but do not discard an otherwise valid local
checkpoint.

## Error Handling

- Reject invalid hazardous PointMaze/environment combinations with argparse
  errors, matching `scripts/train.py`.
- Reject non-positive steps, workers, evaluation episodes, and save intervals
  before creating an environment.
- Preserve the original training exception after attempting W&B finalization.
- Do not change wall-contact or ordinary collision safety semantics.

## Tests

- Callback-free trainer behavior remains unchanged for baseline and PN modes.
- Trainer callback events include numeric step-aligned diagnostics and PN
  safety metrics after evaluation.
- Kaggle parser/config validation covers normal and hazardous PointMaze modes.
- Disabled W&B mode runs without the `wandb` package; online/offline modes
  report a concise dependency or authentication error.
- Run artifact paths remain under the selected output directory.

## Acceptance Criteria

- A Kaggle user can run `python scripts/train_kaggle.py --env PointMaze
  --pn-lmcgs --hazardous-pointmaze --steps 500000 --wandb-project <project>`
  and receive local artifacts plus W&B metrics.
- `--wandb-mode disabled` requires no W&B installation and preserves normal
  training behavior.
- Existing `scripts/train.py`, `scripts/train_pointmaze.py`, and PN-disabled
  training behavior remain unchanged.
