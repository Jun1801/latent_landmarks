"""Kaggle training launcher contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_kaggle.py"


def _load_kaggle_script():
    spec = importlib.util.spec_from_file_location("test_train_kaggle", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kaggle_config_keeps_baseline_default_and_wires_hazard_mode(tmp_path):
    kaggle = _load_kaggle_script()

    baseline = kaggle.build_config(kaggle.build_parser().parse_args([]))
    assert not baseline.pn_lmcgs_enabled
    assert not baseline.pn_pointmaze_hazard_enabled

    args = kaggle.build_parser().parse_args([
        "--env", "PointMaze", "--hazardous-pointmaze", "--output-dir", str(tmp_path),
    ])
    hazardous = kaggle.build_config(args)
    assert hazardous.pn_lmcgs_enabled
    assert hazardous.pn_pointmaze_hazard_enabled


def test_kaggle_main_rejects_invalid_counts_and_hazard_environment(capsys):
    kaggle = _load_kaggle_script()

    with pytest.raises(SystemExit) as zero_steps:
        kaggle.main(["--steps", "0", "--wandb-mode", "disabled"])
    with pytest.raises(SystemExit) as wrong_environment:
        kaggle.main([
            "--env", "AntMaze", "--hazardous-pointmaze", "--wandb-mode", "disabled",
        ])

    assert zero_steps.value.code == wrong_environment.value.code == 2
    stderr = capsys.readouterr().err
    assert "Traceback" not in stderr
    assert "positive" in stderr
    assert "PointMaze" in stderr


def test_prepare_run_directory_writes_resolved_config_under_output_root(tmp_path):
    kaggle = _load_kaggle_script()
    config = {"seed": 0, "env_name": "PointMaze"}

    run_dir = kaggle.prepare_run_directory(tmp_path, "pointmaze-seed0", config)

    assert run_dir == tmp_path / "pointmaze-seed0"
    assert json.loads((run_dir / "config.json").read_text()) == config
    with pytest.raises(FileExistsError, match="--overwrite"):
        kaggle.prepare_run_directory(tmp_path, "pointmaze-seed0", config)
    assert kaggle.prepare_run_directory(tmp_path, "pointmaze-seed0", config, overwrite=True) == run_dir
