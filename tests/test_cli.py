"""Command-line contracts for the optional PN-LMCGS workflow."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    path = _SCRIPTS / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def test_general_train_cli_keeps_pn_disabled_by_default_and_wires_hazard_mode():
    train = _load_script("train.py")

    baseline = train.build_config(train.build_parser().parse_args([]))
    assert not baseline.pn_lmcgs_enabled
    assert not baseline.pn_pointmaze_hazard_enabled

    pn = train.build_config(train.build_parser().parse_args(["--pn-lmcgs"]))
    assert pn.pn_lmcgs_enabled
    assert not pn.pn_pointmaze_hazard_enabled

    hazardous = train.build_config(
        train.build_parser().parse_args(["--env", "PointMaze", "--hazardous-pointmaze"])
    )
    assert hazardous.pn_lmcgs_enabled
    assert hazardous.pn_pointmaze_hazard_enabled

    non_pointmaze = train.build_parser().parse_args([
        "--env", "AntMaze", "--hazardous-pointmaze",
    ])
    with pytest.raises(ValueError, match="PointMaze"):
        train.build_config(non_pointmaze)


def test_general_train_reports_invalid_hazard_environment_as_cli_error(capsys):
    train = _load_script("train.py")

    with pytest.raises(SystemExit) as error:
        train.main(["--env", "AntMaze", "--hazardous-pointmaze"])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "usage:" in stderr
    assert "only valid for the PointMaze environment" in stderr
    assert "Traceback" not in stderr


def test_pointmaze_short_preserves_explicit_steps_and_smoke_training_overrides():
    pointmaze = _load_script("train_pointmaze.py")

    baseline = pointmaze.build_config(pointmaze.build_parser().parse_args([]))
    assert baseline.total_steps == 500_000
    assert not baseline.pn_lmcgs_enabled

    short = pointmaze.build_config(pointmaze.build_parser().parse_args(["--short"]))
    assert short.total_steps == 30_000
    assert short.train_after == 1_000
    assert short.n_grad_steps == 40
    assert short.env_steps_per_opt == 2

    explicit = pointmaze.build_config(
        pointmaze.build_parser().parse_args(["--short", "--steps", "200"])
    )
    assert explicit.total_steps == 200
    assert explicit.train_after == 1_000
    assert explicit.n_grad_steps == 40
    assert explicit.env_steps_per_opt == 2

    pn_smoke = pointmaze.build_config(pointmaze.build_parser().parse_args([
        "--short", "--steps", "200", "--pn-lmcgs",
    ]))
    assert pn_smoke.total_steps == 200
    assert pn_smoke.train_after == 1
    assert pn_smoke.n_grad_steps == 1
    assert pn_smoke.env_steps_per_opt == 100

    hazardous = pointmaze.build_config(
        pointmaze.build_parser().parse_args(["--hazardous-pointmaze"])
    )
    assert hazardous.pn_lmcgs_enabled
    assert hazardous.pn_pointmaze_hazard_enabled


def test_eval_cli_wires_pn_and_hazard_geometry_without_changing_defaults():
    evaluate = _load_script("eval.py")

    baseline = evaluate.build_config(evaluate.build_parser().parse_args([]))
    assert not baseline.pn_lmcgs_enabled
    assert not baseline.pn_pointmaze_hazard_enabled

    pn = evaluate.build_config(evaluate.build_parser().parse_args(["--pn-lmcgs"]))
    assert pn.pn_lmcgs_enabled
    assert not pn.pn_pointmaze_hazard_enabled

    hazardous = evaluate.build_config(
        evaluate.build_parser().parse_args(["--hazardous-pointmaze"])
    )
    assert hazardous.pn_lmcgs_enabled
    assert hazardous.pn_pointmaze_hazard_enabled
