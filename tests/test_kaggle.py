"""Kaggle training launcher contracts."""

from __future__ import annotations

import importlib.util
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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
    assert baseline.env_name == "PointMaze"
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


class _FakeWandb:
    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.finished = False
        self.artifacts = []
        self.run = SimpleNamespace(log_artifact=self.artifacts.append)

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, values, *, step):
        self.logged.append((dict(values), step))

    def finish(self):
        self.finished = True

    class Artifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, path):
            self.files.append(path)


def test_disabled_wandb_never_imports_sdk(monkeypatch):
    kaggle = _load_kaggle_script()
    monkeypatch.setattr(importlib, "import_module", pytest.fail)

    assert kaggle.create_wandb_sink("disabled", "project", None, "run", {}) is None


def test_wandb_sink_prefixes_event_metrics_and_uses_global_step(monkeypatch):
    kaggle = _load_kaggle_script()
    fake = _FakeWandb()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)

    sink = kaggle.create_wandb_sink("offline", "project", None, "run", {"seed": 0})
    sink("evaluation", 42, {"success_rate": 0.5})

    assert fake.init_kwargs == {
        "project": "project", "entity": None, "name": "run",
        "config": {"seed": 0}, "mode": "offline",
    }
    assert fake.logged == [({"evaluation/success_rate": 0.5}, 42)]
    sink.finish()
    assert fake.finished


def test_wandb_sink_logs_final_model_as_artifact(monkeypatch, tmp_path):
    kaggle = _load_kaggle_script()
    fake = _FakeWandb()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)
    sink = kaggle.create_wandb_sink("offline", "project", None, "run", {})

    sink.log_model(model_path)

    assert len(fake.artifacts) == 1
    artifact = fake.artifacts[0]
    assert artifact.name == "run-model"
    assert artifact.type == "model"
    assert artifact.files == [str(model_path)]


def test_main_saves_local_artifacts_without_wandb(tmp_path, monkeypatch):
    kaggle = _load_kaggle_script()

    class FakeTrainer:
        def __init__(self, _env, _cfg, metrics_callback=None):
            self.metrics_callback = metrics_callback
            self.last_eval_metrics = {"safe_success_rate": 0.75}

        def train(self, checkpoint_path, checkpoint_every):
            assert checkpoint_path is None
            assert checkpoint_every == 0
            assert self.metrics_callback is None

        def save(self, path):
            Path(path).write_bytes(b"model")

        def evaluate(self, episodes):
            assert episodes > 0
            return 0.5

    monkeypatch.setattr(kaggle, "make_vec_env", lambda *_args: object())
    monkeypatch.setattr(kaggle, "L3PTrainer", FakeTrainer)

    kaggle.main([
        "--env", "PointMaze", "--steps", "3", "--output-dir", str(tmp_path),
        "--run-name", "smoke", "--wandb-mode", "disabled",
    ])

    run_dir = tmp_path / "smoke"
    assert (run_dir / "model.pt").read_bytes() == b"model"
    assert json.loads((run_dir / "config.json").read_text())["config"]["total_steps"] == 3
    assert (run_dir / "train.log").exists()


def test_main_requires_wandb_api_key_for_online_mode(monkeypatch, tmp_path, capsys):
    kaggle = _load_kaggle_script()
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    with pytest.raises(SystemExit) as error:
        kaggle.main([
            "--output-dir", str(tmp_path), "--wandb-mode", "online",
        ])

    assert error.value.code == 2
    assert "WANDB_API_KEY" in capsys.readouterr().err
