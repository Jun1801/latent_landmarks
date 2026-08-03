# Kaggle W&B Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a generic Kaggle Python launcher that trains L3P or PN-LMCGS, saves portable run artifacts, and optionally streams structured metrics to W&B.

**Architecture:** L3PTrainer gains an optional scalar metrics callback whose absence is a no-op. scripts/train_kaggle.py reuses get_config, make_vec_env, and L3PTrainer, while owning run directories, config serialization, local logs, lazy W&B setup, and model artifacts.

**Tech Stack:** Python 3, NumPy, PyTorch, pytest, optional W&B SDK

---

## File Map

- Modify: l3p/trainer.py - callback registration and train/evaluation/checkpoint events.
- Create: scripts/train_kaggle.py - general Kaggle launcher and W&B adapter.
- Create: requirements-kaggle.txt - opt-in W&B dependency.
- Modify: README.md - Kaggle commands and W&B authentication/modes.
- Modify: tests/test_pn_integration.py - callback regression tests.
- Modify: tests/test_cli.py - parser/config compatibility tests.
- Create: tests/test_kaggle.py - launcher filesystem and fake-W&B tests.

### Task 1: Add an Optional Scalar Trainer Callback

**Files:**
- Modify: l3p/trainer.py:45-290, 1720-1760
- Modify: tests/test_pn_integration.py

- [ ] **Step 1: Write failing event tests**

~~~python
def _event_trainer(pn_enabled: bool):
    cfg = get_config(
        "PointMaze", pn_lmcgs_enabled=pn_enabled, max_episode_steps=1,
        batch_size=2, hidden_units=4, hidden_layers=1, ae_hidden_units=4,
        ae_hidden_layers=1, embedding_size=2,
    )
    return L3PTrainer(make_vec_env(cfg, 1, 0), cfg)

def test_train_emits_scalar_metrics_only_when_callback_is_configured(monkeypatch):
    trainer = _event_trainer(pn_enabled=False)
    events = []
    trainer.metrics_callback = lambda event, step, metrics: events.append(
        (event, step, dict(metrics))
    )
    def collect_once():
        trainer.total_env_steps += 1
        trainer.episodes_collected += 1
        return 1
    monkeypatch.setattr(trainer, "collect", collect_once)
    monkeypatch.setattr(trainer, "update", lambda _n: {"critic": 1.25})
    monkeypatch.setattr(trainer, "evaluate", lambda _n: 0.5)
    trainer.cfg.log_interval = trainer.cfg.eval_interval = 1
    trainer.train(total_steps=1)
    event, step, metrics = next(value for value in events if value[0] == "train")
    assert step == trainer.total_env_steps
    assert metrics["critic"] == 1.25
    assert all(isinstance(value, float) for value in metrics.values())

def test_pn_evaluation_event_contains_safe_success_metrics(monkeypatch):
    trainer = _event_trainer(pn_enabled=True)
    trainer.last_eval_metrics.update(safe_success_rate=0.75, episode_violation_rate=0.25)
    events = []
    trainer.metrics_callback = lambda event, step, metrics: events.append((event, step, metrics))
    def collect_once():
        trainer.total_env_steps += 1
        trainer.episodes_collected += 1
        return 1
    monkeypatch.setattr(trainer, "collect", collect_once)
    monkeypatch.setattr(trainer, "update", lambda _n: {})
    monkeypatch.setattr(trainer, "evaluate", lambda _n: 0.8)
    trainer.cfg.eval_interval = 1
    trainer.train(total_steps=1)
    metrics = next(value[2] for value in events if value[0] == "evaluation")
    assert metrics["success_rate"] == 0.8
    assert metrics["safe_success_rate"] == 0.75
~~~

- [ ] **Step 2: Verify RED**

Run: python3 -m pytest tests/test_pn_integration.py -q -k 'emits_scalar_metrics or evaluation_event'

Expected: FAIL because the callback parameter and events do not exist.

- [ ] **Step 3: Implement the no-op callback boundary**

Extend L3PTrainer.__init__ with metrics_callback: Optional[Callable[[str, int, Dict[str, float]], None]] = None. Add this helper before train:

~~~python
def _emit_metrics(self, event: str, metrics: Dict[str, float]) -> None:
    if self.metrics_callback is None:
        return
    scalars = {}
    for name, value in metrics.items():
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"metric {name!r} must be finite")
        scalars[str(name)] = scalar
    self.metrics_callback(str(event), int(self.total_env_steps), scalars)
~~~

After the existing periodic train print, emit train with logs, episodes_collected, centroids_initialized, and elapsed seconds. After evaluation, emit evaluation with success_rate plus copied last_eval_metrics only for PN. After interval save, emit checkpoint with checkpoint_saved: 1.0. Do not emit paths or non-scalars.

- [ ] **Step 4: Verify GREEN and compatibility**

Run: python3 -m pytest tests/test_pn_integration.py tests/test_modules.py -q

Expected: PASS; callback-free baseline behavior is unchanged.

- [ ] **Step 5: Commit**

~~~bash
git add l3p/trainer.py tests/test_pn_integration.py
git commit -m "feat: add optional trainer metrics callback"
~~~

### Task 2: Create the Kaggle CLI and Local Run Artifacts

**Files:**
- Create: scripts/train_kaggle.py
- Modify: tests/test_cli.py
- Create: tests/test_kaggle.py

- [ ] **Step 1: Write failing CLI and output tests**

~~~python
def test_kaggle_config_keeps_baseline_default_and_wires_hazard_mode(tmp_path):
    kaggle = _load_script("train_kaggle.py")
    baseline = kaggle.build_config(kaggle.build_parser().parse_args([]))
    assert not baseline.pn_lmcgs_enabled
    args = kaggle.build_parser().parse_args([
        "--env", "PointMaze", "--hazardous-pointmaze", "--output-dir", str(tmp_path),
    ])
    assert kaggle.build_config(args).pn_pointmaze_hazard_enabled

def test_prepare_run_directory_writes_config_under_selected_root(tmp_path):
    kaggle = _load_script("train_kaggle.py")
    run_dir = kaggle.prepare_run_directory(tmp_path, "pointmaze-seed0", {"seed": 0})
    assert run_dir == tmp_path / "pointmaze-seed0"
    assert '"seed": 0' in (run_dir / "config.json").read_text()

def test_main_rejects_zero_steps_and_non_pointmaze_hazards(capsys):
    kaggle = _load_script("train_kaggle.py")
    with pytest.raises(SystemExit) as zero_steps:
        kaggle.main(["--steps", "0", "--wandb-mode", "disabled"])
    with pytest.raises(SystemExit) as wrong_env:
        kaggle.main(["--env", "AntMaze", "--hazardous-pointmaze", "--wandb-mode", "disabled"])
    assert zero_steps.value.code == wrong_env.value.code == 2
    assert "Traceback" not in capsys.readouterr().err
~~~

- [ ] **Step 2: Verify RED**

Run: python3 -m pytest tests/test_cli.py tests/test_kaggle.py -q

Expected: FAIL because scripts/train_kaggle.py does not exist.

- [ ] **Step 3: Implement parser, config, and filesystem helpers**

Mirror scripts/train.py options: --env, --steps, --seed, --workers, --eval-episodes, --pn-lmcgs, and --hazardous-pointmaze. Add --output-dir, --run-name, --save-every, --overwrite, --wandb-project, --wandb-entity, --wandb-run-name, and --wandb-mode. Validate supplied numeric counts as positive before environment creation. Use:

~~~python
def default_output_dir() -> Path:
    kaggle_root = Path("/kaggle/working")
    return kaggle_root / "l3p_runs" if kaggle_root.is_dir() else Path("l3p_runs")
~~~

prepare_run_directory must create only output-dir/run-name, reject an existing non-empty directory unless --overwrite, and write sorted config.json. Store train.log, checkpoint.pt, and model.pt there. Keep W&B imports out of module import time.

- [ ] **Step 4: Verify GREEN**

Run: python3 -m pytest tests/test_cli.py tests/test_kaggle.py -q

Expected: PASS without W&B installed.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/train_kaggle.py tests/test_cli.py tests/test_kaggle.py
git commit -m "feat: add Kaggle training launcher"
~~~

### Task 3: Add Lazy W&B Event Logging and Artifacts

**Files:**
- Modify: scripts/train_kaggle.py
- Create: requirements-kaggle.txt
- Modify: tests/test_kaggle.py

- [ ] **Step 1: Write failing fake-SDK tests**

~~~python
def test_disabled_wandb_never_imports_sdk(monkeypatch):
    kaggle = _load_script("train_kaggle.py")
    monkeypatch.setattr(kaggle.importlib, "import_module", pytest.fail)
    assert kaggle.create_wandb_sink("disabled", "project", None, "run", {}) is None

def test_wandb_sink_prefixes_event_and_uses_global_step(monkeypatch):
    kaggle = _load_script("train_kaggle.py")
    fake = FakeWandb()
    monkeypatch.setattr(kaggle.importlib, "import_module", lambda _name: fake)
    sink = kaggle.create_wandb_sink("offline", "project", None, "run", {"seed": 0})
    sink("evaluation", 42, {"success_rate": 0.5})
    assert fake.logged == [({"evaluation/success_rate": 0.5}, 42)]
~~~

- [ ] **Step 2: Verify RED**

Run: python3 -m pytest tests/test_kaggle.py -q -k wandb

Expected: FAIL because the lazy W&B factory does not exist.

- [ ] **Step 3: Implement W&B lifecycle**

Implement create_wandb_sink(mode, project, entity, run_name, config). Return None immediately for disabled; otherwise lazily call importlib.import_module("wandb"), then wandb.init(project=project, entity=entity, name=run_name, config=config, mode=mode). The returned sink maps each scalar to event/metric and calls wandb.log(values, step=step). For a missing SDK, raise RuntimeError that names pip install -r requirements-kaggle.txt and --wandb-mode disabled. For online mode, require WANDB_API_KEY before creating an environment. In finally, call wandb.finish exactly once. After trainer.save(model_path), create a wandb.Artifact containing model.pt; catch upload errors and retain the local model.

Create requirements-kaggle.txt:

~~~text
-r requirements.txt
wandb>=0.17
~~~

- [ ] **Step 4: Verify GREEN**

Run: python3 -m pytest tests/test_kaggle.py -q -k 'wandb or disabled'

Expected: PASS; disabled mode does not import W&B.

- [ ] **Step 5: Commit**

~~~bash
git add scripts/train_kaggle.py requirements-kaggle.txt tests/test_kaggle.py
git commit -m "feat: log Kaggle training to wandb"
~~~

### Task 4: Wire Main, Document Usage, and Verify

**Files:**
- Modify: scripts/train_kaggle.py
- Modify: README.md
- Modify: tests/test_kaggle.py

- [ ] **Step 1: Write a disabled-mode smoke test**

~~~python
def test_main_saves_model_and_config_without_wandb(tmp_path, monkeypatch):
    kaggle = _load_script("train_kaggle.py")
    class FakeTrainer:
        def __init__(self, _env, _cfg, metrics_callback=None):
            self.metrics_callback = metrics_callback
            self.last_eval_metrics = {"safe_success_rate": 0.75}
        def train(self, checkpoint_path, checkpoint_every):
            if self.metrics_callback is not None:
                self.metrics_callback("train", 3, {"critic": 1.0})
        def save(self, path):
            Path(path).write_bytes(b"model")
        def evaluate(self, _episodes):
            return 0.5
    monkeypatch.setattr(kaggle, "make_vec_env", lambda *_args: object())
    monkeypatch.setattr(kaggle, "L3PTrainer", FakeTrainer)
    kaggle.main(["--env", "PointMaze", "--steps", "3", "--output-dir", str(tmp_path),
                 "--run-name", "smoke", "--wandb-mode", "disabled"])
    assert (tmp_path / "smoke" / "model.pt").read_bytes() == b"model"
    assert (tmp_path / "smoke" / "config.json").exists()
~~~

- [ ] **Step 2: Verify RED**

Run: python3 -m pytest tests/test_kaggle.py -q -k saves_model_and_config

Expected: FAIL until main wires the trainer, paths, final evaluation, and final event.

- [ ] **Step 3: Complete launcher and README**

Make main serialize resolved vars(cfg) and launcher arguments, attach the optional W&B sink to L3PTrainer, train with the interval checkpoint path, save model.pt, run final evaluation, and emit a launcher-owned numeric final event. Document:

~~~bash
pip install -r requirements-kaggle.txt
python scripts/train_kaggle.py --env PointMaze --pn-lmcgs \
  --hazardous-pointmaze --steps 500000 --wandb-project latent-landmarks \
  --wandb-mode online
~~~

Document /kaggle/working/l3p_runs, the WANDB_API_KEY Kaggle Secret, offline sync, and --wandb-mode disabled.

- [ ] **Step 4: Run focused compatibility verification**

Run: python3 -m pytest tests/test_cli.py tests/test_kaggle.py tests/test_pn_integration.py -q

Expected: PASS; existing CLI behavior stays unchanged.

- [ ] **Step 5: Run full verification and CPU smoke**

~~~bash
python3 -m compileall -q l3p scripts tests
python3 -m pytest -q tests
python3 scripts/train_kaggle.py --env PointMaze --steps 200 --workers 1 \
  --pn-lmcgs --hazardous-pointmaze --save-every 100 \
  --wandb-mode disabled --output-dir /tmp/l3p-kaggle-smoke --run-name smoke
git diff --check
~~~

Expected: compilation and tests pass; the smoke directory has config.json, train.log, checkpoint.pt, and model.pt; disabled mode never imports W&B.

- [ ] **Step 6: Inspect scope and commit**

Run GitNexus detect_changes when available. If unavailable, run git diff --stat main...HEAD, git status --short, and git diff --check; preserve the user-owned untracked instruction files. Then:

~~~bash
git add scripts/train_kaggle.py README.md tests/test_kaggle.py
git commit -m "docs: document Kaggle training workflow"
~~~
