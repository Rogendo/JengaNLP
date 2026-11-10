"""End-to-end checks for what a training run saves, locally and in MLflow.

Uses ``hf-internal-testing/tiny-random-bert`` (a few MB); skipped when the
Hugging Face Hub is unreachable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jenga_nlp.cli import run_evaluate, run_train
from jenga_nlp.core.config import ExperimentConfig

TINY_MODEL = "hf-internal-testing/tiny-random-bert"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"


@pytest.fixture(scope="module")
def tiny_model_available() -> None:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot load {TINY_MODEL}: {exc}")


def _config(tmp_path: Path, logging: dict | None = None, checkpoints: bool = False) -> ExperimentConfig:
    data = tmp_path / "data.jsonl"
    rows = [{"text": f"message number {i}", "label": i % 2} for i in range(8)]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return ExperimentConfig(
        project_name="artifacts_test",
        model={"base_model": TINY_MODEL},
        tokenizer={"max_length": 16},
        tasks=[
            {
                "name": "clf",
                "type": "single_label_classification",
                "data_path": str(data),
                "label_column": "label",
                "heads": [{"name": "head", "num_labels": 2}],
            }
        ],
        training={
            "output_dir": str(tmp_path / "run"),
            "batch_size": 4,
            "eval_batch_size": 4,
            "num_epochs": 1,
            "device": "cpu",
            "logging": logging or {"service": "none"},
            "checkpoint": {"save_every_n_epochs": 1, "save_best": checkpoints, "max_checkpoints": 1}
            if checkpoints
            else None,
            "data": {"test_size": 0.25, "seed": 0, "num_workers": 0},
        },
    )


def test_model_folder_is_self_contained(tmp_path, tiny_model_available):
    run_train(_config(tmp_path))
    run_dir = tmp_path / "run"

    for name in ("model.pt", "metadata.json", "experiment_config.yaml", "encoder_config", "tokenizer"):
        assert (run_dir / name).exists(), f"missing {name}"

    # Load and predict with the Hub forced offline and an empty HF cache:
    # this only works if nothing has to be downloaded.
    script = textwrap.dedent(
        f"""
        from jenga_nlp.inference.pipeline import InferencePipeline
        p = InferencePipeline.from_checkpoint(
            model_dir={str(run_dir)!r}, config_path={str(run_dir / "experiment_config.yaml")!r}, device="cpu")
        r = p.predict(text="message number 3", task_name="clf").to_dict()
        print("PRED", r["task_results"]["clf"]["heads"]["head"]["prediction"])
        """
    )
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(tmp_path / "empty_hf_home"),
        "PYTHONPATH": os.pathsep.join([str(SRC_DIR), os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PRED" in proc.stdout


def _mlflow_logging(tmp_path: Path, **extra) -> dict:
    return {
        "service": "mlflow",
        "experiment_name": "artifacts_test",
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        **extra,
    }


def _artifact_paths(client, run_id: str, path: str | None = None) -> set[str]:
    found: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            found |= _artifact_paths(client, run_id, item.path)
        else:
            found.add(item.path)
    return found


def test_mlflow_receives_model_eval_results_and_optional_checkpoints(tmp_path, tiny_model_available, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # local MLflow artifact store lands in tmp_path

    # 1. Default: model folder uploaded, checkpoints not.
    cfg = _config(tmp_path, logging=_mlflow_logging(tmp_path), checkpoints=True)
    run_train(cfg)
    client = mlflow.MlflowClient(tracking_uri=cfg.training.logging.tracking_uri)
    exp = client.get_experiment_by_name("artifacts_test")
    train_run = client.search_runs([exp.experiment_id], filter_string="tags.mode = 'train'")[0]
    paths = _artifact_paths(client, train_run.info.run_id)
    for expected in ("model/model.pt", "model/metadata.json", "model/experiment_config.yaml",
                     "model/encoder_config/config.json"):
        assert expected in paths, f"{expected} not in {sorted(paths)}"
    assert any(p.startswith("model/tokenizer/") for p in paths)
    assert not any(p.startswith("checkpoints/") for p in paths)

    # 2. Evaluate: eval_results.json written next to the model and uploaded.
    run_evaluate(cfg, model_dir=tmp_path / "run")
    results = json.loads((tmp_path / "run" / "eval_results.json").read_text())
    assert results["tasks"]["clf"]["eval_samples"] == 2
    assert "eval_loss" in results["metrics"]
    eval_run = client.search_runs([exp.experiment_id], filter_string="tags.mode = 'eval'")[0]
    assert "eval_results.json" in _artifact_paths(client, eval_run.info.run_id)
    assert eval_run.data.metrics["eval_loss"] == pytest.approx(results["metrics"]["eval_loss"])

    # 3. Opt-in: checkpoints uploaded too.
    cfg_ckpt = _config(tmp_path, logging=_mlflow_logging(tmp_path, log_checkpoints=True), checkpoints=True)
    run_train(cfg_ckpt)
    latest = client.search_runs(
        [exp.experiment_id], filter_string="tags.mode = 'train'", order_by=["attributes.start_time DESC"]
    )[0]
    assert any(p.startswith("checkpoints/best/") for p in _artifact_paths(client, latest.info.run_id))


def test_tensorboard_separates_train_and_eval_runs_and_records_params(tmp_path, tiny_model_available):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = _config(tmp_path, logging={"service": "tensorboard", "experiment_name": "tb_test", "log_every_n_steps": 1})
    run_train(cfg)
    run_evaluate(cfg, model_dir=tmp_path / "run")

    run_dirs = sorted(p for p in (tmp_path / "run" / "logs" / "tb_test").iterdir() if p.is_dir())
    names = [p.name for p in run_dirs]
    assert len(run_dirs) == 2, names
    train_dir = next(p for p in run_dirs if not p.name.startswith("artifacts_test-eval"))
    eval_dir = next(p for p in run_dirs if p.name.startswith("artifacts_test-eval"))

    train = EventAccumulator(str(train_dir), size_guidance={"tensors": 0})
    train.Reload()
    assert "train/loss" in train.Tags()["scalars"]
    assert "eval/eval_loss" in train.Tags()["scalars"]
    assert {"run/params/text_summary", "run/config/text_summary"} <= set(train.Tags()["tensors"])

    evaluation = EventAccumulator(str(eval_dir))
    evaluation.Reload()
    assert "eval/eval_loss" in evaluation.Tags()["scalars"]
    assert "train/loss" not in evaluation.Tags()["scalars"]
