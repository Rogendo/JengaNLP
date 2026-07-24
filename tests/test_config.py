"""
test_config.py — verify ExperimentConfig parses correctly from dicts and YAML.

These tests are purely in-memory (no disk I/O beyond tempfiles, no model downloads).
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from jenga_nlp.core.config import (
    ExperimentConfig,
    FusionType,
    HeadConfig,
    LoggingService,
    ModelConfig,
    TaskConfig,
    TaskSamplingStrategy,
    TaskType,
    TokenizerConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Minimal valid config dict fixture
# ---------------------------------------------------------------------------

MINIMAL_TASK = {
    "name": "test_task",
    "type": "single_label_classification",
    "data_path": "data/test.jsonl",
    "heads": [{"name": "test_head", "num_labels": 2}],
}

MINIMAL_CONFIG = {
    "project_name": "test_project",
    "tasks": [MINIMAL_TASK],
}


# ---------------------------------------------------------------------------
# ExperimentConfig — from dict
# ---------------------------------------------------------------------------

def test_experiment_config_minimal():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    assert cfg.project_name == "test_project"
    assert len(cfg.tasks) == 1
    assert cfg.tasks[0].name == "test_task"


def test_experiment_config_defaults():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    assert cfg.model.base_model == "distilbert-base-uncased"
    assert cfg.tokenizer.max_length == 128
    assert cfg.training.num_epochs == 3
    assert cfg.training.batch_size == 16
    assert cfg.training.device == "auto"
    assert cfg.training.task_sampling == TaskSamplingStrategy.ROUND_ROBIN


def test_experiment_config_eval_batch_size_defaults_to_batch_size():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    assert cfg.training.eval_batch_size == cfg.training.batch_size


def test_experiment_config_task_types():
    for task_type in [
        "single_label_classification",
        "multi_label_classification",
        "sentiment",
        "ner",
        "regression",
        "qa_scoring",
        "question_answering",  # backward-compat alias
    ]:
        task = {**MINIMAL_TASK, "type": task_type, "name": f"task_{task_type}"}
        cfg = ExperimentConfig(project_name="p", tasks=[task])
        assert cfg.tasks[0].type == TaskType(task_type)


def test_experiment_config_rejects_duplicate_tasks():
    with pytest.raises(Exception):
        ExperimentConfig(
            project_name="p",
            tasks=[MINIMAL_TASK, MINIMAL_TASK],  # duplicate name
        )


def test_experiment_config_rejects_empty_tasks():
    with pytest.raises(Exception):
        ExperimentConfig(project_name="p", tasks=[])


# ---------------------------------------------------------------------------
# HeadConfig validation
# ---------------------------------------------------------------------------

def test_head_config_valid():
    h = HeadConfig(name="fraud_head", num_labels=2)
    assert h.weight == 1.0
    assert h.dropout == 0.1


def test_head_config_sigmoid_activation():
    h = HeadConfig(name="risk", num_labels=1, activation="sigmoid")
    assert h.activation == "sigmoid"


def test_head_config_rejects_bad_activation():
    with pytest.raises(Exception):
        HeadConfig(name="risk", num_labels=1, activation="relu")


def test_head_config_rejects_inverted_output_range():
    with pytest.raises(Exception):
        HeadConfig(name="risk", num_labels=1, output_min=1.0, output_max=0.0)


def test_head_config_rejects_zero_labels():
    with pytest.raises(Exception):
        HeadConfig(name="x", num_labels=0)


def test_task_config_rejects_duplicate_heads():
    with pytest.raises(Exception):
        TaskConfig(
            name="task",
            type="single_label_classification",
            data_path="data/x.jsonl",
            heads=[
                {"name": "same", "num_labels": 2},
                {"name": "same", "num_labels": 3},
            ],
        )


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

def test_model_config_defaults():
    m = ModelConfig()
    assert m.base_model == "distilbert-base-uncased"
    assert m.hidden_size == 768
    assert m.dropout == 0.1
    assert m.fusion is None


def test_model_config_fusion_none_enum():
    from jenga_nlp.core.config import FusionConfig
    fc = FusionConfig(type="none")
    assert fc.type == FusionType.NONE


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

def test_training_config_device_auto():
    t = TrainingConfig()
    assert t.device == "auto"


def test_training_config_task_sampling_strategies():
    for strategy in ["round_robin", "proportional", "temperature"]:
        t = TrainingConfig(task_sampling=strategy)
        assert t.task_sampling == TaskSamplingStrategy(strategy)


def test_training_config_logging_service_enum():
    t = TrainingConfig(logging={"service": "mlflow", "experiment_name": "test"})
    assert t.logging.service == LoggingService.MLFLOW


# ---------------------------------------------------------------------------
# ExperimentConfig — YAML round-trip
# ---------------------------------------------------------------------------

def test_config_to_yaml_and_back():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        cfg.to_yaml(yaml_path)

        assert yaml_path.exists()
        loaded = ExperimentConfig.from_yaml(yaml_path)

    assert loaded.project_name == cfg.project_name
    assert loaded.tasks[0].name == cfg.tasks[0].name
    assert loaded.model.base_model == cfg.model.base_model


def test_config_from_yaml_string():
    yaml_content = yaml.safe_dump({
        "project_name": "yaml_test",
        "tasks": [
            {
                "name": "fraud",
                "type": "single_label_classification",
                "data_path": "data/messages.jsonl",
                "heads": [{"name": "fraud_head", "num_labels": 2}],
            }
        ],
        "model": {"base_model": "hf-internal-testing/tiny-random-bert"},
        "training": {"num_epochs": 1, "batch_size": 4, "device": "cpu"},
    })

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        cfg = ExperimentConfig.from_yaml(f.name)

    assert cfg.project_name == "yaml_test"
    assert cfg.model.base_model == "hf-internal-testing/tiny-random-bert"
    assert cfg.training.num_epochs == 1
    assert cfg.training.device == "cpu"


def test_config_to_dict():
    cfg = ExperimentConfig(**MINIMAL_CONFIG)
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["project_name"] == "test_project"
    assert "tasks" in d


# ---------------------------------------------------------------------------
# TokenizerConfig
# ---------------------------------------------------------------------------

def test_tokenizer_config_defaults():
    t = TokenizerConfig()
    assert t.max_length == 128
    assert t.padding == "max_length"
    assert t.truncation is True


def test_tokenizer_config_rejects_zero_max_length():
    with pytest.raises(Exception):
        TokenizerConfig(max_length=0)
