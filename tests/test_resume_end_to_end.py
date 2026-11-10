"""End-to-end resume behaviour with a tiny model.

Downloads ``hf-internal-testing/tiny-random-bert`` (a few MB) on first run;
skipped automatically when the Hugging Face Hub is unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jenga_nlp.cli import run_train
from jenga_nlp.core.config import ExperimentConfig
from jenga_nlp.resume import resume_training

TINY_MODEL = "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="module")
def tiny_model_available() -> None:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(TINY_MODEL)
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"cannot load {TINY_MODEL}: {exc}")


def _config(tmp_path: Path, num_epochs: int) -> ExperimentConfig:
    data = tmp_path / "data.jsonl"
    rows = [{"text": f"message number {i}", "label": i % 2} for i in range(8)]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return ExperimentConfig(
        project_name="resume_test",
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
            "num_epochs": num_epochs,
            "device": "cpu",
            "logging": {"service": "none"},
            "checkpoint": {"save_every_n_epochs": 1, "save_best": False, "max_checkpoints": 1},
            "data": {"test_size": 0.25, "seed": 0, "num_workers": 0},
        },
    )


def test_resume_of_finished_run_returns_metrics(tmp_path, tiny_model_available):
    run_train(_config(tmp_path, num_epochs=1))

    results = resume_training(_config(tmp_path, num_epochs=1), tmp_path / "run", device="cpu")

    assert "final_metrics" in results
    assert "eval_loss" in results["final_metrics"]
    assert results["history"] == []  # nothing was retrained


def test_resume_continues_when_epochs_increased(tmp_path, tiny_model_available):
    run_train(_config(tmp_path, num_epochs=1))

    results = resume_training(_config(tmp_path, num_epochs=2), tmp_path / "run", device="cpu")

    assert [h["epoch"] for h in results["history"]] == [2]  # only the new epoch ran
