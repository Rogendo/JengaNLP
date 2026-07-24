#!/usr/bin/env python3
"""Run direct inference against trained checkpoints stored under /tmp.

This is an example for users who want to test saved models without going through
the CLI. It discovers a few known model directories under /tmp, loads each one
with the package's inference pipeline, and runs a couple of sample texts.

Usage:
    python examples/run_inference_from_tmp.py
    python examples/run_inference_from_tmp.py --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu
    python examples/run_inference_from_tmp.py --model-dir /tmp/jenganlp_fixture_multilabel_cpu --task-name mpesa_risk_tags
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL_DIRS = [
    "/tmp/jenganlp_fixture_mpesa_fraud_cpu",
    "/tmp/jenganlp_fixture_multilabel_cpu",
    "/tmp/jenganlp_fixture_ner_cpu",
    "/tmp/jenganlp_fixture_qa_scoring_cpu",
]

DEFAULT_TEXTS = {
    "classification": "Received a suspicious M-PESA message asking for my PIN.",
    "multilabel": "The account was locked after a phishing email and fake login page.",
    "ner": "John lives in Nairobi and called the child helpline for help.",
    "qa": "The agent answered the question clearly and the customer was satisfied.",
}


def _read_task_name(model_dir: str | Path) -> str | None:
    config_path = Path(model_dir) / "experiment_config.yaml"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    tasks = config.get("tasks", [])
    if not tasks:
        return None
    return tasks[0].get("name")


def _load_pipeline(model_dir: str | Path, config_path: str | Path | None = None) -> Any:
    from jenga_nlp.inference.pipeline import InferencePipeline

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if config_path is None:
        config_path = model_dir / "experiment_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return InferencePipeline.from_checkpoint(
        model_dir=model_dir,
        config_path=config_path,
        device="cpu",
    )


def run_one(model_dir: str | Path, text: str, task_name: str | None = None) -> dict[str, Any]:
    pipeline = _load_pipeline(model_dir)
    kwargs: dict[str, Any] = {"text": text, "threshold": 0.5}
    if task_name is not None:
        kwargs["task_name"] = task_name
    result = pipeline.predict(**kwargs)
    return result.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct inference with saved JengaNLP checkpoints")
    parser.add_argument("--model-dir", default=None, help="Optional single model directory to test")
    parser.add_argument("--task-name", default=None, help="Optional task name override")
    parser.add_argument("--text", default=None, help="Optional custom text input")
    args = parser.parse_args()

    model_dirs = [args.model_dir] if args.model_dir else DEFAULT_MODEL_DIRS
    text_input = args.text or None

    for model_dir in model_dirs:
        path = Path(model_dir)
        if not path.exists():
            print(f"[SKIP] Missing model directory: {path}")
            continue

        print(f"\n=== Inference: {path} ===")
        try:
            task_name = args.task_name
            if task_name is None:
                task_name = _read_task_name(path)

            sample_text = text_input or DEFAULT_TEXTS.get(
                "multilabel" if "multilabel" in str(path) else "classification" if "fraud" in str(path) else "ner" if "ner" in str(path) else "qa",
                "This is a sample input.",
            )
            result = run_one(path, sample_text, task_name=task_name)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover - example script convenience
            print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
