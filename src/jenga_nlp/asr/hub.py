"""Hugging Face Hub helpers for ASR models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import WhisperFineTuningConfig


def generate_asr_model_card(config: WhisperFineTuningConfig, metrics: dict[str, Any] | None = None) -> str:
    metric_lines = ""
    if metrics:
        metric_lines = "\n".join(f"- {key}: {value}" for key, value in metrics.items())
    return (
        "---\n"
        f"language: {config.language or 'unknown'}\n"
        "tags:\n"
        "- automatic-speech-recognition\n"
        "- whisper\n"
        "- jenga-nlp\n"
        f"base_model: {config.model_name}\n"
        "license: apache-2.0\n"
        "---\n\n"
        "# JengaNLP Whisper ASR Model\n\n"
        f"Fine-tuned from `{config.model_name}` for `{config.language or 'unknown'}` speech-to-text.\n\n"
        "## Task\n\n"
        f"- ASR task: `{config.task}`\n"
        f"- Dataset source: `{config.dataset.source}`\n"
        f"- Sampling rate: `{config.preprocessing.sampling_rate}`\n\n"
        "## Metrics\n\n"
        f"{metric_lines or 'Metrics were not provided.'}\n\n"
        "## Usage\n\n"
        "```python\n"
        "from transformers import pipeline\n\n"
        "asr = pipeline(\"automatic-speech-recognition\", model=\"YOUR_REPO_ID\")\n"
        "print(asr(\"audio.wav\"))\n"
        "```\n"
    )


def save_asr_model_card(output_dir: str | Path, config: WhisperFineTuningConfig, metrics: dict[str, Any] | None = None) -> Path:
    path = Path(output_dir) / "README.md"
    path.write_text(generate_asr_model_card(config, metrics), encoding="utf-8")
    return path


def push_asr_model_to_hub(model_dir: str | Path, repo_id: str, *, token: str | None = None, private: bool = False) -> str:
    try:
        from huggingface_hub import HfApi, upload_folder
    except Exception as exc:
        raise ImportError("ASR Hub push requires `huggingface_hub`.") from exc

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=str(model_dir), token=token)
    return f"https://huggingface.co/{repo_id}"
