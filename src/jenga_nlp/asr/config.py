"""Configuration for Whisper-based ASR workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..core.config import IntegrationConfig


class WhisperConfig(BaseModel):
    """Minimal config for a Whisper transcription workflow."""

    model_name: str = Field(default="openai/whisper-small")
    language: str | None = Field(default="sw")
    task: str = Field(default="transcribe")
    chunk_length_s: int = Field(default=30, gt=0)
    stride_length_s: int = Field(default=5, ge=0)
    device: str = Field(default="auto")
    batch_size: int = Field(default=8, gt=0)


class WhisperDatasetConfig(BaseModel):
    """Dataset source configuration for Whisper fine-tuning."""

    source: str = Field(default="manifest")
    name: str | None = None
    train_manifest: Path | None = None
    eval_manifest: Path | None = None
    train_split: str = Field(default="train")
    eval_split: str = Field(default="validation")
    audio_column: str = Field(default="audio")
    text_column: str = Field(default="text")
    duration_column: str | None = Field(default="duration_s")
    sentence_column: str = Field(default="auto")
    language_filter: str | None = None
    streaming: bool = Field(default=True)
    trust_remote_code: bool = Field(default=True)
    local_path: Path | None = None
    local_format: str | None = None
    excel_file: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    num_train_samples: int | None = Field(default=None, gt=0)
    num_eval_samples: int | None = Field(default=None, gt=0)


class WhisperPreprocessingConfig(BaseModel):
    """Audio and transcript preprocessing rules for Whisper fine-tuning."""

    sampling_rate: int = Field(default=16000, gt=0)
    min_audio_duration_s: float = Field(default=1.0, ge=0.0)
    max_audio_duration_s: float = Field(default=30.0, gt=0.0)
    max_transcript_chars: int = Field(default=1000, gt=0)
    strip_text: bool = Field(default=True)
    lowercase: bool = Field(default=False)
    normalize_text: bool = Field(default=False)
    validate_audio_paths: bool = Field(default=False)


class WhisperFineTuningConfig(BaseModel):
    """Config for Whisper fine-tuning."""

    model_name: str = Field(default="openai/whisper-small")
    language: str | None = Field(default="sw")
    task: str = Field(default="transcribe")
    output_dir: Path = Field(default=Path("./results/whisper"))
    learning_rate: float = Field(default=1e-5, gt=0.0)
    num_epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=4, gt=0)
    eval_batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    warmup_steps: int = Field(default=100, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    device: str = Field(default="auto")
    fp16: bool = Field(default=False)
    max_steps: int | None = Field(default=None, gt=0)
    eval_steps: int = Field(default=500, gt=0)
    save_steps: int = Field(default=500, gt=0)
    logging_steps: int = Field(default=25, gt=0)
    generation_max_length: int = Field(default=225, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    load_best_model_at_end: bool = Field(default=True)
    metric_for_best_model: str = Field(default="wer")
    greater_is_better: bool = Field(default=False)
    save_total_limit: int = Field(default=3, gt=0)
    optim: str = Field(default="adamw_torch")
    dataloader_num_workers: int = Field(default=0, ge=0)
    push_to_hub: bool = Field(default=False)
    hub_model_id: str | None = None
    hub_strategy: str = Field(default="every_save")
    hub_private: bool = Field(default=False)
    use_peft: bool = Field(default=False)
    peft_config: dict[str, Any] = Field(default_factory=dict)
    max_audio_duration_s: int = Field(default=30, gt=0)
    min_audio_duration_s: int = Field(default=1, gt=0)
    train_manifest: Path | None = None
    eval_manifest: Path | None = None
    dataset: WhisperDatasetConfig = Field(default_factory=WhisperDatasetConfig)
    preprocessing: WhisperPreprocessingConfig = Field(default_factory=WhisperPreprocessingConfig)
    integrations: list[IntegrationConfig] = Field(default_factory=list)

    def resolve_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size

    def resolve_train_data(self) -> Path | None:
        return self.train_manifest or self.dataset.train_manifest

    def resolve_eval_data(self) -> Path | None:
        return self.eval_manifest or self.dataset.eval_manifest

    def to_yaml(self, path: str | Path) -> None:
        data = self.model_dump(mode="json")
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WhisperFineTuningConfig":
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))
