"""Core configuration models for JengaNLP.

This module mirrors the proven experiment schema closely enough to reuse the
stable local training stack while keeping the public package namespace new.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# --- Enums ---


class TaskType(str, Enum):
    SINGLE_LABEL_CLASSIFICATION = "single_label_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    NER = "ner"
    SENTIMENT = "sentiment"
    REGRESSION = "regression"
    QA_SCORING = "qa_scoring"
    # Backward-compatible alias for older configs. This is checklist/QA scoring,
    # not extractive question answering.
    QUESTION_ANSWERING = "question_answering"


class FusionType(str, Enum):
    ATTENTION = "attention"
    CONCATENATION = "concatenation"
    NONE = "none"


class LoggingService(str, Enum):
    TENSORBOARD = "tensorboard"
    MLFLOW = "mlflow"
    NONE = "none"


class TaskSamplingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    PROPORTIONAL = "proportional"
    TEMPERATURE = "temperature"


# --- Config Models ---


class HeadConfig(BaseModel):
    """Configuration for a single prediction head."""

    name: str = Field(..., min_length=1, description="Unique name for this head")
    num_labels: int = Field(..., gt=0, description="Number of output labels/classes")
    weight: float = Field(default=1.0, gt=0.0, description="Loss weight for this head")
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate for this head",
    )
    activation: Optional[str] = Field(
        default=None,
        description="Optional output activation. For regression heads, use 'sigmoid' to bound predictions.",
    )
    output_min: float = Field(default=0.0, description="Minimum output value for this head")
    output_max: float = Field(default=1.0, description="Maximum output value for this head")

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.lower()
        if normalized not in {"sigmoid", "identity", "none"}:
            raise ValueError("activation must be one of: sigmoid, identity, none")
        return normalized

    @model_validator(mode="after")
    def validate_output_range(self) -> "HeadConfig":
        if self.output_max <= self.output_min:
            raise ValueError("output_max must be greater than output_min")
        return self


class TaskConfig(BaseModel):
    """Configuration for a single task."""

    name: str = Field(..., min_length=1, description="Unique name for this task")
    type: TaskType = Field(..., description="Type of NLP task")
    data_path: str = Field(..., min_length=1, description="Path to the task dataset")
    heads: list[HeadConfig] = Field(..., min_length=1, description="Prediction heads for this task")
    text_column: str = Field(default="text", description="Column name containing input text")
    label_column: str = Field(default="labels", description="Column name containing label values")
    label_maps: Optional[dict[str, dict[int, str]]] = Field(
        default=None,
        description="Optional mapping from integer labels to string labels for each head",
    )

    @field_validator("heads")
    @classmethod
    def validate_unique_heads(cls, v: list[HeadConfig]) -> list[HeadConfig]:
        names = [h.name for h in v]
        if len(names) != len(set(names)):
            raise ValueError("Head names must be unique within a task")
        return v


class FusionConfig(BaseModel):
    """Configuration for feature fusion between shared encoder and task heads."""

    type: FusionType = Field(default=FusionType.NONE, description="Type of fusion to apply")
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout rate in the fusion layer",
    )
    use_residual: bool = Field(default=True, description="Whether to use a residual connection")
    num_attention_heads: int = Field(
        default=1,
        gt=0,
        description="Number of attention heads for attention fusion",
    )
    gate_init_value: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Initial value for the fusion gating mechanism",
    )


class ModelConfig(BaseModel):
    """Configuration for the shared model architecture."""

    base_model: str = Field(
        default="distilbert-base-uncased",
        description="HuggingFace model identifier or local model path",
    )
    hidden_size: int = Field(default=768, gt=0, description="Hidden size of encoder representations")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="Dropout rate in the encoder")
    fusion: Optional[FusionConfig] = Field(default=None, description="Fusion layer configuration")
    freeze_encoder_layers: int = Field(
        default=0,
        ge=0,
        description="Number of encoder layers to freeze during training",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to reduce memory usage",
    )


class TokenizerConfig(BaseModel):
    """Configuration for tokenizer preprocessing."""

    max_length: int = Field(
        default=128,
        gt=0,
        le=8192,
        description="Maximum token sequence length",
    )
    padding: str = Field(default="max_length", description="Padding strategy for tokenization")
    truncation: bool = Field(default=True, description="Whether to truncate sequences longer than max_length")


class LoggingConfig(BaseModel):
    """Configuration for experiment logging."""

    service: LoggingService = Field(
        default=LoggingService.TENSORBOARD,
        description="Logging backend service",
    )
    experiment_name: str = Field(
        default="jenga_experiment",
        description="Name of the logging experiment",
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        description="Tracking URI for mlflow or remote logging",
    )
    log_every_n_steps: int = Field(
        default=10,
        gt=0,
        description="Frequency of logging in training steps",
    )


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint saving."""

    save_every_n_epochs: int = Field(
        default=1,
        gt=0,
        description="Save a checkpoint every N epochs",
    )
    save_best: bool = Field(
        default=True,
        description="Save the best checkpoint based on eval performance",
    )
    max_checkpoints: int = Field(
        default=3,
        gt=0,
        description="Maximum number of checkpoints to retain",
    )


class PIIRedactionConfig(BaseModel):
    """Configuration for PII detection and redaction in pipelines."""

    enabled: bool = Field(
        default=False,
        description="Enable PII redaction in the data pipeline",
    )
    strategy: str = Field(
        default="mask",
        description="Redaction strategy to apply to detected PII values",
    )
    detect_types: Optional[list[str]] = Field(
        default=None,
        description="Optional list of PII types to detect; None means detect all supported types",
    )
    hash_salt: str = Field(
        default="jenga-nlp-pii",
        description="Salt value used for deterministic hashing of PII",
    )
    log_detections: bool = Field(
        default=True,
        description="Log detection events without exposing sensitive values",
    )


class IntegrationConfig(BaseModel):
    """Configuration for external integrations."""

    type: str = Field(..., min_length=1, description="Integration type or name")
    enabled: bool = Field(default=True, description="Whether this integration is enabled")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Integration-specific parameters",
    )
    mode: Optional[str] = Field(default=None, description="Optional integration mode")
    scope: Optional[str] = Field(default=None, description="Optional integration scope")


class DataConfig(BaseModel):
    """Configuration for data processing and splitting."""

    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of data reserved for evaluation",
    )
    seed: int = Field(default=42, description="Random seed for data splitting")
    num_workers: int = Field(
        default=0,
        ge=0,
        description="Number of worker processes for data loading",
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin data tensors in memory for faster GPU transfer",
    )
    pii_redaction: PIIRedactionConfig = Field(
        default_factory=PIIRedactionConfig,
        description="PII redaction settings for data preprocessing",
    )


class TrainingConfig(BaseModel):
    """Configuration for the training loop."""

    output_dir: str = Field(
        default="./results",
        description="Directory where training outputs are saved",
    )
    learning_rate: float = Field(default=2e-5, gt=0.0, description="Initial learning rate")
    batch_size: int = Field(default=16, gt=0, description="Training batch size")
    eval_batch_size: Optional[int] = Field(
        default=None,
        gt=0,
        description="Evaluation batch size (defaults to batch_size)",
    )
    num_epochs: int = Field(default=3, gt=0, description="Number of epochs to train")
    weight_decay: float = Field(default=0.01, ge=0.0, description="Weight decay coefficient")
    warmup_steps: int = Field(default=100, ge=0, description="Number of warmup steps")
    max_grad_norm: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum gradient norm for clipping",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        gt=0,
        description="Steps to accumulate gradients before optimizer step",
    )
    use_amp: bool = Field(default=False, description="Use automatic mixed precision training")
    device: str = Field(
        default="auto",
        description="Device selection; use 'auto' to choose between cuda/mps/cpu",
    )
    task_sampling: TaskSamplingStrategy = Field(
        default=TaskSamplingStrategy.ROUND_ROBIN,
        description="Strategy for sampling tasks during training",
    )
    temperature: float = Field(
        default=2.0,
        gt=0.0,
        description="Temperature parameter for proportional task sampling",
    )
    early_stopping_patience: Optional[int] = Field(
        default=None,
        gt=0,
        description="Patience for early stopping on validation performance",
    )
    metric_for_best_model: str = Field(
        default="eval_loss",
        description="Metric name used to select the best model checkpoint",
    )
    greater_is_better: bool = Field(
        default=False,
        description="Whether larger values of the best model metric indicate better performance",
    )
    logging: Optional[LoggingConfig] = Field(
        default=None,
        description="Logging configuration for training",
    )
    checkpoint: Optional[CheckpointConfig] = Field(
        default=None,
        description="Checkpoint saving configuration",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data processing and splitting settings",
    )

    @model_validator(mode="after")
    def fill_eval_batch_size(self) -> "TrainingConfig":
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        return self

    def resolve_device(self) -> str:
        """Resolve the appropriate device when set to 'auto'."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    project_name: str = Field(..., min_length=1, description="Name of the experiment or project")
    tasks: list[TaskConfig] = Field(..., min_length=1, description="Tasks included in the experiment")
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model architecture configuration")
    tokenizer: TokenizerConfig = Field(
        default_factory=TokenizerConfig,
        description="Tokenizer configuration",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training loop configuration",
    )
    integrations: list[IntegrationConfig] = Field(
        default_factory=list,
        description="Optional list of external integrations to enable",
    )

    @field_validator("tasks")
    @classmethod
    def validate_unique_tasks(cls, v: list[TaskConfig]) -> list[TaskConfig]:
        names = [t.name for t in v]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dictionary."""
        return self.model_dump(mode="python")

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = self.model_dump(mode="python")
        _convert_enums(data)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        return cls(**yaml.safe_load(Path(path).read_text()))


def _convert_enums(d: dict[str, Any]) -> None:
    """Recursively convert enum values to primitive strings for YAML serialization."""
    for key, value in d.items():
        if isinstance(value, Enum):
            d[key] = value.value
        elif isinstance(value, dict):
            _convert_enums(value)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _convert_enums(item)
                elif isinstance(item, Enum):
                    d[key][i] = item.value


# def load_config(path: str | Path) -> ExperimentConfig:
#     """Load an experiment config from a YAML file."""
#     return ExperimentConfig.from_yaml(path)

