"""Configuration for domain-adaptive masked-language-model (MLM) pretraining.

Continue MLM training of an XLM-R-family encoder (AfriBERTa, AfroXLMR, ...) on
local text so it learns domain vocabulary and style (Swahili, Sheng,
code-switching, M-PESA / helpline language). The result is a standard Hugging
Face model that can be used as ``model.base_model`` in any JengaNLP task config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..core.config import ConfigModel, LoggingConfig

# Real, sensitive sources may be repeated at most this many times: heavy repetition
# makes masked language models memorise exact strings (including any PII left over).
MAX_SENSITIVE_REPEAT = 2


class CorpusSource(ConfigModel):
    """One source of training text: a local file or a Hugging Face dataset."""

    name: str = Field(..., min_length=1, description="Short name used in reports")
    path: Path | None = Field(default=None, description="Local .txt (one example per line), .jsonl or .csv file")
    format: Literal["auto", "txt", "jsonl", "csv"] = Field(default="auto")
    text_column: str = Field(default="text", description="Text field for jsonl/csv/HF datasets")
    hf_dataset: str | None = Field(default=None, description="Hugging Face dataset name, e.g. wikimedia/wikipedia")
    hf_config: str | None = Field(default=None, description="Dataset config, e.g. 20231101.sw")
    hf_split: str = Field(default="train")
    repeat: int = Field(default=1, ge=1, description="How many times training examples are repeated")
    sensitive: bool = Field(
        default=False,
        description="Real data about people (chats, case notes). Requires NER redaction, a recall check, repeat <= 2",
    )
    max_examples: int | None = Field(default=None, gt=0, description="Cap on examples read from this source")

    @model_validator(mode="after")
    def _one_location(self) -> CorpusSource:
        if (self.path is None) == (self.hf_dataset is None):
            raise ValueError(f"corpus source '{self.name}': set exactly one of `path` or `hf_dataset`")
        return self


class MLMPrivacyConfig(ConfigModel):
    """PII redaction applied to the corpus before training.

    Regex redaction (phones, IDs, KRA PINs, M-PESA IDs, emails, URLs, ...) runs on
    every source. An NER model finds names and places; by default it runs on
    sensitive sources only. Sensitive sources also require a labelled sample on
    which the whole redaction stack must reach ``min_recall`` before training.
    """

    redact: bool = Field(default=True, description="Regex PII redaction on every source")
    detect_types: list[str] | None = Field(
        default=None, description="Built-in PII types to detect (default: all), e.g. ['phone_ke', 'email']"
    )
    extra_patterns: dict[str, str] = Field(
        default_factory=dict, description="Extra regexes to redact, as {TAG: regex}; replaced with [TAG]"
    )
    ner_model: str | None = Field(
        default=None,
        description="JengaNLP NER model folder or Hugging Face token-classification model "
        "(e.g. masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0)",
    )
    ner_task: str | None = Field(default=None, description="NER task name, for multi-task JengaNLP models")
    ner_labels: dict[str, str] | None = Field(
        default=None,
        description="NER label -> redaction tag, e.g. {PER: NAME, VICTIM: NAME, LOC: LOCATION} (default: built-in map)",
    )
    ner_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Low values favour recall")
    ner_batch_size: int = Field(default=16, gt=0)
    ner_device: str = Field(default="auto")
    ner_on_all_sources: bool = Field(default=False, description="Also run the NER model on non-sensitive sources")
    eval_set: Path | None = Field(
        default=None, description="Labelled JSONL (JengaNLP NER format) used to measure PII recall before training"
    )
    min_recall: float = Field(default=0.95, gt=0.0, le=1.0, description="Required recall of PII entities")


class MLMConfig(ConfigModel):
    """Domain-adaptive MLM pretraining run."""

    project_name: str = Field(default="mlm_domain_adaptation")
    base_model: str = Field(..., description="Encoder to adapt, e.g. castorini/afriberta_large")
    output_dir: Path = Field(default=Path("./results/mlm"))
    corpus: list[CorpusSource] = Field(..., min_length=1)

    # Data
    block_size: int = Field(default=128, ge=8, description="Tokens per training example after grouping")
    mlm_probability: float = Field(default=0.15, gt=0.0, lt=1.0)
    whole_word_mask: bool = Field(default=False)
    eval_fraction: float = Field(default=0.05, gt=0.0, lt=0.5, description="Held-out share of each source")
    min_chars: int = Field(default=20, ge=0, description="Drop examples shorter than this after cleaning")
    dedup: bool = Field(default=True, description="Drop exact duplicates (across all sources) before splitting")
    seed: int = Field(default=42)

    # Training
    learning_rate: float = Field(default=1e-4, gt=0.0)
    num_epochs: float = Field(default=3.0, gt=0.0)
    max_steps: int | None = Field(default=None, gt=0)
    batch_size: int = Field(default=32, gt=0)
    eval_batch_size: int | None = Field(default=None, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    warmup_ratio: float = Field(default=0.06, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    bf16: bool = Field(default=False)
    fp16: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=False)
    eval_steps: int = Field(default=500, gt=0)
    save_steps: int = Field(default=500, gt=0)
    logging_steps: int = Field(default=50, gt=0)
    save_total_limit: int = Field(default=2, gt=0)
    keep_checkpoints: bool = Field(
        default=False,
        description="Keep checkpoints/ (weights + optimizer state, for resuming) after a successful run",
    )
    device: str = Field(default="auto", description="auto, cpu or cuda")
    dataloader_num_workers: int = Field(default=0, ge=0)

    # Privacy, tracking, evaluation, publishing
    pii: MLMPrivacyConfig = Field(default_factory=MLMPrivacyConfig)
    logging: LoggingConfig | None = Field(default=None)
    probes_path: Path | None = Field(
        default=None, description='JSONL of {"domain": ..., "text": ...} sentences for per-domain evaluation'
    )
    push_to_hub: bool = Field(default=False)
    hub_model_id: str | None = Field(default=None)
    hub_private: bool = Field(default=True)

    @model_validator(mode="after")
    def _guard_rails(self) -> MLMConfig:
        if self.bf16 and self.fp16:
            raise ValueError("set at most one of bf16 / fp16")
        names = [s.name for s in self.corpus]
        if len(names) != len(set(names)):
            raise ValueError("corpus source names must be unique")
        sensitive = [s for s in self.corpus if s.sensitive]
        for src in sensitive:
            if src.repeat > MAX_SENSITIVE_REPEAT:
                raise ValueError(
                    f"sensitive source '{src.name}' has repeat={src.repeat}; at most {MAX_SENSITIVE_REPEAT} is "
                    "allowed, because heavy repetition makes the model memorise exact text (including names)"
                )
        if sensitive:
            if not self.pii.redact:
                raise ValueError("sensitive sources require pii.redact: true")
            if not self.pii.ner_model:
                raise ValueError(
                    "sensitive sources require pii.ner_model: regex redaction cannot find names or places, "
                    "so set an NER model (a JengaNLP NER model or e.g. masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0)"
                )
            if self.pii.eval_set is None:
                raise ValueError(
                    "sensitive sources require pii.eval_set: a labelled sample (JengaNLP NER JSONL) on which "
                    "PII recall is measured before training"
                )
            if not self.hub_private:
                raise ValueError("models trained on sensitive sources can only be pushed with hub_private: true")
        if self.push_to_hub and not self.hub_model_id:
            raise ValueError("push_to_hub requires hub_model_id")
        return self

    @property
    def has_sensitive_sources(self) -> bool:
        return any(s.sensitive for s in self.corpus)

    def resolve_eval_batch_size(self) -> int:
        return self.eval_batch_size or self.batch_size

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

    @classmethod
    def from_yaml(cls, path: str | Path) -> MLMConfig:
        """Load from YAML; relative file paths resolve against the config's folder if not found from the cwd."""
        path = Path(path)
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        base = path.resolve().parent
        for src in data.get("corpus") or []:
            if isinstance(src, dict) and isinstance(src.get("path"), str):
                src["path"] = _resolve(src["path"], base)
        pii = data.get("pii")
        if isinstance(pii, dict):
            for key in ("eval_set", "ner_model"):  # ner_model may also be a Hub ID; left as-is if not a path
                if isinstance(pii.get(key), str):
                    pii[key] = _resolve(pii[key], base)
        if isinstance(data.get("probes_path"), str):
            data["probes_path"] = _resolve(data["probes_path"], base)
        return cls(**data)


def _resolve(raw: str, config_dir: Path) -> str:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return raw
    relative = config_dir / candidate
    return str(relative.resolve()) if relative.exists() else raw
