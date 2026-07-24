"""Whisper feature extraction helpers."""

from __future__ import annotations

from typing import Any

from .config import WhisperFineTuningConfig


def load_whisper_processor(config: WhisperFineTuningConfig) -> Any:
    try:
        from transformers import WhisperProcessor
    except Exception as exc:
        raise ImportError("Whisper feature extraction requires `transformers`.") from exc
    return WhisperProcessor.from_pretrained(config.model_name, language=config.language, task=config.task)


def prepare_whisper_batch(batch: dict[str, Any], processor: Any) -> dict[str, Any]:
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


def prepare_whisper_dataset(dataset: Any, processor: Any) -> Any:
    remove_columns = ["audio", "sentence", "path", "metadata"]
    if hasattr(dataset, "map"):
        try:
            return dataset.map(lambda batch: prepare_whisper_batch(batch, processor), remove_columns=remove_columns)
        except TypeError:
            return dataset.map(lambda batch: prepare_whisper_batch(batch, processor))
    return [prepare_whisper_batch(dict(row), processor) for row in dataset]
