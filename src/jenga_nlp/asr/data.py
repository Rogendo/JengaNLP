"""Dataset inspection helpers for Whisper ASR workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import WhisperDatasetConfig, WhisperPreprocessingConfig


@dataclass(slots=True)
class ASRManifestSample:
    """A normalized audio/transcript sample from a manifest file."""

    audio: str
    text: str
    duration_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def load_manifest_samples(
    manifest_path: str | Path,
    dataset: WhisperDatasetConfig,
    preprocessing: WhisperPreprocessingConfig,
    *,
    limit: int | None = None,
) -> list[ASRManifestSample]:
    """Load and lightly validate JSONL manifest samples."""
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"ASR manifest not found: {path}")

    samples: list[ASRManifestSample] = []
    for row in _iter_jsonl(path):
        if limit is not None and len(samples) >= limit:
            break

        audio = row.get(dataset.audio_column)
        text = row.get(dataset.text_column)
        if audio is None or text is None:
            continue

        text = str(text)
        if preprocessing.strip_text:
            text = text.strip()
        if preprocessing.lowercase:
            text = text.lower()
        if not text or len(text) > preprocessing.max_transcript_chars:
            continue

        duration = None
        if dataset.duration_column and row.get(dataset.duration_column) is not None:
            duration = float(row[dataset.duration_column])
            if duration < preprocessing.min_audio_duration_s:
                continue
            if duration > preprocessing.max_audio_duration_s:
                continue

        audio_path = str(audio)
        if preprocessing.validate_audio_paths and not Path(audio_path).exists():
            continue

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {dataset.audio_column, dataset.text_column, dataset.duration_column}
        }
        samples.append(ASRManifestSample(audio=audio_path, text=text, duration_s=duration, metadata=metadata))

    return samples


def inspect_manifest(
    manifest_path: str | Path | None,
    dataset: WhisperDatasetConfig,
    preprocessing: WhisperPreprocessingConfig,
    *,
    limit: int | None = None,
) -> dict[str, Any]:
    """Return a lightweight summary for a manifest path without requiring audio dependencies."""
    if manifest_path is None:
        return {"status": "not_configured", "path": None, "num_samples": 0}

    path = Path(manifest_path)
    if not path.exists():
        return {"status": "missing", "path": str(path), "num_samples": 0}

    samples = load_manifest_samples(path, dataset, preprocessing, limit=limit)
    durations = [sample.duration_s for sample in samples if sample.duration_s is not None]
    return {
        "status": "ready",
        "path": str(path),
        "num_samples": len(samples),
        "audio_column": dataset.audio_column,
        "text_column": dataset.text_column,
        "duration_min_s": min(durations) if durations else None,
        "duration_max_s": max(durations) if durations else None,
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path} at line {line_no}")
            yield row
