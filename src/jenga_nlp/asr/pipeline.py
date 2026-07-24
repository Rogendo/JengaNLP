"""Whisper transcription pipeline for JengaNLP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WhisperConfig


@dataclass
class WhisperPipeline:
    """Lazy-loaded transcription pipeline.

    This boundary is intentionally separate from the core NLP stack.
    """

    config: WhisperConfig
    _pipeline: Any = None

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline as hf_pipeline
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Whisper transcription requires `transformers` and its ASR dependencies."
            ) from exc

        device = 0 if self.config.device == "cuda" else -1
        try:
            self._pipeline = hf_pipeline(
                task="automatic-speech-recognition",
                model=self.config.model_name,
                chunk_length_s=self.config.chunk_length_s,
                stride_length_s=self.config.stride_length_s,
                device=device,
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Whisper transcription requires `transformers` and compatible ASR dependencies."
            ) from exc
        return self._pipeline

    def transcribe(self, audio: str | Path, *, language: str | None = None) -> dict[str, Any]:
        """Transcribe a single audio file path."""
        pipe = self._load()
        result = pipe(str(audio))
        text = result.get("text") if isinstance(result, dict) else str(result)
        return {
            "text": text,
            "language": language or self.config.language,
            "model_name": self.config.model_name,
            "audio": str(audio),
        }
