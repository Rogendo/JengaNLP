"""Local speech-to-text serving helpers for JengaNLP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WhisperConfig
from .pipeline import WhisperPipeline


@dataclass
class WhisperServer:
    """Small local serving wrapper around the Whisper transcription pipeline."""

    config: WhisperConfig
    pipeline: WhisperPipeline | None = None

    def _pipeline(self) -> WhisperPipeline:
        if self.pipeline is None:
            self.pipeline = WhisperPipeline(self.config)
        return self.pipeline

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_name": self.config.model_name,
            "language": self.config.language,
            "task": self.config.task,
        }

    def transcribe(self, audio: str | Path, *, language: str | None = None) -> dict[str, Any]:
        return self._pipeline().transcribe(audio, language=language)
