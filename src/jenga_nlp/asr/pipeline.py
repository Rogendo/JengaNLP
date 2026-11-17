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

        device = _resolve_device(self.config.device)
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
        """Transcribe a single audio file path.

        Audio is decoded with JengaNLP's own loader (soundfile/librosa, resampled to 16 kHz), so
        ffmpeg is not required. Decoding is forced to ``language`` (or the configured language)
        instead of letting Whisper guess it.
        """
        from .config import WhisperPreprocessingConfig
        from .loaders import load_audio_file

        pipe = self._load()
        # no duration filter at inference: long files are chunked by the pipeline
        payload = load_audio_file(audio, WhisperPreprocessingConfig(min_audio_duration_s=0.0, max_audio_duration_s=1e9))
        if payload is None:
            raise ValueError(f"Could not read audio file: {audio}")
        lang = language or self.config.language
        generate_kwargs = {"task": self.config.task}
        if lang:
            generate_kwargs["language"] = lang
        result = pipe({"raw": payload["array"], "sampling_rate": payload["sampling_rate"]},
                      generate_kwargs=generate_kwargs)
        text = result.get("text") if isinstance(result, dict) else str(result)
        return {
            "text": text,
            "language": language or self.config.language,
            "model_name": self.config.model_name,
            "audio": str(audio),
        }


def _resolve_device(device: str) -> int:
    """Map cpu/cuda/auto to the transformers pipeline device index (-1 = CPU)."""
    if device == "cpu":
        return -1
    if device.startswith("cuda"):
        return int(device.split(":", 1)[1]) if ":" in device else 0
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:  # pragma: no cover - torch missing
        return -1
