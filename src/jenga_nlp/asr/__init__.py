"""ASR utilities for JengaNLP."""

from .callbacks import ASRActivityCallback
from .checkpoints import find_latest_asr_checkpoint, resolve_asr_checkpoint
from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import WhisperConfig, WhisperDatasetConfig, WhisperFineTuningConfig, WhisperPreprocessingConfig
from .data import ASRManifestSample, inspect_manifest, load_manifest_samples
from .features import load_whisper_processor, prepare_whisper_batch, prepare_whisper_dataset
from .hub import generate_asr_model_card, push_asr_model_to_hub, save_asr_model_card
from .loaders import (
    ASRAudioSample,
    CommonVoiceLocalLoader,
    CustomTSVLoader,
    DDDLocalDatasetLoader,
    HuggingFaceASRLoader,
    ManifestAudioLoader,
    build_asr_dataset,
    build_asr_loader,
    load_audio_file,
    normalize_audio_payload,
    normalize_transcript,
    resample_audio,
)
from .metrics import build_asr_compute_metrics, char_error_rate, compute_wer_cer, word_error_rate
from .pipeline import WhisperPipeline
from .serving import WhisperServer
from .training import WhisperTrainer

__all__ = [
    "ASRActivityCallback",
    "ASRAudioSample",
    "ASRManifestSample",
    "CommonVoiceLocalLoader",
    "CustomTSVLoader",
    "DDDLocalDatasetLoader",
    "DataCollatorSpeechSeq2SeqWithPadding",
    "HuggingFaceASRLoader",
    "ManifestAudioLoader",
    "WhisperConfig",
    "WhisperDatasetConfig",
    "WhisperFineTuningConfig",
    "WhisperPipeline",
    "WhisperPreprocessingConfig",
    "WhisperServer",
    "WhisperTrainer",
    "build_asr_compute_metrics",
    "build_asr_dataset",
    "build_asr_loader",
    "char_error_rate",
    "compute_wer_cer",
    "find_latest_asr_checkpoint",
    "generate_asr_model_card",
    "inspect_manifest",
    "load_audio_file",
    "load_manifest_samples",
    "load_whisper_processor",
    "normalize_audio_payload",
    "normalize_transcript",
    "prepare_whisper_batch",
    "prepare_whisper_dataset",
    "push_asr_model_to_hub",
    "resample_audio",
    "resolve_asr_checkpoint",
    "save_asr_model_card",
    "word_error_rate",
]
