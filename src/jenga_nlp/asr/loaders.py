"""Real dataset loaders and audio preprocessing for Whisper ASR."""

from __future__ import annotations

import csv
import logging
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .config import WhisperDatasetConfig, WhisperFineTuningConfig, WhisperPreprocessingConfig
from .data import _iter_jsonl

logger = logging.getLogger(__name__)

SENTENCE_COLUMNS = ["sentence", "transcription", "text", "actualSentence", "transcript", "transcripts"]


@dataclass(slots=True)
class ASRAudioSample:
    """Normalized sample consumed by Whisper feature extraction."""

    audio: dict[str, Any]
    sentence: str
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"audio": self.audio, "sentence": self.sentence, "path": self.path, "metadata": self.metadata}


def normalize_transcript(text: Any, preprocessing: WhisperPreprocessingConfig) -> str:
    value = str(text)
    if preprocessing.strip_text:
        value = value.strip()
    if preprocessing.normalize_text:
        value = " ".join(value.split())
    if preprocessing.lowercase:
        value = value.lower()
    return value


def resample_audio(array: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio with librosa when available, otherwise use linear interpolation."""
    if source_rate == target_rate:
        return array.astype(np.float32, copy=False)
    try:
        import librosa

        return librosa.resample(array.astype(np.float32), orig_sr=source_rate, target_sr=target_rate).astype(np.float32)
    except Exception:
        duration = len(array) / float(source_rate)
        target_length = max(1, int(round(duration * target_rate)))
        old_x = np.linspace(0.0, duration, num=len(array), endpoint=False)
        new_x = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(new_x, old_x, array).astype(np.float32)


def _read_wav_fallback(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sample_rate


def load_audio_file(path: str | Path, preprocessing: WhisperPreprocessingConfig) -> dict[str, Any] | None:
    """Decode an audio file, convert to mono, filter duration, and resample for Whisper."""
    audio_path = Path(path)
    if not audio_path.exists():
        if preprocessing.validate_audio_paths:
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        logger.warning("Audio file not found: %s", audio_path)
        return None

    try:
        try:
            import soundfile as sf

            array, sampling_rate = sf.read(str(audio_path), always_2d=False)
            array = np.asarray(array, dtype=np.float32)
        except Exception:
            array, sampling_rate = _read_wav_fallback(audio_path)

        if array.ndim > 1:
            array = array.mean(axis=1)
        duration = len(array) / float(sampling_rate)
        if duration < preprocessing.min_audio_duration_s or duration > preprocessing.max_audio_duration_s:
            return None
        array = resample_audio(array, sampling_rate, preprocessing.sampling_rate)
        return {"array": array, "sampling_rate": preprocessing.sampling_rate, "path": str(audio_path)}
    except Exception as exc:
        logger.warning("Could not decode audio %s: %s", audio_path, exc)
        return None


def normalize_audio_payload(audio: Any, preprocessing: WhisperPreprocessingConfig, path: str | None = None) -> dict[str, Any] | None:
    """Normalize an HF-style audio payload or a local file path into Whisper audio dict form."""
    if isinstance(audio, dict):
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate", preprocessing.sampling_rate)
        audio_path = audio.get("path", path)
        if array is None and audio_path:
            return load_audio_file(audio_path, preprocessing)
        if array is None:
            return None
        np_array = np.asarray(array, dtype=np.float32)
        if np_array.ndim > 1:
            np_array = np_array.mean(axis=1)
        duration = len(np_array) / float(sampling_rate)
        if duration < preprocessing.min_audio_duration_s or duration > preprocessing.max_audio_duration_s:
            return None
        np_array = resample_audio(np_array, int(sampling_rate), preprocessing.sampling_rate)
        return {"array": np_array, "sampling_rate": preprocessing.sampling_rate, "path": audio_path or path or ""}
    if isinstance(audio, (str, Path)):
        return load_audio_file(audio, preprocessing)
    return None


def _valid_sentence(sentence: str, preprocessing: WhisperPreprocessingConfig) -> bool:
    return bool(sentence) and len(sentence) <= preprocessing.max_transcript_chars


def _resolve_sentence_column(sample: dict[str, Any], configured: str = "auto") -> str | None:
    if configured != "auto":
        return configured if sample.get(configured) else None
    for column in SENTENCE_COLUMNS:
        if sample.get(column):
            return column
    return None


class ManifestAudioLoader:
    def __init__(self, manifest_path: str | Path, dataset: WhisperDatasetConfig, preprocessing: WhisperPreprocessingConfig, limit: int | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.limit = limit

    def iterate(self) -> Iterator[dict[str, Any]]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"ASR manifest not found: {self.manifest_path}")
        count = 0
        for row in _iter_jsonl(self.manifest_path):
            if self.limit is not None and count >= self.limit:
                break
            audio_value = row.get(self.dataset.audio_column)
            text_value = row.get(self.dataset.text_column)
            if audio_value is None or text_value is None:
                continue
            sentence = normalize_transcript(text_value, self.preprocessing)
            if not _valid_sentence(sentence, self.preprocessing):
                continue
            if self.dataset.duration_column and row.get(self.dataset.duration_column) is not None:
                duration = float(row[self.dataset.duration_column])
                if duration < self.preprocessing.min_audio_duration_s or duration > self.preprocessing.max_audio_duration_s:
                    continue
            audio = normalize_audio_payload(audio_value, self.preprocessing)
            if audio is None:
                continue
            metadata = {
                key: value
                for key, value in row.items()
                if key not in {self.dataset.audio_column, self.dataset.text_column, self.dataset.duration_column}
            }
            yield ASRAudioSample(audio=audio, sentence=sentence, path=audio.get("path"), metadata=metadata).to_dict()
            count += 1


class HuggingFaceASRLoader:
    def __init__(self, dataset: WhisperDatasetConfig, preprocessing: WhisperPreprocessingConfig, split: str, limit: int | None = None) -> None:
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.split = split
        self.limit = limit

    def iterate(self) -> Iterator[dict[str, Any]]:
        try:
            from datasets import Audio, load_dataset
        except Exception as exc:
            raise ImportError("Hugging Face ASR loading requires `datasets`.") from exc
        if not self.dataset.name:
            raise ValueError("dataset.name is required when source='huggingface'")

        ds = load_dataset(
            self.dataset.name,
            split=self.split,
            streaming=self.dataset.streaming,
            trust_remote_code=self.dataset.trust_remote_code,
        )
        try:
            ds = ds.cast_column(self.dataset.audio_column, Audio(sampling_rate=self.preprocessing.sampling_rate))
        except Exception:
            pass
        if self.dataset.language_filter:
            ds = ds.filter(lambda row: row.get("language", "") == self.dataset.language_filter)

        count = 0
        for row in ds:
            if self.limit is not None and count >= self.limit:
                break
            configured_text_column = self.dataset.sentence_column if self.dataset.sentence_column else self.dataset.text_column
            text_column = _resolve_sentence_column(row, configured_text_column)
            if text_column is None:
                text_column = self.dataset.text_column if row.get(self.dataset.text_column) else None
            if text_column is None:
                continue
            sentence = normalize_transcript(row[text_column], self.preprocessing)
            if not _valid_sentence(sentence, self.preprocessing):
                continue
            audio = normalize_audio_payload(row.get(self.dataset.audio_column), self.preprocessing)
            if audio is None:
                continue
            yield ASRAudioSample(audio=audio, sentence=sentence, path=audio.get("path"), metadata={"source": self.dataset.name}).to_dict()
            count += 1


class CommonVoiceLocalLoader:
    def __init__(self, base_path: str | Path, language: str, split: str, preprocessing: WhisperPreprocessingConfig, limit: int | None = None) -> None:
        self.base_path = Path(base_path)
        self.language = language
        self.split = split
        self.preprocessing = preprocessing
        self.limit = limit

    def iterate(self) -> Iterator[dict[str, Any]]:
        lang_dir = self.base_path / self.language
        clips_dir = lang_dir / "clips"
        tsv_path = lang_dir / f"{self.split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Common Voice TSV not found: {tsv_path}")
        csv.field_size_limit(sys.maxsize)
        count = 0
        with tsv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="	")
            for row in reader:
                if self.limit is not None and count >= self.limit:
                    break
                sentence = normalize_transcript(row.get("sentence", ""), self.preprocessing)
                if not _valid_sentence(sentence, self.preprocessing):
                    continue
                audio_path = clips_dir / row.get("path", "")
                audio = load_audio_file(audio_path, self.preprocessing)
                if audio is None:
                    continue
                yield ASRAudioSample(audio=audio, sentence=sentence, path=str(audio_path), metadata={"source": "common_voice"}).to_dict()
                count += 1


class CustomTSVLoader:
    def __init__(self, dataset_path: str | Path, split: str, preprocessing: WhisperPreprocessingConfig, limit: int | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.preprocessing = preprocessing
        self.limit = limit

    def iterate(self) -> Iterator[dict[str, Any]]:
        tsv_path = self.dataset_path / f"{self.split}.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Custom TSV not found: {tsv_path}")
        csv.field_size_limit(sys.maxsize)
        count = 0
        with tsv_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="	")
            for row in reader:
                if self.limit is not None and count >= self.limit:
                    break
                rel_path = row.get("audio", "").strip()
                sentence = normalize_transcript(row.get("transcription", row.get("text", "")), self.preprocessing)
                if not rel_path or not _valid_sentence(sentence, self.preprocessing):
                    continue
                audio_path = self.dataset_path / rel_path
                audio = load_audio_file(audio_path, self.preprocessing)
                if audio is None:
                    continue
                yield ASRAudioSample(audio=audio, sentence=sentence, path=str(audio_path), metadata={"source": "custom_tsv"}).to_dict()
                count += 1


class DDDLocalDatasetLoader:
    AUDIO_COLS = ["audio_path", "file", "filename", "path", "wav_file", "audio_file", "File", "Audio_Path"]
    TEXT_COLS = ["transcription", "sentence", "text", "transcript", "Transcription", "Sentence", "Text"]

    def __init__(self, data_dir: str | Path, preprocessing: WhisperPreprocessingConfig, excel_file: str | None = None, limit: int | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.preprocessing = preprocessing
        self.excel_file = excel_file
        self.limit = limit

    def _find_excel(self) -> Path:
        if self.excel_file:
            return self.data_dir / self.excel_file
        candidates = list(self.data_dir.glob("*.xlsx")) + list(self.data_dir.glob("*.xls"))
        if not candidates:
            raise FileNotFoundError(f"No Excel index file found in {self.data_dir}")
        return candidates[0]

    @staticmethod
    def _resolve(cols: Iterable[str], candidates: list[str]) -> str | None:
        colset = set(cols)
        for candidate in candidates:
            if candidate in colset:
                return candidate
        return None

    def iterate(self) -> Iterator[dict[str, Any]]:
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError("DDD Excel loading requires `pandas` and `openpyxl`.") from exc
        excel_path = self._find_excel()
        df = pd.read_excel(excel_path, dtype=str).fillna("")
        audio_col = self._resolve(df.columns, self.AUDIO_COLS)
        text_col = self._resolve(df.columns, self.TEXT_COLS)
        if audio_col is None or text_col is None:
            raise ValueError(f"Could not find audio/text columns in {excel_path}; columns={df.columns.tolist()}")
        count = 0
        for _, row in df.iterrows():
            if self.limit is not None and count >= self.limit:
                break
            rel_path = str(row[audio_col]).strip()
            sentence = normalize_transcript(row[text_col], self.preprocessing)
            if not rel_path or not _valid_sentence(sentence, self.preprocessing):
                continue
            audio_path = Path(rel_path)
            if not audio_path.is_absolute():
                audio_path = self.data_dir / rel_path
            audio = load_audio_file(audio_path, self.preprocessing)
            if audio is None:
                continue
            yield ASRAudioSample(audio=audio, sentence=sentence, path=str(audio_path), metadata={"source": "local_ddd"}).to_dict()
            count += 1


class MultiSourceLoader:
    def __init__(self, sources: list[dict[str, Any]], config: WhisperFineTuningConfig, split: str, limit: int | None = None) -> None:
        self.sources = sources
        self.config = config
        self.split = split
        self.limit = limit

    def iterate(self) -> Iterator[dict[str, Any]]:
        count = 0
        for source_cfg in self.sources:
            loader = _loader_from_source_dict(source_cfg, self.config, self.split)
            for sample in loader.iterate():
                if self.limit is not None and count >= self.limit:
                    return
                yield sample
                count += 1


def _limit_for(config: WhisperFineTuningConfig, split: str) -> int | None:
    return config.dataset.num_train_samples if split == "train" else config.dataset.num_eval_samples


def _split_name(dataset: WhisperDatasetConfig, split: str) -> str:
    return dataset.train_split if split == "train" else dataset.eval_split


def _loader_from_source_dict(source: dict[str, Any], config: WhisperFineTuningConfig, split: str):
    source_type = source.get("type", source.get("source", "huggingface"))
    source_dataset = config.dataset.model_copy(update={k: v for k, v in source.items() if k != "type"})
    source_dataset.source = source_type
    if "split" in source:
        if split == "train":
            source_dataset.train_split = source["split"]
        else:
            source_dataset.eval_split = source["split"]
    return build_asr_loader(config.model_copy(update={"dataset": source_dataset}), split=split)


def build_asr_loader(config: WhisperFineTuningConfig, *, split: str, manifest_override: str | Path | None = None):
    dataset = config.dataset
    preprocessing = config.preprocessing
    limit = _limit_for(config, split)
    source = dataset.source.lower()

    if manifest_override is not None or source == "manifest":
        manifest = Path(manifest_override) if manifest_override is not None else (dataset.train_manifest if split == "train" else dataset.eval_manifest)
        if manifest is None:
            raise ValueError(f"ASR {split} manifest is not configured")
        return ManifestAudioLoader(manifest, dataset, preprocessing, limit=limit)
    if source == "huggingface":
        return HuggingFaceASRLoader(dataset, preprocessing, _split_name(dataset, split), limit=limit)
    if source in {"local_cv", "common_voice"}:
        if dataset.local_path is None:
            raise ValueError("dataset.local_path is required for local_cv")
        return CommonVoiceLocalLoader(dataset.local_path, config.language or "so", _split_name(dataset, split), preprocessing, limit=limit)
    if source in {"custom_tsv", "local_tsv"}:
        if dataset.local_path is None:
            raise ValueError("dataset.local_path is required for custom_tsv")
        return CustomTSVLoader(dataset.local_path, _split_name(dataset, split), preprocessing, limit=limit)
    if source in {"local_ddd", "ddd_excel"}:
        if dataset.local_path is None:
            raise ValueError("dataset.local_path is required for local_ddd")
        return DDDLocalDatasetLoader(dataset.local_path, preprocessing, excel_file=dataset.excel_file, limit=limit)
    if source == "local":
        local_format = (dataset.local_format or "common_voice").lower()
        mapped = {"common_voice": "local_cv", "custom_tsv": "custom_tsv", "ddd_excel": "local_ddd"}.get(local_format, local_format)
        return build_asr_loader(config.model_copy(update={"dataset": dataset.model_copy(update={"source": mapped})}), split=split)
    if source == "multi":
        return MultiSourceLoader(dataset.sources, config, split, limit=limit)
    raise ValueError(f"Unknown ASR dataset source: {dataset.source}")


def as_dataset(loader: Any):
    """Wrap a loader into a Hugging Face IterableDataset when available."""
    try:
        from datasets import IterableDataset

        return IterableDataset.from_generator(lambda: loader.iterate())
    except Exception:
        return list(loader.iterate())


def build_asr_dataset(config: WhisperFineTuningConfig, *, split: str, manifest_override: str | Path | None = None):
    return as_dataset(build_asr_loader(config, split=split, manifest_override=manifest_override))


def summarize_dataset(dataset: Any, *, limit: int = 5) -> dict[str, Any]:
    """Return a lightweight summary without materializing large datasets."""
    try:
        iterator = iter(dataset)
        count = 0
        for _ in range(limit):
            next(iterator)
            count += 1
        return {"status": "ready", "sampled": count}
    except StopIteration:
        return {"status": "ready", "sampled": 0}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
