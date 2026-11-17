"""ASR (Whisper) regression tests: training-argument compatibility and inference audio handling."""

from __future__ import annotations

import warnings
import wave
from pathlib import Path

import numpy as np
import pytest

from jenga_nlp.asr.config import WhisperConfig, WhisperFineTuningConfig
from jenga_nlp.asr.pipeline import WhisperPipeline, _resolve_device
from jenga_nlp.asr.training import WhisperTrainer
from jenga_nlp.core.config import UnknownConfigKeyWarning


class _ArgsV5:
    """Mimics transformers>=5 Seq2SeqTrainingArguments: eval_strategy, no save_safetensors."""

    def __init__(self, output_dir, per_device_train_batch_size=8, per_device_eval_batch_size=8,
                 gradient_accumulation_steps=1, learning_rate=5e-5, warmup_steps=0, max_grad_norm=1.0,
                 gradient_checkpointing=False, fp16=False, eval_strategy="no", eval_steps=None,
                 save_steps=500, logging_steps=500, load_best_model_at_end=False, metric_for_best_model=None,
                 greater_is_better=None, push_to_hub=False, hub_model_id=None, hub_token=None,
                 hub_strategy="every_save", report_to=None, save_total_limit=None, optim="adamw_torch",
                 predict_with_generate=False, generation_max_length=None, dataloader_num_workers=0,
                 max_steps=-1, num_train_epochs=3, use_cpu=False):
        self.kwargs = {k: v for k, v in locals().items() if k != "self"}


class _ArgsV4:
    """Mimics transformers 4.x: evaluation_strategy and save_safetensors exist."""

    def __init__(self, output_dir, per_device_train_batch_size=8, per_device_eval_batch_size=8,
                 gradient_accumulation_steps=1, learning_rate=5e-5, warmup_steps=0, max_grad_norm=1.0,
                 gradient_checkpointing=False, fp16=False, evaluation_strategy="no", eval_steps=None,
                 save_steps=500, logging_steps=500, load_best_model_at_end=False, metric_for_best_model=None,
                 greater_is_better=None, push_to_hub=False, hub_model_id=None, hub_token=None,
                 hub_strategy="every_save", report_to=None, save_total_limit=None, optim="adamw_torch",
                 predict_with_generate=False, generation_max_length=None, dataloader_num_workers=0,
                 save_safetensors=True, max_steps=-1, num_train_epochs=3, use_cpu=False, no_cuda=False):
        self.kwargs = {k: v for k, v in locals().items() if k != "self"}


def _trainer(tmp_path, **overrides) -> WhisperTrainer:
    return WhisperTrainer(WhisperFineTuningConfig(output_dir=tmp_path, max_steps=10, **overrides))


def test_training_args_work_with_transformers_v5(tmp_path):
    args = _trainer(tmp_path, device="cpu")._build_training_args(_ArgsV5, True, None, None)
    assert args.kwargs["eval_strategy"] == "steps"
    assert args.kwargs["use_cpu"] is True
    assert args.kwargs["max_steps"] == 10


def test_training_args_work_with_transformers_v4(tmp_path):
    args = _trainer(tmp_path, device="cpu")._build_training_args(_ArgsV4, False, None, None)
    assert args.kwargs["evaluation_strategy"] == "no"
    assert args.kwargs["save_safetensors"] is True
    assert args.kwargs["use_cpu"] is True  # preferred over the deprecated no_cuda when both exist


def test_training_args_leave_device_alone_unless_cpu(tmp_path):
    args = _trainer(tmp_path, device="auto")._build_training_args(_ArgsV5, True, None, None)
    assert args.kwargs["use_cpu"] is False


def _write_wav(path, seconds=0.5, rate=8000):
    samples = (np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, int(rate * seconds))) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def test_transcribe_decodes_audio_itself_and_forces_language(tmp_path):
    clip = tmp_path / "clip.wav"
    _write_wav(clip)  # 0.5 s at 8 kHz: shorter than the training minimum, other rate than Whisper's
    calls = {}

    def fake_pipe(inputs, generate_kwargs=None):
        calls.update(inputs=inputs, generate_kwargs=generate_kwargs)
        return {"text": "habari"}

    pipe = WhisperPipeline(WhisperConfig(language="sw"), _pipeline=fake_pipe)
    out = pipe.transcribe(clip)
    assert out["text"] == "habari"
    assert calls["inputs"]["sampling_rate"] == 16000  # resampled, no ffmpeg needed
    assert len(calls["inputs"]["raw"]) == 8000
    assert calls["generate_kwargs"] == {"task": "transcribe", "language": "sw"}

    pipe.transcribe(clip, language="en")
    assert calls["generate_kwargs"]["language"] == "en"


def test_transcribe_missing_file_raises(tmp_path):
    pipe = WhisperPipeline(WhisperConfig(), _pipeline=lambda *a, **k: {"text": ""})
    with pytest.raises(ValueError, match="Could not read audio"):
        pipe.transcribe(tmp_path / "missing.wav")


def test_resolve_device():
    assert _resolve_device("cpu") == -1
    assert _resolve_device("cuda") == 0
    assert _resolve_device("cuda:1") == 1
    assert _resolve_device("auto") in (-1, 0)


WHISPER_CONFIGS = sorted((Path(__file__).resolve().parent.parent / "examples" / "configs").glob("*whisper*.yaml"))


@pytest.mark.parametrize("config", WHISPER_CONFIGS, ids=lambda p: p.name)
def test_whisper_example_config_validates(config: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownConfigKeyWarning)  # a misspelled key must fail, not be ignored
        cfg = WhisperFineTuningConfig.from_yaml(config)
    assert cfg.model_name.startswith("openai/whisper")
    if cfg.dataset.source == "manifest":
        assert cfg.resolve_train_data() and cfg.resolve_eval_data(), "manifest configs need train and eval data"
        assert cfg.resolve_train_data() != cfg.resolve_eval_data(), "eval must not be the training file"


def test_whisper_example_configs_exist() -> None:
    names = {p.name for p in WHISPER_CONFIGS}
    assert {"whisper_tiny_sw_cpu_smoke.yaml", "whisper_small_sw_gpu.yaml"} <= names
