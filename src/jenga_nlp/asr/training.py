"""Whisper fine-tuning trainer for JengaNLP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .callbacks import ASRActivityCallback
from .checkpoints import resolve_asr_checkpoint
from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import WhisperFineTuningConfig
from .data import inspect_manifest
from .features import prepare_whisper_dataset
from .hub import push_asr_model_to_hub, save_asr_model_card
from .loaders import build_asr_dataset, summarize_dataset
from .metrics import build_asr_compute_metrics
from ..integrations.events import ActivityEventStage
from ..integrations.runtime import IntegrationRuntime


@dataclass
class WhisperTrainer:
    """Fine-tunes Whisper-style ASR models with JengaNLP config and integrations."""

    config: WhisperFineTuningConfig

    def _load_runtime(self) -> dict[str, Any]:
        try:
            from transformers import (
                Seq2SeqTrainer,
                Seq2SeqTrainingArguments,
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "Whisper fine-tuning requires `torch` and `transformers` to be installed."
            ) from exc
        return {
            "Seq2SeqTrainer": Seq2SeqTrainer,
            "Seq2SeqTrainingArguments": Seq2SeqTrainingArguments,
            "WhisperForConditionalGeneration": WhisperForConditionalGeneration,
            "WhisperProcessor": WhisperProcessor,
        }

    def train(
        self,
        train_data: str | Path | None = None,
        eval_data: str | Path | None = None,
        *,
        resume_from_checkpoint: str | Path | None = None,
        push_to_hub: bool | None = None,
        hub_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Run Whisper fine-tuning or a scaffold-compatible mocked run."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        runtime = IntegrationRuntime(run_dir=self.config.output_dir, integrations=self.config.integrations)
        runtime.start(
            run_name=f"asr-{self.config.language or 'unknown'}",
            config=self.config.model_dump(mode="json"),
            tags={"modality": "asr", "model_name": self.config.model_name},
        )

        train_summary = self._inspect_input(train_data, split="train")
        eval_summary = self._inspect_input(eval_data, split="eval")
        runtime.emit_info(
            "asr datasets inspected",
            stage=ActivityEventStage.TRAIN,
            payload={"train": train_summary, "eval": eval_summary},
        )

        try:
            deps = self._load_runtime()
            if deps is True:  # Compatibility for existing tests that monkeypatch _load_runtime.
                result = self._scaffold_result(train_data, eval_data, train_summary, eval_summary)
                runtime.emit_info("asr training scaffold ready", stage=ActivityEventStage.TRAIN, payload=result)
                runtime.finish(status="success")
                return result

            result = self._train_with_transformers(
                deps=deps,
                runtime=runtime,
                train_data=train_data,
                eval_data=eval_data,
                resume_from_checkpoint=resume_from_checkpoint,
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
            )
            runtime.finish(status="success")
            return result
        except Exception:
            runtime.finish(status="failed")
            raise

    def _inspect_input(self, data: str | Path | None, *, split: str) -> dict[str, Any]:
        source = self.config.dataset.source.lower()
        manifest = data
        if manifest is None and source == "manifest":
            manifest = self.config.resolve_train_data() if split == "train" else self.config.resolve_eval_data()
        if manifest is not None:
            return inspect_manifest(manifest, self.config.dataset, self.config.preprocessing)
        return {"status": "configured_by_source", "source": self.config.dataset.source, "split": split, "num_samples": None}

    def _scaffold_result(
        self,
        train_data: str | Path | None,
        eval_data: str | Path | None,
        train_summary: dict[str, Any],
        eval_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "scaffold_ready",
            "model_name": self.config.model_name,
            "language": self.config.language,
            "train_data": str(train_data or self.config.resolve_train_data()),
            "eval_data": str(eval_data or self.config.resolve_eval_data()) if (eval_data or self.config.resolve_eval_data()) else None,
            "output_dir": str(self.config.output_dir),
            "dataset": {"source": self.config.dataset.source, "train": train_summary, "eval": eval_summary},
            "preprocessing": self.config.preprocessing.model_dump(mode="json"),
            "integrations": [integration.model_dump(mode="json") for integration in self.config.integrations],
        }

    def _train_with_transformers(
        self,
        *,
        deps: dict[str, Any],
        runtime: IntegrationRuntime,
        train_data: str | Path | None,
        eval_data: str | Path | None,
        resume_from_checkpoint: str | Path | None,
        push_to_hub: bool | None,
        hub_model_id: str | None,
    ) -> dict[str, Any]:
        processor = deps["WhisperProcessor"].from_pretrained(
            self.config.model_name,
            language=self.config.language,
            task=self.config.task,
        )
        model = deps["WhisperForConditionalGeneration"].from_pretrained(
            self.config.model_name,
            low_cpu_mem_usage=True,
            use_cache=False,
        )
        model = self._configure_model(model, processor)

        train_dataset = build_asr_dataset(self.config, split="train", manifest_override=train_data)
        eval_dataset = None
        if eval_data is not None or self.config.dataset.source.lower() != "manifest" or self.config.resolve_eval_data() is not None:
            eval_dataset = build_asr_dataset(self.config, split="eval", manifest_override=eval_data)

        runtime.emit_info(
            "asr raw datasets built",
            stage=ActivityEventStage.TRAIN,
            payload={"train": summarize_dataset(train_dataset), "eval": summarize_dataset(eval_dataset) if eval_dataset is not None else None},
        )

        train_dataset = prepare_whisper_dataset(train_dataset, processor)
        eval_dataset = prepare_whisper_dataset(eval_dataset, processor) if eval_dataset is not None else None
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        training_args = self._build_training_args(deps["Seq2SeqTrainingArguments"], eval_dataset is not None, push_to_hub, hub_model_id)
        checkpoint = resolve_asr_checkpoint(self.config.output_dir, resume_from_checkpoint)

        trainer_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "data_collator": data_collator,
            "compute_metrics": build_asr_compute_metrics(processor),
            "callbacks": [ASRActivityCallback(runtime=runtime)],
        }
        trainer = self._build_trainer(deps["Seq2SeqTrainer"], trainer_kwargs, processor)

        runtime.emit_info(
            "asr training started",
            stage=ActivityEventStage.TRAIN,
            payload={"resume_from_checkpoint": checkpoint, "output_dir": str(self.config.output_dir)},
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = dict(getattr(train_result, "metrics", {}) or {})

        trainer.save_model()
        processor.save_pretrained(str(self.config.output_dir))
        if hasattr(trainer, "log_metrics"):
            trainer.log_metrics("train", metrics)
        if hasattr(trainer, "save_metrics"):
            trainer.save_metrics("train", metrics)
        if hasattr(trainer, "save_state"):
            trainer.save_state()

        save_asr_model_card(self.config.output_dir, self.config, metrics)
        hub_url = None
        should_push = self.config.push_to_hub if push_to_hub is None else push_to_hub
        repo_id = hub_model_id or self.config.hub_model_id
        if should_push and repo_id:
            hub_url = push_asr_model_to_hub(self.config.output_dir, repo_id, private=self.config.hub_private)

        result = {
            "status": "completed",
            "model_name": self.config.model_name,
            "language": self.config.language,
            "metrics": metrics,
            "output_dir": str(self.config.output_dir),
            "checkpoint": checkpoint,
            "hub_url": hub_url,
        }
        runtime.emit_info("asr training complete", stage=ActivityEventStage.TRAIN, payload=result)
        return result

    def _configure_model(self, model: Any, processor: Any) -> Any:
        generation_config = getattr(model, "generation_config", None)
        if generation_config is not None:
            generation_config.language = self.config.language
            generation_config.task = self.config.task
            try:
                generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=self.config.language,
                    task=self.config.task,
                )
            except Exception:
                pass
        if self.config.gradient_checkpointing:
            if hasattr(model, "config"):
                model.config.use_cache = False
            if hasattr(model, "gradient_checkpointing_enable"):
                try:
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except TypeError:
                    model.gradient_checkpointing_enable()
        if self.config.use_peft:
            model = self._apply_peft(model)
        return model

    def _apply_peft(self, model: Any) -> Any:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as exc:
            raise ImportError("Whisper PEFT fine-tuning requires `peft`.") from exc
        cfg = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            **self.config.peft_config,
        }
        lora_config = LoraConfig(
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["target_modules"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        return get_peft_model(model, lora_config)

    def _report_to(self) -> list[str]:
        supported = {"mlflow", "tensorboard", "wandb"}
        return [integration.type for integration in self.config.integrations if integration.enabled and integration.type in supported]

    def _build_training_args(
        self,
        args_cls: Any,
        has_eval_dataset: bool,
        push_to_hub: bool | None,
        hub_model_id: str | None,
    ) -> Any:
        should_push = self.config.push_to_hub if push_to_hub is None else push_to_hub
        kwargs = {
            "output_dir": str(self.config.output_dir),
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.resolve_eval_batch_size(),
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "max_grad_norm": self.config.max_grad_norm,
            "gradient_checkpointing": self.config.gradient_checkpointing,
            "fp16": self.config.fp16,
            "evaluation_strategy": "steps" if has_eval_dataset else "no",
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "logging_steps": self.config.logging_steps,
            "load_best_model_at_end": self.config.load_best_model_at_end and has_eval_dataset,
            "metric_for_best_model": self.config.metric_for_best_model,
            "greater_is_better": self.config.greater_is_better,
            "push_to_hub": should_push,
            "hub_model_id": hub_model_id or self.config.hub_model_id,
            "hub_token": os.environ.get("HF_TOKEN"),
            "hub_strategy": self.config.hub_strategy,
            "report_to": self._report_to() or "none",
            "save_total_limit": self.config.save_total_limit,
            "optim": self.config.optim,
            "predict_with_generate": True,
            "generation_max_length": self.config.generation_max_length,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "save_safetensors": True,
        }
        if self.config.max_steps is not None:
            kwargs["max_steps"] = self.config.max_steps
        else:
            kwargs["num_train_epochs"] = self.config.num_epochs
        try:
            return args_cls(**kwargs)
        except TypeError as exc:
            if "evaluation_strategy" not in str(exc):
                raise
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
            return args_cls(**kwargs)

    @staticmethod
    def _build_trainer(trainer_cls: Any, trainer_kwargs: dict[str, Any], processor: Any) -> Any:
        try:
            return trainer_cls(**trainer_kwargs, processing_class=processor.feature_extractor)
        except TypeError:
            return trainer_cls(**trainer_kwargs, tokenizer=processor.feature_extractor)
