"""Domain-adaptive MLM pretraining with the Hugging Face ``Trainer``.

Tracking goes through JengaNLP's ``LoggingCallback`` (same MLflow / TensorBoard
behaviour as text-task training: run names, params, FAILED/KILLED status), not
through the Trainer's own integrations.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
from pathlib import Path
from typing import Any

from ..training.callbacks import LoggingCallback, failure_status
from .config import MLMConfig
from .data import PreparedCorpus, build_lm_datasets, build_redactor, load_ner_detector, prepare_corpus
from .pii_eval import check_pii_recall

logger = logging.getLogger(__name__)

# Written next to the model; uploaded to MLflow and the Hub with it.
RESULT_FILES = ("mlm_results.json", "redaction_report.json", "pii_recall.json", "experiment_config.yaml", "README.md")


def tracking_params(config: MLMConfig, corpus: PreparedCorpus) -> dict[str, Any]:
    params: dict[str, Any] = {
        "task": "mlm_domain_adaptation",
        "base_model": config.base_model,
        "block_size": config.block_size,
        "mlm_probability": config.mlm_probability,
        "whole_word_mask": config.whole_word_mask,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "max_steps": config.max_steps or "-",
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "precision": "bf16" if config.bf16 else "fp16" if config.fp16 else "fp32",
        "seed": config.seed,
        "train_examples": len(corpus.train_texts),
        "eval_examples": len(corpus.eval_texts),
        "pii_redaction": config.pii.redact,
        "ner_model": config.pii.ner_model or "-",
    }
    for source in config.corpus:
        params[f"source.{source.name}"] = f"repeat={source.repeat},sensitive={source.sensitive}"
    return params


def _training_arguments(config: MLMConfig, output_dir: Path, has_eval: bool) -> Any:
    """Build ``TrainingArguments`` for whichever transformers version is installed."""
    from transformers import TrainingArguments

    supported = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_epochs,
        "max_steps": config.max_steps or -1,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.resolve_eval_batch_size(),
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "gradient_checkpointing": config.gradient_checkpointing,
        "logging_steps": config.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "eval_steps": config.eval_steps,
        "seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "report_to": "none",  # tracking is handled by JengaNLP's LoggingCallback
        "run_name": config.project_name,
    }
    eval_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    kwargs[eval_key] = "steps" if has_eval else "no"
    if "warmup_ratio" in supported:  # transformers < 5
        kwargs["warmup_ratio"] = config.warmup_ratio
    else:  # transformers >= 5: a float in [0, 1) is a ratio of total steps
        kwargs["warmup_steps"] = config.warmup_ratio
    if config.device == "cpu":
        kwargs["use_cpu" if "use_cpu" in supported else "no_cuda"] = True
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in supported})


def _data_collator(tokenizer: Any, config: MLMConfig) -> Any:
    from transformers import DataCollatorForLanguageModeling

    supported = inspect.signature(DataCollatorForLanguageModeling.__init__).parameters
    kwargs: dict[str, Any] = {"tokenizer": tokenizer, "mlm": True, "mlm_probability": config.mlm_probability}
    if config.whole_word_mask:
        if "whole_word_mask" not in supported:
            raise ValueError("whole_word_mask needs transformers >= 5; upgrade or set whole_word_mask: false")
        kwargs["whole_word_mask"] = True
    return DataCollatorForLanguageModeling(**kwargs)


def _tracker_bridge(tracker: LoggingCallback) -> Any:
    """Forward the HF Trainer's logs (train loss, eval loss) to JengaNLP's tracker."""
    from transformers import TrainerCallback

    class _Bridge(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, float] | None = None, **kw: Any):
            if not logs:
                return
            metrics = {}
            for key, value in logs.items():
                if not isinstance(value, (int, float)):
                    continue
                name = {"loss": "train_loss"}.get(key, key)
                metrics[name] = float(value)
                if name == "eval_loss":
                    metrics["eval_perplexity"] = _perplexity(value)
            tracker.log_metrics(metrics, step=state.global_step, prefix="mlm/")

    return _Bridge()


def _perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 50 else float("inf")


class MLMTrainer:
    """Continue masked-language-model training of an encoder on a local corpus."""

    def __init__(self, config: MLMConfig) -> None:
        self.config = config

    def train(self, corpus: PreparedCorpus | None = None) -> dict[str, Any]:
        from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, set_seed

        config = self.config
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(config.seed)

        # PII gate first: with sensitive sources, the redaction stack must reach pii.min_recall on the
        # labelled sample before any text is used. The loaded NER model is reused for the corpus.
        redactors = None
        recall_report = None
        if config.pii.redact and (corpus is None or config.has_sensitive_sources):
            needs_ner = bool(config.pii.ner_model) and (config.has_sensitive_sources or config.pii.ner_on_all_sources)
            ner = load_ner_detector(config) if needs_ner else None
            redactors = {False: build_redactor(config, False, ner), True: build_redactor(config, True, ner)}
            recall_report = check_pii_recall(config, redactors[True])
            if recall_report is not None:
                (output_dir / "pii_recall.json").write_text(json.dumps(recall_report, indent=2))
                logger.info("PII recall %.1f%% on %s (required %.0f%%)", 100 * recall_report["recall"],
                            config.pii.eval_set, 100 * config.pii.min_recall)

        corpus = corpus or prepare_corpus(config, redactors=redactors)
        corpus.write_report(output_dir / "redaction_report.json")
        if not corpus.train_texts:
            raise ValueError("the corpus has no training examples after cleaning, redaction and dedup")
        logger.info(
            "MLM corpus: %d train / %d held-out examples from %d sources",
            len(corpus.train_texts), len(corpus.eval_texts), len(config.corpus),
        )

        tracker = None
        if config.logging is not None:
            tracker = LoggingCallback(
                config.logging,
                str(output_dir),
                run_name=config.project_name,
                params=tracking_params(config, corpus),
                tags={"project_name": config.project_name, "mode": "mlm",
                      **({"pii_recall": f"{recall_report['recall']:.4f}"} if recall_report else {})},
                config_dict=config.model_dump(mode="json"),
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            model = AutoModelForMaskedLM.from_pretrained(config.base_model)
            train_ds, eval_ds = build_lm_datasets(tokenizer, corpus, config.block_size)
            trainer = Trainer(
                model=model,
                args=_training_arguments(config, output_dir, has_eval=eval_ds is not None),
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=_data_collator(tokenizer, config),
                callbacks=[_tracker_bridge(tracker)] if tracker else None,
            )
            train_output = trainer.train()
            results: dict[str, Any] = {
                "base_model": config.base_model,
                "pii_recall": recall_report["recall"] if recall_report else None,
                "train_examples": len(corpus.train_texts),
                "eval_examples": len(corpus.eval_texts),
                "train_blocks": len(train_ds),
                "eval_blocks": len(eval_ds) if eval_ds is not None else 0,
                "global_step": train_output.global_step,
                "train_loss": float(train_output.training_loss),
            }
            if eval_ds is not None:
                eval_loss = float(trainer.evaluate()["eval_loss"])
                results.update(eval_loss=eval_loss, eval_perplexity=_perplexity(eval_loss))
                if tracker:
                    tracker.log_metrics(
                        {"final_eval_loss": eval_loss, "final_eval_perplexity": results["eval_perplexity"]},
                        step=train_output.global_step,
                        prefix="mlm/",
                    )

            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            (output_dir / "training_args.bin").unlink(missing_ok=True)  # internal to the HF Trainer
            if not config.keep_checkpoints:
                # Checkpoints only matter for resuming a crashed run; the final model is saved above.
                import shutil

                shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
            (output_dir / "mlm_results.json").write_text(json.dumps(results, indent=2))
            config.to_yaml(output_dir / "experiment_config.yaml")
            from .hub import write_model_card

            write_model_card(output_dir, config, results, corpus.report)

            if tracker:
                for name in RESULT_FILES:
                    tracker.log_artifact(output_dir / name, artifact_path="mlm")
        except BaseException as exc:
            if tracker:
                tracker.close(status=failure_status(exc))
            raise
        if tracker:
            tracker.close(status="FINISHED")

        if config.push_to_hub:
            from .hub import push_mlm_model

            results["hub_url"] = push_mlm_model(output_dir, config)
        logger.info("MLM model saved to %s", output_dir)
        return results
