"""Inference pipeline for JengaNLP multi-task models.

Loads a trained model and runs inference on raw text, with optional
security integrations (PII redaction, explainability, HITL, audit).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from ..core.config import ExperimentConfig, TaskConfig, TaskType
from ..core.model import MultiTaskModel
from ..utils.device import get_device, move_to_device
from .formatters import format_result

logger = logging.getLogger(__name__)


@dataclass
class HeadResult:
    """Result for a single prediction head."""

    head_name: str
    prediction: Any  # Varies by task type
    confidence: Optional[float] = None
    label: Optional[str] = None
    labels: Optional[dict[str, bool]] = None
    probabilities: Optional[dict[str, float]] = None
    scores: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "head_name": self.head_name,
            "prediction": self.prediction,
            "confidence": self.confidence,
        }
        if self.label is not None:
            d["label"] = self.label
        if self.labels is not None:
            d["labels"] = self.labels
        if self.probabilities is not None:
            d["probabilities"] = self.probabilities
        if self.scores is not None:
            d["scores"] = self.scores
        return d


@dataclass
class TaskResult:
    """Result for a single task."""

    task_name: str
    task_type: str
    heads: dict[str, HeadResult] = field(default_factory=dict)
    routed_for_review: bool = False
    routing_decision: Optional[str] = None
    explanation: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_name": self.task_name,
            "task_type": self.task_type,
            "heads": {name: hr.to_dict() for name, hr in self.heads.items()},
        }
        if self.routed_for_review:
            d["routed_for_review"] = True
            d["routing_decision"] = self.routing_decision
        if self.explanation is not None:
            d["explanation"] = self.explanation
        return d


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    text: str
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    pii_detected: bool = False
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "task_results": {
                name: tr.to_dict() for name, tr in self.task_results.items()
            },
            "pii_detected": self.pii_detected,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _build_head_result(head_name: str, formatted: dict[str, Any]) -> HeadResult:
    """Convert a formatter output dict to a HeadResult dataclass."""
    return HeadResult(
        head_name=head_name,
        prediction=formatted.get("prediction"),
        confidence=formatted.get("confidence"),
        label=formatted.get("label"),
        labels=formatted.get("labels"),
        probabilities=formatted.get("probabilities"),
        scores=formatted.get("scores"),
    )


class InferencePipeline:
    """Load a trained model and run inference on raw text.

    Supports all 6 encoder-based task types. LLM inference (causal_lm)
    is handled separately by the LLM module.
    """

    def __init__(
        self,
        model: MultiTaskModel,
        config: ExperimentConfig,
        tokenizer: Any,
        device: torch.device,
        pii_redactor: Any = None,
        explainability_engine: Any = None,
        hitl_router: Any = None,
        audit_logger: Any = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.pii_redactor = pii_redactor
        self.explainability_engine = explainability_engine
        self.hitl_router = hitl_router
        self.audit_logger = audit_logger

        # Build task_name -> task_id mapping
        self._task_map: dict[str, int] = {
            tc.name: i for i, tc in enumerate(config.tasks)
        }

        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        config_path: str | Path,
        device: str = "auto",
        enable_pii: bool = False,
        enable_explainability: bool = False,
        enable_hitl: bool = False,
        enable_audit: bool = False,
        audit_log_dir: str | Path | None = None,
    ) -> InferencePipeline:
        """Factory: load config, tokenizer, model, and optional security modules.

        Args:
            model_dir: Directory containing saved model (model.pt, metadata.json).
            config_path: Path to experiment YAML config.
            device: Device string ("auto", "cuda", "cpu", "mps").
            enable_pii: Enable PII redaction before inference.
            enable_explainability: Enable explainability engine.
            enable_hitl: Enable HITL routing for uncertain predictions.
            enable_audit: Enable audit logging of inference calls.
            audit_log_dir: Directory for audit logs (defaults to model_dir/audit).

        Returns:
            Configured InferencePipeline instance.
        """
        model_dir = Path(model_dir)
        config_path = Path(config_path)

        # Prefer the saved training config because preprocessing can discover
        # runtime metadata such as NER label maps and final head sizes.
        saved_config_path = model_dir / "experiment_config.yaml"
        effective_config_path = saved_config_path if saved_config_path.exists() else config_path
        if effective_config_path != config_path:
            logger.info("Using saved model config: %s", effective_config_path)

        config = ExperimentConfig.from_yaml(effective_config_path)

        # Resolve device
        resolved_device = get_device(device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)

        # Load model
        model = MultiTaskModel.load(model_dir, config, device=str(resolved_device))

        # Optional security modules
        pii_redactor = None
        if enable_pii:
            from ..data.pii import PIIConfig, PIIRedactor

            pii_redactor = PIIRedactor(PIIConfig(enabled=True))
            logger.info("PII redaction enabled")

        explainability_engine = None
        if enable_explainability:
            from ..security.explainability import ExplainabilityEngine

            explainability_engine = ExplainabilityEngine(model, tokenizer)
            logger.info("Explainability engine enabled")

        hitl_router = None
        if enable_hitl:
            from ..security.hitl import HITLRouter, UncertaintyEstimator

            hitl_router = HITLRouter(uncertainty_estimator=UncertaintyEstimator())
            logger.info("HITL routing enabled")

        audit_logger = None
        if enable_audit:
            from ..security.audit import AuditLogger

            log_dir = Path(audit_log_dir) if audit_log_dir else model_dir / "audit"
            audit_logger = AuditLogger(
                log_dir=log_dir, project_name=config.project_name
            )
            logger.info("Audit logging enabled → %s", log_dir)

        return cls(
            model=model,
            config=config,
            tokenizer=tokenizer,
            device=resolved_device,
            pii_redactor=pii_redactor,
            explainability_engine=explainability_engine,
            hitl_router=hitl_router,
            audit_logger=audit_logger,
        )

    def get_task_names(self) -> list[str]:
        """Return available task names from config."""
        return list(self._task_map.keys())

    def predict(
        self,
        text: str,
        task_name: Optional[str] = None,
        threshold: float = 0.5,
        explain: bool = False,
    ) -> InferenceResult:
        """Run inference on a single text.

        Args:
            text: Raw input text.
            task_name: Specific task to run (default: all tasks).
            threshold: Sigmoid threshold for multi-label/QA.
            explain: Whether to attach explainability info.

        Returns:
            InferenceResult with predictions for requested task(s).
        """
        start_time = time.perf_counter()

        # PII redaction
        pii_detected = False
        processed_text = text
        if self.pii_redactor:
            processed_text = self.pii_redactor.redact(text)
            pii_detected = processed_text != text

        # Determine which tasks to run
        if task_name:
            if task_name not in self._task_map:
                raise ValueError(
                    f"Unknown task '{task_name}'. "
                    f"Available: {self.get_task_names()}"
                )
            task_configs = [
                tc for tc in self.config.tasks if tc.name == task_name
            ]
        else:
            task_configs = list(self.config.tasks)

        # Determine if any task needs offset_mapping (NER)
        needs_offsets = any(tc.type == TaskType.NER for tc in task_configs)

        # Tokenize
        tokenize_kwargs: dict[str, Any] = {
            "padding": self.config.tokenizer.padding,
            "truncation": self.config.tokenizer.truncation,
            "max_length": self.config.tokenizer.max_length,
            "return_tensors": "pt",
        }
        if needs_offsets:
            tokenize_kwargs["return_offsets_mapping"] = True

        encoded = self.tokenizer(processed_text, **tokenize_kwargs)

        # Extract offset_mapping before moving to device (not a tensor the model needs)
        offset_mapping = encoded.pop("offset_mapping", None)

        # Move inputs to device
        encoded = move_to_device(dict(encoded), self.device)

        # Run inference per task
        task_results: dict[str, TaskResult] = {}

        with torch.no_grad():
            for task_config in task_configs:
                task_id = self._task_map[task_config.name]

                output = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    task_id=task_id,
                    labels=None,
                )

                # Format predictions
                formatted_heads = format_result(
                    task_config=task_config,
                    logits=output.logits,
                    sample_idx=0,
                    threshold=threshold,
                    offset_mapping=offset_mapping,
                    attention_mask=encoded["attention_mask"],
                    original_text=processed_text,
                )

                # Build HeadResults
                head_results: dict[str, HeadResult] = {}
                for head_name, fmt in formatted_heads.items():
                    head_results[head_name] = _build_head_result(head_name, fmt)

                task_result = TaskResult(
                    task_name=task_config.name,
                    task_type=task_config.type.value,
                    heads=head_results,
                )

                # HITL routing
                if self.hitl_router and output.logits:
                    first_logits = next(iter(output.logits.values()))
                    from ..security.hitl import RoutingDecision

                    decision, _review_item = self.hitl_router.route(
                        logits=first_logits[0],
                        task_name=task_config.name,
                        input_text=processed_text,
                    )
                    task_result.routed_for_review = (
                        decision != RoutingDecision.AUTO_ACCEPT
                    )
                    task_result.routing_decision = decision.value

                # Explainability
                if explain and self.explainability_engine:
                    try:
                        explanation = self.explainability_engine.explain_prediction(
                            text=processed_text,
                            task_id=task_id,
                        )
                        task_result.explanation = {
                            "token_importance": explanation.token_importance,
                            "top_features": explanation.top_features,
                        }
                    except Exception as e:
                        logger.warning("Explainability failed for %s: %s", task_config.name, e)

                task_results[task_config.name] = task_result

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = InferenceResult(
            text=processed_text,
            task_results=task_results,
            pii_detected=pii_detected,
            processing_time_ms=elapsed_ms,
        )

        # Audit logging
        if self.audit_logger:
            from ..security.audit import AuditAction

            self.audit_logger.log(
                action=AuditAction.INFERENCE_SINGLE,
                resource_type="inference",
                details={
                    "tasks": [tc.name for tc in task_configs],
                    "pii_detected": pii_detected,
                    "processing_time_ms": round(elapsed_ms, 2),
                },
                input_data=processed_text,
                output_data=result.to_json(),
            )

        return result

    def predict_batch(
        self,
        texts: list[str],
        task_name: Optional[str] = None,
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> list[InferenceResult]:
        """Run inference on a batch of texts with proper batching.

        Args:
            texts: List of raw input texts.
            task_name: Specific task to run (default: all tasks).
            batch_size: Batch size for processing.
            threshold: Sigmoid threshold for multi-label/QA.

        Returns:
            List of InferenceResults, one per input text.
        """
        start_time = time.perf_counter()

        # PII redaction
        processed_texts = texts
        pii_flags = [False] * len(texts)
        if self.pii_redactor:
            processed_texts = self.pii_redactor.redact_batch(texts)
            pii_flags = [p != o for p, o in zip(processed_texts, texts)]

        # Determine which tasks to run
        if task_name:
            if task_name not in self._task_map:
                raise ValueError(
                    f"Unknown task '{task_name}'. "
                    f"Available: {self.get_task_names()}"
                )
            task_configs = [
                tc for tc in self.config.tasks if tc.name == task_name
            ]
        else:
            task_configs = list(self.config.tasks)

        needs_offsets = any(tc.type == TaskType.NER for tc in task_configs)

        results: list[InferenceResult] = []

        # Process in batches
        for batch_start in range(0, len(processed_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(processed_texts))
            batch_texts = processed_texts[batch_start:batch_end]
            batch_pii = pii_flags[batch_start:batch_end]
            batch_originals = texts[batch_start:batch_end]

            # Tokenize batch
            tokenize_kwargs: dict[str, Any] = {
                "padding": self.config.tokenizer.padding,
                "truncation": self.config.tokenizer.truncation,
                "max_length": self.config.tokenizer.max_length,
                "return_tensors": "pt",
            }
            if needs_offsets:
                tokenize_kwargs["return_offsets_mapping"] = True

            encoded = self.tokenizer(batch_texts, **tokenize_kwargs)
            offset_mapping = encoded.pop("offset_mapping", None)
            encoded = move_to_device(dict(encoded), self.device)

            # Per-sample results for this batch
            batch_results: list[dict[str, TaskResult]] = [
                {} for _ in range(len(batch_texts))
            ]

            with torch.no_grad():
                for task_config in task_configs:
                    task_id = self._task_map[task_config.name]

                    output = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        task_id=task_id,
                        labels=None,
                    )

                    # Format predictions for each sample in the batch
                    for sample_idx in range(len(batch_texts)):
                        formatted_heads = format_result(
                            task_config=task_config,
                            logits=output.logits,
                            sample_idx=sample_idx,
                            threshold=threshold,
                            offset_mapping=offset_mapping,
                            attention_mask=encoded["attention_mask"],
                            original_text=batch_texts[sample_idx],
                        )

                        head_results: dict[str, HeadResult] = {}
                        for head_name, fmt in formatted_heads.items():
                            head_results[head_name] = _build_head_result(
                                head_name, fmt
                            )

                        task_result = TaskResult(
                            task_name=task_config.name,
                            task_type=task_config.type.value,
                            heads=head_results,
                        )

                        # HITL routing
                        if self.hitl_router and output.logits:
                            first_logits = next(iter(output.logits.values()))
                            from ..security.hitl import RoutingDecision

                            decision, _ = self.hitl_router.route(
                                logits=first_logits[sample_idx],
                                task_name=task_config.name,
                                input_text=batch_texts[sample_idx],
                            )
                            task_result.routed_for_review = (
                                decision != RoutingDecision.AUTO_ACCEPT
                            )
                            task_result.routing_decision = decision.value

                        batch_results[sample_idx][task_config.name] = task_result

            # Build InferenceResults for this batch
            for sample_idx, (txt, pii) in enumerate(
                zip(batch_texts, batch_pii)
            ):
                results.append(
                    InferenceResult(
                        text=txt,
                        task_results=batch_results[sample_idx],
                        pii_detected=pii,
                    )
                )

        total_ms = (time.perf_counter() - start_time) * 1000

        # Set processing time (total / num samples)
        per_sample = total_ms / len(texts) if texts else 0
        for r in results:
            r.processing_time_ms = round(per_sample, 2)

        # Audit logging for batch
        if self.audit_logger:
            from ..security.audit import AuditAction

            self.audit_logger.log(
                action=AuditAction.INFERENCE_BATCH,
                resource_type="inference",
                details={
                    "tasks": [tc.name for tc in task_configs],
                    "num_samples": len(texts),
                    "batch_size": batch_size,
                    "total_time_ms": round(total_ms, 2),
                    "pii_detected_count": sum(pii_flags),
                },
            )

        return results
