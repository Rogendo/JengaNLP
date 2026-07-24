"""Quality Assurance scoring task for call transcript evaluation.

Extends MultiLabelClassificationTask with:
- Sub-metric label names per head (e.g., listening -> [active_listening, paraphrasing, ...])
- Per-sub-metric prediction formatting
- QA-specific evaluation report generation

V1 reference: MultiHeadQAClassifier with 6 heads (opening, listening, proactiveness,
resolution, hold, closing) totaling 17 binary outputs for call transcript scoring.

In V2, this is expressed as a multi-label classification task with 6 heads, each
having different num_labels. The sub-metric names are stored in label_maps.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import numpy as np

from ..core.config import TaskConfig
from .base import TaskOutput
from .classification import MultiLabelClassificationTask

logger = logging.getLogger(__name__)

# Default QA sub-metric names matching V1's MultiHeadQAClassifier
DEFAULT_QA_SUB_LABELS: dict[str, list[str]] = {
    "opening": ["greeting_quality"],
    "listening": [
        "active_listening",
        "paraphrasing",
        "empathy",
        "patience",
        "clarification",
    ],
    "proactiveness": [
        "initiative",
        "anticipation",
        "follow_up",
    ],
    "resolution": [
        "problem_identification",
        "solution_accuracy",
        "completeness",
        "first_call_resolution",
        "escalation_handling",
    ],
    "hold": [
        "hold_notification",
        "hold_duration",
    ],
    "closing": ["closing_quality"],
}


class QAScoringTask(MultiLabelClassificationTask):
    """Quality Assurance multi-head scoring task.

    Each head evaluates a quality area (opening, listening, etc.) with
    one or more binary sub-metrics. Inherits BCEWithLogitsLoss from
    MultiLabelClassificationTask.

    Sub-metric names are read from config.label_maps if provided,
    otherwise defaults to DEFAULT_QA_SUB_LABELS.

    Args:
        config: Task configuration with head definitions.
        hidden_size: Hidden size from the shared encoder.
    """

    def __init__(self, config: TaskConfig, hidden_size: int) -> None:
        super().__init__(config, hidden_size)
        self._sub_labels = self._build_sub_labels()

    def _build_sub_labels(self) -> dict[str, list[str]]:
        """Build sub-metric label mapping from config or defaults."""
        sub_labels: dict[str, list[str]] = {}

        for head in self.config.heads:
            if (
                self.config.label_maps
                and head.name in self.config.label_maps
            ):
                # Read from config label_maps (values are {int: str} dicts)
                label_map = self.config.label_maps[head.name]
                # Sort by key to maintain order
                ordered = sorted(label_map.items(), key=lambda x: int(x[0]))
                sub_labels[head.name] = [v for _, v in ordered]
            elif head.name in DEFAULT_QA_SUB_LABELS:
                # Use defaults
                sub_labels[head.name] = DEFAULT_QA_SUB_LABELS[head.name]
            else:
                # Generate generic names
                sub_labels[head.name] = [
                    f"{head.name}_metric_{i}" for i in range(head.num_labels)
                ]

        return sub_labels

    @property
    def sub_labels(self) -> dict[str, list[str]]:
        """Get sub-metric label names per head."""
        return self._sub_labels

    def format_predictions(
        self,
        logits: dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> dict[str, dict[str, bool]]:
        """Convert raw logits to named binary predictions.

        Args:
            logits: Dict mapping head_name -> logit tensor [batch_size, num_labels].
            threshold: Sigmoid threshold for binary prediction.

        Returns:
            Dict mapping head_name -> {sub_metric_name: True/False}.
            Returns predictions for the first sample in the batch.
        """
        predictions: dict[str, dict[str, bool]] = {}

        for head_name, head_logits in logits.items():
            probs = torch.sigmoid(head_logits[0])  # First sample
            binary = (probs > threshold).cpu().tolist()
            sub_names = self._sub_labels.get(head_name, [])

            predictions[head_name] = {}
            for i, val in enumerate(binary):
                name = sub_names[i] if i < len(sub_names) else f"metric_{i}"
                predictions[head_name][name] = bool(val)

        return predictions

    def format_predictions_batch(
        self,
        logits: dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> list[dict[str, dict[str, bool]]]:
        """Convert raw logits to named binary predictions for entire batch.

        Args:
            logits: Dict mapping head_name -> logit tensor [batch_size, num_labels].
            threshold: Sigmoid threshold for binary prediction.

        Returns:
            List of dicts, one per sample in batch.
        """
        # Get batch size from first head
        first_key = next(iter(logits))
        batch_size = logits[first_key].shape[0]

        results = []
        for b in range(batch_size):
            sample_preds: dict[str, dict[str, bool]] = {}
            for head_name, head_logits in logits.items():
                probs = torch.sigmoid(head_logits[b])
                binary = (probs > threshold).cpu().tolist()
                sub_names = self._sub_labels.get(head_name, [])

                sample_preds[head_name] = {}
                for i, val in enumerate(binary):
                    name = sub_names[i] if i < len(sub_names) else f"metric_{i}"
                    sample_preds[head_name][name] = bool(val)

            results.append(sample_preds)

        return results

    @staticmethod
    def format_evaluation_report(
        metrics: dict[str, float],
        task_name: str = "qa_scoring",
    ) -> str:
        """Format evaluation metrics into a readable QA report.

        Args:
            metrics: Dict from Trainer.evaluate() with keys like
                     "qa_scoring_opening_accuracy", "qa_scoring_listening_f1", etc.
            task_name: Task name prefix in metric keys.

        Returns:
            Formatted multi-line report string.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("QA Scoring Evaluation Report")
        lines.append("=" * 60)

        # Group metrics by head
        head_metrics: dict[str, dict[str, float]] = {}
        prefix = f"{task_name}_"

        for key, value in metrics.items():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            # Format: head_name_metric_name (e.g., opening_accuracy, listening_f1)
            parts = remainder.rsplit("_", 1)
            if len(parts) == 2:
                head_name, metric_name = parts
                if head_name not in head_metrics:
                    head_metrics[head_name] = {}
                head_metrics[head_name][metric_name] = value

        for head_name, hm in sorted(head_metrics.items()):
            lines.append(f"\n  {head_name.upper()}")
            lines.append("  " + "-" * 40)
            for metric_name, value in sorted(hm.items()):
                lines.append(f"    {metric_name:20s}: {value:.4f}")

        # Overall eval_loss if present
        if "eval_loss" in metrics:
            lines.append(f"\n  Overall eval_loss: {metrics['eval_loss']:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)
