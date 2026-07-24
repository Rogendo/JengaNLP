"""Regression task for JengaNLP.

Supports continuous output
prediction with MSE or MAE loss.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..core.config import TaskConfig
from .base import BaseTask, TaskOutput


class RegressionTask(BaseTask):
    """Regression task for continuous value prediction.

    Predicts a continuous value for each input. Supports both
    MSE (mean squared error) and Huber loss.

    Args:
        config: Task configuration with head definitions.
        hidden_size: Hidden size from the shared encoder.
        loss_type: Type of loss function ('mse' or 'huber').
    """

    def __init__(
        self,
        config: TaskConfig,
        hidden_size: int,
        loss_type: str = "mse",
    ) -> None:
        super().__init__(config, hidden_size)
        self.loss_type = loss_type

        for head in config.heads:
            # Regression heads output a single value (num_labels=1 typically)
            self.heads[head.name] = nn.Sequential(
                nn.Dropout(head.dropout),
                nn.Linear(hidden_size, head.num_labels),
            )

    @staticmethod
    def _apply_output_transform(logits: torch.Tensor, head_config: Any) -> torch.Tensor:
        activation = getattr(head_config, "activation", None)
        if activation in (None, "identity", "none"):
            return logits
        if activation == "sigmoid":
            output_min = float(getattr(head_config, "output_min", 0.0))
            output_max = float(getattr(head_config, "output_max", 1.0))
            return torch.sigmoid(logits) * (output_max - output_min) + output_min
        raise ValueError(f"Unsupported regression activation: {activation}")

    def get_forward_output(
        self,
        pooled_output: torch.Tensor,
        sequence_output: torch.Tensor,
        labels: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TaskOutput:
        total_loss = None
        all_logits: dict[str, torch.Tensor] = {}

        if self.loss_type == "huber":
            loss_fct = nn.SmoothL1Loss()
        else:
            loss_fct = nn.MSELoss()

        for head_config in self.config.heads:
            head_name = head_config.name
            raw_logits = self.heads[head_name](pooled_output)
            predictions = self._apply_output_transform(raw_logits, head_config)
            all_logits[head_name] = predictions

            if labels is not None:
                head_labels = labels[head_name] if isinstance(labels, dict) else labels
                loss = loss_fct(predictions.squeeze(-1), head_labels.float())
                weighted_loss = loss * head_config.weight
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        return TaskOutput(loss=total_loss, logits=all_logits)
