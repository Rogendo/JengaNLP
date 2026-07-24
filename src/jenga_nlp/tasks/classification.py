"""Classification tasks for JengaNLP.


- Dynamic hidden_size from config (not hardcoded 768)
- Dropout layers in task heads
- Clean label passing (no kwargs hack)
- Proper loss handling when labels are None
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..core.config import TaskConfig
from .base import BaseTask, TaskOutput


class SingleLabelClassificationTask(BaseTask):
    """Single-label multi-class classification task.

    Each head predicts a single class from num_labels options.
    Uses CrossEntropyLoss.

    Args:
        config: Task configuration with head definitions.
        hidden_size: Hidden size from the shared encoder.
    """

    def __init__(self, config: TaskConfig, hidden_size: int) -> None:
        super().__init__(config, hidden_size)
        for head in config.heads:
            self.heads[head.name] = nn.Sequential(
                nn.Dropout(head.dropout),
                nn.Linear(hidden_size, head.num_labels),
            )

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
        loss_fct = nn.CrossEntropyLoss()

        for head_config in self.config.heads:
            head_name = head_config.name
            logits = self.heads[head_name](pooled_output)
            all_logits[head_name] = logits

            if labels is not None:
                head_labels = labels[head_name] if isinstance(labels, dict) else labels
                loss = loss_fct(logits.view(-1, head_config.num_labels), head_labels.view(-1))
                weighted_loss = loss * head_config.weight
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        return TaskOutput(loss=total_loss, logits=all_logits)


class MultiLabelClassificationTask(BaseTask):
    """Multi-label binary classification task.

    Each head predicts multiple binary labels independently.
    Uses BCEWithLogitsLoss.

    Args:
        config: Task configuration with head definitions.
        hidden_size: Hidden size from the shared encoder.
    """

    def __init__(self, config: TaskConfig, hidden_size: int) -> None:
        super().__init__(config, hidden_size)
        for head in config.heads:
            self.heads[head.name] = nn.Sequential(
                nn.Dropout(head.dropout),
                nn.Linear(hidden_size, head.num_labels),
            )

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
        loss_fct = nn.BCEWithLogitsLoss()

        for head_config in self.config.heads:
            head_name = head_config.name
            logits = self.heads[head_name](pooled_output)
            all_logits[head_name] = logits

            if labels is not None and isinstance(labels, dict) and head_name in labels:
                head_labels = labels[head_name]
                if head_labels is not None:
                    loss = loss_fct(logits, head_labels.float())
                    weighted_loss = loss * head_config.weight
                    total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        return TaskOutput(loss=total_loss, logits=all_logits)
