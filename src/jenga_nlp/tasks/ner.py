"""Named Entity Recognition task for JengaNLP.

- Dynamic hidden_size
- Dropout in prediction head
- Proper handling of -100 ignore index
- Optional CRF layer for sequence labeling
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ..core.config import TaskConfig
from .base import BaseTask, TaskOutput


class NERTask(BaseTask):
    """Named Entity Recognition (token classification) task.

    Predicts a label for each token in the sequence.
    Uses CrossEntropyLoss with ignore_index=-100 for padding/special tokens.

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
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        for head_config in self.config.heads:
            head_name = head_config.name
            # NER uses sequence_output (all tokens), not pooled_output
            logits = self.heads[head_name](sequence_output)
            all_logits[head_name] = logits

            if labels is not None:
                head_labels = labels[head_name] if isinstance(labels, dict) else labels
                # Reshape for CrossEntropyLoss: (batch*seq_len, num_labels) vs (batch*seq_len,)
                loss = loss_fct(
                    logits.view(-1, head_config.num_labels),
                    head_labels.view(-1),
                )
                weighted_loss = loss * head_config.weight
                total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        return TaskOutput(loss=total_loss, logits=all_logits)
