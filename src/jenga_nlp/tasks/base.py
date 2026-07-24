"""Base task definition for the JengaNLP multi-task framework.

This module defines the BaseTask class, which serves as an abstract base for all task types in the JengaNLP framework. Each task type (for example, classification, NER, or regression) inherits from BaseTask and implements get_forward_output to compute task-specific outputs from the shared encoder representations. The TaskOutput dataclass encapsulates the loss and logits returned by each task's forward pass, allowing for a consistent interface across different task types.
Key features:
- Accept hidden_size parameter (not hardcoded 768)
- Proper type annotations with dataclass output
- Dropout in task heads
- Forward method on the module itself for cleaner API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from ..core.config import TaskConfig


@dataclass
class TaskOutput:
    """Output from a task's forward pass.

    Attributes:
        loss: Scalar loss value (None if no labels provided).
        logits: Dict mapping head names to their logit tensors.
        predictions: Optional dict of post-processed predictions.
    """

    loss: Optional[torch.Tensor] = None
    logits: dict[str, torch.Tensor] = field(default_factory=dict)
    predictions: Optional[dict[str, Any]] = None


class BaseTask(nn.Module, ABC):
    """Abstract base class for all tasks.

    Each task defines one or more prediction heads that operate on the
    shared encoder's output. Tasks are registered with the MultiTaskModel
    and called during the forward pass based on task_id.

    Args:
        config: Task configuration.
        hidden_size: Hidden size from the shared encoder (dynamic, not hardcoded).
    """

    def __init__(self, config: TaskConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.name = config.name
        self.type = config.type.value if hasattr(config.type, "value") else config.type
        self.hidden_size = hidden_size
        self.heads = nn.ModuleDict()

    @abstractmethod
    def get_forward_output(
        self,
        pooled_output: torch.Tensor,
        sequence_output: torch.Tensor,
        labels: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TaskOutput:
        """Compute the task-specific forward pass.

        Args:
            pooled_output: [batch_size, hidden_size] - CLS token representation.
            sequence_output: [batch_size, seq_len, hidden_size] - Full sequence.
            labels: Ground truth labels (format depends on task type).
            attention_mask: [batch_size, seq_len] - Attention mask.

        Returns:
            TaskOutput with loss and logits.
        """
        ...
