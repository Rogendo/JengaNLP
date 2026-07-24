"""Sentiment Analysis task for JengaNLP.

Specialized classification task for sentiment analysis with
support for fine-grained sentiment (e.g., 1-5 stars) or
binary sentiment (positive/negative).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..core.config import TaskConfig
from .base import TaskOutput
from .classification import SingleLabelClassificationTask


class SentimentTask(SingleLabelClassificationTask):
    """Sentiment analysis task.

    This is a specialized single-label classification task. It inherits
    all functionality from SingleLabelClassificationTask and can be
    extended with sentiment-specific features like ordinal regression.

    Args:
        config: Task configuration with head definitions.
        hidden_size: Hidden size from the shared encoder.
    """

    def get_forward_output(
        self,
        pooled_output: torch.Tensor,
        sequence_output: torch.Tensor,
        labels: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> TaskOutput:
        output = super().get_forward_output(
            pooled_output=pooled_output,
            sequence_output=sequence_output,
            labels=labels,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Add sentiment-specific predictions (probabilities)
        predictions = {}
        for head_name, logits in output.logits.items():
            probs = torch.softmax(logits, dim=-1)
            predictions[head_name] = {
                "class_id": torch.argmax(probs, dim=-1),
                "probabilities": probs,
            }

        return TaskOutput(
            loss=output.loss,
            logits=output.logits,
            predictions=predictions,
        )
