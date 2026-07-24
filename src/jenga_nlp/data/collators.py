"""Task-specific collate functions for DataLoader.
This module defines collate functions for different task types (classification, NER, regression) that handle padding and batching of inputs and labels. Each collator is designed to work with the expected input format for its task type, ensuring that batches are properly prepared for training or evaluation.
The main classes are:
- ClassificationCollator: Handles single-label and multi-label classification tasks, including multi-head setups.
- NERCollator: Handles token classification tasks, padding labels with -100 for ignored positions.
- RegressionCollator: Handles regression tasks, ensuring proper padding of inputs and labels.
- Uses functools.partial instead of lambda closures (fixes closure bug)
- Named classes instead of standalone functions for cleaner organization
- Proper handling of all task types
"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer


class ClassificationCollator:
    """Collator for classification tasks (single-label and multi-label).

    Handles padding and batching for classification inputs.
    Supports both single tensor labels and dict-of-tensor labels (multi-head).
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        result: dict[str, Any] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        # Handle labels
        first_labels = batch[0].get("labels")
        if first_labels is not None:
            if isinstance(first_labels, dict):
                # Multi-head: labels is a dict of tensors
                labels = {}
                for key in first_labels.keys():
                    labels[key] = torch.stack([item["labels"][key] for item in batch])
                result["labels"] = labels
            else:
                # Single-head: labels is a single tensor
                result["labels"] = torch.stack([item["labels"] for item in batch])

        # Handle multi-label format where labels are top-level keys (labels_headname)
        for key in batch[0]:
            if key.startswith("labels_") and key != "labels":
                result[key] = torch.stack([item[key] for item in batch])

        return result


class NERCollator:
    """Collator for NER (token classification) tasks.

    Pads sequences and labels, using -100 as the ignore index for
    padding positions in labels (required by CrossEntropyLoss).
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }


class RegressionCollator:
    """Collator for regression tasks."""

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        result: dict[str, Any] = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
        }

        if "labels" in batch[0]:
            result["labels"] = torch.stack([item["labels"] for item in batch])

        return result
