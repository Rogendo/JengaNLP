"""Metric computation utilities for JengaNLP.

Provides functions to compute evaluation metrics for different task types.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute metrics for single-label classification.

    Args:
        predictions: [num_samples, num_classes] logits or probabilities.
        labels: [num_samples] ground truth class indices.

    Returns:
        Dict with accuracy, precision, recall, f1.
    """
    preds = np.argmax(predictions, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def compute_multi_label_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute metrics for multi-label classification.

    Args:
        predictions: [num_samples, num_labels] logits.
        labels: [num_samples, num_labels] ground truth binary labels.
        threshold: Threshold for converting logits to binary predictions.

    Returns:
        Dict with micro-averaged precision, recall, f1.
    """
    # Apply sigmoid and threshold
    from scipy.special import expit
    probs = expit(predictions)
    preds = (probs > threshold).astype(int)

    return {
        "precision": float(precision_score(labels, preds, average="micro", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="micro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
    }


def compute_ner_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    id_to_label: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute metrics for NER (token classification).

    Ignores padding tokens (label == -100) and special tokens.

    Args:
        predictions: [num_samples, seq_len, num_labels] logits.
        labels: [num_samples, seq_len] ground truth label IDs.
        id_to_label: Optional mapping from ID to label name.

    Returns:
        Dict with accuracy, precision, recall, f1 (token-level).
    """
    preds = np.argmax(predictions, axis=-1)

    # Flatten and filter out ignored tokens (-100)
    flat_preds = preds.flatten()
    flat_labels = labels.flatten()

    mask = flat_labels != -100
    filtered_preds = flat_preds[mask]
    filtered_labels = flat_labels[mask]

    if len(filtered_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "accuracy": float(accuracy_score(filtered_labels, filtered_preds)),
        "precision": float(precision_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)),
        "recall": float(recall_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)),
        "f1": float(f1_score(filtered_labels, filtered_preds, average="weighted", zero_division=0)),
    }


def compute_regression_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute metrics for regression tasks.

    Args:
        predictions: [num_samples] or [num_samples, 1] predicted values.
        labels: [num_samples] ground truth values.

    Returns:
        Dict with mse, rmse, mae.
    """
    preds = predictions.flatten()
    targets = labels.flatten()

    mse = float(np.mean((preds - targets) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(preds - targets))),
    }
