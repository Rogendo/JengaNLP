"""Per-task-type formatters that convert raw logits to structured predictions.

These are pure functions (no model state). Each formatter takes raw logit
tensors from a task head and produces a dict suitable for HeadResult.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from ..core.config import HeadConfig, TaskConfig, TaskType

logger = logging.getLogger(__name__)


def format_single_label(
    logits: torch.Tensor,
    head_config: HeadConfig,
    label_map: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Format single-label classification logits for one sample.

    Args:
        logits: [num_labels] logit tensor (single sample, already indexed).
        head_config: Head configuration.
        label_map: Optional {label_id: label_name} mapping.

    Returns:
        Dict with prediction, label, confidence, probabilities.
    """
    probs = torch.softmax(logits, dim=-1)
    class_id = int(torch.argmax(probs).item())
    confidence = float(probs[class_id].item())

    # Build probability dict
    probabilities: dict[str, float] = {}
    for i in range(probs.shape[0]):
        key = label_map[i] if label_map and i in label_map else str(i)
        probabilities[key] = round(float(probs[i].item()), 4)

    label = label_map.get(class_id) if label_map else None

    return {
        "prediction": class_id,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
    }


def format_multi_label(
    logits: torch.Tensor,
    head_config: HeadConfig,
    label_map: Optional[dict[int, str]] = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Format multi-label classification logits for one sample.

    Args:
        logits: [num_labels] logit tensor (single sample).
        head_config: Head configuration.
        label_map: Optional {label_id: label_name} mapping.
        threshold: Sigmoid threshold for binary predictions.

    Returns:
        Dict with prediction (list[bool]), confidence, scores, labels.
    """
    scores = torch.sigmoid(logits)
    binary = (scores > threshold).cpu().tolist()
    score_list = [round(float(s), 4) for s in scores.cpu().tolist()]
    confidence = round(float(scores.mean().item()), 4)

    # Build labels dict if label_map available
    labels: Optional[dict[str, bool]] = None
    if label_map:
        labels = {}
        for i, val in enumerate(binary):
            key = label_map.get(i, f"metric_{i}")
            labels[key] = bool(val)

    return {
        "prediction": [bool(v) for v in binary],
        "confidence": confidence,
        "scores": score_list,
        "labels": labels,
    }


def format_qa(
    logits: dict[str, torch.Tensor],
    task_config: TaskConfig,
    sample_idx: int = 0,
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Format QA scoring logits by delegating to QAScoringTask.

    This is a thin wrapper that produces HeadResult-compatible dicts
    for each QA head, using the same logic as QAScoringTask.format_predictions
    but also returning scores and confidence.

    Args:
        logits: Dict mapping head_name -> [batch_size, num_labels] tensor.
        task_config: Task configuration with label_maps.
        sample_idx: Index of the sample in the batch.
        threshold: Sigmoid threshold.

    Returns:
        Dict mapping head_name -> HeadResult-compatible dict.
    """
    from ..tasks.qa import DEFAULT_QA_SUB_LABELS

    heads: dict[str, dict[str, Any]] = {}

    for head_config in task_config.heads:
        head_name = head_config.name
        if head_name not in logits:
            continue

        head_logits = logits[head_name][sample_idx]  # [num_labels]
        scores = torch.sigmoid(head_logits)
        binary = (scores > threshold).cpu().tolist()
        score_list = [round(float(s), 4) for s in scores.cpu().tolist()]
        confidence = round(float(scores.mean().item()), 4)

        # Resolve sub-metric names
        sub_names = _resolve_qa_sub_labels(task_config, head_name, head_config.num_labels)

        prediction: dict[str, bool] = {}
        for i, val in enumerate(binary):
            name = sub_names[i] if i < len(sub_names) else f"metric_{i}"
            prediction[name] = bool(val)

        heads[head_name] = {
            "prediction": prediction,
            "confidence": confidence,
            "scores": score_list,
        }

    return heads


def _resolve_qa_sub_labels(
    task_config: TaskConfig,
    head_name: str,
    num_labels: int,
) -> list[str]:
    """Resolve sub-metric label names for a QA head."""
    from ..tasks.qa import DEFAULT_QA_SUB_LABELS

    if task_config.label_maps and head_name in task_config.label_maps:
        label_map = task_config.label_maps[head_name]
        ordered = sorted(label_map.items(), key=lambda x: int(x[0]))
        return [v for _, v in ordered]
    elif head_name in DEFAULT_QA_SUB_LABELS:
        return DEFAULT_QA_SUB_LABELS[head_name]
    else:
        return [f"{head_name}_metric_{i}" for i in range(num_labels)]


def format_ner(
    logits: torch.Tensor,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    head_config: HeadConfig,
    original_text: str,
    label_map: Optional[dict[int, str]] = None,
) -> dict[str, Any]:
    """Format NER logits for one sample into entity spans.

    Uses offset_mapping to reconstruct character-level entity spans
    from subword token predictions. Groups consecutive B-/I- tags
    into entity spans.

    Args:
        logits: [seq_len, num_labels] logit tensor (single sample).
        offset_mapping: [seq_len, 2] char start/end per token.
        attention_mask: [seq_len] attention mask.
        head_config: Head configuration.
        original_text: Original input text for extracting entity text.
        label_map: {label_id: label_str} e.g. {0: "O", 1: "B-PER", 2: "I-PER"}.

    Returns:
        Dict with prediction (list of entity dicts), confidence.
    """
    probs = torch.softmax(logits, dim=-1)
    pred_ids = torch.argmax(probs, dim=-1).cpu().tolist()
    max_probs = probs.max(dim=-1).values.cpu().tolist()
    offsets = offset_mapping.cpu().tolist()
    mask = attention_mask.cpu().tolist()

    entities: list[dict[str, Any]] = []
    current_entity: Optional[dict[str, Any]] = None

    for idx, (pred_id, prob, (char_start, char_end), m) in enumerate(
        zip(pred_ids, max_probs, offsets, mask)
    ):
        # Skip padding, special tokens (offset [0,0] for CLS/SEP/PAD)
        if m == 0 or (char_start == 0 and char_end == 0):
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        label_str = label_map.get(pred_id, str(pred_id)) if label_map else str(pred_id)

        if label_str.startswith("B-"):
            # Start new entity
            if current_entity:
                entities.append(current_entity)
            entity_label = label_str[2:]
            current_entity = {
                "start": int(char_start),
                "end": int(char_end),
                "label": entity_label,
                "confidence": prob,
                "_count": 1,
            }
        elif label_str.startswith("I-") and current_entity:
            entity_label = label_str[2:]
            if entity_label == current_entity["label"]:
                # Continue current entity
                current_entity["end"] = int(char_end)
                current_entity["confidence"] += prob
                current_entity["_count"] += 1
            else:
                # Label mismatch — close current, start new
                entities.append(current_entity)
                current_entity = {
                    "start": int(char_start),
                    "end": int(char_end),
                    "label": entity_label,
                    "confidence": prob,
                    "_count": 1,
                }
        else:
            # "O" tag or unexpected — close any open entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    # Close final entity if open
    if current_entity:
        entities.append(current_entity)

    # Finalize: average confidence, extract text spans
    for ent in entities:
        ent["confidence"] = round(ent["confidence"] / ent.pop("_count"), 4)
        ent["text"] = original_text[ent["start"]: ent["end"]]

    # Overall confidence is mean across all entities (or 1.0 if no entities)
    overall_confidence = (
        round(sum(e["confidence"] for e in entities) / len(entities), 4)
        if entities
        else 1.0
    )

    return {
        "prediction": entities,
        "confidence": overall_confidence,
    }


def format_regression(
    logits: torch.Tensor,
    head_config: HeadConfig,
) -> dict[str, Any]:
    """Format regression logits for one sample.

    Args:
        logits: [num_labels] logit tensor (typically num_labels=1).
        head_config: Head configuration.

    Returns:
        Dict with prediction (float), confidence (None).
    """
    if logits.numel() == 1:
        value = float(logits.item())
    else:
        value = float(logits[0].item())

    return {
        "prediction": round(value, 4),
        "confidence": None,
    }


def format_result(
    task_config: TaskConfig,
    logits: dict[str, torch.Tensor],
    sample_idx: int = 0,
    threshold: float = 0.5,
    offset_mapping: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    original_text: str = "",
) -> dict[str, dict[str, Any]]:
    """Dispatch to the correct formatter based on task type.

    This is the main entry point for formatting predictions. It routes
    to the appropriate per-task formatter and returns a dict of
    head_name -> HeadResult-compatible dicts.

    Args:
        task_config: Task configuration.
        logits: Dict mapping head_name -> [batch_size, ...] logit tensor.
        sample_idx: Index of the sample in the batch.
        threshold: Sigmoid threshold for multi-label/QA.
        offset_mapping: [batch_size, seq_len, 2] for NER.
        attention_mask: [batch_size, seq_len] for NER.
        original_text: Original input text for NER entity extraction.

    Returns:
        Dict mapping head_name -> formatted result dict.
    """
    task_type = task_config.type

    # QA uses a special multi-head formatter
    if task_type == TaskType.QA_SCORING:
        return format_qa(logits, task_config, sample_idx, threshold)

    heads: dict[str, dict[str, Any]] = {}

    for head_config in task_config.heads:
        head_name = head_config.name
        if head_name not in logits:
            continue

        head_logits = logits[head_name][sample_idx]  # index into batch

        # Get label_map for this head if available
        label_map: Optional[dict[int, str]] = None
        if task_config.label_maps and head_name in task_config.label_maps:
            label_map = task_config.label_maps[head_name]

        if task_type in (TaskType.SINGLE_LABEL_CLASSIFICATION, TaskType.SENTIMENT):
            heads[head_name] = format_single_label(head_logits, head_config, label_map)

        elif task_type == TaskType.MULTI_LABEL_CLASSIFICATION:
            heads[head_name] = format_multi_label(
                head_logits, head_config, label_map, threshold
            )

        elif task_type == TaskType.NER:
            if offset_mapping is None or attention_mask is None:
                logger.warning(
                    "NER formatting requires offset_mapping and attention_mask; "
                    "returning raw argmax predictions."
                )
                pred_ids = torch.argmax(head_logits, dim=-1).cpu().tolist()
                heads[head_name] = {
                    "prediction": pred_ids,
                    "confidence": None,
                }
            else:
                heads[head_name] = format_ner(
                    head_logits,
                    offset_mapping[sample_idx],
                    attention_mask[sample_idx],
                    head_config,
                    original_text,
                    label_map,
                )

        elif task_type == TaskType.REGRESSION:
            heads[head_name] = format_regression(head_logits, head_config)

        else:
            logger.warning("Unknown task type %s — returning raw logits", task_type)
            heads[head_name] = {
                "prediction": head_logits.cpu().tolist(),
                "confidence": None,
            }

    return heads
