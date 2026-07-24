"""WER/CER metrics for ASR without mandatory external downloads."""

from __future__ import annotations

from typing import Any, Sequence


def _edit_distance(source: Sequence[str], target: Sequence[str]) -> int:
    rows = len(source) + 1
    cols = len(target) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if source[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[-1][-1]


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    total_errors = 0
    total_words = 0
    for prediction, reference in zip(predictions, references):
        pred_words = prediction.split()
        ref_words = reference.split()
        total_errors += _edit_distance(pred_words, ref_words)
        total_words += len(ref_words)
    return 0.0 if total_words == 0 else 100.0 * total_errors / total_words


def char_error_rate(predictions: list[str], references: list[str]) -> float:
    total_errors = 0
    total_chars = 0
    for prediction, reference in zip(predictions, references):
        total_errors += _edit_distance(list(prediction), list(reference))
        total_chars += len(reference)
    return 0.0 if total_chars == 0 else 100.0 * total_errors / total_chars


def compute_wer_cer(predictions: list[str], references: list[str]) -> dict[str, float]:
    return {"wer": word_error_rate(predictions, references), "cer": char_error_rate(predictions, references)}


def build_asr_compute_metrics(processor: Any):
    def compute_metrics(pred: Any) -> dict[str, float]:
        import numpy as np

        pred_ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        label_ids = np.array(pred.label_ids, copy=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return compute_wer_cer(pred_str, label_str)

    return compute_metrics
