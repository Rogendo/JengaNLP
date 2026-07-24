"""Unit tests for training metrics module."""

import numpy as np
import pytest

from jenga_nlp.training.metrics import (
    compute_classification_metrics,
    compute_multi_label_metrics,
    compute_ner_metrics,
    compute_regression_metrics,
)


class TestComputeClassificationMetrics:
    """Test classification metrics computation."""

    def test_perfect_predictions(self):
        """compute_classification_metrics should return 1.0 for perfect predictions."""
        # Predictions: [num_samples, num_classes]
        predictions = np.array([
            [1.0, 0.0],  # Predicts class 0
            [0.0, 1.0],  # Predicts class 1
            [1.0, 0.0],  # Predicts class 0
            [0.0, 1.0],  # Predicts class 1
        ])
        labels = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(predictions, labels)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self):
        """compute_classification_metrics should work with random data."""
        predictions = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ])
        labels = np.array([0, 0, 0, 1, 1, 0])
        metrics = compute_classification_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_all_wrong_predictions(self):
        """compute_classification_metrics should handle completely wrong predictions."""
        predictions = np.array([
            [0.0, 1.0],  # Predicts class 1, label is 0
            [1.0, 0.0],  # Predicts class 0, label is 1
            [0.0, 1.0],  # Predicts class 1, label is 0
            [1.0, 0.0],  # Predicts class 0, label is 1
        ])
        labels = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(predictions, labels)
        
        assert metrics["accuracy"] == 0.0

    def test_binary_classification(self):
        """compute_classification_metrics should work for binary classification."""
        predictions = np.array([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
            [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
        ])
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        metrics = compute_classification_metrics(predictions, labels)
        
        # 6 correct out of 8
        assert metrics["accuracy"] == 0.75


class TestComputeMultiLabelMetrics:
    """Test multi-label metrics computation."""

    def test_perfect_predictions(self):
        """compute_multi_label_metrics should return high scores for perfect predictions."""
        predictions = np.array([
            [2.0, -2.0, 2.0],  # After sigmoid + threshold: [1, 0, 1]
            [-2.0, 2.0, -2.0],  # After sigmoid + threshold: [0, 1, 0]
            [2.0, 2.0, -2.0],  # After sigmoid + threshold: [1, 1, 0]
        ])
        labels = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ])
        metrics = compute_multi_label_metrics(predictions, labels)
        
        # Should return precision, recall, f1 (not hamming_loss)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        # Perfect predictions should have f1=1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self):
        """compute_multi_label_metrics should work with random data."""
        predictions = np.array([
            [2.0, -2.0, 2.0],
            [-2.0, 2.0, -2.0],
            [2.0, -2.0, 2.0],
        ])
        labels = np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ])
        metrics = compute_multi_label_metrics(predictions, labels)
        
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["f1"] <= 1

    def test_single_instance(self):
        """compute_multi_label_metrics should work with a single instance."""
        predictions = np.array([[2.0, -2.0, 2.0]])
        labels = np.array([[1, 0, 1]])
        metrics = compute_multi_label_metrics(predictions, labels)
        
        assert metrics["f1"] == 1.0


class TestComputeNERMetrics:
    """Test NER metrics computation."""

    def test_perfect_predictions(self):
        """compute_ner_metrics should return high scores for perfect predictions."""
        # Predictions: [num_samples, seq_len, num_labels]
        predictions = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        ])
        # Labels: [num_samples, seq_len]
        labels = np.array([
            [0, 1, 1, 2, 2, 0],
            [1, 1, 0, 0, 2, 2],
        ])
        metrics = compute_ner_metrics(predictions, labels)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        # Perfect predictions
        assert metrics["f1"] == 1.0

    def test_token_level_accuracy(self):
        """compute_ner_metrics should compute token-level accuracy."""
        predictions = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],
        ])
        labels = np.array([
            [0, 1, 1, 2, 2, 0],
            [1, 1, 0, 0, 2, 2],
        ])
        metrics = compute_ner_metrics(predictions, labels)
        
        # All correct, so F1 should be 1.0
        assert metrics["f1"] == 1.0

    def test_partial_correct(self):
        """compute_ner_metrics should handle partially correct predictions."""
        predictions = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]],  # One token different
        ])
        labels = np.array([
            [0, 1, 1, 2, 2, 0],
            [1, 1, 0, 0, 2, 2],
        ])
        metrics = compute_ner_metrics(predictions, labels)
        
        assert "f1" in metrics
        assert 0 <= metrics["f1"] <= 1

    def test_with_padding_tokens(self):
        """compute_ner_metrics should ignore padding tokens (label == -100)."""
        predictions = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]],  # These should be ignored
        ])
        labels = np.array([
            [0, 1, 1],
            [1, -100, -100],  # -100 are padding tokens
        ])
        metrics = compute_ner_metrics(predictions, labels)
        
        assert "f1" in metrics
        # Should not crash with -100 padding tokens


class TestComputeRegressionMetrics:
    """Test regression metrics computation."""

    def test_perfect_predictions(self):
        """compute_regression_metrics should return 0 MSE for perfect predictions."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_regression_metrics(predictions, labels)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_constant_error(self):
        """compute_regression_metrics should handle constant error."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([2.0, 3.0, 4.0, 5.0])
        metrics = compute_regression_metrics(predictions, labels)
        
        # All predictions are 1 unit off
        assert metrics["mse"] == 1.0
        assert metrics["rmse"] == 1.0
        assert metrics["mae"] == 1.0

    def test_single_value(self):
        """compute_regression_metrics should work with single value."""
        predictions = np.array([1.5])
        labels = np.array([1.5])
        metrics = compute_regression_metrics(predictions, labels)
        
        assert metrics["mse"] == 0.0
        assert metrics["mae"] == 0.0

    def test_multiple_errors(self):
        """compute_regression_metrics should compute correctly with multiple errors."""
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([1.1, 1.9, 3.2, 3.8, 5.0])
        metrics = compute_regression_metrics(predictions, labels)
        
        assert metrics["mse"] > 0
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0

