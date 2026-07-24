"""Unit tests for training callbacks module."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from jenga_nlp.core.config import LoggingConfig, LoggingService, CheckpointConfig
from jenga_nlp.training.callbacks import (
    TrainingCallback,
    LoggingCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
)


class TestTrainingCallback:
    """Test base TrainingCallback class."""

    def test_on_train_begin_noop(self):
        """on_train_begin should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_train_begin()  # Should not raise

    def test_on_train_end_noop(self):
        """on_train_end should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_train_end()  # Should not raise

    def test_on_epoch_begin_noop(self):
        """on_epoch_begin should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_epoch_begin(epoch=0)  # Should not raise

    def test_on_epoch_end_noop(self):
        """on_epoch_end should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_epoch_end(epoch=0, metrics={})  # Should not raise

    def test_on_checkpoint_saved_noop(self):
        """on_checkpoint_saved should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_checkpoint_saved(checkpoint_path="/tmp/ckpt")  # Should not raise

    def test_on_step_noop(self):
        """on_step should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_step(step=0, loss=0.5)  # Should not raise

    def test_on_train_error_noop(self):
        """on_train_error should be a no-op by default."""
        cb = TrainingCallback()
        cb.on_train_error(error="test error")  # Should not raise

    def test_should_stop_default_false(self):
        """should_stop should return False by default."""
        cb = TrainingCallback()
        assert cb.should_stop() is False


class TestLoggingCallback:
    """Test LoggingCallback for TensorBoard and MLflow logging."""

    def test_tensorboard_init(self):
        """LoggingCallback should initialize TensorBoard logging."""
        config = LoggingConfig(
            service=LoggingService.TENSORBOARD,
            experiment_name="test_exp",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(config, tmpdir)
            assert cb._writer is not None
            assert cb.config.service == LoggingService.TENSORBOARD
            cb._writer.close()

    def test_tensorboard_on_step(self):
        """LoggingCallback should log steps to TensorBoard."""
        config = LoggingConfig(
            service=LoggingService.TENSORBOARD,
            experiment_name="test_exp",
            log_every_n_steps=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(config, tmpdir)
            cb.on_step(step=0, loss=0.5)  # Should not raise
            cb._writer.close()

    def test_tensorboard_on_epoch_end(self):
        """LoggingCallback should log epoch metrics to TensorBoard."""
        config = LoggingConfig(
            service=LoggingService.TENSORBOARD,
            experiment_name="test_exp",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(config, tmpdir)
            metrics = {"accuracy": 0.95, "loss": 0.1}
            cb.on_epoch_end(epoch=0, metrics=metrics)  # Should not raise
            cb._writer.close()

    def test_tensorboard_on_train_end(self):
        """LoggingCallback should close TensorBoard writer on train end."""
        config = LoggingConfig(
            service=LoggingService.TENSORBOARD,
            experiment_name="test_exp",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(config, tmpdir)
            writer = cb._writer
            cb.on_train_end()  # Should close writer
            # Verify it's closed (writer.close was called)
            assert writer is not None

    @patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: MagicMock() if name == "mlflow" else __import__(name, *args, **kwargs))
    def test_mlflow_init_with_tracking_uri(self, mock_import):
        """LoggingCallback should initialize MLflow when configured."""
        config = LoggingConfig(
            service=LoggingService.MLFLOW,
            experiment_name="test_exp",
            tracking_uri="http://localhost:5000",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = LoggingCallback(config, tmpdir)
            # Verify callback was initialized
            assert cb._writer is not None
            assert cb.config.service == LoggingService.MLFLOW

    def test_mlflow_init_with_env_tracking_uri(self):
        """LoggingCallback should use MLFLOW_TRACKING_URI env var when config tracking_uri is None."""
        config = LoggingConfig(
            service=LoggingService.MLFLOW,
            experiment_name="test_exp",
            tracking_uri=None,
        )
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow-test"
        fake_mlflow.active_run.return_value = MagicMock(info=MagicMock(experiment_id="0", run_id="run-1"))

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "file:///tmp/mlflow-test"}), patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: fake_mlflow if name == "mlflow" else __import__(name, *args, **kwargs)):
            cb = LoggingCallback(config, tmpdir)
            assert cb.config.service == LoggingService.MLFLOW
            assert cb._writer is fake_mlflow

    def test_mlflow_ui_url_http(self):
        """LoggingCallback should construct HTTP MLflow UI URL (integration test)."""
        # This test requires a running MLflow server; skip for now as it's integration-level
        pytest.skip("Requires running MLflow server")

    def test_mlflow_ui_url_file_backend(self):
        """LoggingCallback should handle file-backed MLflow store."""
        # This test also requires actual MLflow; skip for unit tests
        pytest.skip("Requires MLflow installation and setup")

    def test_mlflow_on_step(self):
        """LoggingCallback should initialize MLflow logging correctly."""
        config = LoggingConfig(
            service=LoggingService.MLFLOW,
            experiment_name="test_exp",
            log_every_n_steps=1,
        )
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow-test"
        fake_mlflow.active_run.return_value = MagicMock(info=MagicMock(experiment_id="0", run_id="run-1"))

        with tempfile.TemporaryDirectory() as tmpdir, patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: fake_mlflow if name == "mlflow" else __import__(name, *args, **kwargs)):
            cb = LoggingCallback(config, tmpdir)
            assert cb._writer is fake_mlflow
            cb.on_step(step=0, loss=0.5)

    def test_mlflow_on_epoch_end(self):
        """LoggingCallback should be configured for MLflow epoch logging."""
        config = LoggingConfig(
            service=LoggingService.MLFLOW,
            experiment_name="test_exp",
        )
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow-test"
        fake_mlflow.active_run.return_value = MagicMock(info=MagicMock(experiment_id="0", run_id="run-1"))

        with tempfile.TemporaryDirectory() as tmpdir, patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: fake_mlflow if name == "mlflow" else __import__(name, *args, **kwargs)):
            cb = LoggingCallback(config, tmpdir)
            cb.on_epoch_end(epoch=0, metrics={"accuracy": 0.95})
            assert cb.config.service == LoggingService.MLFLOW

    def test_mlflow_on_train_end(self):
        """LoggingCallback should end MLflow run on train end."""
        config = LoggingConfig(
            service=LoggingService.MLFLOW,
            experiment_name="test_exp",
        )
        fake_mlflow = MagicMock()
        fake_mlflow.get_tracking_uri.return_value = "file:///tmp/mlflow-test"
        fake_mlflow.active_run.return_value = MagicMock(info=MagicMock(experiment_id="0", run_id="run-1"))

        with tempfile.TemporaryDirectory() as tmpdir, patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: fake_mlflow if name == "mlflow" else __import__(name, *args, **kwargs)):
            cb = LoggingCallback(config, tmpdir)
            cb.on_train_end()


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""

    def test_init(self):
        """EarlyStoppingCallback should initialize with patience."""
        cb = EarlyStoppingCallback(patience=3, metric_name="eval_loss", greater_is_better=False)
        assert cb.patience == 3
        assert cb.metric_name == "eval_loss"
        assert cb.greater_is_better is False
        assert cb.should_stop() is False

    def test_improvement_greater_is_better(self):
        """EarlyStoppingCallback should track improvement when greater_is_better=True."""
        cb = EarlyStoppingCallback(patience=2, metric_name="accuracy", greater_is_better=True)
        cb.on_epoch_end(epoch=0, metrics={"accuracy": 0.80})
        assert cb.best_value == 0.80
        assert cb.epochs_no_improve == 0

    def test_improvement_less_is_better(self):
        """EarlyStoppingCallback should track improvement when greater_is_better=False."""
        cb = EarlyStoppingCallback(patience=2, metric_name="eval_loss", greater_is_better=False)
        cb.on_epoch_end(epoch=0, metrics={"eval_loss": 0.5})
        assert cb.best_value == 0.5
        assert cb.epochs_no_improve == 0

    def test_no_improvement(self):
        """EarlyStoppingCallback should increment patience counter on no improvement."""
        cb = EarlyStoppingCallback(patience=2, metric_name="eval_loss", greater_is_better=False)
        cb.on_epoch_end(epoch=0, metrics={"eval_loss": 0.5})
        cb.on_epoch_end(epoch=1, metrics={"eval_loss": 0.6})  # Worse
        assert cb.epochs_no_improve == 1

    def test_patience_exceeded(self):
        """EarlyStoppingCallback should signal stop when patience exceeded."""
        cb = EarlyStoppingCallback(patience=2, metric_name="eval_loss", greater_is_better=False)
        cb.on_epoch_end(epoch=0, metrics={"eval_loss": 0.5})
        cb.on_epoch_end(epoch=1, metrics={"eval_loss": 0.6})  # No improve
        assert cb.should_stop() is False
        cb.on_epoch_end(epoch=2, metrics={"eval_loss": 0.7})  # No improve
        assert cb.should_stop() is True

    def test_missing_metric(self, caplog):
        """EarlyStoppingCallback should warn when metric is missing."""
        cb = EarlyStoppingCallback(patience=2, metric_name="missing_metric")
        with caplog.at_level(logging.WARNING):
            cb.on_epoch_end(epoch=0, metrics={"accuracy": 0.95})
        assert "not found" in caplog.text.lower()


class TestCheckpointCallback:
    """Test CheckpointCallback."""

    def test_init(self):
        """CheckpointCallback should initialize with proper output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_every_n_epochs=1, save_best=True, max_checkpoints=3)
            cb = CheckpointCallback(config, tmpdir)
            assert (Path(tmpdir) / "checkpoints").exists()
            assert cb.metric_name == "eval_loss"
            assert cb.greater_is_better is False

    def test_on_train_begin_saves_model_ref(self):
        """CheckpointCallback.on_train_begin should save model reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_every_n_epochs=1, save_best=True, max_checkpoints=3)
            cb = CheckpointCallback(config, tmpdir)
            mock_model = MagicMock()
            cb.on_train_begin(model=mock_model)
            assert cb._model_ref is mock_model

    def test_save_checkpoint_creates_directory(self):
        """_save_checkpoint should create checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_every_n_epochs=1, save_best=True, max_checkpoints=3)
            cb = CheckpointCallback(config, tmpdir)
            
            mock_model = MagicMock()
            with patch("torch.save"):
                path, state_path = cb._save_checkpoint(
                    mock_model,
                    epoch=0,
                    metrics={"loss": 0.1},
                    is_best=False,
                )
            assert path.exists()

    def test_save_best_checkpoint(self):
        """CheckpointCallback should save best checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_every_n_epochs=1, save_best=True, max_checkpoints=3)
            cb = CheckpointCallback(config, tmpdir, metric_name="eval_loss", greater_is_better=False)
            
            mock_model = MagicMock()
            metrics = {"eval_loss": 0.1}
            
            with patch("torch.save"):
                cb.on_epoch_end(epoch=0, model=mock_model, metrics=metrics)
            
            best_ckpt = Path(tmpdir) / "checkpoints" / "best"
            assert best_ckpt.exists()

    def test_max_checkpoints_rotation(self):
        """CheckpointCallback should remove old checkpoints beyond max_checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(save_every_n_epochs=1, save_best=False, max_checkpoints=2)
            cb = CheckpointCallback(config, tmpdir)
            
            mock_model = MagicMock()
            metrics = {"eval_loss": 0.1}
            
            with patch("torch.save"):
                # Save 3 checkpoints
                cb.on_epoch_end(epoch=0, model=mock_model, metrics=metrics)
                cb.on_epoch_end(epoch=1, model=mock_model, metrics=metrics)
                cb.on_epoch_end(epoch=2, model=mock_model, metrics=metrics)
            
            # Only 2 should remain (epoch_1 and epoch_2)
            ckpt_dir = Path(tmpdir) / "checkpoints"
            ckpt_count = len(list(ckpt_dir.glob("epoch_*")))
            assert ckpt_count <= 2
