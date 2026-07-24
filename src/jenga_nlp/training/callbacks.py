"""Training callbacks for JengaNLP.

This module also provides modular callbacks for:
- Logging (TensorBoard, MLflow)
- Early stopping
- Checkpoint saving
- Custom user callbacks
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from ..core.config import CheckpointConfig, LoggingConfig, LoggingService

logger = logging.getLogger(__name__)


class TrainingCallback:
    """Base class for training callbacks."""

    def on_train_begin(self, **kwargs: Any) -> None:
        pass

    def on_train_end(self, **kwargs: Any) -> None:
        pass

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        pass

    def on_checkpoint_saved(self, checkpoint_path: str | Path, state_path: str | Path | None = None, **kwargs: Any) -> None:
        pass

    def on_step(self, step: int, loss: float, **kwargs: Any) -> None:
        pass

    def on_train_error(self, error: Exception | str, **kwargs: Any) -> None:
        pass

    def should_stop(self) -> bool:
        return False


class LoggingCallback(TrainingCallback):
    """Callback for logging metrics to TensorBoard or MLflow, with clean callbacks.
    """

    def __init__(self, config: LoggingConfig, output_dir: str) -> None:
        self.config = config
        self._writer: Any = None

        if config.service == LoggingService.TENSORBOARD:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(output_dir, "logs", config.experiment_name)
            self._writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")

        elif config.service == LoggingService.MLFLOW:
            import mlflow

            tracking_uri = config.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(config.experiment_name)
            run = mlflow.start_run()
            # Capture active run and expose a clickable UI URL when possible
            self._writer = mlflow
            try:
                active_run = mlflow.active_run() or run
                resolved_tracking_uri = tracking_uri or mlflow.get_tracking_uri() or ""
                exp_id = active_run.info.experiment_id
                run_id = active_run.info.run_id
                ui_url = None

                # Construct UI URL for common HTTP tracking servers
                if resolved_tracking_uri.startswith("http"):
                    ui_url = f"{resolved_tracking_uri.rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
                # For file-backed stores, include instructions to run the MLflow UI
                elif resolved_tracking_uri.startswith("file:"):
                    ui_url = (
                        f"mlflow ui --backend-store-uri {resolved_tracking_uri} "
                        f"(then open http://127.0.0.1:5000/#/experiments/{exp_id}/runs/{run_id})"
                    )

                if ui_url:
                    logger.info("MLflow UI: %s", ui_url)
                    try:
                        # log the UI url as a run param so it's discoverable in artifacts/parameters
                        mlflow.log_param("mlflow.ui.url", ui_url)
                    except Exception:
                        # Non-fatal: best-effort logging of the UI url
                        logger.debug("Failed to log MLflow UI url as param")
            except Exception:
                logger.debug("Could not determine MLflow run URL")
            logger.info("MLflow experiment: %s", config.experiment_name)

    def on_step(self, step: int, loss: float, **kwargs: Any) -> None:
        if step % self.config.log_every_n_steps != 0:
            return

        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            self._writer.add_scalar("train/loss", loss, step)
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.log_metrics({"train_loss": loss}, step=step)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            for key, value in metrics.items():
                self._writer.add_scalar(f"eval/{key}", value, epoch)
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.log_metrics(metrics, step=epoch)

    def on_train_end(self, **kwargs: Any) -> None:
        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            self._writer.close()
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.end_run()
        logger.info("Logger closed")


class EarlyStoppingCallback(TrainingCallback):
    """Callback for early stopping based on eval metrics, with stopping logic.
    """

    def __init__(
        self,
        patience: int,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.patience = patience
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_value = float("-inf") if greater_is_better else float("inf")
        self.epochs_no_improve = 0
        self._should_stop = False

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        value = metrics.get(self.metric_name)
        if value is None:
            logger.warning(f"Metric '{self.metric_name}' not found for early stopping")
            return

        improved = (
            (self.greater_is_better and value > self.best_value)
            or (not self.greater_is_better and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.epochs_no_improve = 0
            logger.info(f"Metric '{self.metric_name}' improved to {value:.6f}")
        else:
            self.epochs_no_improve += 1
            logger.info(
                f"Metric '{self.metric_name}' did not improve. "
                f"Patience: {self.epochs_no_improve}/{self.patience}"
            )

        if self.epochs_no_improve >= self.patience:
            self._should_stop = True
            logger.info("Early stopping triggered")

    def should_stop(self) -> bool:
        return self._should_stop


class CheckpointCallback(TrainingCallback):
    """Callback for saving model checkpoints.

    intervals and keeps only the N most recent.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        output_dir: str,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir) / "checkpoints"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_value = float("-inf") if greater_is_better else float("inf")
        self._saved_checkpoints: list[Path] = []
        self._model_ref: Any = None
        self.latest_checkpoint_path: Path | None = None
        self.latest_state_path: Path | None = None

    def on_train_begin(self, **kwargs: Any) -> None:
        self._model_ref = kwargs.get("model")

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        model = kwargs.get("model", self._model_ref)
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        scaler = kwargs.get("scaler")
        global_step = kwargs.get("global_step")
        if model is None:
            return

        # Save at interval
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            path, state_path = self._save_checkpoint(
                model, epoch, metrics, is_best=False, optimizer=optimizer, scheduler=scheduler, scaler=scaler, global_step=global_step
            )
            for cb in kwargs.get("callbacks", []):
                checkpoint_event = getattr(cb, "on_checkpoint_saved", None)
                if callable(checkpoint_event):
                    checkpoint_event(checkpoint_path=path, state_path=state_path, epoch=epoch, metrics=metrics, global_step=global_step)

        # Save best model
        if self.config.save_best:
            value = metrics.get(self.metric_name)
            if value is not None:
                improved = (
                    (self.greater_is_better and value > self.best_value)
                    or (not self.greater_is_better and value < self.best_value)
                )
                if improved:
                    self.best_value = value
                    path, state_path = self._save_checkpoint(
                        model, epoch, metrics, is_best=True, optimizer=optimizer, scheduler=scheduler, scaler=scaler, global_step=global_step
                    )
                    for cb in kwargs.get("callbacks", []):
                        checkpoint_event = getattr(cb, "on_checkpoint_saved", None)
                        if callable(checkpoint_event):
                            checkpoint_event(checkpoint_path=path, state_path=state_path, epoch=epoch, metrics=metrics, global_step=global_step)

    def _save_checkpoint(
        self,
        model: Any,
        epoch: int,
        metrics: dict[str, float],
        is_best: bool,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        global_step: int | None = None,
    ) -> tuple[Path, Path | None]:
        import torch

        if is_best:
            save_path = self.output_dir / "best"
        else:
            save_path = self.output_dir / f"epoch_{epoch}"

        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "model.pt")

        # Save metadata
        meta = {"epoch": epoch, "metrics": metrics, "is_best": is_best}
        with open(save_path / "checkpoint_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        state_path: Path | None = None
        if optimizer is not None or scheduler is not None or scaler is not None or global_step is not None:
            state_path = save_path / "training_state.pt"
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "metrics": metrics,
                "is_best": is_best,
            }
            torch.save(state, state_path)

        if not is_best:
            self._saved_checkpoints.append(save_path)
            # Remove old checkpoints beyond max
            while len(self._saved_checkpoints) > self.config.max_checkpoints:
                old_path = self._saved_checkpoints.pop(0)
                if old_path.exists():
                    shutil.rmtree(old_path)
                    logger.debug(f"Removed old checkpoint: {old_path}")

        label = "best" if is_best else f"epoch {epoch}"
        logger.info(f"Checkpoint saved ({label}): {save_path}")
        self.latest_checkpoint_path = save_path
        self.latest_state_path = state_path
        return save_path, state_path
