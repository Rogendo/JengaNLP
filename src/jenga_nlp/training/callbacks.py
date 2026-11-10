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
from typing import Any

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


def failure_status(exc: BaseException) -> str:
    """Tracker status for a run that ended with ``exc``: ``KILLED`` for Ctrl-C, otherwise ``FAILED``."""
    return "KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED"


class LoggingCallback(TrainingCallback):
    """Callback for logging metrics to TensorBoard or MLflow, with clean callbacks.
    """

    def __init__(
        self,
        config: LoggingConfig,
        output_dir: str,
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        config_dict: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self._writer: Any = None
        self._closed = False

        if config.service == LoggingService.TENSORBOARD:
            from datetime import datetime

            from torch.utils.tensorboard import SummaryWriter

            # One sub-directory per run, so `tensorboard --logdir <output_dir>/logs`
            # lists train and eval runs separately instead of merging them.
            run_dir = f"{run_name or 'run'}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            log_dir = os.path.join(output_dir, "logs", config.experiment_name, run_dir)
            self._writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
            self._tensorboard_write_run_info(params, tags, config_dict)

        elif config.service == LoggingService.MLFLOW:
            import mlflow

            tracking_uri = config.tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            experiment = mlflow.set_experiment(config.experiment_name)
            try:
                # MLflow 3 opens experiments in its GenAI view unless told otherwise;
                # mark JengaNLP experiments as model training (never override a user's choice).
                if "mlflow.experimentKind" not in (experiment.tags or {}):
                    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
            except Exception:
                logger.debug("Could not set MLflow experiment kind tag")
            run = mlflow.start_run(run_name=run_name, tags=tags or None)
            try:
                if params:
                    # MLflow rejects values longer than 6000 chars; keep params short.
                    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
                if config_dict:
                    mlflow.log_dict(config_dict, "experiment_config.yaml")
            except Exception as exc:  # Non-fatal: never fail training over logging
                logger.warning("Could not log MLflow params/config: %s", exc)
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

    def _tensorboard_write_run_info(
        self,
        params: dict[str, Any] | None,
        tags: dict[str, str] | None,
        config_dict: dict[str, Any] | None,
    ) -> None:
        """Record hyperparameters, tags and the full config in TensorBoard's Text tab."""
        try:
            rows = {**(tags or {}), **(params or {})}
            if rows:
                table = "| key | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows.items())
                self._writer.add_text("run/params", table, 0)
            if config_dict:
                import yaml

                config_yaml = yaml.safe_dump(config_dict, sort_keys=False)
                self._writer.add_text("run/config", "```yaml\n" + config_yaml + "```", 0)
        except Exception as exc:  # Non-fatal
            logger.warning("Could not write run info to TensorBoard: %s", exc)

    def on_step(self, step: int, loss: float, **kwargs: Any) -> None:
        if step % self.config.log_every_n_steps != 0:
            return

        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            self._writer.add_scalar("train/loss", loss, step)
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.log_metrics({"train_loss": loss}, step=step)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        # Use the global step (same axis as train_loss) when the trainer provides it,
        # so train and eval curves line up in TensorBoard/MLflow charts.
        step = kwargs.get("global_step", epoch)
        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            for key, value in metrics.items():
                self._writer.add_scalar(f"eval/{key}", value, step)
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.log_metrics(metrics, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0, prefix: str = "") -> None:
        """Log a metrics dict outside the training loop (e.g. standalone evaluation)."""
        if self._closed or not self._writer:
            return
        numeric = {f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if self.config.service == LoggingService.TENSORBOARD:
            # Same "eval/" namespace the training loop uses, so charts group together.
            for key, value in numeric.items():
                self._writer.add_scalar(key if prefix else f"eval/{key}", value, step)
        elif self.config.service == LoggingService.MLFLOW:
            self._writer.log_metrics(numeric, step=step)

    def close(self, status: str = "FINISHED") -> None:
        """Flush and close the logger. Safe to call more than once.

        Args:
            status: MLflow run status to record: ``FINISHED``, ``FAILED`` or ``KILLED``.
        """
        if self._closed:
            return
        self._closed = True
        if self.config.service == LoggingService.TENSORBOARD and self._writer:
            self._writer.close()
        elif self.config.service == LoggingService.MLFLOW and self._writer:
            self._writer.end_run(status=status)
        logger.info("Logger closed (%s)", status)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Upload a file or folder to the tracker (MLflow only; a no-op for TensorBoard)."""
        if self._closed or self.config.service != LoggingService.MLFLOW or not self._writer:
            return
        path = Path(path)
        try:
            if path.is_dir():
                self._writer.log_artifacts(str(path), artifact_path=artifact_path)
            elif path.is_file():
                self._writer.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as exc:  # Non-fatal: never fail training over artifact upload
            logger.warning("Could not upload %s to MLflow: %s", path, exc)

    def on_train_end(self, output_dir: str | None = None, status: str = "FINISHED", **kwargs: Any) -> None:
        if output_dir and status == "FINISHED" and self.config.service == LoggingService.MLFLOW:
            out = Path(output_dir)
            if self.config.log_model:
                for name in MODEL_ARTIFACTS:
                    self.log_artifact(out / name, artifact_path="model/" + name if (out / name).is_dir() else "model")
                logger.info("Uploaded model folder to MLflow artifacts (model/)")
            if self.config.log_checkpoints and (out / "checkpoints").is_dir():
                self.log_artifact(out / "checkpoints", artifact_path="checkpoints")
                logger.info("Uploaded checkpoints to MLflow artifacts (checkpoints/)")
        self.close(status=status)


# Files that make up a self-contained, loadable JengaNLP model folder.
MODEL_ARTIFACTS = ("model.pt", "metadata.json", "experiment_config.yaml", "encoder_config", "tokenizer")


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
        # Created on first save, so evaluation-only trainers don't create empty folders.
        self.output_dir = Path(output_dir) / "checkpoints"
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
