"""MLflow adapter for JengaNLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseAdapter
from ..events import ActivityEvent


class MLflowAdapter(BaseAdapter):
    def __init__(self, tracking_uri: str | None = None, experiment_name: str | None = None) -> None:
        super().__init__(name="mlflow")
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name or "jenga_nlp"
        self._mlflow = None

    def is_available(self) -> bool:
        try:
            import mlflow  # noqa: F401
        except Exception:
            return False
        return True

    def _load(self):
        if self._mlflow is None:
            import mlflow

            self._mlflow = mlflow
        return self._mlflow

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any:
        mlflow = self._load()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        run = mlflow.start_run(run_name=run_name)
        self._run = run
        self.log_params(config)
        if tags:
            mlflow.set_tags(tags)
        return run

    def handle_event(self, event: ActivityEvent) -> None:
        if self._mlflow is None:
            self._load()
        super().handle_event(event)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._mlflow is None:
            self._load()
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is None:
            self._load()
        flat = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(flat)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        if self._mlflow is None:
            self._load()
        self._mlflow.log_artifact(str(path), artifact_path=name)

    def finish(self, status: str = "success") -> None:
        if self._mlflow is not None:
            self._mlflow.end_run(status=status)
        super().finish(status=status)
