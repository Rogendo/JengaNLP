"""Weights & Biases adapter for JengaNLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseAdapter
from ..events import ActivityEvent


class WandBAdapter(BaseAdapter):
    def __init__(self, project: str = "jenga_nlp", entity: str | None = None) -> None:
        super().__init__(name="wandb")
        self.project = project
        self.entity = entity
        self._wandb = None

    def is_available(self) -> bool:
        try:
            import wandb  # noqa: F401
        except Exception:
            return False
        return True

    def _load(self):
        if self._wandb is None:
            import wandb

            self._wandb = wandb
        return self._wandb

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any:
        wandb = self._load()
        self._run = wandb.init(project=self.project, entity=self.entity, name=run_name, config=config, tags=list((tags or {}).values()))
        return self._run

    def handle_event(self, event: ActivityEvent) -> None:
        wandb = self._load()
        if event.event_type.value in {"step_logged", "metrics_logged"}:
            metrics = event.payload.get("metrics", {}) if isinstance(event.payload, dict) else {}
            wandb.log(metrics, step=event.step or event.epoch or 0)
        elif event.event_type.value == "checkpoint_saved":
            path = event.payload.get("artifact_path") if isinstance(event.payload, dict) else None
            if path:
                wandb.save(str(path))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        wandb = self._load()
        wandb.log(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        wandb = self._load()
        if self._run is not None:
            wandb.config.update(params)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        wandb = self._load()
        wandb.save(str(path))

    def finish(self, status: str = "success") -> None:
        if self._wandb is not None:
            self._wandb.finish()
        super().finish(status=status)
