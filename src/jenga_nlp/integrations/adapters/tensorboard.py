"""TensorBoard adapter for JengaNLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseAdapter
from ..events import ActivityEvent


class TensorBoardAdapter(BaseAdapter):
    def __init__(self, log_dir: str | Path) -> None:
        super().__init__(name="tensorboard")
        self.log_dir = Path(log_dir)
        self._writer = None

    def is_available(self) -> bool:
        try:
            from torch.utils.tensorboard import SummaryWriter  # noqa: F401
        except Exception:
            return False
        return True

    def _load_writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        return self._writer

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any:
        writer = self._load_writer()
        self._run = {"run_name": run_name, "tags": tags or {}}
        self.log_params(config)
        writer.add_text("run/name", run_name, 0)
        return self._run

    def handle_event(self, event: ActivityEvent) -> None:
        writer = self._load_writer()
        if event.event_type.value == "step_logged":
            metrics = event.payload.get("metrics", {}) if isinstance(event.payload, dict) else {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"train/{key}", float(value), event.step or 0)
        elif event.event_type.value == "metrics_logged":
            metrics = event.payload.get("metrics", {}) if isinstance(event.payload, dict) else {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"eval/{key}", float(value), event.epoch or 0)
        elif event.event_type.value == "checkpoint_saved":
            writer.add_text("checkpoint/path", str(event.payload.get("artifact_path", "")), event.epoch or 0)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        writer = self._load_writer()
        for key, value in metrics.items():
            writer.add_scalar(key, float(value), step or 0)

    def log_params(self, params: dict[str, Any]) -> None:
        writer = self._load_writer()
        for key, value in params.items():
            writer.add_text(f"params/{key}", str(value), 0)

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        writer = self._load_writer()
        writer.add_text(f"artifacts/{name or Path(path).name}", str(path), 0)

    def finish(self, status: str = "success") -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        super().finish(status=status)
