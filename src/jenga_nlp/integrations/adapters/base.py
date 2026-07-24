"""Shared adapter protocol for JengaNLP integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..events import ActivityEvent


@runtime_checkable
class BackendAdapter(Protocol):
    """Common interface for tracking and labeling backends."""

    name: str

    def is_available(self) -> bool: ...

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any: ...

    def handle_event(self, event: ActivityEvent) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_artifact(self, path: str | Path, name: str | None = None) -> None: ...

    def finish(self, status: str = "success") -> None: ...


@dataclass(slots=True)
class BaseAdapter:
    """Small convenience base for event-driven adapters."""

    name: str
    _run: Any = field(init=False, default=None, repr=False)

    def is_available(self) -> bool:
        return True

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any:
        self._run = {"run_name": run_name, "config": config, "tags": tags or {}}
        return self._run

    def handle_event(self, event: ActivityEvent) -> None:
        if event.event_type.value == "step_logged":
            metrics = event.payload.get("metrics") if isinstance(event.payload, dict) else None
            if isinstance(metrics, dict):
                self.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, step=event.step)
        elif event.event_type.value == "metrics_logged":
            metrics = event.payload.get("metrics") if isinstance(event.payload, dict) else None
            if isinstance(metrics, dict):
                self.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, step=event.step)
        elif event.event_type.value == "checkpoint_saved":
            artifact = event.payload.get("artifact_path") if isinstance(event.payload, dict) else None
            if artifact:
                self.log_artifact(artifact, name=event.payload.get("artifact_name"))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        return None

    def finish(self, status: str = "success") -> None:
        self._run = None
