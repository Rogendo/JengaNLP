"""Label Studio adapter for JengaNLP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseAdapter
from ..events import ActivityEvent


class LabelStudioAdapter(BaseAdapter):
    def __init__(self, export_dir: str | Path) -> None:
        super().__init__(name="labelstudio")
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def is_available(self) -> bool:
        return True

    def start_run(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> Any:
        self._run = {"run_name": run_name, "config": config, "tags": tags or {}}
        return self._run

    def handle_event(self, event: ActivityEvent) -> None:
        if event.event_type.value == "metrics_logged":
            return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_params(self, params: dict[str, Any]) -> None:
        return None

    def log_artifact(self, path: str | Path, name: str | None = None) -> None:
        destination = self.export_dir / f"{name or Path(path).stem}.json"
        destination.write_text(json.dumps({"artifact_path": str(path), "name": name}, ensure_ascii=False, indent=2))

    def finish(self, status: str = "success") -> None:
        super().finish(status=status)
