"""Runtime helpers for loading adapters and emitting events."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .events import ActivityEvent, ActivityEventLevel, ActivityEventStage, ActivityEventType
from .registry import IntegrationConfig, build_adapter
from .sinks import ActivityLogger, ConsoleEventSink, JSONLEventSink


@dataclass
class IntegrationRuntime:
    """Owns activity logging and optional backend adapters for a run."""

    run_dir: str | Path
    integrations: list[IntegrationConfig] = field(default_factory=list)
    logger: ActivityLogger = field(init=False)
    adapters: list[Any] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        run_dir = Path(self.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        event_log = JSONLEventSink(run_dir / "activity.jsonl")
        console = ConsoleEventSink()
        self.logger = ActivityLogger(sinks=[console, event_log])
        self.adapters = [adapter for cfg in self.integrations if (adapter := build_adapter(cfg, base_dir=run_dir)) is not None]

    def start(self, run_name: str, config: dict[str, Any], tags: dict[str, str] | None = None) -> None:
        for adapter in self.adapters:
            if adapter.is_available():
                adapter.start_run(run_name=run_name, config=config, tags=tags)
        self.emit(
            ActivityEvent(
                stage=ActivityEventStage.SYSTEM,
                event_type=ActivityEventType.INFO,
                message=f"run started: {run_name}",
                payload={"run_name": run_name},
            )
        )

    def emit(self, event: ActivityEvent) -> None:
        self.logger.emit(event)
        for adapter in self.adapters:
            if adapter.is_available():
                adapter.handle_event(event)

    def emit_info(self, message: str, *, stage: ActivityEventStage = ActivityEventStage.SYSTEM, payload: dict[str, Any] | None = None) -> None:
        self.emit(
            ActivityEvent(
                stage=stage,
                event_type=ActivityEventType.INFO,
                level=ActivityEventLevel.INFO,
                message=message,
                payload=payload or {},
            )
        )

    def finish(self, status: str = "success") -> None:
        self.emit_info(f"run finished: {status}", stage=ActivityEventStage.SYSTEM, payload={"status": status})
        for adapter in self.adapters:
            if adapter.is_available():
                adapter.finish(status=status)
        self.logger.close()
