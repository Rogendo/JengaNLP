"""Sinks and fan-out helpers for activity events."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .events import ActivityEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class ActivityEventSink(Protocol):
    """Consumes activity events."""

    def handle(self, event: ActivityEvent) -> None: ...

    def close(self) -> None: ...


@dataclass
class ConsoleEventSink:
    """Human-readable console sink."""

    logger_name: str = "jenga_nlp.activity"
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = logging.getLogger(self.logger_name)

    def handle(self, event: ActivityEvent) -> None:
        parts = [f"[{event.stage.value}]", event.event_type.value, event.message]
        if event.epoch is not None:
            parts.append(f"epoch={event.epoch}")
        if event.step is not None:
            parts.append(f"step={event.step}")
        self._logger.info(" ".join(parts))

    def close(self) -> None:
        return None


@dataclass
class JSONLEventSink:
    """Writes events to a JSONL file."""

    path: str | Path
    _fh: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def handle(self, event: ActivityEvent) -> None:
        self._fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


@dataclass
class MultiEventSink:
    """Fan-out sink for multiple event sinks."""

    sinks: list[ActivityEventSink]

    def handle(self, event: ActivityEvent) -> None:
        for sink in self.sinks:
            sink.handle(event)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


@dataclass
class ActivityLogger:
    """Routes activity events to one or more sinks."""

    sink: ActivityEventSink | None = None
    sinks: list[ActivityEventSink] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sink is not None:
            self.sinks.insert(0, self.sink)
        self._sink = MultiEventSink(self.sinks)

    def emit(self, event: ActivityEvent) -> None:
        self._sink.handle(event)

    def close(self) -> None:
        self._sink.close()
