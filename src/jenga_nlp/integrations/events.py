"""Activity event schema for JengaNLP."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ActivityEventLevel(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ActivityEventStage(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    INFER = "infer"
    CHECKPOINT = "checkpoint"
    BACKEND = "backend"
    LABELING = "labeling"
    SYSTEM = "system"


class ActivityEventType(StrEnum):
    TRAIN_STARTED = "train_started"
    EPOCH_STARTED = "epoch_started"
    STEP_LOGGED = "step_logged"
    METRICS_LOGGED = "metrics_logged"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    RUN_FAILED = "run_failed"
    RUN_FINISHED = "run_finished"
    INFO = "info"


@dataclass(slots=True)
class ActivityEvent:
    """Structured event emitted by training and integration workflows."""

    stage: ActivityEventStage
    event_type: ActivityEventType
    message: str
    level: ActivityEventLevel = ActivityEventLevel.INFO
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_id: str = field(default_factory=lambda: uuid4().hex)
    epoch: int | None = None
    step: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "stage": self.stage.value,
            "event_type": self.event_type.value,
            "level": self.level.value,
            "message": self.message,
            "epoch": self.epoch,
            "step": self.step,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActivityEvent":
        return cls(
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            run_id=data.get("run_id", uuid4().hex),
            stage=ActivityEventStage(data["stage"]),
            event_type=ActivityEventType(data["event_type"]),
            level=ActivityEventLevel(data.get("level", "info")),
            message=data.get("message", ""),
            epoch=data.get("epoch"),
            step=data.get("step"),
            payload=data.get("payload", {}) or {},
        )

