"""Training callbacks that emit activity events for JengaNLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .events import ActivityEvent, ActivityEventLevel, ActivityEventStage, ActivityEventType
from .runtime import IntegrationRuntime


@dataclass
class ActivityEventCallback:
    """Bridge the trainer callback system to the activity-event stream."""

    runtime: IntegrationRuntime

    def on_train_begin(self, **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.TRAIN,
                event_type=ActivityEventType.TRAIN_STARTED,
                level=ActivityEventLevel.INFO,
                message="training started",
                payload={"keys": list(kwargs.keys())},
            )
        )

    def on_epoch_begin(self, epoch: int, **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.TRAIN,
                event_type=ActivityEventType.EPOCH_STARTED,
                level=ActivityEventLevel.INFO,
                message=f"epoch {epoch + 1} started",
                epoch=epoch,
                payload={"keys": list(kwargs.keys())},
            )
        )

    def on_step(self, step: int, loss: float, **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.TRAIN,
                event_type=ActivityEventType.STEP_LOGGED,
                level=ActivityEventLevel.INFO,
                message=f"step {step} loss {loss:.4f}",
                step=step,
                payload={"loss": loss, "keys": list(kwargs.keys())},
            )
        )

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.EVAL,
                event_type=ActivityEventType.METRICS_LOGGED,
                level=ActivityEventLevel.INFO,
                message=f"epoch {epoch + 1} metrics logged",
                epoch=epoch,
                payload={"metrics": metrics, "keys": list(kwargs.keys())},
            )
        )


    def on_checkpoint_saved(self, checkpoint_path: str, state_path: str | None = None, **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.CHECKPOINT,
                event_type=ActivityEventType.CHECKPOINT_SAVED,
                level=ActivityEventLevel.INFO,
                message=f"checkpoint saved: {checkpoint_path}",
                epoch=kwargs.get("epoch"),
                step=kwargs.get("global_step"),
                payload={
                    "checkpoint_path": str(checkpoint_path),
                    "state_path": str(state_path) if state_path else None,
                    "metrics": kwargs.get("metrics", {}),
                },
            )
        )

    def on_train_end(self, **kwargs: Any) -> None:
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.SYSTEM,
                event_type=ActivityEventType.RUN_FINISHED,
                level=ActivityEventLevel.INFO,
                message="training finished",
                payload={"keys": list(kwargs.keys())},
            )
        )

    def should_stop(self) -> bool:
        return False
