"""Transformers callbacks for ASR activity events and checkpoint metadata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checkpoints import write_checkpoint_metadata
from ..integrations.events import ActivityEvent, ActivityEventLevel, ActivityEventStage, ActivityEventType
from ..integrations.runtime import IntegrationRuntime

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - optional dependency path
    class TrainerCallback:  # type: ignore[no-redef]
        pass


class ASRActivityCallback(TrainerCallback):
    """Bridge Hugging Face Seq2SeqTrainer events into JengaNLP activity logs."""

    def __init__(self, runtime: IntegrationRuntime, max_nan_count: int = 3) -> None:
        self.runtime = runtime
        self.max_nan_count = max_nan_count
        self.nan_count = 0

    def on_train_begin(self, args, state, control, **kwargs: Any):
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.TRAIN,
                event_type=ActivityEventType.TRAIN_STARTED,
                message="asr training started",
                payload={"max_steps": getattr(state, "max_steps", None)},
            )
        )

    def on_log(self, args, state, control, logs=None, **kwargs: Any):
        logs = logs or {}
        loss = logs.get("loss")
        if isinstance(loss, (int, float)) and (math.isnan(loss) or math.isinf(loss)):
            self.nan_count += 1
            if self.nan_count >= self.max_nan_count:
                control.should_training_stop = True
                self.runtime.emit(
                    ActivityEvent(
                        stage=ActivityEventStage.TRAIN,
                        event_type=ActivityEventType.RUN_FAILED,
                        level=ActivityEventLevel.ERROR,
                        message="asr training stopped due to repeated NaN/Inf loss",
                        step=getattr(state, "global_step", None),
                        payload={"loss": loss, "nan_count": self.nan_count},
                    )
                )
        elif isinstance(loss, (int, float)):
            self.nan_count = 0
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.TRAIN,
                event_type=ActivityEventType.STEP_LOGGED,
                message="asr training log",
                step=getattr(state, "global_step", None),
                payload={"metrics": {k: v for k, v in logs.items() if isinstance(v, (int, float))}},
            )
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs: Any):
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.EVAL,
                event_type=ActivityEventType.METRICS_LOGGED,
                message="asr evaluation metrics",
                step=getattr(state, "global_step", None),
                payload={"metrics": metrics or {}},
            )
        )

    def on_save(self, args, state, control, **kwargs: Any):
        checkpoint_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        metadata_path = write_checkpoint_metadata(
            checkpoint_path,
            {"step": state.global_step, "best_metric": getattr(state, "best_metric", None)},
        )
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.CHECKPOINT,
                event_type=ActivityEventType.CHECKPOINT_SAVED,
                message=f"asr checkpoint saved: {checkpoint_path}",
                step=state.global_step,
                payload={"checkpoint_path": str(checkpoint_path), "metadata_path": str(metadata_path)},
            )
        )

    def on_train_end(self, args, state, control, **kwargs: Any):
        self.runtime.emit(
            ActivityEvent(
                stage=ActivityEventStage.SYSTEM,
                event_type=ActivityEventType.RUN_FINISHED,
                message="asr training finished",
                step=getattr(state, "global_step", None),
                payload={"best_metric": getattr(state, "best_metric", None)},
            )
        )
