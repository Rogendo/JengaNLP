"""Task registry for automatic task discovery and instantiation.

Maps TaskType enum values to their corresponding task classes.
"""

from __future__ import annotations

import logging
from typing import Type

from ..core.config import TaskConfig, TaskType
from .base import BaseTask
from .classification import MultiLabelClassificationTask, SingleLabelClassificationTask
from .ner import NERTask
from .qa import QAScoringTask
from .regression import RegressionTask
from .sentiment import SentimentTask

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registry that maps task types to their implementation classes.

    Supports registration of custom task types for extensibility.
    """

    _registry: dict[str, Type[BaseTask]] = {
        TaskType.SINGLE_LABEL_CLASSIFICATION.value: SingleLabelClassificationTask,
        TaskType.MULTI_LABEL_CLASSIFICATION.value: MultiLabelClassificationTask,
        TaskType.NER.value: NERTask,
        TaskType.SENTIMENT.value: SentimentTask,
        TaskType.REGRESSION.value: RegressionTask,
        TaskType.QA_SCORING.value: QAScoringTask,
        TaskType.QUESTION_ANSWERING.value: QAScoringTask,
    }

    @classmethod
    def register(cls, task_type: str, task_class: Type[BaseTask]) -> None:
        """Register a custom task type.

        Args:
            task_type: String identifier for the task type.
            task_class: The task class to register.
        """
        cls._registry[task_type] = task_class
        logger.info(f"Registered task type: {task_type} -> {task_class.__name__}")

    @classmethod
    def create_task(cls, config: TaskConfig, hidden_size: int) -> BaseTask:
        """Create a task instance from its config.

        Args:
            config: Task configuration.
            hidden_size: Hidden size from the shared encoder.

        Returns:
            Instantiated task.

        Raises:
            ValueError: If the task type is not registered.
        """
        task_type = config.type.value if hasattr(config.type, "value") else config.type

        if task_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown task type: '{task_type}'. Available: {available}")

        task_class = cls._registry[task_type]
        logger.info(f"Creating task '{config.name}' of type '{task_type}' with hidden_size={hidden_size}")
        return task_class(config=config, hidden_size=hidden_size)

    @classmethod
    def available_types(cls) -> list[str]:
        """Return list of available task types."""
        return list(cls._registry.keys())
