"""Task utilities for JengaNLP."""

from .base import BaseTask, TaskOutput
from .classification import SingleLabelClassificationTask, MultiLabelClassificationTask
from .ner import NERTask
from .qa import QAScoringTask
from .sentiment import SentimentTask
from .regression import RegressionTask
from .registry import TaskRegistry

__all__ = [
    "BaseTask",
    "TaskOutput",
    "SingleLabelClassificationTask",
    "MultiLabelClassificationTask",
    "NERTask",
    "QAScoringTask",
    "SentimentTask",
    "RegressionTask",
    "TaskRegistry",
]
