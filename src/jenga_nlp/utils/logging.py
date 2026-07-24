"""Logging helpers for JengaNLP."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, name: str = "jenga_nlp") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger

