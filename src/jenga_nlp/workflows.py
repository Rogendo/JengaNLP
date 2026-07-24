"""High-level workflow helpers for JengaNLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .client.client import JengaClient
from .core.config import ExperimentConfig


def ship_model(
    config: str | Path | ExperimentConfig,
    text: str,
    *,
    output_dir: str | Path | None = None,
    device: str | None = None,
    task_name: str | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Train a model and immediately run inference from the saved checkpoint.

    This is the narrow Phase 1 from-data-to-inference workflow.
    """
    client = JengaClient()
    cfg = client.load_config(config)
    train_result = client.train(cfg, output_dir=output_dir, device=device)

    model_dir = Path(cfg.training.output_dir)
    prediction = client.predict(
        cfg,
        model_dir=model_dir,
        text=text,
        task_name=task_name,
        threshold=threshold,
    )

    return {
        "train_result": train_result,
        "prediction": prediction,
        "model_dir": str(model_dir),
    }
