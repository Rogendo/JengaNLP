"""Resume helpers for JengaNLP training runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.config import ExperimentConfig


def find_latest_training_state(run_dir: str | Path) -> Path:
    """Find the newest training_state.pt under a run directory."""
    root = Path(run_dir)
    candidates = [p for p in root.rglob("training_state.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No training_state.pt found under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resume_training(
    config: str | Path | ExperimentConfig,
    run_dir: str | Path,
    *,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resume training from the latest saved training state."""
    from .client.client import JengaClient
    from .cli import run_train

    client = JengaClient()
    cfg = client.load_config(config)
    if output_dir:
        cfg.training.output_dir = str(output_dir)
    if device:
        cfg.training.device = device

    state_path = find_latest_training_state(run_dir)
    resume_from = state_path.parent
    return run_train(cfg, output_dir=cfg.training.output_dir, device=device, resume_from=resume_from)
