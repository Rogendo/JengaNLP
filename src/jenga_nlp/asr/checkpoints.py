"""Checkpoint helpers for ASR training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def find_latest_asr_checkpoint(output_dir: str | Path) -> Path | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = []
    for path in root.glob("checkpoint-*"):
        try:
            step = int(path.name.split("-", 1)[1])
        except Exception:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def resolve_asr_checkpoint(output_dir: str | Path, resume_from_checkpoint: str | Path | None) -> str | None:
    if resume_from_checkpoint is None:
        return None
    if str(resume_from_checkpoint) == "latest":
        latest = find_latest_asr_checkpoint(output_dir)
        return str(latest) if latest is not None else None
    return str(resume_from_checkpoint)


def write_checkpoint_metadata(checkpoint_dir: str | Path, metadata: dict[str, Any]) -> Path:
    path = Path(checkpoint_dir) / "checkpoint_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **metadata}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
