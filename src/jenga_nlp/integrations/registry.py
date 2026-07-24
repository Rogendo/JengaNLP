"""Adapter registry for JengaNLP integrations."""

from __future__ import annotations

from pathlib import Path

from ..core.config import IntegrationConfig
from .adapters import LabelStudioAdapter, MLflowAdapter, TensorBoardAdapter, WandBAdapter
from .base import BackendAdapter


ADAPTERS: dict[str, type[BackendAdapter]] = {
    "mlflow": MLflowAdapter,
    "tensorboard": TensorBoardAdapter,
    "wandb": WandBAdapter,
    "labelstudio": LabelStudioAdapter,
}


def build_adapter(config: IntegrationConfig, *, base_dir: str | Path | None = None) -> BackendAdapter | None:
    if not config.enabled:
        return None
    adapter_cls = ADAPTERS.get(config.type.lower())
    if adapter_cls is None:
        raise ValueError(f"Unsupported integration type: {config.type}")

    params = dict(config.params)
    if base_dir is not None:
        params.setdefault("base_dir", str(base_dir))
        params.setdefault("log_dir", str(Path(base_dir) / "tensorboard"))
        params.setdefault("export_dir", str(Path(base_dir) / "labelstudio"))
    if adapter_cls is MLflowAdapter:
        return adapter_cls(
            tracking_uri=params.get("tracking_uri"),
            experiment_name=params.get("experiment_name"),
        )
    if adapter_cls is TensorBoardAdapter:
        return adapter_cls(log_dir=params.get("log_dir", params.get("path", "./runs/tensorboard")))
    if adapter_cls is WandBAdapter:
        return adapter_cls(project=params.get("project", "jenga_nlp"), entity=params.get("entity"))
    if adapter_cls is LabelStudioAdapter:
        return adapter_cls(export_dir=params.get("export_dir", params.get("path", "./runs/labelstudio")))
    return adapter_cls()  # pragma: no cover
