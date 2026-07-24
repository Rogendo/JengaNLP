"""Local-first convenience client for JengaNLP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import ExperimentConfig
from ..utils.logging import setup_logging

logger = setup_logging()


class JengaClient:
    """Thin convenience wrapper around the JengaNLP core."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        self.base_path = Path(base_path) if base_path else None

    def load_config(self, config: str | Path | ExperimentConfig) -> ExperimentConfig:
        if isinstance(config, ExperimentConfig):
            return config
        return ExperimentConfig.from_yaml(config)

    def train(self, config: str | Path | ExperimentConfig, **kwargs: Any) -> dict[str, Any]:
        from ..cli import run_train

        return run_train(self.load_config(config), **kwargs)

    def evaluate(self, config: str | Path | ExperimentConfig, model_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
        from ..cli import run_evaluate

        return run_evaluate(self.load_config(config), model_dir=model_dir, **kwargs)

    def resume(
        self,
        config: str | Path | ExperimentConfig,
        run_dir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        from ..resume import resume_training

        return resume_training(self.load_config(config), run_dir=run_dir, **kwargs)

    def push_to_hub(
        self,
        model_dir: str | Path,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool = False,
        commit_message: str = "Upload JengaNLP model",
    ) -> str:
        from ..export.huggingface import push_model_to_hub

        return push_model_to_hub(
            model_dir=model_dir,
            repo_id=repo_id,
            token=token,
            private=private,
            commit_message=commit_message,
        )

    def predict(
        self,
        config: str | Path | ExperimentConfig,
        model_dir: str | Path,
        text: str,
        **kwargs: Any,
    ) -> Any:
        from ..cli import run_predict

        return run_predict(config=self.load_config(config), model_dir=model_dir, text=text, **kwargs)
