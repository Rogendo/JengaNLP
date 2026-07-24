"""Optional Hugging Face Hub export helpers for JengaNLP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _build_readme(repo_id: str, model_dir: Path) -> str:
    return f"""---
language: en
tags:
- jenga-nlp
- text-classification
- named-entity-recognition
- pytorch
- transformers
license: mit
---

# {repo_id}

This model was trained with JengaNLP.

## Artifacts

- Source directory: `{model_dir}`
- Training config: `experiment_config.yaml`

## Usage

```python
from jenga_nlp import JengaClient

client = JengaClient()
prediction = client.predict(
    config="experiment_config.yaml",
    model_dir="{model_dir}",
    text="Your input text here",
)
print(prediction)
```
"""


def push_model_to_hub(
    model_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "Upload JengaNLP model",
) -> str:
    """Push a trained model directory to the Hugging Face Hub.

    The model directory should already contain the trained artifacts.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("huggingface_hub is required for push_to_hub") from exc

    model_path = Path(model_dir)
    readme_path = model_path / "README.md"
    readme_path.write_text(_build_readme(repo_id, model_path))

    effective_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not effective_token:
        raise ValueError("A Hugging Face token is required to push a model to the Hub")

    api = HfApi(token=effective_token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_path),
        commit_message=commit_message,
        token=effective_token,
    )
    return f"https://huggingface.co/{repo_id}"
