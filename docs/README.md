# JengaNLP Framework Documentation

This folder documents the `jenga_nlp` Python framework from the perspective of a user who installed or cloned the package.

Start here if you want to train, evaluate, monitor, export, or run inference with JengaNLP models.

## Guides

| Guide | Purpose |
|---|---|
| [quickstart.md](quickstart.md) | Smallest local train -> evaluate -> predict workflow. |
| [configuration.md](configuration.md) | Explains the YAML config and how it maps to core modules. |
| [inference.md](inference.md) | Explains CLI inference, Python inference, batch inference, and production loading. |

## Package Boundary

JengaNLP is the new package namespace:

```python
from jenga_nlp import JengaClient
from jenga_nlp.inference.pipeline import InferencePipeline
```

Do not use old `jenga_ai` imports for the new framework docs.

The old JengaAI platform/product story has been moved to:

```text
docs/jenga-ai-platform-story.md
```

## Core Runtime Map

```text
jenga_nlp/core/config.py       YAML schema and Pydantic config models
jenga_nlp/data/processor.py    CSV/JSON/JSONL loading and task preprocessing
jenga_nlp/tasks/registry.py    Task type -> task class resolution
jenga_nlp/core/model.py        Shared encoder + task heads
jenga_nlp/training/trainer.py  Training loop, metrics, callbacks, checkpoints
jenga_nlp/inference/pipeline.py Load once and predict
jenga_nlp/client/client.py     Thin convenience client
jenga_nlp/cli.py               Command line entrypoint
```

## Current Stable Path

The most stable path today is text NLP:

```text
config.yaml -> train -> evaluate -> predict -> InferencePipeline
```

ASR/Whisper is present but should be treated as experimental until real audio/GPU validation is complete.
