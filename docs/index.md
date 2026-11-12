---
title: "JengaNLP: Config-Driven Multi-Task NLP Framework for Python"
description: "JengaNLP trains one shared transformer encoder with many task heads - classification, sentiment, NER, regression and call QA scoring - from a single YAML file, then serves it with load-once inference."
---

# JengaNLP: multi-task NLP from one YAML file

JengaNLP trains **one shared transformer encoder with many task heads** (classification, sentiment, NER, regression and checklist-style QA scoring) from a single YAML file. It then serves the model with a **load-once inference pipeline**.

```bash
pip install jenga-nlp
```

!!! note "Status: alpha"
    The core text workflow (config → train → evaluate → predict → `InferencePipeline`) is tested and supported. ASR/Whisper, Weights & Biases, Label Studio, explainability and human-in-the-loop features are experimental.

## What JengaNLP adds on top of Hugging Face

JengaNLP is built on PyTorch and Hugging Face Transformers. For a single classifier, the Hugging Face `Trainer` is a fine choice. JengaNLP is aimed at the cases below:

- **Multi-task models by default.** One encoder, several tasks, several heads per task, per-head loss weights and task sampling strategies, all declared in YAML.
- **Checklist QA scoring.** Score call-centre transcripts against weighted rubric sections as multi-head binary checklists.
- **Production inference included.** Load once, predict single or batch, get structured per-head results with confidences, and flag PII in inputs.
- **One artefact layout.** Training, evaluation, resume, checkpoints and inference all read the same saved config and model directory.

## Start here

| Guide | What you will do |
|---|---|
| [Quickstart](quickstart.md) | Train, evaluate and run a text classifier on CPU in five minutes. |
| [Task types](tasks/index.md) | Classification, sentiment, multi-label, NER, regression, QA scoring and multi-task learning. |
| [Configuration](configuration.md) | Every section of the YAML config and how it maps to the code. |
| [Production inference](inference.md) | Load once in Python, batch prediction and HTTP serving. |
| [User guide](usage-guide.md) | Every workflow in depth, including experimental ones. |

## How it fits together

```text
config.yaml -> train -> evaluate -> predict -> InferencePipeline
```

```text
jenga_nlp/core/config.py        YAML schema and Pydantic config models
jenga_nlp/data/processor.py     CSV/JSON/JSONL loading and task preprocessing
jenga_nlp/tasks/registry.py     Task type -> task class resolution
jenga_nlp/core/model.py         Shared encoder + task heads
jenga_nlp/training/trainer.py   Training loop, metrics, callbacks, checkpoints
jenga_nlp/inference/pipeline.py Load once and predict
jenga_nlp/client/client.py      Thin convenience client
jenga_nlp/cli.py                Command line entrypoint
```

## Roadmap

1. **African-language base models.** Adapt AfriBERTa / AfroXLMR to Swahili, Sheng and Swahili–English code-switched text.
2. **Benchmarks.** Public results on African-language datasets, including multi-task vs single-task baselines.
3. **Templates with pretrained checkpoints.** Mobile-money fraud, call-centre QA and Swahili sentiment, on the Hugging Face Hub.

## Authors

- **Peter** ([@Rogendo](https://github.com/Rogendo)), creator and maintainer
- **Fredrick Kioko** ([@its-kios09](https://github.com/its-kios09)), contributor

Contributions are welcome on [GitHub](https://github.com/Rogendo/JengaNLP).
