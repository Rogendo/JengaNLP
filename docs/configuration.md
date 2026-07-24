# Configuration Guide

JengaNLP uses YAML configs so training can be reproduced from a file.

A config becomes an `ExperimentConfig` object, then flows into data processing, model creation, training, checkpoints, and inference.

## Minimal Shape

```yaml
project_name: my_project

model:
  base_model: bert-base-multilingual-cased

tokenizer:
  max_length: 128
  padding: max_length
  truncation: true

tasks:
  - name: my_task
    type: single_label_classification
    data_path: data/train.jsonl
    text_column: text
    label_column: label
    heads:
      - name: classifier
        num_labels: 2

training:
  output_dir: ./results/my_project
  learning_rate: 0.00002
  batch_size: 16
  num_epochs: 3
  device: auto
```

## Core Sections

| Section | Consumed By | Purpose |
|---|---|---|
| `project_name` | CLI, logs, integrations | Human-readable run name. |
| `model` | `MultiTaskModel` | Base encoder and model behavior. |
| `tokenizer` | `DataProcessor`, `InferencePipeline` | Token length, padding, truncation. |
| `tasks` | `DataProcessor`, `TaskRegistry`, formatters | Defines task type, data, columns, heads. |
| `training` | `Trainer` | Optimizer, batch size, epochs, output, callbacks. |
| `integrations` | `IntegrationRuntime` | MLflow, TensorBoard, W&B, Label Studio artifact adapters. |

## Task Types

| Type | Label Shape | Use Case |
|---|---|---|
| `single_label_classification` | integer class id | topic, fraud, spam |
| `multi_label_classification` | list/vector of 0/1 labels | multiple flags per text |
| `sentiment` | integer class id | negative/neutral/positive |
| `ner` | entity span list | names, places, organizations |
| `regression` | float | risk, severity, quality score |
| `qa_scoring` | dict of checklist vectors | call-center QA/compliance |

`question_answering` currently maps to `qa_scoring` for backward compatibility. It is not extractive question answering yet.

## Data Formats

Supported text dataset files:

- `.jsonl`
- `.json`
- `.csv`

Single-label example:

```json
{"text":"Share your PIN now","label":1}
```

Multi-label example:

```json
{"text":"Threatening scam message","labels":[1,1,0]}
```

QA scoring example:

```json
{
  "text":"Thank you for calling. How can I help?",
  "labels": {
    "opening": [1],
    "listening": [1, 0, 1],
    "closing": [0]
  }
}
```

## Bounded Regression

Use bounded regression when the target is a constrained score, for example `0.0` to `1.0`.

```yaml
heads:
  - name: risk
    num_labels: 1
    activation: sigmoid
    output_min: 0.0
    output_max: 1.0
```

## Logging

Simple direct logging:

```yaml
training:
  logging:
    service: mlflow
    experiment_name: my_project
    tracking_uri: file:///tmp/jenganlp_mlruns
```

Adapter logging:

```yaml
integrations:
  - type: tensorboard
    enabled: true
    params:
      log_dir: /tmp/jenganlp_tensorboard
```
