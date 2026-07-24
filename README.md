# JengaNLP Usage README

JengaNLP is a text-first NLP framework for local training, multi-task models, multi-head classification, NER, regression scoring, QA/compliance scoring, Whisper ASR workflows, experiment tracking, and production inference.

This guide explains how to start logging services, train models, evaluate, predict, resume runs, understand the YAML config, and decide when production code must import `jenga_nlp` versus when a standard Hugging Face model can be used directly.


## Deep-Dive Developer Guides

For deeper onboarding and implementation understanding:

- [Config To Core Deep Dive](docs/jenganlp-config-to-core-deep-dive.md): explains how YAML maps into `ExperimentConfig`, data processors, task registry, task heads, `MultiTaskModel`, trainer, callbacks, integrations, inference, ASR modules, and more than 30 package files.
- [Production Inference And Standard Model Strategy](docs/jenganlp-production-inference-and-standard-models.md): explains serving patterns, scalability, batching, CPU/GPU workers, model versioning, monitoring, on-prem deployment, and when to import JengaNLP versus standard Hugging Face packages.


## Current Scope

JengaNLP currently has two broad workflow families:

| Workflow family | Use JengaNLP for training? | Need JengaNLP for inference? | Why |
|---|---:|---:|---|
| Custom multi-task text model | Yes | Yes | The model is `MultiTaskModel`: shared encoder plus JengaNLP task heads and saved task metadata. |
| Single-label, multi-label, sentiment, NER, regression, QA scoring heads | Yes | Yes | These use JengaNLP task classes, label maps, formatters, and checkpoint layout. |
| Whisper transcription serving | Optional | No, if exported as standard Whisper | Whisper is already a standard Hugging Face architecture. |
| Whisper fine-tuning | Optional orchestration layer | No, if saved/pushed as standard Whisper | JengaNLP can manage configs, preprocessing, metrics, and tracking, but the model itself remains standard. |
| Standard BERT/DistilBERT/MLM fine-tuning | Optional/future orchestration | No, if saved as standard Transformers artifacts | Standard models can be loaded with `transformers` without JengaNLP. |
| Fine-tuned base model reused for downstream custom tasks | Yes for downstream stage | Yes for downstream custom model | A standard fine-tuned encoder can become `model.base_model` for JengaNLP custom heads. |

The practical rule is:

- If you trained a **custom JengaNLP multi-task/multi-head model**, production inference imports `jenga_nlp`.
- If you trained a **standard Hugging Face model** such as Whisper or `AutoModelForSequenceClassification`, production inference can use `transformers` directly.
- If you use a standard fine-tuned encoder as a base for a JengaNLP multi-task classifier, the final downstream model again needs `jenga_nlp`.

## Installation

### From PyPI (recommended)

```bash
pip install jenga-nlp
```

### From source (development)

```bash
git clone https://github.com/Rogendo/JengaNLP.git
cd JengaNLP
pip install -e ".[dev]"
```

### Verify installation

```bash
python -m jenga_nlp          # prints CLI quickstart
jenga-nlp --help             # also works after install
```

The project pins `setuptools<81` because TensorBoard imports `pkg_resources` in some environments:

```bash
pip install "setuptools>=68,<81"
```

## CLI Quick Start

Show help:

```bash
python -m jenga_nlp
python -m jenga_nlp -h
python -m jenga_nlp train -h
python -m jenga_nlp predict -h
python -m jenga_nlp asr train -h
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --device cpu
```

Evaluate:

```bash
python -m jenga_nlp evaluate \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu \
  --task-name mpesa_fraud_detection \
  --text "Urgent M-PESA reversal request asking the customer to share their PIN"
```

Resume:

```bash
python -m jenga_nlp resume \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --run-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu
```

Backend status:

```bash
python -m jenga_nlp backends status \
  --config configs/fixture_logging_mlflow_cpu.yaml
```

Initialize configured backends:

```bash
python -m jenga_nlp backends start \
  --config configs/fixture_logging_mlflow_cpu.yaml
```

Important: `backends start` initializes/checks adapters. It does not daemonize MLflow, TensorBoard, W&B, or Label Studio services. Start those services with their own commands as shown below.


## Notebook-Style Walkthrough Script

For beginner-ready notebook examples, start with:

```bash
ls notebooks/
python -u notebooks/01_local_train_predict_beginner.py
```

The `notebooks/` folder contains documented examples for local training, cloud/Colab training with MLflow/TensorBoard/ngrok, load-once production-style inference, and standard Hugging Face usage for Whisper/MLM models.

For local testing, teaching, and later upload to Colab/Kaggle/online GPU machines, use:

```bash
python -u examples/jenganlp_notebook_walkthrough.py \
  --config configs/fixture_multitask_cpu.yaml \
  --device cpu
```

The file is a normal Python script, but it is written with `# %%` cells so notebook-aware editors can run it section by section. Use `python -u` when you want live, unbuffered explanations in terminals or remote GPU sessions.

What it demonstrates:

- loads and explains the YAML config
- checks configured logging/integration backends
- prints MLflow/TensorBoard service commands when relevant
- trains through `JengaClient`
- evaluates the saved model
- predicts through the simple client API
- reloads once with `InferencePipeline` for production-style inference
- optionally pushes to Hugging Face Hub with `--push-to-hub --repo-id your-org/model`

Example for an online GPU:

```bash
python -u examples/jenganlp_notebook_walkthrough.py \
  --config configs/your_real_gpu_config.yaml \
  --device cuda \
  --output-dir results/gpu_run
```

## Colab, Notebook, Pipeline, And HTTP Serving Are Different Workflows

A Colab or Jupyter notebook can run JengaNLP in two normal ways:

1. **Notebook shell commands** using `!python -m jenga_nlp ...`.
2. **Notebook Python API calls** using `JengaClient` or `InferencePipeline`.

Those are not the same thing as `curl`. `curl` only makes sense after you have started an HTTP server, for example a FastAPI app or a future `jenga serve` endpoint. Most notebook users should not use curl for inference. They should train, save the model, then load the saved model directory or Hugging Face repo directly in Python.

### Pattern A: Shell-command notebook cells

This is the closest equivalent to terminal usage, but inside Colab or Jupyter:

```python
# Colab / notebook cell
!pip install -e .
!python -m jenga_nlp train --config configs/fixture_multitask_cpu.yaml --device cpu
!python -m jenga_nlp evaluate --config configs/fixture_multitask_cpu.yaml --model-dir /tmp/jenganlp_fixture_multitask_cpu --device cpu
!python -m jenga_nlp predict \
  --config configs/fixture_multitask_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multitask_cpu \
  --task-name mpesa_fraud_detection \
  --text "Urgent M-PESA reversal request asking for PIN"
```

Use this when:

- you are teaching or demoing the framework;
- you want the notebook to look like a reproducible terminal recipe;
- you are uploading the same commands later to Colab, Kaggle, RunPod, or a cloud GPU VM;
- you want simple train/evaluate/predict cells without writing much Python glue.

Tradeoff: each CLI call starts a new Python process. For inference, the model is loaded again for each command. That is acceptable for smoke tests, but not ideal for repeated prediction.

### Pattern B: Python API notebook cells

This is better when the notebook is a real pipeline and you want to keep objects in memory:

```python
from jenga_nlp.client import JengaClient

client = JengaClient()

train_result = client.train(
    "configs/fixture_multitask_cpu.yaml",
    device="cpu",
    output_dir="/tmp/jenganlp_fixture_multitask_cpu",
)

metrics = client.evaluate(
    "configs/fixture_multitask_cpu.yaml",
    model_dir="/tmp/jenganlp_fixture_multitask_cpu",
    device="cpu",
)

prediction = client.predict(
    "configs/fixture_multitask_cpu.yaml",
    model_dir="/tmp/jenganlp_fixture_multitask_cpu",
    task_name="mpesa_fraud_detection",
    text="Urgent M-PESA reversal request asking for PIN",
)

print(train_result)
print(metrics)
print(prediction)
```

Use this when:

- you want one notebook to behave like an end-to-end ML pipeline;
- you want Python variables for metrics, predictions, model paths, and Hub repo IDs;
- you want to branch based on metrics, for example push only if `f1` passes a threshold;
- you want cleaner integration with pandas, MLflow, W&B, TensorBoard, or Hugging Face Hub.

### Pattern C: Load once for repeated notebook inference

If you will run many predictions in a notebook, load the model once using `InferencePipeline`:

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="/tmp/jenganlp_fixture_multitask_cpu",
    config_path="configs/fixture_multitask_cpu.yaml",
    device="cpu",
)

texts = [
    "Urgent M-PESA reversal request asking for PIN",
    "Your payment has been received successfully",
]

results = pipeline.predict_batch(
    texts=texts,
    task_name="mpesa_fraud_detection",
    batch_size=16,
)

for item in results:
    print(item.to_dict())
```

This is the notebook equivalent of production inference: load once, predict many times.

### Pattern D: Direct Hugging Face loading after Hub push

For JengaNLP-native multi-task models, the production or notebook code still needs `jenga_nlp` because the task heads and output formatting are custom:

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="your-org/jenganlp-multitask-model",
    config_path="configs/fixture_multitask_cpu.yaml",
    device="cpu",
)

result = pipeline.predict(
    text="Urgent M-PESA reversal request asking for PIN",
    task_name="mpesa_fraud_detection",
)
```

For standard exported models such as Whisper or MLM BERT, users can load the model with standard Hugging Face packages instead:

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="your-org/somali-whisper-small-v1",
)

print(asr("sample.wav"))
```

That distinction is important: JengaNLP-native models require the JengaNLP inference layer; standard Hugging Face models can be used without importing JengaNLP after training and export.

### Pattern E: HTTP serving and curl

Use curl only when there is a running server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then another terminal, notebook cell, or service can call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Urgent M-PESA reversal request asking for PIN","task_name":"mpesa_fraud_detection"}'
```

This is a deployment or integration test workflow, not the default notebook inference workflow.

### MLflow and ngrok in Colab

The older Colab-style script at `/home/naynek/Downloads/bert_jengaai_mlflow_ngrok.py` follows this pattern:

1. clone or install the framework;
2. install `mlflow` and `pyngrok`;
3. start MLflow locally inside the notebook runtime;
4. expose MLflow UI through ngrok for browser access;
5. train using shell commands such as `!python -m jenga_nlp train ...`;
6. zip or copy training artifacts to Google Drive;
7. optionally push the saved model directory to Hugging Face Hub;
8. load the saved local path or Hub repo in Python for inference.

The important point is that ngrok is only for viewing the MLflow UI from outside Colab. It does not replace model loading, and it does not mean prediction must use curl.


## Starting Logging And Labeling Services

JengaNLP has two logging paths:

| Path | Config block | Best for | Backends |
|---|---|---|---|
| Built-in training callback | `training.logging` | Simple direct MLflow/TensorBoard logging | MLflow, TensorBoard, none |
| Pluggable activity adapters | `integrations` | Extensible event-driven logging and artifacts | MLflow, TensorBoard, W&B, Label Studio |

Prefer `integrations` for new workflows because it is the extensible layer. Keep `training.logging` for simple smoke tests and backward compatibility.

### MLflow

File-based MLflow needs no server during training. Example config:

```yaml
training:
  logging:
    service: mlflow
    experiment_name: fixture_logging_mlflow_cpu
    tracking_uri: file:///tmp/jenganlp_mlruns
    log_every_n_steps: 1
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_logging_mlflow_cpu.yaml \
  --device cpu
```

Start the MLflow UI:

```bash
mlflow ui \
  --backend-store-uri /tmp/jenganlp_mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

For a shared team server:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

Then point configs to:

```yaml
integrations:
  - type: mlflow
    enabled: true
    params:
      tracking_uri: http://localhost:5000
      experiment_name: my_project
```

### TensorBoard

Example config:

```yaml
training:
  logging:
    service: tensorboard
    experiment_name: fixture_logging_tensorboard_cpu
    log_every_n_steps: 1
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_logging_tensorboard_cpu.yaml \
  --device cpu
```

Start TensorBoard:

```bash
tensorboard \
  --logdir /tmp/jenganlp_fixture_logging_tensorboard_cpu/logs \
  --host 127.0.0.1 \
  --port 6006
```

Open:

```text
http://127.0.0.1:6006
```

If TensorBoard fails with `ModuleNotFoundError: No module named 'pkg_resources'`, install compatible setuptools:

```bash
pip install "setuptools>=68,<81"
```

### Weights & Biases

W&B is configured through the pluggable adapter layer:

```yaml
integrations:
  - type: wandb
    enabled: true
    params:
      project: jenganlp-demo
      entity: null
```

Login first:

```bash
wandb login
```

Then train normally:

```bash
python -m jenga_nlp train \
  --config configs/my_wandb_config.yaml \
  --device cpu
```

Use W&B when you want hosted experiment dashboards, reports, and team comparison views.

### Label Studio

The current Label Studio adapter exports artifacts/metadata into a directory. It does not yet run a full Label Studio ML backend server.

Config:

```yaml
integrations:
  - type: labelstudio
    enabled: true
    params:
      export_dir: /tmp/jenganlp_labelstudio
```

Check adapter availability:

```bash
python -m jenga_nlp backends status \
  --config configs/my_labelstudio_config.yaml
```

If you run Label Studio separately:

```bash
pip install label-studio
label-studio start --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

Use Label Studio for annotation, human review, and dataset improvement loops. The next production step is a true Label Studio ML backend that calls `InferencePipeline` for pre-annotations.

## The YAML Config: What Every Section Means

A JengaNLP text config is parsed into `ExperimentConfig` from `jenga_nlp/core/config.py`.

Minimal shape:

```yaml
project_name: mpesa_fraud_demo

model:
  base_model: distilbert-base-uncased

tokenizer:
  max_length: 128
  padding: max_length
  truncation: true

tasks:
  - name: mpesa_fraud_detection
    type: single_label_classification
    data_path: data/messages.jsonl
    text_column: text
    label_column: label
    heads:
      - name: fraud_head
        num_labels: 2

training:
  output_dir: results/mpesa_fraud_demo
  learning_rate: 0.00002
  batch_size: 16
  num_epochs: 3
  device: auto
  logging:
    service: tensorboard
    experiment_name: mpesa_fraud_demo
    log_every_n_steps: 10
  checkpoint:
    save_every_n_epochs: 1
    save_best: true
    max_checkpoints: 3
  data:
    test_size: 0.2
    seed: 42
```

### `project_name`

This names the run. It is used in logs, activity events, backend runs, and generated output metadata.

### `model`

```yaml
model:
  base_model: distilbert-base-uncased
  dropout: 0.1
  freeze_encoder_layers: 0
  gradient_checkpointing: false
  fusion:
    type: none
```

`base_model` is the shared Hugging Face encoder. For custom JengaNLP text models, all tasks share this encoder and attach task-specific heads.

Examples:

```yaml
base_model: distilbert-base-uncased
base_model: bert-base-multilingual-cased
base_model: xlm-roberta-base
base_model: Rogendo/my-finetuned-language-encoder
```

Use a small model for CPU smoke tests:

```yaml
base_model: hf-internal-testing/tiny-random-bert
```

Use a real model for actual quality:

```yaml
base_model: distilbert-base-uncased
```

### `tokenizer`

```yaml
tokenizer:
  max_length: 256
  padding: max_length
  truncation: true
```

`max_length` controls the maximum token count per input. Longer values preserve more text but cost more memory and latency.

For production:

- Use the shortest `max_length` that preserves task quality.
- For call transcripts, use larger values or chunking.
- For SMS fraud and short messages, `64` or `128` is often enough.

### `tasks`

Each task defines one supervised output contract. Multi-task training is just a list of tasks sharing one encoder.

Task fields:

| Field | Meaning |
|---|---|
| `name` | Stable task identifier used in training, inference, and metrics. |
| `type` | Task implementation to use. |
| `data_path` | CSV, JSON, or JSONL training file. |
| `text_column` | Column containing raw text. |
| `label_column` | Column containing labels. |
| `heads` | One or more prediction heads for the task. |
| `label_maps` | Optional output label names. NER may populate this during preprocessing. |

Supported text task types:

| Type | Use when | Label shape |
|---|---|---|
| `single_label_classification` | Exactly one class is correct | integer class id |
| `sentiment` | One sentiment/rating class is correct | integer class id |
| `multi_label_classification` | Many independent labels can be true | binary vector or dict of vectors |
| `ner` | Extract spans/entities from text | entity span list |
| `regression` | Predict a continuous score | float |
| `qa_scoring` | Multi-head compliance/checklist scoring | dict of binary vectors |

`question_answering` remains as a backward-compatible alias for `qa_scoring`, but new configs should use `qa_scoring`.

### `heads`

Heads are the output layers for a task.

Single-label example:

```yaml
heads:
  - name: fraud_head
    num_labels: 2
```

Multi-label example:

```yaml
heads:
  - name: flags
    num_labels: 4
```

QA scoring example:

```yaml
heads:
  - name: opening
    num_labels: 1
    weight: 1.0
  - name: listening
    num_labels: 5
    weight: 1.5
  - name: resolution
    num_labels: 5
    weight: 2.0
```

Regression with bounded output:

```yaml
heads:
  - name: risk_score
    num_labels: 1
    activation: sigmoid
    output_min: 0.0
    output_max: 1.0
```

Use bounded regression when the output must be constrained, such as a risk score from `0.0` to `1.0`. Without `activation: sigmoid`, regression is unbounded and may produce values outside the label range.

### `training`

```yaml
training:
  output_dir: results/my_run
  learning_rate: 0.00002
  batch_size: 16
  eval_batch_size: 16
  num_epochs: 3
  weight_decay: 0.01
  warmup_steps: 100
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1
  use_amp: false
  device: auto
  task_sampling: round_robin
  early_stopping_patience: null
  metric_for_best_model: eval_loss
  greater_is_better: false
```

Important fields:

| Field | Why it matters |
|---|---|
| `output_dir` | Where model, checkpoints, logs, saved config, and activity JSONL are written. |
| `num_epochs` | More epochs can improve quality but increase overfitting risk. |
| `batch_size` | Higher is faster but needs more memory. |
| `gradient_accumulation_steps` | Simulates larger batches on small GPUs/CPU. |
| `device` | `auto`, `cpu`, `cuda`, or `mps`. |
| `task_sampling` | How multi-task batches are scheduled. |
| `metric_for_best_model` | Which metric checkpointing and early stopping watch. |

Task sampling:

| Strategy | Meaning | Use when |
|---|---|---|
| `round_robin` | One batch per task in order | Balanced multi-task training |
| `proportional` | Larger tasks appear more often | Dataset sizes reflect importance |
| `temperature` | Smoothed proportional sampling | You want large tasks to dominate less |

### `training.logging`

This is the direct legacy logging callback:

```yaml
training:
  logging:
    service: mlflow
    experiment_name: my_experiment
    tracking_uri: file:///tmp/jenganlp_mlruns
    log_every_n_steps: 10
```

Supported values:

```yaml
service: mlflow
service: tensorboard
service: none
```

### `training.checkpoint`

```yaml
training:
  checkpoint:
    save_every_n_epochs: 1
    save_best: true
    max_checkpoints: 3
```

Checkpoints are written to:

```text
<output_dir>/checkpoints/epoch_0/
<output_dir>/checkpoints/best/
```

Each checkpoint can include:

```text
model.pt
checkpoint_meta.json
training_state.pt
```

`training_state.pt` is what enables resume.

### `training.data`

```yaml
training:
  data:
    test_size: 0.2
    seed: 42
    num_workers: 0
    pin_memory: false
```

JengaNLP currently splits each task dataset into train/eval using `test_size` and `seed`.

For deterministic smoke tests:

```yaml
seed: 42
num_workers: 0
pin_memory: false
```

### `integrations`

This is the pluggable adapter layer:

```yaml
integrations:
  - type: mlflow
    enabled: true
    params:
      tracking_uri: file:///tmp/jenganlp_mlruns
      experiment_name: my_project
  - type: tensorboard
    enabled: true
    params:
      log_dir: results/my_project/tensorboard
  - type: wandb
    enabled: false
    params:
      project: my_project
      entity: null
  - type: labelstudio
    enabled: false
    params:
      export_dir: results/my_project/labelstudio
```

The runtime emits structured activity events to:

```text
<output_dir>/activity.jsonl
```

Adapters receive these events and can log metrics, parameters, checkpoints, or artifacts.

## How Config Maps To Core Modules

The CLI is intentionally thin. The YAML config drives the core runtime.

Training path:

```text
YAML config
  -> ExperimentConfig.from_yaml()
  -> transformers.AutoTokenizer.from_pretrained(model.base_model)
  -> DataProcessor.process()
  -> TaskRegistry.create_task()
  -> MultiTaskModel.from_config()
  -> Trainer.train()
  -> callbacks + IntegrationRuntime
  -> model/checkpoints/logs/activity saved to output_dir
```

Main modules:

| Module | Responsibility |
|---|---|
| `jenga_nlp/core/config.py` | Pydantic config schema and validation. |
| `jenga_nlp/data/processor.py` | Loads CSV/JSON/JSONL, validates columns, tokenizes, creates labels, splits train/eval. |
| `jenga_nlp/tasks/registry.py` | Maps task type strings to task classes. |
| `jenga_nlp/tasks/*.py` | Loss and head logic for classification, sentiment, NER, regression, QA scoring. |
| `jenga_nlp/core/model.py` | Shared encoder, optional fusion, and task heads. |
| `jenga_nlp/training/trainer.py` | Optimizer, scheduler, task sampling, epochs, metrics, checkpoints, resume. |
| `jenga_nlp/training/callbacks.py` | Direct logging, early stopping, checkpoint callbacks. |
| `jenga_nlp/integrations/runtime.py` | Activity event logging and pluggable adapter fanout. |
| `jenga_nlp/inference/pipeline.py` | Loads saved config/model and runs inference. |
| `jenga_nlp/inference/formatters.py` | Converts logits into structured predictions. |
| `jenga_nlp/client/client.py` | Thin Python convenience wrapper. |

Inference path:

```text
model_dir + config
  -> InferencePipeline.from_checkpoint()
  -> prefer model_dir/experiment_config.yaml if present
  -> MultiTaskModel.load()
  -> tokenizer encodes text
  -> selected task heads run
  -> formatters convert logits to labels/entities/scores
```

The saved `experiment_config.yaml` is important. NER can discover label maps during preprocessing, and the trained config may differ from the original YAML. Inference prefers the saved config when available.

## Text Task Workflows

### Single-Label Classification

Use when exactly one label is correct.

Dataset:

```json
{"text": "Send your M-PESA PIN now", "label": 1}
{"text": "Your payment has been received", "label": 0}
```

Config:

```yaml
tasks:
  - name: mpesa_fraud_detection
    type: single_label_classification
    data_path: fixtures/sample_classification.jsonl
    text_column: text
    label_column: label
    heads:
      - name: fraud_head
        num_labels: 2
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu \
  --task-name mpesa_fraud_detection \
  --text "Share your PIN to reverse the transaction"
```

### Sentiment

`sentiment` is a domain-specific single-label classification task. Use it when labels mean polarity/rating.

Examples:

```text
negative, neutral, positive
1-star, 2-star, 3-star, 4-star, 5-star
angry, calm, satisfied
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_mpesa_sentiment_cpu.yaml \
  --device cpu
```

### Multi-Label Classification

Use when multiple labels can be true at the same time.

Dataset:

```json
{"text": "Urgent PIN request by SMS", "labels": [1, 1, 0, 1]}
```

Config:

```yaml
tasks:
  - name: message_flags
    type: multi_label_classification
    data_path: fixtures/sample_multilabel.jsonl
    text_column: text
    label_column: labels
    heads:
      - name: flags
        num_labels: 4
```

Predict with threshold:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_multilabel_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multilabel_cpu \
  --task-name message_flags \
  --threshold 0.5 \
  --text "Urgent account closure warning asking for PIN"
```

Lower the threshold if you want higher recall:

```bash
--threshold 0.25
```

### NER

Use NER when the output is spans in text.

Dataset:

```json
{
  "text": "Jane works at Safaricom in Nairobi",
  "labels": [
    {"start": 0, "end": 4, "label": "PERSON"},
    {"start": 14, "end": 22, "label": "ORG"},
    {"start": 26, "end": 33, "label": "LOCATION"}
  ]
}
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_ner_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_ner_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_ner_cpu \
  --task-name mpesa_ner \
  --text "Jane received KES 500 from Safaricom in Nairobi"
```

NER preprocessing discovers IOB labels from spans and saves the updated label map into the trained config.

### Regression

Use regression when the output is a continuous score.

Examples:

- fraud risk score
- urgency score
- toxicity severity
- call satisfaction estimate
- compliance risk from `0.0` to `1.0`

Bounded regression config:

```yaml
tasks:
  - name: text_risk_regression
    type: regression
    data_path: fixtures/sample_regression.jsonl
    text_column: text
    label_column: score
    heads:
      - name: risk_score
        num_labels: 1
        activation: sigmoid
        output_min: 0.0
        output_max: 1.0
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_regression_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_regression_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_regression_cpu \
  --task-name text_risk_regression \
  --text "Urgent M-PESA reversal requires your PIN immediately"
```

### QA Scoring / Compliance Scoring

Use `qa_scoring` for quality assurance and compliance checklists. This is not extractive question answering.

It is a multi-head, multi-label task:

```yaml
tasks:
  - name: call_qa_scoring
    type: qa_scoring
    data_path: fixtures/sample_qa_transcripts.jsonl
    text_column: transcript
    label_column: labels
    heads:
      - name: opening
        num_labels: 1
      - name: listening
        num_labels: 5
      - name: proactiveness
        num_labels: 3
      - name: resolution
        num_labels: 5
      - name: hold
        num_labels: 2
      - name: closing
        num_labels: 1
```

Dataset:

```json
{
  "transcript": "Thank you for calling. I understand the issue...",
  "labels": {
    "opening": [1],
    "listening": [1, 1, 0, 1, 0],
    "resolution": [1, 0, 1, 1, 0]
  }
}
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_qa_scoring_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_qa_scoring_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_qa_scoring_cpu \
  --task-name call_qa_scoring \
  --text "Thank you for calling. I understand your concern and I will help resolve it."
```

### Multi-Task Training

Multi-task training combines tasks in one config:

```yaml
tasks:
  - name: mpesa_fraud_detection
    type: single_label_classification
  - name: mpesa_sentiment
    type: sentiment
  - name: call_qa_scoring
    type: qa_scoring
```

Train:

```bash
python -m jenga_nlp train \
  --config configs/fixture_multitask_cpu.yaml \
  --device cpu
```

Predict one task:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_multitask_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multitask_cpu \
  --task-name mpesa_fraud_detection \
  --text "Urgent M-PESA reversal request asking for PIN"
```

Predict all tasks by omitting `--task-name`:

```bash
python -m jenga_nlp predict \
  --config configs/fixture_multitask_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multitask_cpu \
  --text "Thank you for calling. Please share your PIN."
```

For production, run only the required task when possible. Running all tasks increases latency.

## Python API

Use `JengaClient` for a simple local-first API:

```python
from jenga_nlp import JengaClient

client = JengaClient()

result = client.train(
    "configs/fixture_mpesa_fraud_cpu.yaml",
    device="cpu",
)
print(result)

metrics = client.evaluate(
    "configs/fixture_mpesa_fraud_cpu.yaml",
    model_dir="/tmp/jenganlp_fixture_mpesa_fraud_cpu",
    device="cpu",
)
print(metrics)

prediction = client.predict(
    "configs/fixture_mpesa_fraud_cpu.yaml",
    model_dir="/tmp/jenganlp_fixture_mpesa_fraud_cpu",
    task_name="mpesa_fraud_detection",
    text="Urgent M-PESA reversal request asking for PIN",
)
print(prediction)
```

Use lower-level modules when building a custom service:

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="/tmp/jenganlp_fixture_multitask_cpu",
    config_path="configs/fixture_multitask_cpu.yaml",
    device="cpu",
)

result = pipeline.predict(
    text="Urgent M-PESA reversal request asking for PIN",
    task_name="mpesa_fraud_detection",
)
print(result.to_dict())
```

Batch inference:

```python
results = pipeline.predict_batch(
    texts=[
        "Share your PIN now",
        "Your payment has been received",
    ],
    task_name="mpesa_fraud_detection",
    batch_size=32,
)
```

## Creating Multi-Task Models Programmatically

You do not have to write YAML manually. You can create the same config from Python:

```python
from jenga_nlp.core.config import (
    ExperimentConfig,
    HeadConfig,
    ModelConfig,
    TaskConfig,
    TokenizerConfig,
    TrainingConfig,
)

config = ExperimentConfig(
    project_name="mpesa_multitask",
    model=ModelConfig(base_model="distilbert-base-uncased"),
    tokenizer=TokenizerConfig(max_length=128),
    tasks=[
        TaskConfig(
            name="fraud",
            type="single_label_classification",
            data_path="data/fraud.jsonl",
            text_column="text",
            label_column="label",
            heads=[HeadConfig(name="fraud_head", num_labels=2)],
        ),
        TaskConfig(
            name="sentiment",
            type="sentiment",
            data_path="data/sentiment.jsonl",
            text_column="text",
            label_column="label",
            heads=[HeadConfig(name="sentiment_head", num_labels=3)],
        ),
    ],
    training=TrainingConfig(
        output_dir="results/mpesa_multitask",
        num_epochs=3,
        batch_size=16,
        device="auto",
    ),
)
```

Then:

```python
from jenga_nlp import JengaClient

client = JengaClient()
client.train(config)
```

This is useful when a platform or notebook UI generates configs dynamically.

## Hugging Face Hub Export

JengaNLP can push a trained model directory:

```python
from jenga_nlp import JengaClient

client = JengaClient()
url = client.push_to_hub(
    model_dir="/tmp/jenganlp_fixture_multitask_cpu",
    repo_id="your-org/your-model-name",
    token=None,
    private=False,
)
print(url)
```

Token resolution:

```bash
export HF_TOKEN=hf_...
```

Do not hardcode or commit Hugging Face tokens.

Important compatibility note:

- Custom JengaNLP multi-task models pushed to Hub still need `jenga_nlp` for inference.
- Standard Hugging Face models pushed to Hub can be loaded directly with `transformers`.

## Whisper Workflows

Whisper is different from the custom text heads. Whisper already has a standard architecture and processor in Hugging Face.

### Local Transcription

Transcribe an audio file:

```bash
python -m jenga_nlp asr serve \
  --model-name openai/whisper-small \
  --language so \
  --audio path/to/audio.wav
```

This loads a Whisper pipeline and prints transcription output.

### Whisper Fine-Tuning

Example config:

```yaml
model_name: openai/whisper-small
language: so
task: transcribe
output_dir: results/whisper-somali
learning_rate: 1.0e-5
num_epochs: 3
batch_size: 4
eval_batch_size: 8
gradient_accumulation_steps: 2
fp16: true
gradient_checkpointing: true

dataset:
  source: manifest
  train_manifest: data/somali_train.jsonl
  eval_manifest: data/somali_eval.jsonl
  audio_column: audio
  text_column: text
  duration_column: duration_s

preprocessing:
  sampling_rate: 16000
  min_audio_duration_s: 1.0
  max_audio_duration_s: 30.0
  strip_text: true
  lowercase: false

integrations:
  - type: mlflow
    enabled: true
    params:
      tracking_uri: file:./mlruns
      experiment_name: whisper-somali-training
  - type: tensorboard
    enabled: true
    params:
      log_dir: results/whisper-somali/tensorboard
```

Manifest row:

```json
{"audio": "/path/to/audio.wav", "text": "transcript text", "duration_s": 12.3}
```

Train:

```bash
python -m jenga_nlp asr train \
  --config configs/jenganlp_whisper_somali.yaml
```

Override data paths:

```bash
python -m jenga_nlp asr train \
  --config configs/jenganlp_whisper_somali.yaml \
  --train-data data/somali_train.jsonl \
  --eval-data data/somali_eval.jsonl
```

Resume:

```bash
python -m jenga_nlp asr train \
  --config configs/jenganlp_whisper_somali.yaml \
  --resume latest
```

Push to Hub:

```bash
python -m jenga_nlp asr train \
  --config configs/jenganlp_whisper_somali.yaml \
  --push-to-hub \
  --hub-model-id your-org/whisper-somali
```

Once exported as a standard Whisper model, a developer can use it without JengaNLP:

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="your-org/whisper-somali",
)

print(asr("audio.wav"))
```

That is the key difference: Whisper does not need custom JengaNLP task heads.

## Standard Transformers Workflows

Some workflows should remain standard Hugging Face workflows:

- masked language modeling
- causal language modeling
- vanilla sequence classification
- vanilla token classification
- vanilla seq2seq
- Whisper ASR

For example, a masked language model can be trained with `AutoModelForMaskedLM` and pushed to the Hub. Later, you can use that fine-tuned encoder as the base for a JengaNLP downstream classifier:

```yaml
model:
  base_model: your-org/domain-adapted-bert
```

This gives a clean two-stage workflow:

1. Use standard packages to adapt a base model to the language/domain.
2. Use JengaNLP to build custom multi-task heads for classification, NER, QA scoring, and regression.

## Production Inference

Do not use the CLI for high-throughput production inference. The CLI loads the model for each command, which is fine for tests but inefficient for production.

Use `InferencePipeline` and load once at application startup:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from jenga_nlp.inference.pipeline import InferencePipeline

app = FastAPI()

pipeline = InferencePipeline.from_checkpoint(
    model_dir="/models/mpesa_multitask",
    config_path="/models/mpesa_multitask/experiment_config.yaml",
    device="cuda",
)

class PredictRequest(BaseModel):
    text: str
    task_name: str | None = None
    threshold: float = 0.5

@app.post("/predict")
def predict(req: PredictRequest):
    result = pipeline.predict(
        text=req.text,
        task_name=req.task_name,
        threshold=req.threshold,
    )
    return result.to_dict()
```

Run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Production practices:

| Concern | Recommendation |
|---|---|
| Startup latency | Load model once at process startup. |
| Throughput | Use `predict_batch` where possible. |
| Latency | Pass `task_name` to avoid running unnecessary heads. |
| GPU usage | Prefer one model process per GPU unless memory allows more. |
| CPU usage | Use multiple workers if model fits memory and traffic requires it. |
| Token length | Tune `tokenizer.max_length`; long inputs increase cost. |
| Multi-task serving | Serve one shared model when tasks share the same domain. |
| Model quality | Use real base models and enough data; tiny-random fixtures only test mechanics. |
| Observability | Log request count, latency, selected task, model version, and errors. |
| Safety | Add PII handling and audit logging before regulated deployments. |

For custom JengaNLP models, production code imports:

```python
from jenga_nlp.inference.pipeline import InferencePipeline
```

For standard Whisper or standard Transformers models, production code can import only:

```python
from transformers import pipeline
```

This distinction keeps the framework useful without forcing unnecessary coupling.

## Resume And Checkpoints

Training checkpoints are saved under:

```text
<output_dir>/checkpoints/
```

Resume from a run directory:

```bash
python -m jenga_nlp resume \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --run-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu
```

Resume from a specific checkpoint:

```bash
python -m jenga_nlp train \
  --config configs/fixture_mpesa_fraud_cpu.yaml \
  --device cpu
```

The trainer internally looks for:

```text
training_state.pt
model.pt
```

Use resume after interruption, machine restart, or training crash. If the config changes materially between original training and resume, treat it as a new run.

## Activity Event Layer

Every run writes structured events to:

```text
<output_dir>/activity.jsonl
```

Events include:

- run started
- tokenizer loading
- datasets processed
- training started
- epoch started
- step loss logged
- evaluation metrics logged
- checkpoint saved
- training finished

Why this exists:

- CLI users see readable logs.
- Adapters receive structured events instead of parsing text logs.
- Future platform sync can replay activity events to a web dashboard.
- Debugging becomes easier because run history is saved as JSONL.

The activity layer is the bridge between the core trainer and external systems.

## Troubleshooting

### `python -m jenga_nlp` says command is required

Use:

```bash
python -m jenga_nlp -h
```

or one of:

```bash
python -m jenga_nlp train ...
python -m jenga_nlp evaluate ...
python -m jenga_nlp predict ...
```

### `--model-dir /tmp/ jenganlp_model` fails

The path must be one argument:

```bash
--model-dir /tmp/jenganlp_model
```

Do not split the directory with a space.

### Training is very fast

Fixture configs use:

```yaml
base_model: hf-internal-testing/tiny-random-bert
```

That model is intentionally tiny and random. It proves the pipeline works, but it does not produce production quality.

For real training, use:

```yaml
base_model: distilbert-base-uncased
```

or a multilingual/domain model:

```yaml
base_model: bert-base-multilingual-cased
base_model: xlm-roberta-base
```

### TensorBoard fails with `pkg_resources`

Install compatible setuptools:

```bash
pip install "setuptools>=68,<81"
```

### Hugging Face downloads warn about unauthenticated requests

Set a token for higher rate limits:

```bash
export HF_TOKEN=hf_...
```

Never commit tokens.

### Predictions are not semantically good

Check:

- Are you using `hf-internal-testing/tiny-random-bert`? That is only for smoke tests.
- Do you have enough data?
- Are label shapes correct?
- Are labels balanced?
- Did you train enough epochs?
- Is `max_length` cutting off important text?
- Is the task type correct?

## Recommended Development Sequence

For a new text project:

1. Start with one task and a tiny fixture-sized dataset.
2. Verify the config parses.
3. Train on CPU with `hf-internal-testing/tiny-random-bert`.
4. Confirm model artifacts, metrics, activity logs, and inference output.
5. Switch to a real base model.
6. Add MLflow or TensorBoard.
7. Add more data.
8. Add more tasks or heads only after the single-task path is stable.
9. Run production inference through `InferencePipeline`, not the CLI.
10. Push to Hugging Face only after the local model loads and predicts correctly.

For Whisper:

1. Validate manifest rows and audio paths.
2. Verify audio can decode and resample to 16 kHz.
3. Run a tiny overfit test.
4. Enable WER/CER metrics.
5. Add MLflow/TensorBoard.
6. Train the real model.
7. Push as a standard Whisper model.
8. Serve with `transformers.pipeline` or JengaNLP ASR wrapper.

## What Is Still Evolving

The implemented framework is usable for local text workflows, but these areas are still evolving:

- Full Label Studio ML backend server.
- Production HTTP serving command for text models.
- Platform API integration.
- Standard Hugging Face AutoModel-compatible export for custom JengaNLP heads.
- First-class masked language modeling workflow.
- More advanced production monitoring for drift, latency, and data quality.

The current product spine is sound: config-driven local training, custom multi-task text heads, adapters, checkpoints, resume, inference, and Whisper orchestration. The next step is to keep the framework narrow enough that each workflow is complete end to end before adding more task families.
