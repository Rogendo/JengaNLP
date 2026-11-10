# JengaNLP

**Config-driven multi-task NLP.** Train one shared transformer encoder with many task heads — classification, sentiment, NER, regression and checklist-style QA scoring — from a single YAML file, then serve it with a load-once inference pipeline.

> **Status: alpha (2.0.0).** The core text workflow — config → train → evaluate → predict → `InferencePipeline` — is tested and is the supported path. Everything else is marked experimental below.

## What JengaNLP is, honestly

JengaNLP is built on PyTorch and Hugging Face Transformers. For a single classifier, the Hugging Face `Trainer` is a fine choice, and JengaNLP is mostly a convenience layer on top of the same ideas.

Where JengaNLP does more than a training wrapper:

- **Multi-task models are the default, not a research project.** One encoder, several tasks, several heads per task, per-head loss weights, and task sampling strategies (`round_robin`, `proportional`, `temperature`) — all declared in YAML.
- **Checklist QA scoring (`qa_scoring`).** Score call-centre transcripts against weighted rubric sections (opening, listening, resolution, …) as multi-head binary checklists — a task type you would otherwise hand-build.
- **Production inference included.** `InferencePipeline` loads once, predicts single or batch, returns structured per-head results with confidences, and flags PII in inputs.
- **One artefact layout.** Training, evaluation, resume, checkpoints and inference all read the same saved config and model directory.

**What is not here yet:** pretrained JengaNLP checkpoints, published benchmarks, and African-language base models and datasets. Those are the roadmap, not current features — see [Roadmap](#roadmap).

## Install

```bash
pip install jenga-nlp
```

Requires Python 3.10+. Optional extras:

| Extra | Installs | When you need it |
|---|---|---|
| `jenga-nlp[asr]` | accelerate, peft, librosa, soundfile | Whisper ASR workflows (experimental) |
| `jenga-nlp[wandb]` | `wandb` | Weights & Biases experiment tracking |
| `jenga-nlp[dev]` | pytest, ruff, mypy | Working on JengaNLP itself |

From source:

```bash
git clone https://github.com/Rogendo/JengaNLP.git
cd JengaNLP
pip install -e ".[dev]"
```

## Quickstart (5 minutes, CPU)

**1. Data** — JSONL, JSON or CSV with a text column and a label column:

```bash
mkdir -p jenganlp_demo
cat > jenganlp_demo/messages.jsonl <<'JSONL'
{"text":"Urgent M-PESA reversal request. Share your PIN now.","label":1}
{"text":"Your payment has been received successfully.","label":0}
{"text":"Congratulations, send your PIN to claim the prize.","label":1}
{"text":"Please visit the nearest branch for help.","label":0}
{"text":"Account blocked. Reply with your secret code.","label":1}
{"text":"Your statement is ready for download.","label":0}
JSONL
```

**2. Config** — run every command from the folder that contains `jenganlp_demo/`:

```bash
cat > jenganlp_demo/config.yaml <<'YAML'
project_name: demo_mpesa_fraud
model:
  base_model: distilbert-base-uncased
tokenizer:
  max_length: 64
tasks:
  - name: mpesa_fraud_detection
    type: single_label_classification
    data_path: jenganlp_demo/messages.jsonl
    text_column: text
    label_column: label
    heads:
      - name: fraud_head
        num_labels: 2
training:
  output_dir: jenganlp_demo/model
  batch_size: 2
  num_epochs: 3
  device: cpu
  logging:
    service: none
  data:
    test_size: 0.33
YAML
```

**3. Train, evaluate, predict:**

```bash
jenga-nlp train    --config jenganlp_demo/config.yaml
jenga-nlp evaluate --config jenganlp_demo/config.yaml --model-dir jenganlp_demo/model
jenga-nlp predict  --config jenganlp_demo/config.yaml --model-dir jenganlp_demo/model \
  --task-name mpesa_fraud_detection --text "Urgent M-PESA reversal request asking for PIN"
```

**4. Load once in Python** (notebooks, batch jobs, services):

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="jenganlp_demo/model",
    config_path="jenganlp_demo/config.yaml",
    device="cpu",
)
print(pipeline.predict(text="Urgent M-PESA reversal request asking for PIN",
                       task_name="mpesa_fraud_detection").to_dict())
```

> Six examples prove the pipeline works, not that the model is good. A perfect score on a handful of eval samples is meaningless — use hundreds of labelled examples per class before trusting predictions.

## Task types

| `type` | Labels look like | Example use |
|---|---|---|
| `single_label_classification` | `1` | Fraud vs legitimate message |
| `multi_label_classification` | `{"risk_tags": [1, 1, 0, 1]}` | Several risk tags per message |
| `sentiment` | `2` | Negative / neutral / positive |
| `ner` | entity spans | Names, amounts, phone numbers |
| `regression` | `0.73` | Risk or quality score |
| `qa_scoring` | `{"opening": [1], "resolution": [1, 0, 1, 1, 0]}` | Call-centre QA checklists |

Multi-task: list several tasks under `tasks:` — they share the encoder. See [`examples/configs/fixture_multitask_cpu.yaml`](https://github.com/Rogendo/JengaNLP/blob/main/examples/configs/fixture_multitask_cpu.yaml).

## Stable vs experimental

| Area | Status |
|---|---|
| YAML config, text training, evaluation, resume, checkpoints | **Stable** — covered by tests |
| `InferencePipeline` (single + batch), CLI predict | **Stable** — covered by tests |
| MLflow / TensorBoard logging | **Stable** |
| Hugging Face Hub export | Experimental |
| Whisper ASR fine-tuning and serving | Experimental — not yet validated on real audio/GPU; needs `ffmpeg` installed to read audio files |
| Domain-adaptive MLM pretraining (`jenga-nlp mlm`) | Experimental — tested on CPU with small models; not yet run on GPU-scale corpora |
| Weights & Biases, Label Studio adapters | Experimental |
| Explainability, human-in-the-loop routing, audit logging | Experimental |

Experimental features may change or move out of the core package.

## Documentation

- [Quickstart](https://github.com/Rogendo/JengaNLP/blob/main/docs/quickstart.md)
- [Configuration reference](https://github.com/Rogendo/JengaNLP/blob/main/docs/configuration.md)
- [Inference and production serving](https://github.com/Rogendo/JengaNLP/blob/main/docs/inference.md)
- [Domain-adaptive pretraining (MLM)](https://github.com/Rogendo/JengaNLP/blob/main/docs/domain-adaptation.md) — adapt AfriBERTa / AfroXLMR to Kenyan text (Swahili, Sheng, code-switching)
- [Full usage guide](https://github.com/Rogendo/JengaNLP/blob/main/docs/usage-guide.md) — every workflow, including experimental ones
- [Example configs](https://github.com/Rogendo/JengaNLP/blob/main/examples/configs/) and [fixtures](https://github.com/Rogendo/JengaNLP/blob/main/examples/fixtures/)

## Roadmap

1. **African-language base models** — adapt an African-language encoder (AfriBERTa / AfroXLMR) to Swahili, Sheng and Swahili–English code-switched text. The training workflow is available as `jenga-nlp mlm` (experimental); adapted models and their evaluations are next.
2. **Benchmarks** — public results on African-language datasets, including multi-task vs single-task baselines.
3. **Ready-made templates with pretrained checkpoints** — mobile-money fraud, call-centre QA, Swahili sentiment, on the Hugging Face Hub.

## License

MIT — see [LICENSE](https://github.com/Rogendo/JengaNLP/blob/main/LICENSE).
