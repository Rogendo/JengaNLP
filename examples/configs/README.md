# JengaNLP Example Configs

Example YAML configs for the `jenga-nlp` CLI. Run commands from the repository root.
Relative `data_path` values are resolved against this folder, so the `../fixtures/...` paths work from anywhere.

## CPU fixture configs (run in minutes, bundled data)

| Config | Task |
|--------|------|
| `fixture_mpesa_fraud_cpu.yaml` | M-PESA fraud detection (binary classification), logs to MLflow (`sqlite:///mlflow.db`) |
| `fixture_mpesa_sentiment_cpu.yaml` | Sentiment (3 classes) |
| `fixture_multilabel_cpu.yaml` | Multi-label risk tags |
| `fixture_ner_cpu.yaml` | Named entity recognition |
| `fixture_regression_cpu.yaml` | Regression score |
| `fixture_qa_scoring_cpu.yaml` | Call transcript QA scoring (6 heads) |
| `fixture_multitask_cpu.yaml` | Fraud + sentiment + QA scoring on one shared encoder |
| `fixture_logging_mlflow_cpu.yaml` / `fixture_logging_tensorboard_cpu.yaml` | Logging backend examples |

## Larger examples

| Config | Task |
|--------|------|
| `multi_task_classification.yaml` | Fraud detection with TensorBoard logging |
| `multi_task_3way_fusion.yaml` | 3 tasks with attention fusion |
| `qa_scoring.yaml` | QA scoring — **dataset not bundled**, set `data_path` to your own data |
| `jenganlp_whisper_somali.yaml` | Whisper ASR fine-tuning (experimental; use `jenga-nlp asr train`) |

## Commands

```bash
# Train
jenga-nlp train --config examples/configs/fixture_mpesa_fraud_cpu.yaml

# Override output directory or device
jenga-nlp train --config examples/configs/fixture_qa_scoring_cpu.yaml --output-dir ./my_results --device cpu

# Evaluate a saved model
jenga-nlp evaluate --config examples/configs/fixture_qa_scoring_cpu.yaml --model-dir ./my_results

# Predict
jenga-nlp predict --config examples/configs/fixture_mpesa_fraud_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu --text "Share your M-PESA PIN to claim your prize"
```

The standalone scripts `examples/run_experiment.py` and `examples/run_qa_training.py` accept the same `--config`, `--output-dir` and `--device` options, plus `-v` for debug logging.

## Config schema

All configs are validated by `ExperimentConfig` in `src/jenga_nlp/core/config.py` (Pydantic v2).
