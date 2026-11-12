---
title: "Multi-Task Learning in Python: Fraud Detection, Sentiment and QA in One Model"
description: "Train one shared transformer encoder for fraud detection, sentiment analysis and call QA scoring at once with JengaNLP multi-task learning."
---

# Multi-Task Learning in Python: Fraud Detection, Sentiment and QA in One Model

Multi-task learning trains one shared encoder with a separate head for each task, so a single model serves several predictions and loads once in production.

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
  --config examples/configs/fixture_multitask_cpu.yaml \
  --device cpu
```

Predict one task:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_multitask_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multitask_cpu \
  --task-name mpesa_fraud_detection \
  --text "Urgent M-PESA reversal request asking for PIN"
```

Predict all tasks by omitting `--task-name`:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_multitask_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multitask_cpu \
  --text "Thank you for calling. Please share your PIN."
```

For production, run only the required task when possible. Running all tasks increases latency.

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
