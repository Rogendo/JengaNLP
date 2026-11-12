---
title: "Text Classification in Python: M-Pesa Fraud Detection"
description: "Train a transformer text classifier in Python from a YAML config, using mobile money (M-Pesa) fraud detection as the worked example."
---

# Text Classification in Python: M-Pesa Fraud Detection

Single-label text classification assigns exactly one label to each text. This page trains a fraud detector for mobile money (M-Pesa) SMS messages with JengaNLP.

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
  --config examples/configs/fixture_mpesa_fraud_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_mpesa_fraud_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu \
  --task-name mpesa_fraud_detection \
  --text "Share your PIN to reverse the transaction"
```

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
