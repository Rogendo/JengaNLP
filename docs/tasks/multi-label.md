---
title: "Multi-Label Text Classification in Python"
description: "Train a multi-label text classifier in Python with JengaNLP, where several tags can be true for one message, and tune the decision threshold."
---

# Multi-Label Text Classification in Python

Multi-label classification predicts several independent tags per text, for example several risk flags on one SMS message.

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
  --config examples/configs/fixture_multilabel_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_multilabel_cpu \
  --task-name message_flags \
  --threshold 0.5 \
  --text "Urgent account closure warning asking for PIN"
```

Lower the threshold if you want higher recall:

```bash
--threshold 0.25
```

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
