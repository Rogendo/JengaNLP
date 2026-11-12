---
title: "Text Regression in Python: Predict Risk Scores from Text"
description: "Predict continuous scores such as fraud risk, urgency or satisfaction from text with a bounded transformer regression head in JengaNLP."
---

# Text Regression in Python: Predict Risk Scores from Text

Text regression predicts a number instead of a label, such as a fraud-risk score between 0 and 1.

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
  --config examples/configs/fixture_regression_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_regression_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_regression_cpu \
  --task-name text_risk_regression \
  --text "Urgent M-PESA reversal requires your PIN immediately"
```

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
