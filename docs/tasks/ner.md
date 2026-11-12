---
title: "Named Entity Recognition (NER) Training in Python"
description: "Train a custom named entity recognition (NER) model in Python from character-span annotations with JengaNLP."
---

# Named Entity Recognition (NER) Training in Python

Train a token-classification NER model from character-span labels. JengaNLP converts spans to IOB tags for you.

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
  --config examples/configs/fixture_ner_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_ner_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_ner_cpu \
  --task-name mpesa_ner \
  --text "Jane received KES 500 from Safaricom in Nairobi"
```

NER preprocessing discovers IOB labels from spans and saves the updated label map into the trained config.

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
