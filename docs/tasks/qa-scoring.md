---
title: "Call Centre QA and Compliance Scoring with NLP"
description: "Automatically score call-centre transcripts against a QA or compliance checklist (opening, listening, resolution, closing) with JengaNLP."
---

# Call Centre QA and Compliance Scoring with NLP

Score call-centre transcripts against a quality-assurance checklist, with one head per rubric section.

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
  --config examples/configs/fixture_qa_scoring_cpu.yaml \
  --device cpu
```

Predict:

```bash
python -m jenga_nlp predict \
  --config examples/configs/fixture_qa_scoring_cpu.yaml \
  --model-dir /tmp/jenganlp_fixture_qa_scoring_cpu \
  --task-name call_qa_scoring \
  --text "Thank you for calling. I understand your concern and I will help resolve it."
```

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
