---
title: "NLP Task Types in JengaNLP"
description: "Overview of the NLP tasks JengaNLP can train from YAML - text classification, sentiment, multi-label, NER, regression, call QA scoring and multi-task learning."
---

# NLP task types

Every task is declared under `tasks:` in the YAML config. List several tasks to train them together on one shared encoder.

| Task | What it does |
|---|---|
| [Classification](classification.md) | Train a transformer text classifier in Python from a YAML config, using mobile money (M-Pesa) fraud detection as the worked example. |
| [Sentiment analysis](sentiment.md) | Fine-tune a transformer for sentiment analysis in Python with JengaNLP: negative/neutral/positive, star ratings or custom polarity labels. |
| [Multi-label classification](multi-label.md) | Train a multi-label text classifier in Python with JengaNLP, where several tags can be true for one message, and tune the decision threshold. |
| [Named entity recognition](ner.md) | Train a custom named entity recognition (NER) model in Python from character-span annotations with JengaNLP. |
| [Text regression](regression.md) | Predict continuous scores such as fraud risk, urgency or satisfaction from text with a bounded transformer regression head in JengaNLP. |
| [Call centre QA scoring](qa-scoring.md) | Automatically score call-centre transcripts against a QA or compliance checklist (opening, listening, resolution, closing) with JengaNLP. |
| [Multi-task learning](multi-task.md) | Train one shared transformer encoder for fraud detection, sentiment analysis and call QA scoring at once with JengaNLP multi-task learning. |
