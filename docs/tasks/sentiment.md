---
title: "Sentiment Analysis in Python with a Fine-Tuned Transformer"
description: "Fine-tune a transformer for sentiment analysis in Python with JengaNLP: negative/neutral/positive, star ratings or custom polarity labels."
---

# Sentiment Analysis in Python with a Fine-Tuned Transformer

Train a sentiment analysis model on your own labelled text with one YAML config and one command.

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
  --config examples/configs/fixture_mpesa_sentiment_cpu.yaml \
  --device cpu
```

## Next steps

- [Configuration reference](../configuration.md) for every YAML option.
- [Production inference](../inference.md) to load the trained model once and serve predictions.
- [All task types](index.md)
