# Quickstart

This quickstart creates a tiny dataset, writes a config, trains a CPU model, evaluates it, and runs inference.

The model uses `hf-internal-testing/tiny-random-bert`, which is intentionally tiny and random. Use it to verify workflow mechanics, not production quality.

## 1. Create A Dataset

```bash
mkdir -p /tmp/jenganlp_demo
cat > /tmp/jenganlp_demo/messages.jsonl <<'JSONL'
{"text":"Urgent M-PESA reversal request. Share your PIN now.","label":1}
{"text":"Your payment has been received successfully.","label":0}
{"text":"Congratulations, send your PIN to claim the prize.","label":1}
{"text":"Please visit the nearest branch for help.","label":0}
{"text":"Account blocked. Reply with your secret code.","label":1}
{"text":"Your statement is ready for download.","label":0}
JSONL
```

## 2. Create A Config

```bash
cat > /tmp/jenganlp_demo/config.yaml <<'YAML'
project_name: demo_mpesa_fraud

model:
  base_model: hf-internal-testing/tiny-random-bert

tokenizer:
  max_length: 64
  padding: max_length
  truncation: true

tasks:
  - name: mpesa_fraud_detection
    type: single_label_classification
    data_path: /tmp/jenganlp_demo/messages.jsonl
    text_column: text
    label_column: label
    heads:
      - name: fraud_head
        num_labels: 2

training:
  output_dir: /tmp/jenganlp_demo/model
  learning_rate: 0.00005
  batch_size: 2
  eval_batch_size: 2
  num_epochs: 1
  device: cpu
  logging:
    service: none
    experiment_name: demo_mpesa_fraud
  checkpoint:
    save_every_n_epochs: 1
    save_best: true
    max_checkpoints: 1
  data:
    test_size: 0.33
    seed: 42
    num_workers: 0
    pin_memory: false
YAML
```

## 3. Train

```bash
python -m jenga_nlp train --config /tmp/jenganlp_demo/config.yaml --device cpu
```

## 4. Evaluate

```bash
python -m jenga_nlp evaluate \
  --config /tmp/jenganlp_demo/config.yaml \
  --model-dir /tmp/jenganlp_demo/model \
  --device cpu
```

## 5. Predict

```bash
python -m jenga_nlp predict \
  --config /tmp/jenganlp_demo/config.yaml \
  --model-dir /tmp/jenganlp_demo/model \
  --task-name mpesa_fraud_detection \
  --text "Urgent M-PESA reversal request asking for PIN"
```

## 6. Load Once In Python

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="/tmp/jenganlp_demo/model",
    config_path="/tmp/jenganlp_demo/config.yaml",
    device="cpu",
)

result = pipeline.predict(
    text="Urgent M-PESA reversal request asking for PIN",
    task_name="mpesa_fraud_detection",
)
print(result.to_dict())
```

Use this Python loading pattern for repeated notebook inference and production services.
