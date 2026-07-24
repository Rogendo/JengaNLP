# Inference Guide

JengaNLP supports three inference styles.

## 1. CLI Smoke Test

Use this after training to verify that a saved model loads:

```bash
python -m jenga_nlp predict \
  --config config.yaml \
  --model-dir ./results/my_model \
  --task-name my_task \
  --text "Your text here"
```

The CLI reloads the model each time. It is useful for checks, not high-throughput serving.

## 2. Python Load-Once Inference

Use this for notebooks, scripts, batch jobs, and production services:

```python
from jenga_nlp.inference.pipeline import InferencePipeline

pipeline = InferencePipeline.from_checkpoint(
    model_dir="./results/my_model",
    config_path="./results/my_model/experiment_config.yaml",
    device="cpu",
)

result = pipeline.predict(text="Your text here", task_name="my_task")
print(result.to_dict())
```

Batch prediction:

```python
results = pipeline.predict_batch(
    texts=["text one", "text two"],
    task_name="my_task",
    batch_size=32,
)
```

## 3. Production HTTP Wrapper

JengaNLP does not yet provide a stable text `serve` command. For now, wrap `InferencePipeline` in your own FastAPI app:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from jenga_nlp.inference.pipeline import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline.from_checkpoint(
    model_dir="/models/my_model",
    config_path="/models/my_model/experiment_config.yaml",
    device="cuda",
)

class PredictRequest(BaseModel):
    text: str
    task_name: str | None = None

@app.post("/predict")
def predict(req: PredictRequest):
    result = pipeline.predict(text=req.text, task_name=req.task_name)
    return result.to_dict()
```

Only use `curl` after an HTTP server is running. Notebook users should load the model directly in Python.

## JengaNLP-Native vs Standard Hugging Face Models

JengaNLP-native multi-task models require `jenga_nlp` at inference time because task heads and formatters are custom.

Standard exported models, such as Whisper or plain `AutoModelForSequenceClassification`, can usually be loaded with `transformers` directly.
