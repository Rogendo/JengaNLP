# Jenga-AI V2 Configuration Files

Example YAML configs for running experiments via the CLI.

## Available Configs

| Config | Description | Command |
|--------|-------------|---------|
| `multi_task_classification.yaml` | M-PESA fraud detection (binary classification) | `python -m jenga_ai train --config configs/multi_task_classification.yaml` |
| `qa_scoring.yaml` | Call transcript QA scoring (6 heads, 17 sub-metrics) | `python -m jenga_ai train --config configs/qa_scoring.yaml` |
| `llm_finetuning.yaml` | LLM fine-tuning with LoRA (distilgpt2) | `python -m jenga_ai llm-train --config configs/llm_finetuning.yaml` |

## CLI Options

```bash
# Override output directory
python -m jenga_ai train --config configs/qa_scoring.yaml --output-dir ./my_results

# Override device
python -m jenga_ai train --config configs/qa_scoring.yaml --device cpu

# Verbose logging
python -m jenga_ai -v train --config configs/qa_scoring.yaml

# Evaluate a saved model
python -m jenga_ai evaluate --config configs/qa_scoring.yaml --model-dir results/qa_scoring
```

## Creating Custom Configs

All configs follow Pydantic v2 schemas with validation. See:
- `jenga_ai/core/config.py` for multi-task config schema (`ExperimentConfig`)
- `jenga_ai/llm/config.py` for LLM config schema (`LLMConfig`)
