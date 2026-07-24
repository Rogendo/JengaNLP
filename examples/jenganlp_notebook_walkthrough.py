#!/usr/bin/env python3
"""Notebook-style JengaNLP walkthrough.

This file is intentionally written like a notebook, but saved as a regular
Python script. Editors such as VS Code can run the `# %%` sections as cells,
while CI, local machines, Colab, Kaggle, and cloud GPU boxes can run it with:

    python examples/jenganlp_notebook_walkthrough.py

The goal is to explain as it runs:
- load a YAML config
- inspect task/model/training choices
- check configured backend adapters
- train
- evaluate
- predict via JengaClient
- load once for production-style inference
- optionally push to Hugging Face Hub
"""

from __future__ import annotations

# %% Imports and path setup
import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Allow running directly from a cloned repo without installing first.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def explain(title: str, body: str) -> None:
    """Print notebook-style narrative output."""
    print("\n" + "=" * 88, flush=True)
    print(title, flush=True)
    print("-" * 88, flush=True)
    print(body.strip(), flush=True)


def compact_dict(data: dict[str, Any], max_items: int = 8) -> dict[str, Any]:
    """Keep console output readable for larger metric dictionaries."""
    items = list(data.items())
    if len(items) <= max_items:
        return data
    head = dict(items[:max_items])
    head["..."] = f"{len(items) - max_items} more keys"
    return head


# %% CLI arguments
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a notebook-style JengaNLP walkthrough from config to inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/fixture_multitask_cpu.yaml",
        help="JengaNLP YAML config to run.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device override: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--text",
        default="Urgent M-PESA reversal request asking the customer to share their PIN.",
        help="Text used for post-training inference.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Task to run during inference. Defaults to the first task in the config.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for multi-label and QA scoring heads.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and reuse the configured output_dir/model_dir.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained model directory to Hugging Face Hub.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face repo id, required with --push-to-hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/use a private Hugging Face repo.",
    )
    return parser


# %% Config summary helpers
def print_config_summary(config: Any) -> None:
    print(f"Project: {config.project_name}")
    print(f"Base model: {config.model.base_model}")
    print(f"Tokenizer max_length: {config.tokenizer.max_length}")
    print(f"Training output_dir: {config.training.output_dir}")
    print(f"Training device: {config.training.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Task sampling: {config.training.task_sampling}")

    print("\nTasks:")
    for task in config.tasks:
        heads = ", ".join(f"{head.name}:{head.num_labels}" for head in task.heads)
        print(f"  - {task.name} | type={task.type.value} | data={task.data_path} | heads={heads}")

    if config.training.logging:
        print("\nDirect training.logging:")
        print(f"  service={config.training.logging.service.value}")
        print(f"  experiment={config.training.logging.experiment_name}")
        print(f"  tracking_uri={config.training.logging.tracking_uri}")
    else:
        print("\nDirect training.logging: none")

    if config.integrations:
        print("\nPluggable integrations:")
        for integration in config.integrations:
            state = "enabled" if integration.enabled else "disabled"
            print(f"  - {integration.type}: {state} params={integration.params}")
    else:
        print("\nPluggable integrations: none configured")


def print_backend_status(config: Any) -> None:
    from jenga_nlp.integrations.registry import build_adapter

    if not config.integrations:
        print("No `integrations:` block configured. Direct `training.logging` may still log.")
        return

    for integration in config.integrations:
        if not integration.enabled:
            print(f"{integration.type}: disabled")
            continue
        adapter = build_adapter(integration, base_dir=config.training.output_dir)
        if adapter is None:
            print(f"{integration.type}: not built")
            continue
        status = "available" if adapter.is_available() else "missing optional dependency"
        print(f"{adapter.name}: {status}")


def print_service_hints(config: Any) -> None:
    output_dir = Path(config.training.output_dir)
    logging_config = config.training.logging

    if logging_config and logging_config.service.value == "mlflow":
        tracking_uri = logging_config.tracking_uri or "file:./mlruns"
        if tracking_uri.startswith("file://"):
            mlruns = tracking_uri.removeprefix("file://")
            print("\nMLflow UI command:")
            print(f"  mlflow ui --backend-store-uri {mlruns} --host 127.0.0.1 --port 5000")

    if logging_config and logging_config.service.value == "tensorboard":
        logdir = output_dir / "logs"
        print("\nTensorBoard command:")
        print(f"  tensorboard --logdir {logdir} --host 127.0.0.1 --port 6006")

    for integration in config.integrations:
        if not integration.enabled:
            continue
        if integration.type.lower() == "tensorboard":
            logdir = integration.params.get("log_dir", str(output_dir / "tensorboard"))
            print("\nTensorBoard command:")
            print(f"  tensorboard --logdir {logdir} --host 127.0.0.1 --port 6006")
        elif integration.type.lower() == "mlflow":
            tracking_uri = integration.params.get("tracking_uri")
            if tracking_uri and tracking_uri.startswith("file:"):
                mlruns = tracking_uri.removeprefix("file:")
                print("\nMLflow UI command:")
                print(f"  mlflow ui --backend-store-uri {mlruns} --host 127.0.0.1 --port 5000")


# %% Main walkthrough
def main() -> None:
    args = build_parser().parse_args()

    from jenga_nlp import ExperimentConfig, JengaClient
    from jenga_nlp.inference.pipeline import InferencePipeline

    explain(
        "1. Load the YAML config",
        """
        The YAML config is the user-facing abstraction. Internally it becomes
        `ExperimentConfig` from `jenga_nlp.core.config`, which validates model,
        tokenizer, task, training, checkpoint, and integration settings.
        """,
    )
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = ExperimentConfig.from_yaml(config_path)
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.device:
        config.training.device = args.device

    selected_task = args.task_name or config.tasks[0].name
    print_config_summary(config)

    explain(
        "2. Check backend/logging configuration",
        """
        `training.logging` is the direct built-in logger for MLflow/TensorBoard.
        `integrations` is the newer pluggable adapter layer that receives
        structured activity events and can fan out to MLflow, TensorBoard, W&B,
        and Label Studio-style artifact exports.
        """,
    )
    print_backend_status(config)
    print_service_hints(config)

    client = JengaClient()
    model_dir = Path(config.training.output_dir)

    if args.skip_train:
        explain(
            "3. Skip training",
            """
            `--skip-train` is useful when the model directory already contains
            `model.pt`, `metadata.json`, and `experiment_config.yaml`.
            """,
        )
        if not (model_dir / "model.pt").exists():
            raise FileNotFoundError(f"Cannot skip training; no model.pt found in {model_dir}")
    else:
        explain(
            "3. Train",
            """
            `JengaClient.train()` calls the same core path as the CLI:
            tokenizer -> DataProcessor -> TaskRegistry -> MultiTaskModel ->
            Trainer -> callbacks/adapters/checkpoints.
            """,
        )
        train_result = client.train(config)
        print("Training result:")
        print(compact_dict(train_result))

    if args.skip_eval:
        explain(
            "4. Skip evaluation",
            "Evaluation is skipped by request. The next step will run inference from the saved checkpoint.",
        )
    else:
        explain(
            "4. Evaluate",
            """
            Evaluation reloads the same config, rebuilds eval datasets through
            `DataProcessor`, loads the saved model, and computes task-specific
            metrics such as accuracy/F1, NER metrics, QA scoring F1, or
            regression MSE/RMSE/MAE.
            """,
        )
        metrics = client.evaluate(config, model_dir=model_dir, device=config.training.device)
        print("Evaluation metrics:")
        print(compact_dict(metrics, max_items=12))

    explain(
        "5. Predict with JengaClient",
        """
        This is the simple developer API. Use it for notebooks, scripts, demos,
        and low-throughput checks. For production, load `InferencePipeline` once
        and reuse it instead of reloading per request.
        """,
    )
    prediction = client.predict(
        config,
        model_dir=model_dir,
        text=args.text,
        task_name=selected_task,
        threshold=args.threshold,
    )
    print("Prediction:")
    print(prediction)

    explain(
        "6. Production-style inference",
        """
        Production services should import JengaNLP once, load the model once at
        process startup, then call `pipeline.predict()` or `predict_batch()`.
        This is how a FastAPI service or worker should use custom JengaNLP
        multi-task/multi-head models.
        """,
    )
    pipeline = InferencePipeline.from_checkpoint(
        model_dir=model_dir,
        config_path=config_path,
        device=config.training.device,
    )
    batch_results = pipeline.predict_batch(
        texts=[
            args.text,
            "Customer asks about branch working hours and account balance.",
        ],
        task_name=selected_task,
        threshold=args.threshold,
        batch_size=2,
    )
    for idx, result in enumerate(batch_results, start=1):
        print(f"Batch result {idx}:")
        print(result.to_dict())

    if args.push_to_hub:
        explain(
            "7. Optional Hugging Face Hub push",
            """
            This step is intentionally opt-in. Custom JengaNLP models can be
            uploaded as artifact directories, but consumers still need JengaNLP
            for inference because the architecture contains custom task heads.
            """,
        )
        if not args.repo_id:
            raise ValueError("--repo-id is required with --push-to-hub")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before --push-to-hub")
        hub_url = client.push_to_hub(model_dir=model_dir, repo_id=args.repo_id, token=token, private=args.private)
        print(f"Pushed to: {hub_url}")
    else:
        explain(
            "7. Hub push skipped",
            """
            Add `--push-to-hub --repo-id your-org/your-model` when you are ready
            to upload artifacts. For GPU notebooks, keep the same script and
            switch `--device cuda` plus a real config/base model.
            """,
        )

    explain(
        "Done",
        f"""
        Model directory: {model_dir}
        Saved config: {model_dir / "experiment_config.yaml"}
        Activity log: {model_dir / "activity.jsonl"}

        Next useful commands:
          python -m jenga_nlp evaluate --config {config_path} --model-dir {model_dir} --device {config.training.device}
          python -m jenga_nlp predict --config {config_path} --model-dir {model_dir} --task-name {selected_task} --text "{args.text}"
        """,
    )


if __name__ == "__main__":
    main()
