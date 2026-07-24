#!/usr/bin/env python3
"""Train -> push_to_hub -> inference demo for JengaNLP.

This example is intentionally concrete:
- uses a local fixture dataset
- trains a small binary classifier
- pushes the resulting artifact directory to Hugging Face Hub
- runs inference from the saved local checkpoint path

Prerequisites:
- install the project dependencies
- export HF_TOKEN (or pass --token)
- choose a repo ID you can create on Hugging Face
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="JengaNLP HF demo")
    parser.add_argument("--config", required=True, help="Path to a JengaNLP YAML config")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/jenganlp-demo")
    parser.add_argument("--token", default=None, help="Hugging Face token (defaults to HF_TOKEN env var)")
    parser.add_argument("--text", default="Received a suspicious M-PESA message asking for my PIN.", help="Text to classify after training")
    parser.add_argument("--task-name", default="fraud_detection", help="Task name for inference")
    parser.add_argument("--device", default=None, help="Override device (auto/cuda/cpu/mps)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("Missing Hugging Face token. Set HF_TOKEN or pass --token.")
        sys.exit(1)

    from jenga_nlp import ExperimentConfig, JengaClient

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    client = JengaClient()
    config = ExperimentConfig.from_yaml(config_path)

    logger.info("Training model...")
    train_result = client.train(config, output_dir=args.output_dir, device=args.device)

    model_dir = Path(config.training.output_dir)
    logger.info("Training complete. Model dir: %s", model_dir)
    logger.info("Pushing model to Hugging Face Hub: %s", args.repo_id)

    hub_url = client.push_to_hub(
        model_dir=model_dir,
        repo_id=args.repo_id,
        token=token,
        private=False,
    )
    logger.info("Pushed to %s", hub_url)

    logger.info("Running local inference from the saved checkpoint...")
    prediction = client.predict(
        config=config,
        model_dir=model_dir,
        text=args.text,
        task_name=args.task_name,
    )

    print("\n=== Training Result ===")
    print(train_result)
    print("\n=== Hugging Face Hub URL ===")
    print(hub_url)
    print("\n=== Prediction ===")
    print(prediction)


if __name__ == "__main__":
    main()
