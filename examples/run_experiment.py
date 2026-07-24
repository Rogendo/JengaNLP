#!/usr/bin/env python3
"""Run a multi-task experiment from a YAML config file.

Usage:
    python examples/run_experiment.py --config configs/multi_task_classification.yaml
    python examples/run_experiment.py --config configs/qa_scoring.yaml --output-dir ./my_results
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jenga-AI multi-task experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--device", default=None, help="Override device (auto/cuda/cpu/mps)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from transformers import AutoTokenizer

    from jenga_ai.core.config import ExperimentConfig, TaskType
    from jenga_ai.core.model import MultiTaskModel
    from jenga_ai.data.processor import DataProcessor
    from jenga_ai.training.trainer import Trainer

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    config = ExperimentConfig.from_yaml(config_path)

    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.device:
        config.training.device = args.device

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Jenga-AI V2 — Multi-Task Experiment")
    logger.info("=" * 60)
    logger.info("Project: %s", config.project_name)
    logger.info("Tasks: %s", [t.name for t in config.tasks])
    logger.info("Model: %s", config.model.base_model)
    logger.info("Epochs: %d", config.training.num_epochs)
    logger.info("Output: %s", output_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)

    # Process data
    logger.info("Processing data...")
    processor = DataProcessor(config, tokenizer)
    train_datasets, eval_datasets, config = processor.process()

    # Create model
    logger.info("Creating model...")
    model = MultiTaskModel.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Train
    logger.info("Starting training...")
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
    )
    results = trainer.train()

    # Save config
    config.to_yaml(output_dir / "experiment_config.yaml")

    # Print QA report if applicable
    for task_config in config.tasks:
        if task_config.type == TaskType.QA:
            from jenga_ai.tasks.qa import QAScoringTask

            report = QAScoringTask.format_evaluation_report(
                results.get("final_metrics", {}),
                task_name=task_config.name,
            )
            print(report)

    logger.info("Experiment complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
