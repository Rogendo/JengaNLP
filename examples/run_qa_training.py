#!/usr/bin/env python3
"""Run QA call transcript scoring training.

Trains a multi-head model for call quality assurance with 6 quality areas:
opening, listening, proactiveness, resolution, hold, closing.

Usage:
    python examples/run_qa_training.py --config configs/qa_scoring.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jenga-AI QA scoring training")
    parser.add_argument("--config", required=True, help="Path to QA YAML config")
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
    from jenga_ai.tasks.qa import QAScoringTask
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

    # Verify QA task type
    qa_tasks = [t for t in config.tasks if t.type == TaskType.QA]
    if not qa_tasks:
        logger.error("No QA (question_answering) tasks found in config")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Jenga-AI V2 — QA Call Scoring Training")
    logger.info("=" * 60)
    logger.info("Project: %s", config.project_name)
    logger.info("Model: %s", config.model.base_model)

    for task in qa_tasks:
        total_metrics = sum(h.num_labels for h in task.heads)
        logger.info(
            "QA Task '%s': %d heads, %d total sub-metrics",
            task.name, len(task.heads), total_metrics,
        )
        for head in task.heads:
            logger.info("  - %s: %d sub-metrics (weight=%.1f)", head.name, head.num_labels, head.weight)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)

    # Process data
    logger.info("Processing QA data...")
    processor = DataProcessor(config, tokenizer)
    train_datasets, eval_datasets, config = processor.process()

    # Create model
    logger.info("Creating multi-head QA model...")
    model = MultiTaskModel.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Train
    logger.info("Starting QA training...")
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

    # Print QA evaluation report
    final_metrics = results.get("final_metrics", {})
    for task in qa_tasks:
        report = QAScoringTask.format_evaluation_report(final_metrics, task_name=task.name)
        print(report)

    logger.info("QA training complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
