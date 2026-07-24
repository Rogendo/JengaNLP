#!/usr/bin/env python3
"""Run LLM fine-tuning from a YAML config file.

Usage:
    python examples/run_llm_finetuning.py --config configs/llm_finetuning.yaml
    python examples/run_llm_finetuning.py --config configs/llm_finetuning.yaml --output-dir ./my_llm
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Jenga-AI LLM fine-tuning")
    parser.add_argument("--config", required=True, help="Path to LLM YAML config")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from jenga_ai.llm.config import LLMConfig
    from jenga_ai.llm.data import LLMDataProcessor
    from jenga_ai.llm.model_factory import load_model_and_tokenizer
    from jenga_ai.llm.trainer import LLMTrainer

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    config = LLMConfig.from_yaml(config_path)

    if args.output_dir:
        config.training.output_dir = args.output_dir

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Jenga-AI V2 — LLM Fine-Tuning")
    logger.info("=" * 60)
    logger.info("Model: %s", config.model_name)
    logger.info("LoRA: enabled=%s, rank=%d", config.lora.enabled, config.lora.rank)
    logger.info("Output: %s", output_dir)

    # Load tokenizer for data processing
    _, tokenizer = load_model_and_tokenizer(config)

    # Process data
    logger.info("Processing data...")
    data_processor = LLMDataProcessor(config, tokenizer)
    train_dataset, eval_dataset = data_processor.load_and_prepare()

    logger.info("Train samples: %d", len(train_dataset))
    if eval_dataset:
        logger.info("Eval samples: %d", len(eval_dataset))

    # Train
    logger.info("Starting LLM training...")
    trainer = LLMTrainer(config)
    metrics = trainer.train(train_dataset, eval_dataset)

    # Save
    trainer.save(str(output_dir))

    logger.info("Training complete. Metrics: %s", metrics)
    logger.info("Model saved to %s", output_dir)


if __name__ == "__main__":
    main()
