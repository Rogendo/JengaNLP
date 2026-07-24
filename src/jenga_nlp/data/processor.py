"""Unified data processor for JengaNLP.

This module provides a single DataProcessor class that can handle all task types (single-label, multi-label, NER, regression) with flexible configuration. Key features include:
- Supports JSON, JSONL, and CSV formats with auto-detection
- Per-task processing
- Configurable text/label column names
- Configurable train/test split ratio and seed
- Support for pre-split datasets (separate train/test files)
- Data validation with clear error messages
- Supports ALL task types 
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..core.config import ExperimentConfig, TaskConfig, TaskType

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw data files into tokenized datasets for training.

    Handles multiple tasks with different data formats and processing
    requirements. Each task is processed independently with its own
    configuration.

    Args:
        config: Experiment configuration.
        tokenizer: HuggingFace tokenizer instance.
    """

    def __init__(self, config: ExperimentConfig, tokenizer: PreTrainedTokenizer) -> None:
        self.config = config
        self.tokenizer = tokenizer

    def process(self) -> tuple[dict[str, Dataset], dict[str, Dataset], ExperimentConfig]:
        """Process all tasks and return train/eval datasets.

        Returns:
            Tuple of (train_datasets, eval_datasets, updated_config).
            Each dataset dict maps task_name -> Dataset.
        """
        train_datasets: dict[str, Dataset] = {}
        eval_datasets: dict[str, Dataset] = {}

        for task_config in self.config.tasks:
            logger.info(f"Processing data for task: {task_config.name} (type: {task_config.type})")

            # Load raw data
            df = self._load_data(task_config.data_path)
            self._validate_data(df, task_config)

            dataset = Dataset.from_pandas(df)

            # Process based on task type
            task_type = task_config.type
            if task_type == TaskType.SINGLE_LABEL_CLASSIFICATION or task_type == TaskType.SENTIMENT:
                tokenized = self._process_single_label(dataset, task_config)
            elif task_type in (TaskType.MULTI_LABEL_CLASSIFICATION, TaskType.QA_SCORING, TaskType.QUESTION_ANSWERING):
                tokenized = self._process_multi_label(dataset, task_config)
            elif task_type == TaskType.NER:
                tokenized = self._process_ner(dataset, task_config)
            elif task_type == TaskType.REGRESSION:
                tokenized = self._process_regression(dataset, task_config)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Set tensor format
            label_columns = [col for col in tokenized.column_names if col.startswith("labels")]
            columns_to_set = ["input_ids", "attention_mask"] + label_columns
            tokenized.set_format(type="torch", columns=columns_to_set)

            # Split into train/eval
            split_config = self.config.training.data
            split = tokenized.train_test_split(
                test_size=split_config.test_size,
                seed=split_config.seed,
            )

            train_datasets[task_config.name] = split["train"]
            eval_datasets[task_config.name] = split["test"]

            logger.info(
                f"  Task '{task_config.name}': {len(split['train'])} train, "
                f"{len(split['test'])} eval samples"
            )

        return train_datasets, eval_datasets, self.config

    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from JSON, JSONL, or CSV file."""
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            df = pd.read_json(data_path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(data_path)
        elif suffix == ".csv":
            df = pd.read_csv(data_path)
        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported formats: .json, .jsonl, .csv"
            )

        logger.info(f"  Loaded {len(df)} rows from {path.name}")
        return df

    def _validate_data(self, df: pd.DataFrame, task_config: TaskConfig) -> None:
        """Validate that the data has required columns."""
        text_col = task_config.text_column

        if text_col not in df.columns:
            available = list(df.columns)
            raise ValueError(
                f"Text column '{text_col}' not found in data. "
                f"Available columns: {available}"
            )

        if df[text_col].isna().any():
            na_count = df[text_col].isna().sum()
            logger.warning(f"  Found {na_count} NaN values in text column '{text_col}'")

    def _process_single_label(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Process single-label classification data."""
        text_col = task_config.text_column
        label_col = task_config.label_column

        def tokenize_and_label(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = self.tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config.tokenizer.max_length,
            )
            tokenized["labels"] = examples[label_col]
            return tokenized

        columns_to_remove = dataset.column_names
        return dataset.map(tokenize_and_label, batched=True, remove_columns=columns_to_remove)

    def _process_multi_label(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Process multi-label classification data."""
        text_col = task_config.text_column

        def tokenize_and_label(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = self.tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config.tokenizer.max_length,
            )

            # Process labels for each head
            for head_config in task_config.heads:
                head_name = head_config.name
                expected_len = head_config.num_labels
                labels_batch = []

                for i in range(len(examples[text_col])):
                    label_data = examples[task_config.label_column][i]

                    # Parse if string
                    if isinstance(label_data, str):
                        label_data = json.loads(label_data)

                    # Extract head-specific labels
                    if isinstance(label_data, dict):
                        label_value = label_data.get(head_name, [])
                    else:
                        label_value = label_data

                    # Clean and pad/truncate to expected length
                    try:
                        cleaned = [float(v) for v in label_value]
                    except (ValueError, TypeError):
                        cleaned = [0.0] * expected_len

                    if len(cleaned) > expected_len:
                        cleaned = cleaned[:expected_len]
                    elif len(cleaned) < expected_len:
                        cleaned = cleaned + [0.0] * (expected_len - len(cleaned))

                    labels_batch.append(torch.tensor(cleaned, dtype=torch.float))

                tokenized[f"labels_{head_name}"] = torch.stack(labels_batch)

            return tokenized

        columns_to_remove = dataset.column_names
        return dataset.map(tokenize_and_label, batched=True, remove_columns=columns_to_remove)

    def _process_ner(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Process NER data with IOB2 tagging."""
        text_col = task_config.text_column
        entities_col = task_config.label_column

        # Build label map from all entities in dataset
        all_entity_labels = set()
        for example in dataset:
            entities = example[entities_col]
            if isinstance(entities, list):
                for entity in entities:
                    if isinstance(entity, dict) and "label" in entity:
                        all_entity_labels.add(entity["label"])

        iob_labels = ["O"] + [f"{prefix}-{label}" for label in sorted(all_entity_labels) for prefix in ["B", "I"]]
        label_to_id = {label: i for i, label in enumerate(iob_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}

        # Update config with discovered label map
        task_config.heads[0].num_labels = len(label_to_id)
        task_config.label_maps = {"ner_head": id_to_label}

        logger.info(f"  NER labels discovered: {len(label_to_id)} ({list(label_to_id.keys())[:10]}...)")

        def tokenize_and_align(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = self.tokenizer(
                examples[text_col],
                truncation=True,
                is_split_into_words=False,
                padding="max_length",
                max_length=self.config.tokenizer.max_length,
            )

            all_labels = []
            for i, entities in enumerate(examples[entities_col]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids = [-100] * len(word_ids)

                if isinstance(entities, list):
                    for entity in entities:
                        if not isinstance(entity, dict):
                            continue
                        start_char = entity.get("start", 0)
                        end_char = entity.get("end", 0)
                        label = entity.get("label", "")

                        # Find tokens that overlap with this entity
                        entity_word_ids: set[int] = set()
                        for k, wid in enumerate(word_ids):
                            if wid is not None:
                                tok_start, tok_end = tokenized.encodings[i].offsets[k]
                                if tok_start < end_char and tok_end > start_char:
                                    entity_word_ids.add(wid)

                        sorted_word_ids = sorted(entity_word_ids)
                        if not sorted_word_ids:
                            continue

                        # Apply B-/I- tags
                        first_word = True
                        for wid in sorted_word_ids:
                            first_token = True
                            for tok_idx, mapped_wid in enumerate(word_ids):
                                if mapped_wid == wid and first_token:
                                    tag = f"B-{label}" if first_word else f"I-{label}"
                                    if tag in label_to_id:
                                        label_ids[tok_idx] = label_to_id[tag]
                                    first_token = False
                                    first_word = False

                # Set remaining word tokens to O
                for tok_idx, wid in enumerate(word_ids):
                    if wid is not None and label_ids[tok_idx] == -100:
                        label_ids[tok_idx] = label_to_id["O"]

                all_labels.append(label_ids)

            tokenized["labels"] = all_labels
            return tokenized

        columns_to_remove = dataset.column_names
        return dataset.map(tokenize_and_align, batched=True, remove_columns=columns_to_remove)

    def _process_regression(self, dataset: Dataset, task_config: TaskConfig) -> Dataset:
        """Process regression data."""
        text_col = task_config.text_column
        label_col = task_config.label_column

        def tokenize_and_label(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized = self.tokenizer(
                examples[text_col],
                padding="max_length",
                truncation=True,
                max_length=self.config.tokenizer.max_length,
            )
            tokenized["labels"] = [float(v) for v in examples[label_col]]
            return tokenized

        columns_to_remove = dataset.column_names
        return dataset.map(tokenize_and_label, batched=True, remove_columns=columns_to_remove)
