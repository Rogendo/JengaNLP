"""Unit tests for data processor module."""

import pytest
from transformers import AutoTokenizer

from jenga_nlp.core.config import ExperimentConfig


class TestDataProcessor:
    """Test DataProcessor class."""

    @pytest.fixture
    def tokenizer(self):
        """Load a small tokenizer for testing."""
        return AutoTokenizer.from_pretrained("distilbert-base-uncased")

    @pytest.fixture
    def experiment_config(self):
        """Create a basic experiment config for testing."""
        from jenga_nlp.core.config import TaskConfig, HeadConfig, TaskType
        
        head = HeadConfig(name="default_head", num_labels=2)
        task = TaskConfig(
            name="test_task",
            type=TaskType.SINGLE_LABEL_CLASSIFICATION,
            data_path="/tmp/dummy.jsonl",
            heads=[head],
        )
        
        return ExperimentConfig(
            project_name="test_project",
            model={},
            tokenizer={},
            tasks=[task],
            training={},
        )

    def test_data_processor_import(self):
        """DataProcessor should be importable."""
        from jenga_nlp.data.processor import DataProcessor
        assert DataProcessor is not None

    def test_experiment_config_required(self, experiment_config, tokenizer):
        """DataProcessor requires ExperimentConfig and tokenizer."""
        from jenga_nlp.data.processor import DataProcessor
        processor = DataProcessor(experiment_config, tokenizer)
        assert processor.config == experiment_config
        assert processor.tokenizer == tokenizer

    def test_tokenizer_preserved(self, experiment_config, tokenizer):
        """DataProcessor should preserve the tokenizer instance."""
        from jenga_nlp.data.processor import DataProcessor
        processor = DataProcessor(experiment_config, tokenizer)
        assert processor.tokenizer is tokenizer

    def test_config_preserved(self, experiment_config, tokenizer):
        """DataProcessor should preserve the experiment config."""
        from jenga_nlp.data.processor import DataProcessor
        processor = DataProcessor(experiment_config, tokenizer)
        assert processor.config is experiment_config

    def test_processor_has_process_method(self, experiment_config, tokenizer):
        """DataProcessor should have a process method."""
        from jenga_nlp.data.processor import DataProcessor
        processor = DataProcessor(experiment_config, tokenizer)
        assert hasattr(processor, "process")
        assert callable(processor.process)
