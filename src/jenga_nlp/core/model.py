"""Multi-task model for JengaNLP.
Architecture:
    Input → Shared Encoder → [Optional Fusion] → Task Head → Output
    
    The model processes one task at a time per forward pass, identified
    
    """

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .config import ExperimentConfig, FusionType, ModelConfig
from .fusion import create_fusion
from ..tasks.base import BaseTask, TaskOutput
from ..tasks.registry import TaskRegistry

logger = logging.getLogger(__name__)


class MultiTaskModel(nn.Module):
    """Multi-task model with shared encoder and task-specific heads.

    Architecture:
        Input → Shared Encoder → [Optional Fusion] → Task Head → Output

    The model processes one task at a time per forward pass, identified
    by task_id. Each task has its own prediction head(s).

    Args:
        model_config: Model configuration.
        tasks: List of task instances.
    """

    def __init__(self, model_config: ModelConfig, tasks: list[BaseTask]) -> None:
        super().__init__()

        # Load shared encoder
        self.encoder_config = AutoConfig.from_pretrained(model_config.base_model)
        self.encoder = AutoModel.from_pretrained(model_config.base_model, config=self.encoder_config)

        # Auto-detect hidden size from encoder
        self.hidden_size = self.encoder_config.hidden_size
        if model_config.hidden_size != self.hidden_size:
            logger.info(
                f"Config hidden_size={model_config.hidden_size} overridden by encoder "
                f"hidden_size={self.hidden_size}"
            )

        # Task heads
        self.tasks = nn.ModuleList(tasks)

        # Fusion mechanism
        self.fusion: Optional[nn.Module] = None
        if model_config.fusion and model_config.fusion.type != FusionType.NONE:
            self.fusion = create_fusion(
                fusion_type=model_config.fusion.type,
                hidden_size=self.hidden_size,
                num_tasks=len(tasks),
                config=model_config.fusion,
            )

        # Freeze encoder layers if requested
        if model_config.freeze_encoder_layers > 0:
            self._freeze_encoder_layers(model_config.freeze_encoder_layers)

        # Enable gradient checkpointing if requested
        if model_config.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self._model_config = model_config

    def _freeze_encoder_layers(self, num_layers: int) -> None:
        """Freeze the first N layers of the encoder."""
        frozen = 0
        # Freeze embeddings
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            frozen += 1

        # Freeze encoder layers
        encoder_layers = None
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            encoder_layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer") and hasattr(self.encoder.transformer, "layer"):
            encoder_layers = self.encoder.transformer.layer

        if encoder_layers is not None:
            for i, layer in enumerate(encoder_layers):
                if frozen >= num_layers:
                    break
                for param in layer.parameters():
                    param.requires_grad = False
                frozen += 1

        logger.info(f"Froze {frozen} encoder layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor | dict[str, torch.Tensor]] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> TaskOutput:
        """Forward pass for a single task's batch.

        Args:
            input_ids: [batch_size, seq_len] input token IDs.
            attention_mask: [batch_size, seq_len] attention mask.
            task_id: Integer index of the task to run.
            labels: Ground truth labels (format depends on task type).
            token_type_ids: Optional [batch_size, seq_len] token type IDs.

        Returns:
            TaskOutput with loss and logits.
        """
        # Build encoder inputs
        encoder_inputs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Only pass token_type_ids if the model supports it
        if token_type_ids is not None and hasattr(self.encoder_config, "type_vocab_size"):
            if self.encoder_config.type_vocab_size > 0:
                encoder_inputs["token_type_ids"] = token_type_ids

        # Encode
        encoder_outputs = self.encoder(**encoder_inputs)
        sequence_output = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Apply fusion if available
        if self.fusion is not None:
            sequence_output = self.fusion(sequence_output, task_id)

        # Pool: use CLS token
        pooled_output = sequence_output[:, 0]  # [batch, hidden]

        # Run task-specific head
        task = self.tasks[task_id]
        task_output = task.get_forward_output(
            pooled_output=pooled_output,
            sequence_output=sequence_output,
            labels=labels,
            attention_mask=attention_mask,
        )

        return task_output

    def save(self, save_dir: str | Path) -> None:
        """Save the complete model to a directory.

        Saves:
        - Encoder weights
        - Task head weights
        - Fusion weights
        - Model metadata (task names, hidden_size, etc.)
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save full model state dict
        torch.save(self.state_dict(), save_path / "model.pt")

        # Save encoder config
        self.encoder_config.save_pretrained(save_path / "encoder_config")

        # Save metadata
        metadata = {
            "base_model": self._model_config.base_model,
            "hidden_size": self.hidden_size,
            "num_tasks": len(self.tasks),
            "task_names": [t.name for t in self.tasks],
            "task_types": [t.type for t in self.tasks],
            "has_fusion": self.fusion is not None,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def load(
        cls,
        load_dir: str | Path,
        experiment_config: ExperimentConfig,
        device: str = "cpu",
    ) -> "MultiTaskModel":
        """Load a saved model from a directory.

        Args:
            load_dir: Directory containing saved model.
            experiment_config: Experiment config for rebuilding task heads.
            device: Device to load model onto.

        Returns:
            Loaded MultiTaskModel instance.
        """
        load_path = Path(load_dir)

        # Read metadata
        with open(load_path / "metadata.json") as f:
            metadata = json.load(f)

        # Detect hidden_size from metadata
        hidden_size = metadata["hidden_size"]

        # Create tasks from config
        tasks = [
            TaskRegistry.create_task(task_config, hidden_size)
            for task_config in experiment_config.tasks
        ]

        # Create model
        model = cls(model_config=experiment_config.model, tasks=tasks)

        # Load state dict
        state_dict = torch.load(load_path / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        logger.info(f"Model loaded from {load_path}")
        return model

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "MultiTaskModel":
        """Create a new model from an experiment config.

        This is the primary way to create a model. It:
        1. Loads the encoder to detect hidden_size
        2. Creates task instances with the correct hidden_size
        3. Creates the model with fusion if configured

        Args:
            config: Complete experiment configuration.

        Returns:
            New MultiTaskModel instance.
        """
        # Auto-detect hidden size from encoder
        encoder_config = AutoConfig.from_pretrained(config.model.base_model)
        hidden_size = encoder_config.hidden_size

        # Create tasks with the correct hidden size
        tasks = [
            TaskRegistry.create_task(task_config, hidden_size)
            for task_config in config.tasks
        ]

        return cls(model_config=config.model, tasks=tasks)
