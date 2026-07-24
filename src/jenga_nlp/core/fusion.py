"""Fusion mechanisms for multi-task learning.
This module implements different strategies for fusing shared encoder outputs with task-specific information. The main classes are:
- AttentionFusion: Uses an attention mechanism to combine shared and task-specific representations.
- ConcatenationFusion: Concatenates shared output with task embedding and projects back to hidden size.
- NoFusion: A simple passthrough that returns the shared output unchanged.
The create_fusion function serves as a factory to instantiate the desired fusion mechanism based on configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionConfig, FusionType

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for task-specific representations.

    Fuses the shared encoder output with a learned task embedding using
    an attention mechanism. Includes residual connection and gating.

    Args:
        hidden_size: Hidden dimension of the encoder.
        num_tasks: Number of tasks.
        config: Fusion configuration.
    """

    def __init__(self, hidden_size: int, num_tasks: int, config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.config = config or FusionConfig()

        # Task embeddings - cached, no tensor creation each forward pass
        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.dropout)

        # Learnable gate for balancing shared vs task-specific
        if self.config.use_residual:
            gate_value = self.config.gate_init_value
            self.gate = nn.Parameter(torch.tensor(gate_value))

        # Layer norm for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, shared_output: torch.Tensor, task_id: int) -> torch.Tensor:
        """Fuse shared representation with task-specific embedding.

        Args:
            shared_output: [batch_size, seq_len, hidden_size] from encoder.
            task_id: Integer ID of the current task.

        Returns:
            [batch_size, seq_len, hidden_size] fused representation.
        """
        batch_size, seq_len, _ = shared_output.shape

        # Get task embedding - use LongTensor on the correct device
        task_idx = torch.tensor([task_id], dtype=torch.long, device=shared_output.device)
        task_emb = self.task_embeddings(task_idx)  # [1, hidden_size]

        # Expand to match sequence: [batch_size, seq_len, hidden_size]
        task_emb_expanded = task_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        # Concatenate shared + task embedding: [batch_size, seq_len, hidden_size * 2]
        combined = torch.cat([shared_output, task_emb_expanded], dim=-1)

        # Compute attention weights: [batch_size, seq_len, 1]
        attn_scores = self.attention(combined)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Apply attention: [batch_size, seq_len, hidden_size]
        attended = shared_output * attn_weights
        attended = self.dropout(attended)

        # Residual connection with learnable gate
        if self.config.use_residual:
            gate = torch.sigmoid(self.gate)
            output = gate * attended + (1 - gate) * shared_output
        else:
            output = attended

        # Layer normalization
        output = self.layer_norm(output)

        return output


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion.

    Concatenates shared output with task embedding and projects back
    to hidden_size. Simpler and faster than attention fusion.

    Args:
        hidden_size: Hidden dimension of the encoder.
        num_tasks: Number of tasks.
        config: Fusion configuration.
    """

    def __init__(self, hidden_size: int, num_tasks: int, config: Optional[FusionConfig] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config or FusionConfig()

        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, shared_output: torch.Tensor, task_id: int) -> torch.Tensor:
        batch_size, seq_len, _ = shared_output.shape

        task_idx = torch.tensor([task_id], dtype=torch.long, device=shared_output.device)
        task_emb = self.task_embeddings(task_idx)
        task_emb_expanded = task_emb.unsqueeze(1).expand(batch_size, seq_len, -1)

        combined = torch.cat([shared_output, task_emb_expanded], dim=-1)
        output = self.projection(combined)
        output = self.dropout(output)

        if self.config.use_residual:
            output = output + shared_output

        output = self.layer_norm(output)
        return output


class NoFusion(nn.Module):
    """Passthrough fusion (no fusion applied).

    Simply returns the shared output unchanged. Useful as a baseline.
    """

    def forward(self, shared_output: torch.Tensor, task_id: int) -> torch.Tensor:
        return shared_output


def create_fusion(
    fusion_type: FusionType,
    hidden_size: int,
    num_tasks: int,
    config: Optional[FusionConfig] = None,
) -> nn.Module:
    """Factory function to create the appropriate fusion mechanism.

    Args:
        fusion_type: Type of fusion to create.
        hidden_size: Hidden size of the encoder.
        num_tasks: Number of tasks.
        config: Optional fusion configuration.

    Returns:
        Fusion module instance.
    """
    if fusion_type == FusionType.ATTENTION:
        logger.info(f"Creating AttentionFusion (hidden={hidden_size}, tasks={num_tasks})")
        return AttentionFusion(hidden_size, num_tasks, config)
    elif fusion_type == FusionType.CONCATENATION:
        logger.info(f"Creating ConcatenationFusion (hidden={hidden_size}, tasks={num_tasks})")
        return ConcatenationFusion(hidden_size, num_tasks, config)
    elif fusion_type == FusionType.NONE:
        logger.info("Creating NoFusion (passthrough)")
        return NoFusion()
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
