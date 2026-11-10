"""Fusion mechanisms for multi-task learning.
This module implements different strategies for fusing shared encoder outputs with task-specific information. The main classes are:
- AttentionFusion: Task embedding queries the tokens (masked attention); the task context is gated in.
- ConcatenationFusion: Concatenates shared output with task embedding and projects back to hidden size.
- NoFusion: A simple passthrough that returns the shared output unchanged.
The create_fusion function serves as a factory to instantiate the desired fusion mechanism based on configuration.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import FusionConfig, FusionType

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """Task-query attention fusion.

    Each task owns a learned embedding that acts as an attention *query* over
    the encoder's token representations (padding is masked out). The resulting
    task-specific context vector is projected and added to every token through
    a learnable gate, then layer-normalised::

        ctx_t = MultiHeadAttention(query=e_t, keys=values=tokens, mask=padding)
        out   = LayerNorm(tokens + sigmoid(gate) * W(ctx_t))

    Because the task contribution is an additive, input-dependent vector, it is
    not cancelled by the LayerNorm, so pooled (CLS) and token features differ
    per task. ``W`` is zero-initialised, so the module starts as
    ``LayerNorm(tokens)`` and learns how much task conditioning to add.

    Args:
        hidden_size: Hidden dimension of the encoder.
        num_tasks: Number of tasks.
        config: Fusion configuration (``num_attention_heads`` must divide ``hidden_size``).
    """

    def __init__(self, hidden_size: int, num_tasks: int, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.config = config or FusionConfig()

        num_heads = self.config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"fusion.num_attention_heads={num_heads} must divide the encoder hidden size {hidden_size}"
            )

        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)
        nn.init.normal_(self.task_embeddings.weight, std=0.02)

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=self.config.dropout, batch_first=True
        )
        self.context_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

        self.dropout = nn.Dropout(self.config.dropout)

        # The residual is always used (FusionConfig rejects use_residual: false for attention).
        # Store the gate as a logit so sigmoid(gate) starts exactly at gate_init_value.
        init = min(max(self.config.gate_init_value, 1e-4), 1 - 1e-4)
        self.gate = nn.Parameter(torch.logit(torch.tensor(init)))

        self.layer_norm = nn.LayerNorm(hidden_size)
        # Diagnostics only: set to True to keep the last attention weights in
        # ``last_attention_weights`` (computing them disables PyTorch's fast path).
        self.store_attention_weights = False
        self.last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        shared_output: torch.Tensor,
        task_id: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse shared token representations with task-specific attention context.

        Args:
            shared_output: [batch_size, seq_len, hidden_size] from the encoder.
            task_id: Integer ID of the current task.
            attention_mask: Optional [batch_size, seq_len], 1 for real tokens, 0 for padding.

        Returns:
            [batch_size, seq_len, hidden_size] task-conditioned representation.
        """
        batch_size = shared_output.shape[0]
        task_idx = torch.tensor([task_id], dtype=torch.long, device=shared_output.device)
        query = self.task_embeddings(task_idx).unsqueeze(0).expand(batch_size, 1, -1)  # [B, 1, H]

        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        context, weights = self.attention(
            query,
            shared_output,
            shared_output,
            key_padding_mask=key_padding_mask,
            need_weights=self.store_attention_weights,
        )  # context: [B, 1, H]; weights: [B, 1, seq_len] or None
        if self.store_attention_weights and weights is not None:
            self.last_attention_weights = weights.detach()

        task_vector = torch.sigmoid(self.gate) * self.dropout(self.context_proj(context))  # [B, 1, H]

        return self.layer_norm(shared_output + task_vector)


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion.

    Concatenates shared output with task embedding and projects back
    to hidden_size. Simpler and faster than attention fusion.

    Args:
        hidden_size: Hidden dimension of the encoder.
        num_tasks: Number of tasks.
        config: Fusion configuration.
    """

    def __init__(self, hidden_size: int, num_tasks: int, config: FusionConfig | None = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config or FusionConfig()

        self.task_embeddings = nn.Embedding(num_tasks, hidden_size)
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(self.config.dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, shared_output: torch.Tensor, task_id: int, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, shared_output: torch.Tensor, task_id: int, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return shared_output


def create_fusion(
    fusion_type: FusionType,
    hidden_size: int,
    num_tasks: int,
    config: FusionConfig | None = None,
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
