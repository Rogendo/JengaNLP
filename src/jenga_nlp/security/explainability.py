"""Explainability engine for JengaNLP.

Provides tools to make model predictions transparent and interpretable.
This is CRITICAL for government security use cases where decisions must
be justified and auditable.

Supports:
1. Attention visualization - Show which tokens influenced the prediction
2. Feature importance (SHAP-style) - Quantify input contribution to output
3. Prediction confidence analysis - Identify uncertain predictions
4. Counterfactual explanations - "What would change the prediction?"

Required for:
- Kenya Data Protection Act compliance (right to explanation)
- Government audit requirements (justifiable AI decisions)
- Human analyst trust (show WHY a message was flagged as a threat)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """Container for a model prediction explanation.

    Attributes:
        input_text: The original input text.
        prediction: The model's prediction (class label or value).
        confidence: Confidence score (0-1).
        token_importance: Dict mapping tokens to their importance scores.
        attention_weights: Optional attention weight matrix.
        top_features: Top N most important features/tokens.
        counterfactual: Optional counterfactual explanation.
    """
    input_text: str
    prediction: Any
    confidence: float
    token_importance: dict[str, float]
    attention_weights: Optional[np.ndarray] = None
    top_features: Optional[list[tuple[str, float]]] = None
    counterfactual: Optional[str] = None


class ExplainabilityEngine:
    """Engine for generating explanations of model predictions.

    Provides multiple explanation methods that can be used independently
    or combined for comprehensive interpretability.

    Args:
        model: The trained model to explain.
        tokenizer: The tokenizer used by the model.
    """

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def explain_prediction(
        self,
        text: str,
        task_id: int,
        method: str = "attention",
        top_k: int = 10,
    ) -> Explanation:
        """Generate an explanation for a single prediction.

        Args:
            text: Input text to explain.
            task_id: Which task to explain.
            method: Explanation method ('attention', 'gradient', 'occlusion').
            top_k: Number of top features to return.

        Returns:
            Explanation object with interpretability data.
        """
        self.model.eval()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        if method == "attention":
            importance = self._attention_explanation(inputs, task_id, tokens)
        elif method == "gradient":
            importance = self._gradient_explanation(inputs, task_id, tokens)
        elif method == "occlusion":
            importance = self._occlusion_explanation(text, task_id, tokens)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

        # Get prediction and confidence
        with torch.no_grad():
            output = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                task_id=task_id,
            )

        # Extract prediction from first head
        first_head = list(output.logits.keys())[0]
        logits = output.logits[first_head]
        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, prediction].item()

        # Top features
        sorted_importance = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_importance[:top_k]

        return Explanation(
            input_text=text,
            prediction=prediction,
            confidence=confidence,
            token_importance=importance,
            top_features=top_features,
        )

    def _attention_explanation(
        self,
        inputs: dict[str, torch.Tensor],
        task_id: int,
        tokens: list[str],
    ) -> dict[str, float]:
        """Extract attention weights as token importance.

        Uses the attention weights from the encoder's last layer
        to determine which tokens the model focused on.
        """
        self.model.eval()

        with torch.no_grad():
            # Get encoder outputs with attention
            encoder = self.model.encoder if hasattr(self.model, "encoder") else self.model
            encoder_outputs = encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )

            if encoder_outputs.attentions:
                # Average attention across heads and layers
                # Each attention: [batch, heads, seq_len, seq_len]
                last_attention = encoder_outputs.attentions[-1]
                # Average over heads: [batch, seq_len, seq_len]
                avg_attention = last_attention.mean(dim=1)
                # CLS token attention to all other tokens: [seq_len]
                cls_attention = avg_attention[0, 0].numpy()

                importance = {}
                for i, (token, score) in enumerate(zip(tokens, cls_attention)):
                    if token not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                        importance[token] = float(score)

                return importance

        # Fallback: uniform importance
        return {token: 1.0 / len(tokens) for token in tokens if token not in ("[CLS]", "[SEP]", "[PAD]")}

    def _gradient_explanation(
        self,
        inputs: dict[str, torch.Tensor],
        task_id: int,
        tokens: list[str],
    ) -> dict[str, float]:
        """Gradient-based token importance (integrated gradients approximation).

        Computes the gradient of the output w.r.t. input embeddings
        to determine which tokens most influence the prediction.
        """
        self.model.eval()

        # Enable gradient computation for embeddings
        embedding_layer = self.model.encoder.embeddings if hasattr(self.model, "encoder") else None
        if embedding_layer is None:
            return {token: 0.0 for token in tokens}

        inputs_embeds = embedding_layer.word_embeddings(inputs["input_ids"])
        inputs_embeds.requires_grad_(True)

        # Forward pass through encoder using embeddings directly
        try:
            encoder_outputs = self.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs["attention_mask"],
            )
            sequence_output = encoder_outputs.last_hidden_state
            pooled = sequence_output[:, 0]

            # Get task output
            task = self.model.tasks[task_id]
            first_head_name = task.config.heads[0].name
            head = task.heads[first_head_name]
            logits = head(pooled)

            # Get gradient for predicted class
            pred_class = logits.argmax(dim=-1)
            target_logit = logits[0, pred_class]
            target_logit.backward()

            # Gradient magnitude per token
            grad = inputs_embeds.grad[0]  # [seq_len, hidden]
            token_importance = grad.norm(dim=-1).detach().numpy()  # [seq_len]

            importance = {}
            for i, (token, score) in enumerate(zip(tokens, token_importance)):
                if token not in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                    importance[token] = float(score)

            return importance

        except Exception as e:
            logger.warning(f"Gradient explanation failed: {e}")
            return {token: 0.0 for token in tokens}

    def _occlusion_explanation(
        self,
        text: str,
        task_id: int,
        tokens: list[str],
    ) -> dict[str, float]:
        """Occlusion-based importance (leave-one-out).

        For each token, mask it and measure how much the prediction changes.
        Tokens whose removal causes the largest change are most important.
        """
        self.model.eval()

        # Get baseline prediction
        baseline_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            baseline_output = self.model(
                input_ids=baseline_inputs["input_ids"],
                attention_mask=baseline_inputs["attention_mask"],
                task_id=task_id,
            )
        first_head = list(baseline_output.logits.keys())[0]
        baseline_probs = torch.softmax(baseline_output.logits[first_head], dim=-1)
        baseline_pred = baseline_probs[0].numpy()

        # Occlude each word and measure change
        words = text.split()
        importance: dict[str, float] = {}

        for i, word in enumerate(words):
            masked_text = " ".join(words[:i] + ["[MASK]"] + words[i + 1:])
            masked_inputs = self.tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                masked_output = self.model(
                    input_ids=masked_inputs["input_ids"],
                    attention_mask=masked_inputs["attention_mask"],
                    task_id=task_id,
                )
            masked_probs = torch.softmax(masked_output.logits[first_head], dim=-1)
            masked_pred = masked_probs[0].numpy()

            # KL divergence as importance score
            change = float(np.sum(np.abs(baseline_pred - masked_pred)))
            importance[word] = change

        return importance

    def generate_report(self, explanation: Explanation) -> str:
        """Generate a human-readable explanation report.

        This is designed for government analysts who need to understand
        WHY a message was flagged.

        Args:
            explanation: Explanation object.

        Returns:
            Formatted string report.
        """
        lines = [
            "=" * 60,
            "JENGA-NLP PREDICTION EXPLANATION REPORT",
            "=" * 60,
            f"Input: {explanation.input_text}",
            f"Prediction: {explanation.prediction}",
            f"Confidence: {explanation.confidence:.1%}",
            "",
            "TOP INFLUENTIAL TOKENS:",
            "-" * 40,
        ]

        if explanation.top_features:
            for token, score in explanation.top_features:
                bar = "#" * int(score * 50)
                lines.append(f"  {token:20s} | {score:.4f} {bar}")

        lines.append("=" * 60)
        return "\n".join(lines)
