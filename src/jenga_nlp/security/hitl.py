"""Human-in-the-Loop (HITL) module for JengaNLP.

Provides tools for routing uncertain or high-stakes predictions to
human analysts for review. This is essential for:

1. National security decisions - No fully automated threat classification
2. Building trust -  analysts can verify and override AI
3. Active learning - Human corrections improve the model over time
4. Compliance - Kenya Data Protection Act requires human oversight for
   automated decisions that significantly affect individuals

Architecture:
    Model Prediction → Uncertainty Check → Route Decision
                                              ↓
                            High Confidence → Auto-accept (with audit log)
                            Low Confidence  → Queue for human review
                            High Stakes     → Always route to human
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ReviewPriority(str, Enum):
    """Priority levels for human review queue."""
    CRITICAL = "critical"      # Must review immediately (threat detected)
    HIGH = "high"              # Review within hours
    MEDIUM = "medium"          # Review within a day
    LOW = "low"                # Review when available


class RoutingDecision(str, Enum):
    """Decision on how to handle a prediction."""
    AUTO_ACCEPT = "auto_accept"          # Confident, auto-process
    HUMAN_REVIEW = "human_review"        # Route to analyst
    HUMAN_REQUIRED = "human_required"    # Always requires human (policy)
    REJECTED = "rejected"                # Below minimum confidence


@dataclass
class ReviewItem:
    """An item queued for human review.

    Attributes:
        item_id: Unique identifier.
        input_text: The input that generated the prediction.
        prediction: The model's prediction.
        confidence: Model confidence (0-1).
        uncertainty: Uncertainty score (higher = less certain).
        priority: Review priority.
        task_name: Which task generated this.
        explanation: Optional explanation of the prediction.
        reviewer: Who reviewed it (filled after review).
        review_decision: Human's decision (filled after review).
        review_notes: Human's notes (filled after review).
    """
    item_id: str = ""
    input_text: str = ""
    prediction: Any = None
    confidence: float = 0.0
    uncertainty: float = 0.0
    priority: ReviewPriority = ReviewPriority.MEDIUM
    task_name: str = ""
    explanation: Optional[dict[str, Any]] = None
    reviewer: Optional[str] = None
    review_decision: Optional[str] = None
    review_notes: Optional[str] = None


class UncertaintyEstimator:
    """Estimate prediction uncertainty using multiple methods.

    Methods:
    1. Softmax entropy - Higher entropy = more uncertain
    2. Margin-based - Small margin between top-2 classes = uncertain
    3. MC Dropout - Enable dropout at inference for uncertainty

    Args:
        method: Uncertainty estimation method.
        threshold: Confidence threshold below which predictions are flagged.
    """

    def __init__(
        self,
        method: str = "entropy",
        confidence_threshold: float = 0.8,
        mc_samples: int = 10,
    ) -> None:
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.mc_samples = mc_samples

    def estimate(self, logits: torch.Tensor) -> tuple[float, float]:
        """Estimate uncertainty from model logits.

        Args:
            logits: [num_classes] raw model outputs.

        Returns:
            Tuple of (confidence, uncertainty).
        """
        probs = torch.softmax(logits, dim=-1)

        if self.method == "entropy":
            return self._entropy_uncertainty(probs)
        elif self.method == "margin":
            return self._margin_uncertainty(probs)
        else:
            return self._entropy_uncertainty(probs)

    def _entropy_uncertainty(self, probs: torch.Tensor) -> tuple[float, float]:
        """Compute entropy-based uncertainty.

        Low entropy = confident (one class dominates)
        High entropy = uncertain (uniform distribution)
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = np.log(probs.shape[-1])
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        confidence = 1.0 - normalized_entropy
        uncertainty = normalized_entropy
        return confidence, uncertainty

    def _margin_uncertainty(self, probs: torch.Tensor) -> tuple[float, float]:
        """Compute margin-based uncertainty.

        Small margin between top-2 predictions = uncertain.
        """
        sorted_probs, _ = torch.sort(probs, descending=True)
        if sorted_probs.shape[-1] >= 2:
            margin = (sorted_probs[..., 0] - sorted_probs[..., 1]).item()
        else:
            margin = sorted_probs[..., 0].item()

        confidence = float(sorted_probs[..., 0].item())
        uncertainty = 1.0 - margin
        return confidence, uncertainty

    def is_uncertain(self, logits: torch.Tensor) -> bool:
        """Check if a prediction should be flagged for review."""
        confidence, _ = self.estimate(logits)
        return confidence < self.confidence_threshold


class HITLRouter:
    """Routes predictions to auto-accept or human review based on
    confidence, uncertainty, and policy rules.

    Supports:
    - Confidence-based routing (below threshold → human)
    - Policy-based routing (certain task types always require human)
    - Priority assignment based on prediction content
    - Review queue management

    Args:
        uncertainty_estimator: UncertaintyEstimator instance.
        always_review_tasks: Task names that always require human review.
        critical_labels: Label names/IDs that trigger critical priority.
    """

    def __init__(
        self,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        always_review_tasks: Optional[list[str]] = None,
        critical_labels: Optional[list[Any]] = None,
    ) -> None:
        self.estimator = uncertainty_estimator or UncertaintyEstimator()
        self.always_review_tasks = set(always_review_tasks or [])
        self.critical_labels = set(critical_labels or [])
        self.review_queue: list[ReviewItem] = []

    def route(
        self,
        logits: torch.Tensor,
        task_name: str,
        input_text: str = "",
    ) -> tuple[RoutingDecision, ReviewItem]:
        """Decide how to handle a prediction.

        Args:
            logits: Model output logits.
            task_name: Name of the task.
            input_text: Original input text.

        Returns:
            Tuple of (routing_decision, review_item).
        """
        confidence, uncertainty = self.estimator.estimate(logits)

        probs = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()

        # Determine priority
        if prediction in self.critical_labels:
            priority = ReviewPriority.CRITICAL
        elif uncertainty > 0.5:
            priority = ReviewPriority.HIGH
        elif uncertainty > 0.3:
            priority = ReviewPriority.MEDIUM
        else:
            priority = ReviewPriority.LOW

        item = ReviewItem(
            input_text=input_text,
            prediction=prediction,
            confidence=confidence,
            uncertainty=uncertainty,
            priority=priority,
            task_name=task_name,
        )

        # Routing decision
        if task_name in self.always_review_tasks:
            decision = RoutingDecision.HUMAN_REQUIRED
        elif self.estimator.is_uncertain(logits):
            decision = RoutingDecision.HUMAN_REVIEW
        elif prediction in self.critical_labels:
            decision = RoutingDecision.HUMAN_REVIEW
        else:
            decision = RoutingDecision.AUTO_ACCEPT

        if decision != RoutingDecision.AUTO_ACCEPT:
            self.review_queue.append(item)
            logger.info(
                f"Prediction routed to human review: task={task_name}, "
                f"confidence={confidence:.2f}, priority={priority.value}"
            )

        return decision, item

    def get_pending_reviews(
        self,
        priority: Optional[ReviewPriority] = None,
        limit: int = 50,
    ) -> list[ReviewItem]:
        """Get items pending human review.

        Args:
            priority: Filter by priority level.
            limit: Maximum items to return.

        Returns:
            List of ReviewItems.
        """
        items = self.review_queue
        if priority:
            items = [i for i in items if i.priority == priority]

        # Sort by priority (critical first)
        priority_order = {
            ReviewPriority.CRITICAL: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3,
        }
        items.sort(key=lambda x: priority_order.get(x.priority, 99))

        return items[:limit]

    def submit_review(
        self,
        item_id: str,
        reviewer: str,
        decision: str,
        notes: str = "",
    ) -> bool:
        """Submit a human review decision.

        Args:
            item_id: ID of the review item.
            reviewer: Who reviewed it.
            decision: The human's decision.
            notes: Additional notes.

        Returns:
            True if the review was submitted successfully.
        """
        for item in self.review_queue:
            if item.item_id == item_id:
                item.reviewer = reviewer
                item.review_decision = decision
                item.review_notes = notes
                logger.info(f"Review submitted by {reviewer}: {decision}")
                return True

        logger.warning(f"Review item not found: {item_id}")
        return False
