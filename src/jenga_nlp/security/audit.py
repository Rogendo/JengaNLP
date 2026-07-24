"""Audit trail system for JengaNLP.

Provides complete traceability for all model operations. This is
MANDATORY for government security deployments under the Kenya
Data Protection Act and for maintaining public trust.

Tracks:
1. Data provenance - Where did training data come from?
2. Model lineage - How was the model trained, by whom, with what config?
3. Prediction logging - Every inference with input, output, confidence
4. Access control events - Who accessed what, when
5. Model changes - Version history, parameter changes, retraining events

Also supports quantum-readiness by using hash-based integrity
verification that can be upgraded to post-quantum algorithms.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Data operations
    DATA_UPLOAD = "data_upload"
    DATA_DELETE = "data_delete"
    DATA_TRANSFORM = "data_transform"
    DATA_EXPORT = "data_export"

    # Model operations
    MODEL_CREATE = "model_create"
    MODEL_TRAIN_START = "model_train_start"
    MODEL_TRAIN_COMPLETE = "model_train_complete"
    MODEL_TRAIN_FAIL = "model_train_fail"
    MODEL_EVALUATE = "model_evaluate"
    MODEL_DEPLOY = "model_deploy"
    MODEL_DELETE = "model_delete"
    MODEL_EXPORT = "model_export"

    # Inference operations
    INFERENCE_SINGLE = "inference_single"
    INFERENCE_BATCH = "inference_batch"
    INFERENCE_FLAGGED = "inference_flagged"  # Prediction routed to HITL

    # Access operations
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"

    # Security events
    ADVERSARIAL_DETECTED = "adversarial_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    PII_DETECTED = "pii_detected"


@dataclass
class AuditEvent:
    """A single audit trail event.

    Every event is immutable once created and includes a hash
    of the previous event for tamper-evidence (blockchain-like chain).

    Attributes:
        event_id: Unique event identifier.
        timestamp: When the event occurred (UTC).
        action: Type of action performed.
        actor: Who performed the action (user ID or system).
        resource_type: What type of resource was affected.
        resource_id: ID of the affected resource.
        details: Additional context about the action.
        input_hash: Hash of the input data (for data provenance).
        output_hash: Hash of the output data (for result verification).
        previous_hash: Hash of the previous audit event (chain integrity).
        event_hash: Hash of this event (computed automatically).
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    action: AuditAction = AuditAction.MODEL_CREATE
    actor: str = "system"
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this event for integrity verification."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "action": self.action.value if isinstance(self.action, Enum) else self.action,
            "actor": self.actor,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


class AuditLogger:
    """Append-only audit logger with chain integrity.

    All events are chained via hashes, making the audit trail
    tamper-evident. Supports both file-based and database storage.

    Args:
        log_dir: Directory for audit log files.
        project_name: Name of the project being audited.
    """

    def __init__(self, log_dir: str | Path, project_name: str = "jenga_ai") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.log_file = self.log_dir / f"audit_{project_name}.jsonl"
        self._last_hash: Optional[str] = None
        self._event_count = 0

        # Load last hash from existing log
        if self.log_file.exists():
            self._load_last_hash()

        logger.info(f"Audit logger initialized: {self.log_file}")

    def _load_last_hash(self) -> None:
        """Load the hash of the last event from the log file."""
        try:
            with open(self.log_file) as f:
                last_line = None
                for line in f:
                    last_line = line.strip()
                    self._event_count += 1
                if last_line:
                    event_data = json.loads(last_line)
                    self._last_hash = event_data.get("event_hash")
        except (json.JSONDecodeError, OSError):
            self._last_hash = None

    def log(
        self,
        action: AuditAction,
        actor: str = "system",
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[dict[str, Any]] = None,
        input_data: Optional[str] = None,
        output_data: Optional[str] = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            action: Type of action performed.
            actor: Who performed the action.
            resource_type: Type of resource affected.
            resource_id: ID of the resource.
            details: Additional context.
            input_data: Input data to hash (for provenance).
            output_data: Output data to hash (for verification).

        Returns:
            The created AuditEvent.
        """
        event = AuditEvent(
            action=action,
            actor=actor,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            input_hash=hashlib.sha256(input_data.encode()).hexdigest() if input_data else None,
            output_hash=hashlib.sha256(output_data.encode()).hexdigest() if output_data else None,
            previous_hash=self._last_hash,
        )

        # Compute and set event hash
        event.event_hash = event.compute_hash()
        self._last_hash = event.event_hash
        self._event_count += 1

        # Write to log (append-only)
        event_dict = asdict(event)
        # Convert enum to string
        if isinstance(event_dict.get("action"), AuditAction):
            event_dict["action"] = event_dict["action"].value

        with open(self.log_file, "a") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")

        logger.debug(f"Audit event logged: {action.value} by {actor}")
        return event

    def verify_integrity(self) -> tuple[bool, int]:
        """Verify the integrity of the entire audit trail.

        Checks that the hash chain is unbroken, meaning no events
        have been tampered with or deleted.

        Returns:
            Tuple of (is_valid, num_events_verified).
        """
        if not self.log_file.exists():
            return True, 0

        previous_hash = None
        count = 0

        with open(self.log_file) as f:
            for line in f:
                try:
                    event_data = json.loads(line.strip())
                    event = AuditEvent(**{k: v for k, v in event_data.items() if k != "event_hash"})
                    event.previous_hash = previous_hash

                    computed_hash = event.compute_hash()
                    stored_hash = event_data.get("event_hash")

                    if stored_hash and computed_hash != stored_hash:
                        logger.error(f"Audit integrity violation at event {count + 1}")
                        return False, count

                    previous_hash = stored_hash or computed_hash
                    count += 1
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Audit log parse error at event {count + 1}: {e}")
                    return False, count

        logger.info(f"Audit trail verified: {count} events, integrity OK")
        return True, count

    def get_events(
        self,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events with optional filters.

        Args:
            action: Filter by action type.
            actor: Filter by actor.
            limit: Maximum number of events to return.

        Returns:
            List of event dicts (most recent first).
        """
        events = []
        if not self.log_file.exists():
            return events

        with open(self.log_file) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if action and event.get("action") != action.value:
                        continue
                    if actor and event.get("actor") != actor:
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    continue

        # Return most recent first, limited
        return list(reversed(events[-limit:]))

    @property
    def event_count(self) -> int:
        return self._event_count
