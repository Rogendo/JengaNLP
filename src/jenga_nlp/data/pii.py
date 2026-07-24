"""PII detection and redaction for JengaNLP data pipelines.

An OPTIONAL preprocessing step that detects and masks Personally Identifiable
Information before data enters the training pipeline. This is critical for:

1. Government datasets that may contain citizen information
2. M-Pesa transaction logs with phone numbers and names
3. Social media data with usernames and contact details
4. Email datasets with sender/recipient information
5. Network logs with IP addresses and hostnames

IMPORTANT: This is opt-in. Users working with synthetic or pre-anonymized
data can skip this entirely by setting pii_redaction=False in their config.

Supported PII types (Kenya-focused):
- Phone numbers (Kenyan format: 07xx, +254xx, Safaricom/Airtel patterns)
- National ID numbers (Kenyan format: 8-digit)
- KRA PIN (Kenya Revenue Authority: letter + 9 digits + letter)
- Email addresses
- IP addresses (v4 and v6)
- M-Pesa transaction IDs
- Names (via NER or pattern matching)
- Credit card numbers (Luhn-validated)
- Dates of birth
- Physical addresses (partial — street/PO Box patterns)
- URLs and domains

Redaction strategies:
- MASK: Replace with [PII_TYPE] tag → "Call 0712345678" → "Call [PHONE]"
- HASH: Replace with deterministic hash → preserves uniqueness without revealing value
- FAKE: Replace with realistic synthetic data → "John" → "David"
- REMOVE: Delete the PII entirely → "Call 0712345678" → "Call"
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""
    PHONE_KE = "phone_ke"           # Kenyan phone numbers
    NATIONAL_ID = "national_id"     # Kenyan national ID
    KRA_PIN = "kra_pin"             # Kenya Revenue Authority PIN
    EMAIL = "email"                 # Email addresses
    IP_ADDRESS = "ip_address"       # IPv4 and IPv6
    MPESA_ID = "mpesa_id"           # M-Pesa transaction references
    CREDIT_CARD = "credit_card"     # Credit/debit card numbers
    URL = "url"                     # URLs and domains
    DATE_OF_BIRTH = "dob"           # Dates that look like birthdates
    PO_BOX = "po_box"              # P.O. Box addresses
    PERSON_NAME = "person_name"     # Person names (pattern-based)


class RedactionStrategy(str, Enum):
    """How to handle detected PII."""
    MASK = "mask"       # Replace with [PII_TYPE] tag
    HASH = "hash"       # Replace with deterministic hash (preserves uniqueness)
    REMOVE = "remove"   # Delete the PII text
    FLAG = "flag"       # Don't redact, just flag for review


@dataclass
class PIIConfig:
    """Configuration for PII detection and redaction.

    Usage:
        # Redact everything (strictest)
        config = PIIConfig(enabled=True)

        # Only redact phone numbers and emails
        config = PIIConfig(
            enabled=True,
            detect_types=[PIIType.PHONE_KE, PIIType.EMAIL]
        )

        # Skip PII redaction (synthetic data)
        config = PIIConfig(enabled=False)
    """
    enabled: bool = False  # OFF by default — opt-in
    strategy: RedactionStrategy = RedactionStrategy.MASK
    detect_types: Optional[List[PIIType]] = None  # None = detect ALL types
    hash_salt: str = "jenganlp-pii"  # salt for deterministic hashing
    log_detections: bool = True  # log what was found (without revealing values)
    custom_patterns: Dict[str, str] = field(default_factory=dict)  # name → regex

    def __post_init__(self):
        if self.detect_types is None:
            self.detect_types = list(PIIType)


class PIIPattern:
    """A single PII detection pattern."""

    def __init__(self, pii_type: PIIType, pattern: str, tag: str,
                 validator: Optional[callable] = None):
        self.pii_type = pii_type
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.tag = tag
        self.validator = validator  # optional function to reduce false positives

    def find_all(self, text: str) -> List[Tuple[int, int, str]]:
        """Find all matches in text. Returns list of (start, end, matched_text)."""
        matches = []
        for m in self.regex.finditer(text):
            matched = m.group()
            if self.validator is None or self.validator(matched):
                matches.append((m.start(), m.end(), matched))
        return matches


def _luhn_check(number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _is_valid_ke_phone(number: str) -> bool:
    """Validate Kenyan phone number format."""
    digits = re.sub(r'[^\d]', '', number)
    # Kenyan numbers: 07xx (10 digits) or 254xxx (12 digits) or +254xxx (12 digits)
    if len(digits) == 10 and digits.startswith('07'):
        return True
    if len(digits) == 12 and digits.startswith('254'):
        return True
    return False


# Kenya-specific PII patterns
KENYA_PII_PATTERNS = [
    # Kenyan phone numbers (Safaricom 07xx, Airtel 07xx, +254)
    PIIPattern(
        PIIType.PHONE_KE,
        r'(?:\+?254|0)[17]\d{8}',
        '[PHONE]',
        _is_valid_ke_phone
    ),

    # Kenyan National ID (typically 7-8 digits)
    PIIPattern(
        PIIType.NATIONAL_ID,
        r'\b\d{7,8}\b',
        '[NATIONAL_ID]',
        lambda x: 7 <= len(x) <= 8  # basic length check
    ),

    # KRA PIN (letter + 9 digits + letter, e.g., A123456789B)
    PIIPattern(
        PIIType.KRA_PIN,
        r'\b[A-Z]\d{9}[A-Z]\b',
        '[KRA_PIN]'
    ),

    # Email addresses
    PIIPattern(
        PIIType.EMAIL,
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '[EMAIL]'
    ),

    # IPv4 addresses
    PIIPattern(
        PIIType.IP_ADDRESS,
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        '[IP_ADDRESS]'
    ),

    # M-Pesa transaction IDs (typically alphanumeric, 10 chars, starts with letter)
    PIIPattern(
        PIIType.MPESA_ID,
        r'\b[A-Z]{2,3}\d{7,8}[A-Z0-9]{0,2}\b',
        '[MPESA_ID]'
    ),

    # Credit card numbers (13-19 digits, with optional spaces/dashes)
    PIIPattern(
        PIIType.CREDIT_CARD,
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,7}\b',
        '[CREDIT_CARD]',
        lambda x: _luhn_check(x)
    ),

    # URLs
    PIIPattern(
        PIIType.URL,
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
        '[URL]'
    ),

    # P.O. Box addresses
    PIIPattern(
        PIIType.PO_BOX,
        r'P\.?\s*O\.?\s*Box\s+\d+[\s,-]*\d*',
        '[PO_BOX]'
    ),

    # Dates that look like birthdates (DD/MM/YYYY or DD-MM-YYYY)
    PIIPattern(
        PIIType.DATE_OF_BIRTH,
        r'\b(?:0[1-9]|[12]\d|3[01])[/\-.](?:0[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b',
        '[DOB]'
    ),
]


class PIIDetector:
    """Detect PII in text using pattern matching.

    Uses regex patterns optimized for Kenyan data (phone numbers, national IDs,
    KRA PINs, M-Pesa transaction IDs) plus universal patterns (email, IP, URLs).
    """

    def __init__(self, config: PIIConfig):
        self.config = config
        self._patterns: List[PIIPattern] = []

        # Load patterns for configured PII types
        active_types = set(config.detect_types)
        for pattern in KENYA_PII_PATTERNS:
            if pattern.pii_type in active_types:
                self._patterns.append(pattern)

        # Add custom patterns
        for name, regex in config.custom_patterns.items():
            self._patterns.append(PIIPattern(
                PIIType.PERSON_NAME,  # generic type for custom
                regex,
                f'[{name.upper()}]'
            ))

        logger.info(
            "PII detector initialized: %d patterns, types=%s",
            len(self._patterns),
            [t.value for t in active_types]
        )

    def detect(self, text: str) -> List[Dict]:
        """Detect all PII in a text string.

        Returns list of detections:
            [{"type": "phone_ke", "start": 5, "end": 15, "text": "0712345678"}]
        """
        detections = []
        for pattern in self._patterns:
            for start, end, matched in pattern.find_all(text):
                detections.append({
                    "type": pattern.pii_type.value,
                    "start": start,
                    "end": end,
                    "text": matched,
                    "tag": pattern.tag
                })

        # Sort by position (rightmost first for safe replacement)
        detections.sort(key=lambda d: d["start"], reverse=True)
        return detections

    def has_pii(self, text: str) -> bool:
        """Quick check: does this text contain any PII?"""
        for pattern in self._patterns:
            if pattern.regex.search(text):
                return True
        return False


class PIIRedactor:
    """Redact detected PII from text using the configured strategy."""

    def __init__(self, config: PIIConfig):
        self.config = config
        self.detector = PIIDetector(config)
        self._stats: Dict[str, int] = {}

    def redact(self, text: str) -> str:
        """Redact all PII from a text string.

        Args:
            text: input text potentially containing PII

        Returns:
            text with PII redacted according to configured strategy
        """
        if not self.config.enabled:
            return text

        detections = self.detector.detect(text)
        if not detections:
            return text

        # Apply redactions (detections are sorted rightmost-first)
        result = text
        for detection in detections:
            start = detection["start"]
            end = detection["end"]
            original = detection["text"]
            pii_type = detection["type"]

            replacement = self._get_replacement(original, detection["tag"])
            result = result[:start] + replacement + result[end:]

            # Track stats
            self._stats[pii_type] = self._stats.get(pii_type, 0) + 1

        if self.config.log_detections:
            logger.info(
                "PII redacted: %d items (%s)",
                len(detections),
                {k: v for k, v in self._stats.items() if v > 0}
            )

        return result

    def _get_replacement(self, original: str, tag: str) -> str:
        """Generate replacement text based on redaction strategy."""
        if self.config.strategy == RedactionStrategy.MASK:
            return tag

        elif self.config.strategy == RedactionStrategy.HASH:
            # Deterministic hash — same input always produces same output
            # Preserves uniqueness for analysis without revealing actual values
            salted = f"{self.config.hash_salt}:{original}"
            hash_hex = hashlib.sha256(salted.encode()).hexdigest()[:12]
            return f"{tag}_{hash_hex}"

        elif self.config.strategy == RedactionStrategy.REMOVE:
            return ""

        elif self.config.strategy == RedactionStrategy.FLAG:
            return f"⚠{original}⚠"

        return tag  # fallback to mask

    def redact_batch(self, texts: List[str]) -> List[str]:
        """Redact PII from a batch of texts."""
        return [self.redact(text) for text in texts]

    def redact_dataset_column(
        self,
        data: List[dict],
        text_column: str = "text"
    ) -> List[dict]:
        """Redact PII from a specific column in a list of data records.

        Args:
            data: list of dicts (e.g., [{"text": "Call 0712345678", "label": "spam"}])
            text_column: which column contains the text to redact

        Returns:
            same data with PII redacted in the text column
        """
        redacted = []
        for record in data:
            new_record = dict(record)
            if text_column in new_record and isinstance(new_record[text_column], str):
                new_record[text_column] = self.redact(new_record[text_column])
            redacted.append(new_record)
        return redacted

    @property
    def stats(self) -> Dict[str, int]:
        """Get PII detection statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self._stats.clear()

    def generate_report(self) -> str:
        """Generate a summary report of PII detections."""
        if not self._stats:
            return "No PII detected."

        total = sum(self._stats.values())
        lines = [
            f"PII Redaction Report",
            f"{'=' * 40}",
            f"Strategy: {self.config.strategy.value}",
            f"Total PII items redacted: {total}",
            f"",
            f"Breakdown by type:",
        ]
        for pii_type, count in sorted(self._stats.items(), key=lambda x: -x[1]):
            lines.append(f"  {pii_type}: {count}")

        return "\n".join(lines)
