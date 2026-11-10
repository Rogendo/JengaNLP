"""Corpus redaction for MLM pretraining: built-in PII types plus a names list.

The general-purpose ``PIIRedactor`` (``jenga_nlp.data.pii``) has no built-in
person-name detection, so real chats or case notes would keep names of the
people in them. For pretraining corpora this module adds:

- a user-supplied names list, redacted as ``[NAME]`` (whole words, any case);
- overlap-safe replacement (the longest match wins when patterns overlap);
- quiet per-type counters for the redaction report (no per-call logging, and
  the matched text is never stored).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ..data.pii import PIIConfig, PIIDetector, PIIType, RedactionStrategy

NAME_TAG = "[NAME]"


def load_names(path: str | Path) -> list[str]:
    """Read one name per line; blank lines and lines starting with '#' are ignored."""
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    return names


def compile_names_pattern(names: list[str]) -> re.Pattern[str] | None:
    """One case-insensitive whole-word regex for all names (longest first, so full names win)."""
    unique = sorted({n for n in names if n}, key=len, reverse=True)
    if not unique:
        return None
    alternation = "|".join(re.escape(n) for n in unique)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE)


class CorpusRedactor:
    """Redact built-in PII types, extra regexes and listed names from corpus text."""

    def __init__(
        self,
        names: list[str] | None = None,
        extra_patterns: dict[str, str] | None = None,
        detect_types: list[str] | None = None,
    ) -> None:
        types = [PIIType(t) for t in detect_types] if detect_types else None
        self._detector = PIIDetector(
            PIIConfig(
                enabled=True,
                strategy=RedactionStrategy.MASK,
                detect_types=types,
                log_detections=False,
                custom_patterns=dict(extra_patterns or {}),
            )
        )
        self._names = compile_names_pattern(names or [])
        self.counts: Counter[str] = Counter()

    def redact(self, text: str) -> str:
        spans: list[tuple[int, int, str, str]] = []  # (start, end, tag, kind)
        for det in self._detector.detect(text):
            kind = det["type"]
            if kind == PIIType.PERSON_NAME.value:  # extra_patterns are reported by their own tag
                kind = det["tag"].strip("[]").lower()
            spans.append((det["start"], det["end"], det["tag"], kind))
        if self._names is not None:
            for match in self._names.finditer(text):
                spans.append((match.start(), match.end(), NAME_TAG, "name"))
        if not spans:
            return text

        # Overlap-safe: keep the longest span, then the earliest; drop anything overlapping it.
        spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
        kept: list[tuple[int, int, str, str]] = []
        for span in spans:
            if all(span[1] <= k[0] or span[0] >= k[1] for k in kept):
                kept.append(span)

        result = text
        for start, end, tag, kind in sorted(kept, key=lambda s: s[0], reverse=True):
            result = result[:start] + tag + result[end:]
            self.counts[kind] += 1
        return result
