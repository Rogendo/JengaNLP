"""Measure how much PII the redaction stack actually removes, and gate training on it.

The labelled set uses JengaNLP's NER JSONL format::

    {"text": "Amina called from Kisumu", "entities": [{"start": 0, "end": 5, "label": "NAME"}, ...]}

Gold entities whose label maps to a redaction tag (``pii.ner_labels``, e.g.
NAME, PER, VICTIM, LOCATION) count as PII. An entity is **caught** only if every
character of it is redacted; a partially redacted name is a miss.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import MLMConfig
from .redaction import DEFAULT_NER_LABELS, CorpusRedactor


class PIIRecallError(ValueError):
    """Raised when the PII detectors miss too much of the labelled sample."""


def load_labelled(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row.get("text"), str):
                rows.append({"text": row["text"], "entities": row.get("entities") or []})
    if not rows:
        raise ValueError(f"{path}: no labelled examples")
    return rows


def evaluate_redaction(
    redactor: CorpusRedactor, labelled: list[dict[str, Any]], label_map: dict[str, str] | None = None
) -> dict[str, Any]:
    """Recall of PII entities (fully redacted) per tag, plus how much non-PII text was redacted."""
    labels = {k.upper(): v.upper() for k, v in (label_map or DEFAULT_NER_LABELS).items()}
    texts = [row["text"] for row in labelled]
    predicted = redactor.spans_batch(texts)

    total: Counter[str] = Counter()
    caught: Counter[str] = Counter()
    redacted_chars = redacted_outside_pii = 0
    for row, spans in zip(labelled, predicted):
        covered = set()
        for span in spans:
            covered.update(range(span.start, span.end))
        gold_pii_chars = set()
        for ent in row["entities"]:
            tag = labels.get(str(ent["label"]).upper())
            if tag is None:
                continue
            chars = set(range(int(ent["start"]), int(ent["end"])))
            gold_pii_chars |= chars
            total[tag] += 1
            if chars <= covered:
                caught[tag] += 1
        redacted_chars += len(covered)
        redacted_outside_pii += len(covered - gold_pii_chars)

    n_total, n_caught = sum(total.values()), sum(caught.values())
    return {
        "examples": len(labelled),
        "pii_entities": n_total,
        "caught": n_caught,
        "recall": (n_caught / n_total) if n_total else None,
        "per_tag": {tag: {"entities": total[tag], "caught": caught[tag], "recall": caught[tag] / total[tag]}
                    for tag in sorted(total)},
        # Over-redaction is cheap for pretraining text, but very high values damage the corpus.
        "share_of_redacted_chars_outside_gold_pii": (redacted_outside_pii / redacted_chars) if redacted_chars else 0.0,
    }


def check_pii_recall(config: MLMConfig, redactor: CorpusRedactor | None = None) -> dict[str, Any] | None:
    """For configs with sensitive sources: score the sensitive-source redactor; raise if recall is too low."""
    if not config.has_sensitive_sources:
        return None
    from .data import build_redactor

    redactor = redactor or build_redactor(config, sensitive=True)
    report = evaluate_redaction(redactor, load_labelled(config.pii.eval_set), config.pii.ner_labels)
    report["min_recall"] = config.pii.min_recall
    report["passed"] = report["recall"] is not None and report["recall"] >= config.pii.min_recall
    if report["recall"] is None:
        raise PIIRecallError(f"{config.pii.eval_set}: no entities with PII labels to measure recall on")
    if not report["passed"]:
        per_tag = ", ".join(f"{t}: {v['caught']}/{v['entities']}" for t, v in report["per_tag"].items())
        raise PIIRecallError(
            f"PII recall {report['recall']:.1%} is below pii.min_recall {config.pii.min_recall:.0%} "
            f"on {config.pii.eval_set} ({per_tag}). Sensitive sources would keep personal data; "
            "improve the NER model, lower pii.ner_threshold, or leave these sources out."
        )
    return report
