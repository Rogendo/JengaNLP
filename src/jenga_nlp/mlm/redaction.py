"""PII redaction for MLM pretraining corpora: pluggable span detectors.

A masked language model learns to reproduce text it sees often, so personal
data must be removed from the corpus *before* training. Redaction combines
detectors, each good at a different kind of PII:

- ``RegexDetector``: fixed formats (Kenyan phones, national IDs, KRA PINs,
  M-PESA IDs, emails, URLs, ...) via ``jenga_nlp.data.pii``.
- ``NERDetector``: names, places and other free-form identifiers, using a
  JengaNLP NER model folder or any Hugging Face token-classification model
  (e.g. ``masakhane/afroxlmr-large-ner-masakhaner-1.0_2.0``).

``CorpusRedactor`` merges their spans (longest wins on overlap), replaces them
with tags such as ``[NAME]``, and keeps per-type counts. Matched text is never
stored.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..data.pii import PIIConfig, PIIDetector, PIIType, RedactionStrategy


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tag: str   # replacement, e.g. "[NAME]"
    kind: str  # counter key, e.g. "name", "phone_ke"


class SpanDetector(Protocol):
    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]: ...


# NER labels (MasakhaNER, JengaNLP fixtures, common schemes) -> redaction tag.
# ORG and DATE are not redacted by default: organisations are rarely personal,
# and dates are mostly harmless without other identifiers. Override with pii.ner_labels.
DEFAULT_NER_LABELS: dict[str, str] = {
    "PER": "NAME", "PERSON": "NAME", "NAME": "NAME", "VICTIM": "NAME", "PERPETRATOR": "NAME",
    "LOC": "LOCATION", "LOCATION": "LOCATION", "LANDMARK": "LOCATION", "GPE": "LOCATION",
    "AGE": "AGE", "PHONE_NUMBER": "PHONE",
}


class RegexDetector:
    """Fixed-format PII from ``jenga_nlp.data.pii`` (plus optional extra regexes)."""

    def __init__(self, detect_types: list[str] | None = None, extra_patterns: dict[str, str] | None = None):
        self._detector = PIIDetector(
            PIIConfig(
                enabled=True,
                strategy=RedactionStrategy.MASK,
                detect_types=[PIIType(t) for t in detect_types] if detect_types else None,
                log_detections=False,
                custom_patterns=dict(extra_patterns or {}),
            )
        )

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        out = []
        for text in texts:
            spans = []
            for det in self._detector.detect(text):
                kind = det["type"]
                if kind == PIIType.PERSON_NAME.value:  # extra_patterns report under their own tag
                    kind = det["tag"].strip("[]").lower()
                spans.append(Span(det["start"], det["end"], det["tag"], kind))
            out.append(spans)
        return out


_WORD_CHAR = re.compile(r"\w")


def widen_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend a span to whole-word boundaries, so a partially tagged name is fully removed."""
    while start > 0 and _WORD_CHAR.match(text[start - 1]):
        start -= 1
    while end < len(text) and _WORD_CHAR.match(text[end]):
        end += 1
    return start, end


class NERDetector:
    """Names, places and other entities from an NER model.

    ``model`` is a JengaNLP NER model folder (with ``metadata.json``) or a Hugging
    Face token-classification model ID/folder. Entities whose label maps to a tag
    in ``labels`` and whose confidence is at least ``threshold`` are redacted.
    A low threshold favours recall, which is what PII removal needs.
    """

    def __init__(
        self,
        model: str | Path,
        labels: dict[str, str] | None = None,
        threshold: float = 0.3,
        task_name: str | None = None,
        batch_size: int = 16,
        device: str = "auto",
        predict_fn: Callable[[Sequence[str]], list[list[dict[str, Any]]]] | None = None,
    ) -> None:
        self.labels = {k.upper(): v.upper() for k, v in (labels or DEFAULT_NER_LABELS).items()}
        self.threshold = threshold
        self.batch_size = batch_size
        # predict_fn returns, per text, entities as {"start", "end", "label", "score"}.
        self._predict = predict_fn or _load_ner(str(model), task_name, batch_size, device)

    def detect_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        out: list[list[Span]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            for text, entities in zip(chunk, self._predict(chunk)):
                spans = []
                for ent in entities:
                    label = str(ent["label"]).upper().removeprefix("B-").removeprefix("I-")
                    tag = self.labels.get(label)
                    if tag is None or float(ent.get("score", 1.0)) < self.threshold:
                        continue
                    s, e = widen_to_words(text, int(ent["start"]), int(ent["end"]))
                    if e > s:
                        spans.append(Span(s, e, f"[{tag}]", tag.lower()))
                out.append(spans)
        return out


def _load_ner(model: str, task_name: str | None, batch_size: int, device: str) -> Callable:
    from ..utils.device import get_device

    resolved = get_device(device)
    path = Path(model)
    if (path / "metadata.json").exists():  # JengaNLP NER model
        from ..inference.pipeline import InferencePipeline

        pipeline = InferencePipeline.from_checkpoint(
            model_dir=path, config_path=path / "experiment_config.yaml", device=str(resolved)
        )
        ner_tasks = [t.name for t in pipeline.config.tasks if getattr(t.type, "value", t.type) == "ner"]
        if task_name is None:
            if len(ner_tasks) != 1:
                raise ValueError(f"{model}: set pii.ner_task to one of the NER tasks {ner_tasks}")
            task_name = ner_tasks[0]

        def predict(texts: Sequence[str]) -> list[list[dict[str, Any]]]:
            results = pipeline.predict_batch(list(texts), task_name=task_name, batch_size=batch_size)
            out = []
            for result in results:
                heads = result.to_dict()["task_results"][task_name]["heads"]
                out.append([
                    {"start": e["start"], "end": e["end"], "label": e["label"], "score": e.get("confidence", 1.0)}
                    for head in heads.values() for e in head["prediction"]
                ])
            return out

        return predict

    from transformers import pipeline as hf_pipeline  # Hugging Face token-classification model

    ner = hf_pipeline(
        "token-classification", model=model, aggregation_strategy="simple",
        device=0 if resolved.type == "cuda" else -1,
    )

    def predict_hf(texts: Sequence[str]) -> list[list[dict[str, Any]]]:
        results = ner(list(texts), batch_size=batch_size)
        if texts and results and isinstance(results[0], dict):  # single text returns a flat list
            results = [results]
        return [
            [{"start": e["start"], "end": e["end"], "label": e["entity_group"], "score": e["score"]} for e in ents]
            for ents in results
        ]

    return predict_hf


class CorpusRedactor:
    """Apply detectors to texts; merge overlapping spans (longest wins); count redactions."""

    def __init__(self, detectors: Sequence[SpanDetector]) -> None:
        self.detectors = list(detectors)
        self.counts: Counter[str] = Counter()

    def spans_batch(self, texts: Sequence[str]) -> list[list[Span]]:
        merged: list[list[Span]] = [[] for _ in texts]
        for detector in self.detectors:
            for i, spans in enumerate(detector.detect_batch(texts)):
                merged[i].extend(spans)
        return [_non_overlapping(spans) for spans in merged]

    def redact_batch(self, texts: Sequence[str]) -> list[str]:
        out = []
        for text, spans in zip(texts, self.spans_batch(texts)):
            for span in sorted(spans, key=lambda s: s.start, reverse=True):
                text = text[: span.start] + span.tag + text[span.end :]
                self.counts[span.kind] += 1
            out.append(text)
        return out

    def redact(self, text: str) -> str:
        return self.redact_batch([text])[0]


def _non_overlapping(spans: list[Span]) -> list[Span]:
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(span.end <= k.start or span.start >= k.end for k in kept):
            kept.append(span)
    return kept
