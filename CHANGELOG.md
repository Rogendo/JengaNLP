# Changelog

All notable changes to **JengaNLP** will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `jenga-nlp mlm`: domain-adaptive masked-language-model pretraining for XLM-R-family encoders
  (AfriBERTa, AfroXLMR): `train`, `evaluate` (base vs adapted pseudo-perplexity per domain) and
  `tokenizer-report` (sub-word fragmentation) and `pii-check`. Corpus pipeline with regex PII
  redaction, cross-source dedup and a held-out split taken before repetition.
- NER-based PII redaction for MLM corpora: a JengaNLP NER model or any Hugging Face
  token-classification model (e.g. MasakhaNER) finds names and places. Sources marked `sensitive`
  require it plus a labelled sample, and training refuses to start unless PII recall on that sample
  reaches `pii.min_recall`; they are also capped at 2x repetition and private-only on the Hub.
  Output is a standard Hugging Face model usable as `model.base_model`. See `docs/domain-adaptation.md`.

### Fixed
- The NER decoder dropped entities whose first token was tagged `I-` (no preceding `B-`), losing recall.
- Kenyan phone redaction missed numbers written with spaces or hyphens (`+254 722 000 111`,
  `0722-000-111`) and never redacted `01x` numbers.

---

## [2.0.0] — 2026-11-11

### Added
- Full multi-task text NLP framework with shared encoder + pluggable task heads
- Task types: `single_label_classification`, `multi_label_classification`, `sentiment`, `ner`, `regression`, `qa_scoring`
- `ExperimentConfig` — Pydantic v2 YAML-driven configuration system
- `MultiTaskModel` — shared encoder with independent task heads and optional fusion layers
- `InferencePipeline` — load-once, predict-many production inference with formatters
- `JengaClient` — high-level Python API for train / evaluate / predict
- CLI: `python -m jenga_nlp train | evaluate | predict | resume | backends | asr`
- Pluggable integrations layer: MLflow, TensorBoard, Weights & Biases, Label Studio
- Activity event system: typed events, sinks (Console, JSONL, Multi), callbacks
- ASR module: Whisper fine-tuning (`WhisperTrainer`), local serving (`WhisperServer`), manifest-based dataset loaders, WER/CER metrics, HuggingFace Hub push
- PII redaction in `data/pii.py`
- Security module: audit logging, explainability, HITL review
- Training resume: `find_latest_training_state`, `resume_training`
- `ship_model` workflow: train → evaluate → push to Hub
- PyPI packaging with `jenga-nlp` CLI entry point
- Unknown config keys now raise an `UnknownConfigKeyWarning` naming the section and the closest
  valid key (they are still ignored), so typos such as `fussion:` no longer fail silently

### Changed
- Attention fusion redesigned: each task's embedding now attends over the tokens (padding masked,
  `num_attention_heads` honoured) and the task context is gated into every token. The previous
  version had no task-specific effect; models trained with it must be retrained.
- New `model.fusion.lr_multiplier`: fusion trains faster than the pretrained encoder (default 10 for attention
  fusion, 1 for concatenation)
- `training.task_sampling` now implements `proportional` and `temperature` (previously always round-robin)
- CLI: `predict` accepts `--device`; `evaluate` writes its activity log to the evaluated model folder
- Crashed or interrupted runs end as `FAILED` / `KILLED` in MLflow (they were recorded as `FINISHED`), and
  `evaluate` always closes its tracker run; the MLflow integration adapter no longer passes an invalid status
- Resuming a run whose optimizer layout changed restarts the optimizer cleanly instead of crashing
- Attention fusion always uses its residual connection: `use_residual: false` is rejected for
  `type: attention` (still supported for concatenation). Custom callbacks with an
  `on_train_end(self, model)` signature keep working.
- Refactored from monolithic codebase into modular sub-packages
- Config parsing migrated to Pydantic v2

---

## [1.x.x] — Prior releases (internal / JengaAI backend)

Earlier versions were part of the JengaAI backend monorepo and not published to PyPI.
