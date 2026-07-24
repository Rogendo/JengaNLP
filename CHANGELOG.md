# Changelog

All notable changes to **JengaNLP** will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2024-07-23

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

### Changed
- Refactored from monolithic codebase into modular sub-packages
- Config parsing migrated to Pydantic v2

---

## [1.x.x] — Prior releases (internal / JengaAI backend)

Earlier versions were part of the JengaAI backend monorepo and not published to PyPI.
