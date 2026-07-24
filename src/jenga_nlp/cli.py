"""CLI for JengaNLP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .core.config import ExperimentConfig
from .asr.config import WhisperConfig, WhisperFineTuningConfig
from .asr.serving import WhisperServer
from .asr.training import WhisperTrainer
from .integrations.callbacks import ActivityEventCallback
from .integrations.events import ActivityEventStage
from .integrations.registry import build_adapter
from .integrations.runtime import IntegrationRuntime
from .resume import resume_training
from .utils.logging import setup_logging

logger = setup_logging()


CLI_QUICKSTART = """JengaNLP command line

Usage:
  python -m jenga_nlp <command> [options]

Common commands:
  python -m jenga_nlp train --config configs/fixture_mpesa_fraud_cpu.yaml --device cpu
  python -m jenga_nlp evaluate --config configs/fixture_mpesa_fraud_cpu.yaml --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu --device cpu
  python -m jenga_nlp predict --config configs/fixture_mpesa_fraud_cpu.yaml --model-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu --text "Your text here"
  python -m jenga_nlp resume --config configs/fixture_mpesa_fraud_cpu.yaml --run-dir /tmp/jenganlp_fixture_mpesa_fraud_cpu
  python -m jenga_nlp backends status --config configs/fixture_mpesa_fraud_cpu.yaml
  python -m jenga_nlp asr serve --model-name openai/whisper-small --audio sample.wav

Commands:
  train       Train a text model from a YAML config.
  evaluate    Evaluate a saved model directory.
  predict     Run inference with a saved model directory.
  resume      Resume training from a saved run directory.
  backends    Check or initialize configured integrations.
  asr         Whisper speech-to-text serving and fine-tuning commands.

For command-specific help:
  python -m jenga_nlp train -h
  python -m jenga_nlp predict -h
  python -m jenga_nlp asr train -h
"""


class JengaArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that gives actionable errors to non-expert CLI users."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"\nError: {message}\n\n{CLI_QUICKSTART}\n")


def _build_runtime(config: ExperimentConfig) -> IntegrationRuntime:
    return IntegrationRuntime(run_dir=config.training.output_dir, integrations=config.integrations)


def run_train(config: ExperimentConfig, output_dir: str | Path | None = None, device: str | None = None, resume_from: str | Path | None = None) -> dict[str, Any]:
    from .core.model import MultiTaskModel
    from .data.processor import DataProcessor
    from .training.trainer import Trainer

    if output_dir:
        config.training.output_dir = str(output_dir)
    if device:
        config.training.device = device

    runtime = _build_runtime(config)
    runtime.start(run_name=config.project_name, config=config.model_dump(mode="python"), tags={"project_name": config.project_name})
    runtime.emit_info("loading tokenizer")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    processor = DataProcessor(config, tokenizer)
    train_datasets, eval_datasets, _ = processor.process()
    runtime.emit_info(
        "datasets processed",
        payload={"train_tasks": list(train_datasets.keys()), "eval_tasks": list(eval_datasets.keys())},
    )
    model = MultiTaskModel.from_config(config)
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        callbacks=[ActivityEventCallback(runtime=runtime)],
        resume_from=resume_from,
    )
    results = trainer.train()
    runtime.emit_info("training complete", stage=ActivityEventStage.TRAIN, payload={"metrics": results})
    config.to_yaml(Path(config.training.output_dir) / "experiment_config.yaml")
    runtime.finish(status="success")
    return results


def run_evaluate(config: ExperimentConfig, model_dir: str | Path, device: str | None = None) -> dict[str, Any]:
    from .core.model import MultiTaskModel
    from .data.processor import DataProcessor
    from .training.trainer import Trainer

    if device:
        config.training.device = device
    runtime = _build_runtime(config)
    runtime.start(
        run_name=f"{config.project_name}-eval",
        config=config.model_dump(mode="python"),
        tags={"project_name": config.project_name, "mode": "eval"},
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model)
    _, eval_datasets, _ = DataProcessor(config, tokenizer).process()
    runtime.emit_info("evaluation datasets processed", payload={"eval_tasks": list(eval_datasets.keys())})
    model = MultiTaskModel.load(model_dir, config, device=config.training.resolve_device())
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_datasets={},
        eval_datasets=eval_datasets,
        callbacks=[ActivityEventCallback(runtime=runtime)],
    )
    result = trainer.evaluate()
    runtime.emit_info("evaluation complete", stage=ActivityEventStage.EVAL, payload={"metrics": result})
    runtime.finish(status="success")
    return result


def run_predict(
    config: ExperimentConfig,
    model_dir: str | Path,
    text: str,
    task_name: str | None = None,
    threshold: float = 0.5,
) -> Any:
    import tempfile

    from .inference.pipeline import InferencePipeline
    from .utils.device import get_device

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "experiment_config.yaml"
        config.to_yaml(config_path)
        pipeline = InferencePipeline.from_checkpoint(model_dir=model_dir, config_path=config_path, device=str(get_device("auto")))
        result = pipeline.predict(text=text, task_name=task_name, threshold=threshold)
        return result.to_dict()


def run_backends(config: ExperimentConfig, command: str) -> None:
    for integration in config.integrations:
        adapter = build_adapter(integration, base_dir=config.training.output_dir)
        if adapter is None:
            continue
        status = "available" if adapter.is_available() else "missing-dependency"
        if command == "start":
            print(f"{adapter.name}: {status}")
            if adapter.is_available():
                adapter.start_run(run_name=f"{config.project_name}-{adapter.name}", config=config.model_dump(mode="python"))
                adapter.finish(status="success")
        elif command == "status":
            print(f"{adapter.name}: {status}")
        elif command == "stop":
            print(f"{adapter.name}: stopped")


def run_asr_train(
    config: WhisperFineTuningConfig,
    train_data: str | Path | None = None,
    eval_data: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    push_to_hub: bool | None = None,
    hub_model_id: str | None = None,
) -> dict[str, Any]:
    resolved_train_data = train_data or config.resolve_train_data()
    resolved_eval_data = eval_data or config.resolve_eval_data()
    if config.dataset.source.lower() == "manifest" and resolved_train_data is None:
        raise ValueError("ASR training requires --train-data or dataset.train_manifest in the config")
    trainer = WhisperTrainer(config)
    return trainer.train(
        resolved_train_data,
        resolved_eval_data,
        resume_from_checkpoint=resume_from_checkpoint,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = JengaArgumentParser(
        prog="jenga_nlp",
        description="Train, evaluate, resume, serve, and run inference for JengaNLP models.",
        epilog=CLI_QUICKSTART,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, parser_class=JengaArgumentParser)

    train = sub.add_parser("train", help="Train a text model from a YAML config.")
    train.add_argument("--config", required=True)
    train.add_argument("--output-dir")
    train.add_argument("--device")

    evalp = sub.add_parser("evaluate", help="Evaluate a saved model directory.")
    evalp.add_argument("--config", required=True)
    evalp.add_argument("--model-dir", required=True)
    evalp.add_argument("--device")

    predict = sub.add_parser("predict", help="Run inference with a saved model directory.")
    predict.add_argument("--config", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--task-name")
    predict.add_argument("--threshold", type=float, default=0.5)

    resume = sub.add_parser("resume", help="Resume training from an existing run directory.")
    resume.add_argument("--config", required=True)
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--output-dir")
    resume.add_argument("--device")

    asr = sub.add_parser("asr", help="Whisper speech-to-text commands.")
    asr_sub = asr.add_subparsers(dest="asr_command", required=True, parser_class=JengaArgumentParser)

    asr_serve = asr_sub.add_parser("serve", help="Run local Whisper transcription.")
    asr_serve.add_argument("--model-name", default="openai/whisper-small")
    asr_serve.add_argument("--language", default="sw")
    asr_serve.add_argument("--device", default="auto")
    asr_serve.add_argument("--chunk-length-s", type=int, default=30)
    asr_serve.add_argument("--stride-length-s", type=int, default=5)
    asr_serve.add_argument("--audio", help="Optional audio file to transcribe immediately")

    asr_train = asr_sub.add_parser("train", help="Fine-tune a Whisper model.")
    asr_train.add_argument("--config", required=True)
    asr_train.add_argument("--train-data")
    asr_train.add_argument("--eval-data")
    asr_train.add_argument("--output-dir")
    asr_train.add_argument("--device")
    asr_train.add_argument("--resume", help="Checkpoint path or 'latest'")
    asr_train.add_argument("--push-to-hub", action="store_true")
    asr_train.add_argument("--hub-model-id")

    backends = sub.add_parser("backends", help="Manage configured integration backends.")
    backend_sub = backends.add_subparsers(dest="backend_command", required=True, parser_class=JengaArgumentParser)

    backend_start = backend_sub.add_parser("start", help="Initialize configured backends.")
    backend_start.add_argument("--config", required=True)

    backend_status = backend_sub.add_parser("status", help="Show backend availability.")
    backend_status.add_argument("--config", required=True)

    backend_stop = backend_sub.add_parser("stop", help="Stop configured backends.")
    backend_stop.add_argument("--config", required=True)

    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        print(CLI_QUICKSTART)
        return

    args = parser.parse_args(argv)
    if args.command == "train":
        config = ExperimentConfig.from_yaml(args.config)
        run_train(config, output_dir=args.output_dir, device=args.device)
    elif args.command == "evaluate":
        config = ExperimentConfig.from_yaml(args.config)
        print(run_evaluate(config, model_dir=args.model_dir, device=args.device))
    elif args.command == "predict":
        config = ExperimentConfig.from_yaml(args.config)
        print(run_predict(config=config, model_dir=args.model_dir, text=args.text, task_name=args.task_name, threshold=args.threshold))
    elif args.command == "resume":
        config = ExperimentConfig.from_yaml(args.config)
        print(resume_training(config, run_dir=args.run_dir, output_dir=args.output_dir, device=args.device))
    elif args.command == "asr":
        if args.asr_command == "serve":
            server = WhisperServer(
                WhisperConfig(
                    model_name=args.model_name,
                    language=args.language,
                    device=args.device,
                    chunk_length_s=args.chunk_length_s,
                    stride_length_s=args.stride_length_s,
                )
            )
            print(server.health())
            if args.audio:
                print(server.transcribe(args.audio))
        elif args.asr_command == "train":
            config = WhisperFineTuningConfig.from_yaml(args.config)
            if args.output_dir:
                config.output_dir = Path(args.output_dir)
            if args.device:
                config.device = args.device
            print(
                run_asr_train(
                    config,
                    train_data=args.train_data,
                    eval_data=args.eval_data,
                    resume_from_checkpoint=args.resume,
                    push_to_hub=args.push_to_hub if args.push_to_hub else None,
                    hub_model_id=args.hub_model_id,
                )
            )
    elif args.command == "backends":
        config = ExperimentConfig.from_yaml(args.config)
        run_backends(config, args.backend_command)
