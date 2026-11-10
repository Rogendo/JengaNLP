"""
test_cli.py — verify the CLI builds correctly and --help exits cleanly.

These tests do not execute any ML operations. They only exercise
the argument parser and help text output.
"""

import subprocess
import sys

import pytest

from jenga_nlp.cli import build_parser, CLI_QUICKSTART


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------

def test_parser_builds():
    parser = build_parser()
    assert parser is not None


def test_parser_has_train_subcommand():
    parser = build_parser()
    # Access sub-actions to confirm train is registered
    subactions = [
        action
        for action in parser._actions
        if hasattr(action, "_name_parser_map")
    ]
    assert len(subactions) > 0
    subcommands = list(subactions[0]._name_parser_map.keys())
    assert "train" in subcommands


def test_parser_has_all_subcommands():
    parser = build_parser()
    subactions = [
        action
        for action in parser._actions
        if hasattr(action, "_name_parser_map")
    ]
    subcommands = set(subactions[0]._name_parser_map.keys())
    expected = {"train", "evaluate", "predict", "resume", "backends", "asr"}
    assert expected.issubset(subcommands)


# ---------------------------------------------------------------------------
# CLI quickstart string
# ---------------------------------------------------------------------------

def test_cli_quickstart_nonempty():
    assert isinstance(CLI_QUICKSTART, str)
    assert len(CLI_QUICKSTART) > 50


def test_cli_quickstart_mentions_commands():
    for keyword in ["train", "evaluate", "predict", "resume", "asr", "backends"]:
        assert keyword in CLI_QUICKSTART


# ---------------------------------------------------------------------------
# `python -m jenga_nlp` via subprocess (integration-level)
# ---------------------------------------------------------------------------

def test_module_help_exits_zero():
    """python -m jenga_nlp -h must exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "train" in result.stdout or "train" in result.stderr


def test_module_no_args_exits_zero():
    """python -m jenga_nlp with no args must print quickstart and exit 0."""
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Unexpected non-zero exit: {result.stderr}"


def test_train_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp", "train", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_predict_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp", "predict", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--text" in result.stdout


def test_asr_train_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp", "asr", "train", "-h"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_unknown_command_exits_nonzero():
    result = subprocess.run(
        [sys.executable, "-m", "jenga_nlp", "does-not-exist"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# backends command output
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from jenga_nlp.cli import main  # noqa: E402

_CONFIGS = Path(__file__).resolve().parent.parent / "examples" / "configs"


@pytest.mark.parametrize("command_name", ["backends", "backend"])
def test_backends_status_reports_logging_and_integrations(capsys, command_name):
    main([command_name, "status", "--config", str(_CONFIGS / "fixture_mpesa_fraud_cpu.yaml")])
    out = capsys.readouterr().out
    assert "training logging: mlflow (sqlite:///mlflow.db)" in out
    assert "integrations: none configured" in out


def test_predict_accepts_device():
    args = build_parser().parse_args(
        ["predict", "--config", "c.yaml", "--model-dir", "m", "--text", "hi", "--device", "cpu"]
    )
    assert args.device == "cpu"
    without_device = build_parser().parse_args(["predict", "--config", "c.yaml", "--model-dir", "m", "--text", "hi"])
    assert without_device.device is None
