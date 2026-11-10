"""Static checks that keep the examples in sync with the jenga_nlp package.

These catch stale imports (e.g. the pre-2.0 ``jenga_ai`` package name),
syntax errors, and configs that no longer validate — without training anything.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from jenga_nlp.core.config import ExperimentConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
CONFIGS = EXAMPLES / "configs"

EXAMPLE_SCRIPTS = sorted(EXAMPLES.glob("*.py"))
# Whisper configs use WhisperFineTuningConfig, not ExperimentConfig.
TEXT_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if "whisper" not in p.name)
OLD_PACKAGES = ("jenga_ai", "multitask_bert")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_script_has_no_old_package_imports(script: Path) -> None:
    stale = {m for m in _imported_modules(script) if m.split(".")[0] in OLD_PACKAGES}
    assert not stale, f"{script.name} imports removed packages: {sorted(stale)}"


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS, ids=lambda p: p.name)
def test_example_script_jenga_nlp_imports_resolve(script: Path) -> None:
    import importlib

    for module in sorted(m for m in _imported_modules(script) if m.startswith("jenga_nlp")):
        importlib.import_module(module)


@pytest.mark.parametrize(
    "path",
    sorted([*CONFIGS.glob("*.yaml"), CONFIGS / "README.md"]),
    ids=lambda p: p.name,
)
def test_config_docs_do_not_reference_old_cli(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"python -m (jenga_ai|multitask_bert)\b", text), f"{path.name} references the old CLI"


@pytest.mark.parametrize("config", TEXT_CONFIGS, ids=lambda p: p.name)
def test_example_config_validates(config: Path) -> None:
    cfg = ExperimentConfig.from_yaml(config)
    assert cfg.tasks, f"{config.name} defines no tasks"
