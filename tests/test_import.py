"""
test_import.py — verify every public symbol listed in __all__ is importable.

These tests run without GPU, network, or large model downloads.
They are purely import-level smoke tests.
"""

import importlib


def test_package_importable():
    """jenga_nlp itself must import without error."""
    import jenga_nlp  # noqa: F401


def test_version_defined():
    """__version__ must be a non-empty string."""
    import jenga_nlp
    assert isinstance(jenga_nlp.__version__, str)
    assert len(jenga_nlp.__version__) > 0


def test_all_exports_present():
    """Every symbol listed in __all__ must be importable from jenga_nlp."""
    import jenga_nlp

    missing = []
    for name in jenga_nlp.__all__:
        if not hasattr(jenga_nlp, name):
            missing.append(name)

    assert missing == [], f"Missing exports from jenga_nlp.__all__: {missing}"


def test_core_submodule():
    mod = importlib.import_module("jenga_nlp.core")
    assert hasattr(mod, "ExperimentConfig")


def test_tasks_submodule():
    mod = importlib.import_module("jenga_nlp.tasks")
    assert hasattr(mod, "TaskRegistry")


def test_training_submodule():
    mod = importlib.import_module("jenga_nlp.training")
    # training/__init__.py may be minimal; just ensure it imports cleanly
    assert mod is not None


def test_inference_submodule():
    mod = importlib.import_module("jenga_nlp.inference")
    assert hasattr(mod, "InferencePipeline")


def test_client_submodule():
    mod = importlib.import_module("jenga_nlp.client")
    assert hasattr(mod, "JengaClient")


def test_integrations_submodule():
    mod = importlib.import_module("jenga_nlp.integrations")
    assert hasattr(mod, "MLflowAdapter")
    assert hasattr(mod, "TensorBoardAdapter")
    assert hasattr(mod, "ActivityLogger")


def test_asr_submodule():
    mod = importlib.import_module("jenga_nlp.asr")
    assert hasattr(mod, "WhisperTrainer")
    assert hasattr(mod, "WhisperServer")
    assert hasattr(mod, "WhisperPipeline")


def test_data_submodule():
    mod = importlib.import_module("jenga_nlp.data")
    assert hasattr(mod, "DataProcessor")


def test_export_submodule():
    mod = importlib.import_module("jenga_nlp.export")
    assert hasattr(mod, "push_model_to_hub")


def test_utils_submodule():
    mod = importlib.import_module("jenga_nlp.utils")
    assert mod is not None
