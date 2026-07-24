"""Examples-level smoke tests for loading saved checkpoints from /tmp."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.run_inference_from_tmp import DEFAULT_MODEL_DIRS, run_one


@pytest.mark.parametrize("model_dir", DEFAULT_MODEL_DIRS)
def test_run_one_smoke_for_known_checkpoint(model_dir: str) -> None:
    """Each example checkpoint should load and produce a prediction payload."""
    path = Path(model_dir)
    if not path.exists():
        pytest.skip(f"checkpoint not present: {path}")

    task_name = None
    if "multilabel" in str(path):
        task_name = "mpesa_risk_tags"
    elif "ner" in str(path):
        task_name = "child_helpline_ner"

    sample_text = {
        "multilabel": "The account was locked after a phishing email and fake login page.",
        "fraud": "Received a suspicious M-PESA message asking for my PIN.",
        "ner": "John lives in Nairobi and called the child helpline for help.",
        "qa": "The agent answered the question clearly and the customer was satisfied.",
    }

    payload = run_one(path, sample_text["fraud" if "fraud" in str(path) else "multilabel" if "multilabel" in str(path) else "ner" if "ner" in str(path) else "qa"], task_name=task_name)

    assert "text" in payload
    assert "task_results" in payload
    assert isinstance(payload["task_results"], dict)
