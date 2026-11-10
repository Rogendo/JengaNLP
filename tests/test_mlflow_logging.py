"""MLflow logging through LoggingCallback, against a real temporary SQLite store."""

from __future__ import annotations

import pytest

mlflow = pytest.importorskip("mlflow")

from jenga_nlp.core.config import LoggingConfig, LoggingService  # noqa: E402
from jenga_nlp.training.callbacks import LoggingCallback  # noqa: E402


@pytest.fixture()
def tracking_uri(tmp_path, monkeypatch):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.chdir(tmp_path)  # keep local artifacts inside tmp_path
    yield uri
    if mlflow.active_run():
        mlflow.end_run()


def _callback(tmp_path, tracking_uri, **kwargs):
    cfg = LoggingConfig(service=LoggingService.MLFLOW, experiment_name="jenga_test", tracking_uri=tracking_uri,
                        log_every_n_steps=1)
    return LoggingCallback(cfg, str(tmp_path), **kwargs)


def test_training_run_records_name_params_tags_config_and_metrics(tmp_path, tracking_uri):
    cb = _callback(
        tmp_path,
        tracking_uri,
        run_name="my_project",
        params={"learning_rate": 5e-5, "base_model": "distilbert-base-uncased"},
        tags={"mode": "train"},
        config_dict={"project_name": "my_project"},
    )
    run_id = mlflow.active_run().info.run_id
    cb.on_step(step=1, loss=0.9)
    cb.on_step(step=2, loss=0.5)
    cb.on_epoch_end(epoch=0, metrics={"eval_loss": 0.4}, global_step=2)
    cb.on_train_end()

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert run.info.run_name == "my_project"
    assert run.info.status == "FINISHED"
    assert run.data.params == {"learning_rate": "5e-05", "base_model": "distilbert-base-uncased"}
    assert run.data.tags["mode"] == "train"
    assert [m.step for m in client.get_metric_history(run_id, "train_loss")] == [1, 2]
    assert [m.step for m in client.get_metric_history(run_id, "eval_loss")] == [2]  # global step, not epoch
    assert "experiment_config.yaml" in [a.path for a in client.list_artifacts(run_id)]

    experiment = client.get_experiment(run.info.experiment_id)
    assert experiment.tags.get("mlflow.experimentKind") == "custom_model_development"


def test_standalone_metrics_are_logged_and_run_closed(tmp_path, tracking_uri):
    cb = _callback(tmp_path, tracking_uri, run_name="my_project-eval", tags={"mode": "eval"})
    run_id = mlflow.active_run().info.run_id
    cb.log_metrics({"eval_loss": 0.3, "accuracy": 0.9, "not_a_number": "x"})  # type: ignore[dict-item]
    cb.close()
    cb.close()  # idempotent

    run = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics == {"eval_loss": 0.3, "accuracy": 0.9}
    assert mlflow.active_run() is None


def test_existing_experiment_kind_is_not_overridden(tmp_path, tracking_uri):
    mlflow.set_tracking_uri(tracking_uri)
    exp_id = mlflow.create_experiment("jenga_test")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.set_experiment_tag(exp_id, "mlflow.experimentKind", "genai_development")

    _callback(tmp_path, tracking_uri).close()

    assert client.get_experiment(exp_id).tags["mlflow.experimentKind"] == "genai_development"


@pytest.mark.parametrize("status", ["FAILED", "KILLED"])
def test_run_status_is_recorded_on_close(tmp_path, tracking_uri, status):
    cb = _callback(tmp_path, tracking_uri)
    run_id = mlflow.active_run().info.run_id
    cb.on_train_end(model=None, status=status)
    assert mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id).info.status == status
    assert mlflow.active_run() is None


@pytest.mark.parametrize(("runtime_status", "mlflow_status"), [("success", "FINISHED"), ("failed", "FAILED")])
def test_mlflow_integration_adapter_maps_runtime_status(tmp_path, tracking_uri, runtime_status, mlflow_status):
    from jenga_nlp.integrations.adapters.mlflow import MLflowAdapter

    adapter = MLflowAdapter(tracking_uri=tracking_uri, experiment_name="adapter_test")
    run = adapter.start_run(run_name="r", config={})
    run_id = (run or mlflow.active_run()).info.run_id
    adapter.finish(status=runtime_status)
    assert mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id).info.status == mlflow_status
