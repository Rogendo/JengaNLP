"""W&B adapter: training/eval activity events must reach wandb.log with usable steps.

Uses a fake ``wandb`` module, so no account, network or wandb install is needed.
"""

from __future__ import annotations

from jenga_nlp.core.config import IntegrationConfig
from jenga_nlp.integrations.adapters.wandb import WandBAdapter
from jenga_nlp.integrations.callbacks import ActivityEventCallback
from jenga_nlp.integrations.events import ActivityEventStage
from jenga_nlp.integrations.runtime import IntegrationRuntime


class FakeWandb:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int | None]] = []

    def log(self, data, step=None):
        self.logged.append((dict(data), step))


def _runtime_with_fake_wandb(tmp_path) -> tuple[IntegrationRuntime, FakeWandb]:
    runtime = IntegrationRuntime(run_dir=tmp_path, integrations=[IntegrationConfig(type="wandb")])
    adapter = next(a for a in runtime.adapters if isinstance(a, WandBAdapter))
    fake = FakeWandb()
    adapter._wandb = fake
    return runtime, fake


def test_training_loss_and_epoch_metrics_reach_wandb_on_one_step_axis(tmp_path):
    runtime, fake = _runtime_with_fake_wandb(tmp_path)
    cb = ActivityEventCallback(runtime=runtime)

    for step in (1, 2, 3, 4):
        cb.on_step(step=step, loss=1.0 / step)
    cb.on_epoch_end(epoch=0, metrics={"eval_loss": 0.4, "accuracy": 0.9}, global_step=4)

    assert ({"train/loss": 1.0}, 1) in fake.logged
    assert ({"train/loss": 0.25}, 4) in fake.logged
    assert ({"eval/eval_loss": 0.4, "eval/accuracy": 0.9}, 4) in fake.logged
    steps = [s for _, s in fake.logged]
    assert steps == sorted(steps), "W&B drops steps that go backwards"
    assert all(data for data, _ in fake.logged), "no empty log calls"


def test_standalone_evaluation_metrics_reach_wandb(tmp_path):
    runtime, fake = _runtime_with_fake_wandb(tmp_path)

    runtime.emit_info("evaluation complete", stage=ActivityEventStage.EVAL, payload={"metrics": {"eval_loss": 0.3}})

    assert fake.logged == [({"eval/eval_loss": 0.3}, None)]
