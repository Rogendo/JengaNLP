"""Attention fusion must condition on the task; task sampling must follow the configured strategy."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from jenga_nlp.core.config import DataConfig, FusionConfig, FusionType, TaskSamplingStrategy
from jenga_nlp.core.fusion import AttentionFusion, create_fusion
from jenga_nlp.training.trainer import Trainer

HIDDEN, TASKS = 16, 4


def _fusion(**cfg) -> AttentionFusion:
    torch.manual_seed(0)
    return AttentionFusion(HIDDEN, TASKS, FusionConfig(type=FusionType.ATTENTION, dropout=0.0, **cfg))


def _train_steps(fusion: AttentionFusion, steps: int = 5) -> None:
    """A few optimiser steps where each task must map the same input to a different target."""
    opt = torch.optim.Adam(fusion.parameters(), lr=1e-2)
    x = torch.randn(3, 6, HIDDEN)
    targets = torch.randn(TASKS, HIDDEN)
    for _ in range(steps):
        for t in range(TASKS):
            opt.zero_grad()
            loss = (fusion(x, t)[:, 0] - targets[t]).pow(2).mean()
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Attention fusion
# ---------------------------------------------------------------------------

def test_starts_as_layernorm_and_gate_starts_at_configured_value():
    fusion = _fusion(gate_init_value=0.3)
    x = torch.randn(2, 5, HIDDEN)
    assert torch.allclose(fusion(x, 0), fusion.layer_norm(x), atol=1e-6)  # zero-init projection
    assert torch.sigmoid(fusion.gate).item() == pytest.approx(0.3, abs=1e-6)


def test_output_depends_on_task_after_training():
    fusion = _fusion()
    _train_steps(fusion)
    fusion.eval()
    x = torch.randn(2, 7, HIDDEN)
    with torch.no_grad():
        outs = [fusion(x, t) for t in range(TASKS)]
    for t in range(1, TASKS):
        assert (outs[0][:, 0] - outs[t][:, 0]).abs().max() > 1e-3, f"CLS for task {t} identical to task 0"
        assert (outs[0][:, 3] - outs[t][:, 3]).abs().max() > 1e-3, f"token features for task {t} identical"


def test_task_embeddings_receive_gradient():
    fusion = _fusion()
    _train_steps(fusion, steps=1)  # moves the zero-initialised projection off zero
    fusion.zero_grad()
    fusion(torch.randn(2, 5, HIDDEN), 1)[:, 0].sum().backward()
    assert fusion.task_embeddings.weight.grad[1].norm() > 0
    assert fusion.task_embeddings.weight.grad[0].norm() == 0  # only the active task's row


def test_padding_tokens_are_ignored():
    fusion = _fusion()
    _train_steps(fusion)
    fusion.eval()
    fusion.store_attention_weights = True
    x = torch.randn(1, 6, HIDDEN)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    x_other_padding = x.clone()
    x_other_padding[:, 4:] = torch.randn(1, 2, HIDDEN) * 50
    with torch.no_grad():
        a = fusion(x, 2, attention_mask=mask)[:, :4]
        b = fusion(x_other_padding, 2, attention_mask=mask)[:, :4]
        masked_weights = fusion.last_attention_weights
        c = fusion(x_other_padding, 2)[:, :4]  # without the mask, padding leaks in
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, c, atol=1e-3)
    assert masked_weights[0, 0, 4:].abs().max() == 0


def test_num_attention_heads_is_used_and_validated():
    assert _fusion(num_attention_heads=4).attention.num_heads == 4
    with pytest.raises(ValueError, match="must divide"):
        AttentionFusion(HIDDEN, TASKS, FusionConfig(type=FusionType.ATTENTION, num_attention_heads=3))


def test_all_fusion_types_accept_attention_mask():
    x, mask = torch.randn(2, 5, HIDDEN), torch.ones(2, 5, dtype=torch.long)
    for kind in (FusionType.ATTENTION, FusionType.CONCATENATION, FusionType.NONE):
        out = create_fusion(kind, HIDDEN, TASKS, FusionConfig(type=kind))(x, 1, attention_mask=mask)
        assert out.shape == x.shape


def test_pre_2_0_attention_fusion_checkpoint_gives_clear_error(tmp_path):
    import json

    from jenga_nlp.core.config import ExperimentConfig
    from jenga_nlp.core.model import MultiTaskModel

    (tmp_path / "metadata.json").write_text(json.dumps({"hidden_size": HIDDEN}))
    torch.save({"fusion.attention.0.weight": torch.zeros(1)}, tmp_path / "model.pt")
    cfg = ExperimentConfig(
        project_name="old",
        tasks=[{"name": "t", "type": "single_label_classification", "data_path": "x.jsonl",
                "heads": [{"name": "h", "num_labels": 2}]}],
    )
    with pytest.raises(ValueError, match="pre-2.0 attention fusion"):
        MultiTaskModel.load.__func__(_NoBuildModel, tmp_path, cfg)


class _NoBuildModel(torch.nn.Module):
    """Stand-in for MultiTaskModel so the test doesn't download an encoder."""

    def __init__(self, **kwargs):
        super().__init__()


# ---------------------------------------------------------------------------
# Task sampling
# ---------------------------------------------------------------------------

def _schedule(sizes: dict[str, int], strategy, temperature: float = 2.0, epoch: int = 0, batch_size: int = 1):
    loaders = {name: DataLoader(list(range(n)), batch_size=batch_size) for name, n in sizes.items()}
    fake = SimpleNamespace(
        train_dataloaders=loaders,
        training_config=SimpleNamespace(task_sampling=strategy, temperature=temperature, data=DataConfig(seed=7)),
    )
    return Trainer._build_task_schedule(fake, epoch)


def test_round_robin_matches_previous_interleaving():
    assert _schedule({"a": 3, "b": 1, "c": 2}, TaskSamplingStrategy.ROUND_ROBIN) == ["a", "b", "c", "a", "c", "a"]


def test_proportional_follows_dataset_sizes():
    sizes = {"big": 800, "small": 200}
    counts = Counter(_schedule(sizes, TaskSamplingStrategy.PROPORTIONAL))
    assert sum(counts.values()) == 1000  # same epoch length as round robin
    assert counts["big"] / 1000 == pytest.approx(0.8, abs=0.04)


def test_temperature_flattens_towards_uniform():
    sizes = {"big": 900, "small": 100}
    share = Counter(_schedule(sizes, TaskSamplingStrategy.TEMPERATURE, temperature=2.0))["small"] / 1000
    expected = 100 ** 0.5 / (900 ** 0.5 + 100 ** 0.5)  # 0.25, vs 0.10 proportional
    assert share == pytest.approx(expected, abs=0.04)
    near_uniform = Counter(_schedule(sizes, TaskSamplingStrategy.TEMPERATURE, temperature=100.0))["small"] / 1000
    assert near_uniform == pytest.approx(0.5, abs=0.06)


def test_sampled_schedules_are_reproducible_and_vary_by_epoch():
    sizes = {"a": 50, "b": 30, "c": 20}
    first = _schedule(sizes, TaskSamplingStrategy.PROPORTIONAL, epoch=0)
    assert first == _schedule(sizes, TaskSamplingStrategy.PROPORTIONAL, epoch=0)
    assert first != _schedule(sizes, TaskSamplingStrategy.PROPORTIONAL, epoch=1)


# ---------------------------------------------------------------------------
# Fusion learning-rate multiplier
# ---------------------------------------------------------------------------

def _optimizer_for(lr_multiplier: float):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(HIDDEN, HIDDEN)
    model.fusion = _fusion()
    fake = SimpleNamespace(
        model=model,
        config=SimpleNamespace(model=SimpleNamespace(fusion=FusionConfig(type=FusionType.ATTENTION,
                                                                        lr_multiplier=lr_multiplier))),
        training_config=SimpleNamespace(learning_rate=1e-4, weight_decay=0.0, warmup_steps=0),
    )
    optimizer, _ = Trainer._create_optimizer_and_scheduler(fake, num_training_steps=10)
    return model, optimizer


def test_fusion_lr_multiplier_creates_faster_param_group():
    model, optimizer = _optimizer_for(20.0)
    assert len(optimizer.param_groups) == 2
    encoder_group, fusion_group = optimizer.param_groups
    assert fusion_group["initial_lr"] == pytest.approx(20 * encoder_group["initial_lr"])
    assert {id(p) for p in fusion_group["params"]} == {id(p) for p in model.fusion.parameters()}
    all_params = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert sorted(all_params) == sorted(id(p) for p in model.parameters())  # nothing lost or duplicated


def test_default_multiplier_keeps_single_param_group():
    _, optimizer = _optimizer_for(1.0)
    assert len(optimizer.param_groups) == 1


def test_lr_multiplier_auto_default_depends_on_fusion_type():
    assert FusionConfig(type=FusionType.ATTENTION).effective_lr_multiplier() == 10.0
    assert FusionConfig(type=FusionType.CONCATENATION).effective_lr_multiplier() == 1.0
    assert FusionConfig(type=FusionType.NONE).effective_lr_multiplier() == 1.0
    assert FusionConfig(type=FusionType.CONCATENATION, lr_multiplier=5).effective_lr_multiplier() == 5.0
    assert FusionConfig(type=FusionType.ATTENTION, lr_multiplier=1).effective_lr_multiplier() == 1.0


def test_concatenation_fusion_keeps_single_param_group_by_default():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(HIDDEN, HIDDEN)
    model.fusion = create_fusion(FusionType.CONCATENATION, HIDDEN, TASKS, FusionConfig(type=FusionType.CONCATENATION))
    fake = SimpleNamespace(
        model=model,
        config=SimpleNamespace(model=SimpleNamespace(fusion=FusionConfig(type=FusionType.CONCATENATION))),
        training_config=SimpleNamespace(learning_rate=1e-4, weight_decay=0.0, warmup_steps=0),
    )
    optimizer, _ = Trainer._create_optimizer_and_scheduler(fake, num_training_steps=10)
    assert len(optimizer.param_groups) == 1


def test_attention_weights_are_not_computed_unless_requested():
    fusion = _fusion()
    fusion(torch.randn(2, 5, HIDDEN), 0)
    assert fusion.last_attention_weights is None
    fusion.store_attention_weights = True
    fusion(torch.randn(2, 5, HIDDEN), 0)
    assert fusion.last_attention_weights is not None


def test_resume_rejects_pre_2_0_attention_fusion_weights(tmp_path):
    torch.save({"fusion.attention.0.weight": torch.zeros(1)}, tmp_path / "model.pt")
    fake = SimpleNamespace(device="cpu", model=torch.nn.Linear(1, 1))
    with pytest.raises(ValueError, match="pre-2.0 attention fusion"):
        Trainer._load_resume_model_weights(fake, tmp_path)
