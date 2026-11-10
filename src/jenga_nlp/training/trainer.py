"""Multi-task trainer for JengaNLP.


- Mixed-precision training (AMP) with torch.cuda.amp
- Gradient clipping (configurable max_grad_norm)
- Gradient accumulation (configurable steps)
- Checkpoint saving/loading (resume from crash)
- Real eval loss computation (not fake 1-f1)
- Configurable task sampling: round-robin, proportional, temperature
- Callback system for modular extensibility
- Proper Python logging (not print statements)
- Model saving at end of training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from ..core.config import ExperimentConfig, TaskSamplingStrategy, TaskType
from ..core.model import MultiTaskModel
from ..data.collators import ClassificationCollator, NERCollator, RegressionCollator
from ..utils.device import get_device, move_to_device
from .callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TrainingCallback,
)
from .metrics import (
    compute_classification_metrics,
    compute_multi_label_metrics,
    compute_ner_metrics,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Multi-task trainer with AMP, gradient clipping, checkpoints, and callbacks.

    Args:
        config: Experiment configuration.
        model: MultiTaskModel instance.
        tokenizer: HuggingFace tokenizer.
        train_datasets: Dict mapping task_name -> training Dataset.
        eval_datasets: Dict mapping task_name -> evaluation Dataset.
        callbacks: Optional list of additional callbacks.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: MultiTaskModel,
        tokenizer: PreTrainedTokenizer,
        train_datasets: dict[str, Any],
        eval_datasets: dict[str, Any],
        callbacks: Optional[list[TrainingCallback]] = None,
        resume_from: str | Path | None = None,
        run_name: str | None = None,
        run_tags: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.training_config = config.training
        self.run_name = run_name or config.project_name
        self.run_tags = {"project_name": config.project_name, "mode": "train", **(run_tags or {})}

        # Device setup
        self.device = get_device(self.training_config.device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Task mapping
        self.task_map = {task.name: i for i, task in enumerate(config.tasks)}

        # Dataloaders
        self.train_dataloaders = self._create_dataloaders(train_datasets, is_eval=False)
        self.eval_dataloaders = self._create_dataloaders(eval_datasets, is_eval=True)

        # AMP setup
        self.use_amp = self.training_config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Callbacks
        self.callbacks: list[TrainingCallback] = callbacks or []
        self._setup_builtin_callbacks()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.training_history: list[dict[str, Any]] = []
        self.resume_from = Path(resume_from) if resume_from else None
        self._resume_state: dict[str, Any] | None = None
        if self.resume_from is not None:
            self._resume_state = self._load_resume_state(self.resume_from)
            self._load_resume_model_weights(self.resume_from)
            self.global_step = int(self._resume_state.get("global_step", 0) or 0)
            self.current_epoch = int(self._resume_state.get("epoch", 0) or 0) + 1

    def _setup_builtin_callbacks(self) -> None:
        """Set up built-in callbacks from config."""
        # Logging callback
        if self.training_config.logging:
            config_dict = self.config.model_dump(mode="json")
            self.callbacks.append(
                LoggingCallback(
                    self.training_config.logging,
                    self.training_config.output_dir,
                    run_name=self.run_name,
                    params=_tracking_params(self.config),
                    tags={**self.run_tags, "tasks": ",".join(t.name for t in self.config.tasks)},
                    config_dict=config_dict,
                )
            )

        # Early stopping callback
        if self.training_config.early_stopping_patience:
            self.callbacks.append(
                EarlyStoppingCallback(
                    patience=self.training_config.early_stopping_patience,
                    metric_name=self.training_config.metric_for_best_model,
                    greater_is_better=self.training_config.greater_is_better,
                )
            )

        # Checkpoint callback
        if self.training_config.checkpoint:
            self.callbacks.append(
                CheckpointCallback(
                    config=self.training_config.checkpoint,
                    output_dir=self.training_config.output_dir,
                    metric_name=self.training_config.metric_for_best_model,
                    greater_is_better=self.training_config.greater_is_better,
                )
            )

    def log_final_metrics(
        self, metrics: dict[str, Any], artifact_files: list[str | Path] | None = None
    ) -> None:
        """Send metrics (and result files) from a standalone ``evaluate()`` to the tracker, then close it."""
        for cb in self.callbacks:
            if isinstance(cb, LoggingCallback):
                cb.log_metrics(metrics)
                for path in artifact_files or []:
                    cb.log_artifact(path)
                cb.close()

    def _create_dataloaders(self, datasets: dict[str, Any], is_eval: bool) -> dict[str, DataLoader]:
        """Create task-specific dataloaders with proper collators."""
        dataloaders = {}
        batch_size = (
            self.training_config.eval_batch_size if is_eval else self.training_config.batch_size
        )

        for task_name, dataset in datasets.items():
            task_config = next(t for t in self.config.tasks if t.name == task_name)

            # Select collator based on task type (uses classes, not lambdas)
            if task_config.type == TaskType.NER:
                collator = NERCollator(self.tokenizer)
            elif task_config.type == TaskType.REGRESSION:
                collator = RegressionCollator(self.tokenizer)
            else:
                collator = ClassificationCollator(self.tokenizer)

            dataloaders[task_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=not is_eval,
                collate_fn=collator,
                num_workers=self.training_config.data.num_workers,
                pin_memory=self.training_config.data.pin_memory and self.device.type == "cuda",
            )

        return dataloaders


    def _load_resume_state(self, resume_from: Path) -> dict[str, Any]:
        import torch

        state_candidates = []
        if resume_from.is_file() and resume_from.name == "training_state.pt":
            state_candidates.append(resume_from)
        elif resume_from.is_dir():
            state_candidates.append(resume_from / "training_state.pt")
            state_candidates.extend(sorted(resume_from.rglob("training_state.pt")))

        for candidate in state_candidates:
            if candidate.exists():
                return torch.load(candidate, map_location="cpu")

        raise FileNotFoundError(f"No training_state.pt found under {resume_from}")

    def _load_resume_model_weights(self, resume_from: Path) -> None:
        import torch

        model_candidates = []
        if resume_from.is_file() and resume_from.name == "model.pt":
            model_candidates.append(resume_from)
        elif resume_from.is_file() and resume_from.name == "training_state.pt":
            model_candidates.append(resume_from.parent / "model.pt")
        elif resume_from.is_dir():
            model_candidates.append(resume_from / "model.pt")
            model_candidates.extend(sorted(resume_from.rglob("model.pt")))

        for candidate in model_candidates:
            if candidate.exists():
                state_dict = torch.load(candidate, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights from {candidate}")
                return

        raise FileNotFoundError(f"No model.pt found under {resume_from}")

    def _create_optimizer_and_scheduler(self, num_training_steps: int) -> tuple[AdamW, Any]:
        """Create optimizer and learning rate scheduler."""
        lr = self.training_config.learning_rate
        fusion = getattr(self.model, "fusion", None)
        fusion_cfg = self.config.model.fusion
        fusion_params = list(fusion.parameters()) if fusion is not None else []
        if fusion_params and fusion_cfg is not None and fusion_cfg.lr_multiplier != 1.0:
            # Fusion layers start from scratch, so they may need a higher learning
            # rate than the pretrained encoder to learn within a fine-tuning run.
            fusion_ids = {id(p) for p in fusion_params}
            param_groups: Any = [
                {"params": [p for p in self.model.parameters() if id(p) not in fusion_ids], "lr": lr},
                {"params": fusion_params, "lr": lr * fusion_cfg.lr_multiplier},
            ]
        else:
            param_groups = self.model.parameters()
        optimizer = AdamW(param_groups, lr=lr, weight_decay=self.training_config.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, scheduler

    def train(self) -> dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with training history and final metrics.
        """
        num_steps_per_epoch = sum(len(dl) for dl in self.train_dataloaders.values())
        total_steps = num_steps_per_epoch * self.training_config.num_epochs
        effective_steps = total_steps // self.training_config.gradient_accumulation_steps

        optimizer, scheduler = self._create_optimizer_and_scheduler(effective_steps)
        if self._resume_state is not None:
            if self._resume_state.get("optimizer_state_dict") is not None:
                try:
                    optimizer.load_state_dict(self._resume_state["optimizer_state_dict"])
                except ValueError as exc:  # e.g. parameter groups changed (fusion lr_multiplier)
                    logger.warning(f"Could not restore optimizer state ({exc}); continuing with a fresh optimizer")
            if self._resume_state.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(self._resume_state["scheduler_state_dict"])
            if self._resume_state.get("scaler_state_dict") is not None and self.scaler is not None:
                self.scaler.load_state_dict(self._resume_state["scaler_state_dict"])
            logger.info(f"Resuming from epoch {self.current_epoch} and step {self.global_step}")

        logger.info(f"Starting training: {self.training_config.num_epochs} epochs, "
                     f"{total_steps} total steps, device={self.device}")
        if self.use_amp:
            logger.info("Automatic Mixed Precision (AMP) enabled")

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_begin(model=self.model, resume_from=str(self.resume_from) if self.resume_from else None)

        progress = tqdm(
            total=total_steps,
            initial=min(self.global_step, total_steps),
            desc="Training",
        )

        # Defined up front so resuming an already-finished run (where the epoch
        # loop below does not execute) still returns valid final metrics.
        eval_metrics: dict[str, Any] = {}
        if self.current_epoch >= self.training_config.num_epochs:
            logger.info(
                f"Run already completed {self.training_config.num_epochs} epoch(s); "
                "nothing left to train. Increase training.num_epochs to continue training."
            )
            if self.eval_dataloaders:
                eval_metrics = self.evaluate()

        try:
            for epoch in range(self.current_epoch, self.training_config.num_epochs):
                self.current_epoch = epoch
                self.model.train()

                for cb in self.callbacks:
                    cb.on_epoch_begin(epoch=epoch)

                iterators = {name: iter(dl) for name, dl in self.train_dataloaders.items()}
                epoch_losses: list[float] = []

                for task_name in self._build_task_schedule(epoch):
                    try:
                        batch = next(iterators[task_name])
                    except StopIteration:
                        # Sampled schedules can revisit a small task: start a fresh pass over it.
                        iterators[task_name] = iter(self.train_dataloaders[task_name])
                        batch = next(iterators[task_name])

                    loss = self._training_step(batch, task_name, optimizer, scheduler)

                    epoch_losses.append(loss)
                    self.global_step += 1

                    # Callbacks
                    for cb in self.callbacks:
                        cb.on_step(step=self.global_step, loss=loss)

                    progress.update(1)
                    progress.set_postfix(loss=f"{loss:.4f}", epoch=epoch + 1)

                # Evaluate
                eval_metrics = {}
                if self.eval_dataloaders:
                    eval_metrics = self.evaluate()

                # Add training loss to metrics
                eval_metrics["train_loss_avg"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0

                # Store history
                self.training_history.append({"epoch": epoch + 1, **eval_metrics})

                logger.info(f"Epoch {epoch + 1}/{self.training_config.num_epochs} - Metrics: {eval_metrics}")

                # Epoch-end callbacks
                for cb in self.callbacks:
                    cb.on_epoch_end(
                        epoch=epoch,
                        metrics=eval_metrics,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=self.scaler,
                        global_step=self.global_step,
                        callbacks=self.callbacks,
                    )

                # Check early stopping
                if any(cb.should_stop() for cb in self.callbacks):
                    logger.info("Training stopped early")
                    break

        except BaseException as exc:
            progress.close()
            if isinstance(exc, Exception):
                for cb in self.callbacks:
                    on_error = getattr(cb, "on_train_error", None)
                    if callable(on_error):
                        on_error(exc, model=self.model, global_step=self.global_step, epoch=self.current_epoch)
            # Still close loggers/trackers, but there is no finished model to publish.
            for cb in self.callbacks:
                cb.on_train_end(model=self.model)
            raise

        progress.close()

        # Save the final, self-contained model folder *before* end-of-training
        # callbacks run, so trackers (e.g. MLflow) can upload it.
        output_dir = Path(self.training_config.output_dir)
        self.model.save(output_dir)
        self._save_tokenizer(output_dir)
        self.config.to_yaml(output_dir / "experiment_config.yaml")

        for cb in self.callbacks:
            cb.on_train_end(model=self.model, output_dir=str(output_dir))

        return {"history": self.training_history, "final_metrics": eval_metrics}

    def _build_task_schedule(self, epoch: int) -> list[str]:
        """Return the order in which tasks are trained during one epoch.

        The schedule always has ``sum(len(dataloader))`` steps, so the learning-rate
        schedule is the same for every strategy.

        - ``round_robin``: one batch per task in turn until each task is exhausted.
        - ``proportional``: each step picks a task with probability proportional to
          its number of training examples.
        - ``temperature``: probability proportional to ``size ** (1 / temperature)``;
          ``temperature > 1`` flattens the distribution, upsampling small tasks.

        Sampled schedules are reproducible: the generator is seeded from
        ``training.data.seed`` and the epoch.
        """
        loaders = self.train_dataloaders
        names = list(loaders.keys())
        batches = {name: len(dl) for name, dl in loaders.items()}
        total_steps = sum(batches.values())
        strategy = self.training_config.task_sampling

        if strategy == TaskSamplingStrategy.ROUND_ROBIN or len(names) <= 1:
            schedule: list[str] = []
            remaining = dict(batches)
            while any(remaining.values()):
                for name in names:
                    if remaining[name]:
                        schedule.append(name)
                        remaining[name] -= 1
            return schedule

        sizes = torch.tensor([float(len(loaders[n].dataset)) for n in names])
        exponent = 1.0 if strategy == TaskSamplingStrategy.PROPORTIONAL else 1.0 / self.training_config.temperature
        probs = sizes.pow(exponent)
        probs = probs / probs.sum()

        seed = self.training_config.data.seed if self.training_config.data is not None else 0
        generator = torch.Generator().manual_seed(int(seed) * 1000 + epoch)
        picks = torch.multinomial(probs, total_steps, replacement=True, generator=generator)
        return [names[i] for i in picks.tolist()]

    def _save_tokenizer(self, output_dir: Path) -> None:
        """Save the tokenizer next to the weights so inference does not need the Hub."""
        from ..core.model import TOKENIZER_DIRNAME

        save_pretrained = getattr(self.tokenizer, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(str(output_dir / TOKENIZER_DIRNAME))

    def _training_step(
        self,
        batch: dict[str, Any],
        task_name: str,
        optimizer: AdamW,
        scheduler: Any,
    ) -> float:
        """Execute a single training step."""
        task_id = self.task_map[task_name]

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = move_to_device(labels, self.device)

        # Also handle multi-label format (labels_headname keys)
        if labels is None:
            label_keys = [k for k in batch.keys() if k.startswith("labels_")]
            if label_keys:
                labels = {k.replace("labels_", ""): batch[k].to(self.device) for k in label_keys}

        # Forward pass with optional AMP
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id,
                    labels=labels,
                )
                loss = outputs.loss / self.training_config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task_id=task_id,
                labels=labels,
            )
            loss = outputs.loss / self.training_config.gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.training_config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm,
            )

            if self.use_amp:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        return loss.item() * self.training_config.gradient_accumulation_steps

    def evaluate(self) -> dict[str, float]:
        """Run evaluation on all tasks.

        Returns:
            Dict of metric_name -> value for all tasks.
        """
        self.model.eval()
        all_metrics: dict[str, float] = {}
        total_eval_loss = 0.0
        total_eval_batches = 0

        for task_name, dataloader in self.eval_dataloaders.items():
            task_config = next(t for t in self.config.tasks if t.name == task_name)

            all_preds: dict[str, list[np.ndarray]] = {h.name: [] for h in task_config.heads}
            all_labels: dict[str, list[np.ndarray]] = {h.name: [] for h in task_config.heads}
            task_eval_loss = 0.0
            task_batches = 0

            with torch.no_grad():
                for batch in dataloader:
                    task_id = self.task_map[task_name]
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch.get("labels")
                    if labels is not None:
                        labels = move_to_device(labels, self.device)

                    # Handle multi-label format
                    if labels is None:
                        label_keys = [k for k in batch.keys() if k.startswith("labels_")]
                        if label_keys:
                            labels = {k.replace("labels_", ""): batch[k].to(self.device) for k in label_keys}

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        task_id=task_id,
                        labels=labels,
                    )

                    # Accumulate real eval loss
                    if outputs.loss is not None:
                        task_eval_loss += outputs.loss.item()
                        task_batches += 1

                    # Collect predictions and labels per head
                    for head_config in task_config.heads:
                        head_name = head_config.name
                        if head_name in outputs.logits:
                            logits = outputs.logits[head_name].cpu().numpy()
                            all_preds[head_name].append(logits)

                            if isinstance(labels, dict) and head_name in labels:
                                all_labels[head_name].append(labels[head_name].cpu().numpy())
                            elif isinstance(labels, torch.Tensor):
                                all_labels[head_name].append(labels.cpu().numpy())

            # Compute task metrics
            for head_config in task_config.heads:
                head_name = head_config.name
                if not all_preds[head_name]:
                    continue

                preds_np = np.concatenate(all_preds[head_name], axis=0)
                labels_np = np.concatenate(all_labels[head_name], axis=0)

                if task_config.type == TaskType.NER:
                    metrics = compute_ner_metrics(preds_np, labels_np)
                elif task_config.type in (TaskType.MULTI_LABEL_CLASSIFICATION, TaskType.QA_SCORING, TaskType.QUESTION_ANSWERING):
                    metrics = compute_multi_label_metrics(preds_np, labels_np)
                elif task_config.type == TaskType.REGRESSION:
                    metrics = compute_regression_metrics(preds_np, labels_np)
                else:
                    metrics = compute_classification_metrics(preds_np, labels_np)

                for metric_name, value in metrics.items():
                    all_metrics[f"{task_name}_{head_name}_{metric_name}"] = value

            total_eval_loss += task_eval_loss
            total_eval_batches += task_batches

        # Compute real average eval loss
        all_metrics["eval_loss"] = (
            total_eval_loss / total_eval_batches if total_eval_batches > 0 else 0.0
        )

        return all_metrics

    def close(self) -> None:
        """Clean up resources."""
        for cb in self.callbacks:
            cb.on_train_end()


def _tracking_params(config: ExperimentConfig) -> dict[str, Any]:
    """Flatten the hyperparameters worth comparing across runs into tracker params."""
    t = config.training
    params: dict[str, Any] = {
        "base_model": config.model.base_model,
        "dropout": config.model.dropout,
        "max_length": config.tokenizer.max_length,
        "learning_rate": t.learning_rate,
        "batch_size": t.batch_size,
        "eval_batch_size": t.eval_batch_size,
        "num_epochs": t.num_epochs,
        "weight_decay": t.weight_decay,
        "warmup_steps": t.warmup_steps,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "max_grad_norm": t.max_grad_norm,
        "use_amp": t.use_amp,
        "device": t.device,
        "task_sampling": getattr(t.task_sampling, "value", t.task_sampling),
        "num_tasks": len(config.tasks),
    }
    fusion = config.model.fusion
    params["fusion"] = getattr(fusion.type, "value", fusion.type) if fusion is not None else "none"
    if fusion is not None and params["fusion"] != "none":
        params["fusion.lr_multiplier"] = fusion.lr_multiplier
        params["fusion.num_attention_heads"] = fusion.num_attention_heads
        params["fusion.gate_init_value"] = fusion.gate_init_value
    if t.data is not None:
        params["test_size"] = t.data.test_size
        params["seed"] = t.data.seed
    for task in config.tasks:
        params[f"task.{task.name}.type"] = getattr(task.type, "value", task.type)
        params[f"task.{task.name}.heads"] = ",".join(f"{h.name}:{h.num_labels}" for h in task.heads)
    return params
