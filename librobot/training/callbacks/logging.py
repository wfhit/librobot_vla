"""Logging callbacks for training."""

import json
import time
from pathlib import Path
from typing import Any, Optional

from .base import AbstractCallback


class LoggingCallback(AbstractCallback):
    """Basic logging callback."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_interval: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            log_dir: Directory to save logs
            log_interval: Logging interval (batches)
            verbose: Print to console
        """
        super().__init__()
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_interval = log_interval
        self.verbose = verbose

        self.history: dict[str, list[float]] = {}
        self.epoch_metrics: dict[str, float] = {}
        self._batch_count = 0
        self._epoch_start_time = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Called at training start."""
        if self.verbose:
            print("Training started...")

    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Called at epoch start."""
        self._epoch_start_time = time.time()
        self._batch_count = 0
        self.epoch_metrics = {}

        if self.verbose:
            print(f"\nEpoch {epoch + 1}")
            print("-" * 40)

    def on_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Called at batch end."""
        self._batch_count += 1
        logs = logs or {}

        # Accumulate metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.epoch_metrics:
                    self.epoch_metrics[key] = []
                if isinstance(self.epoch_metrics[key], list):
                    self.epoch_metrics[key].append(value)

        # Log at intervals
        if self.verbose and self._batch_count % self.log_interval == 0:
            loss = logs.get('loss', 0)
            print(f"  Batch {batch}: loss = {loss:.4f}")

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Called at epoch end."""
        logs = logs or {}
        epoch_time = time.time() - self._epoch_start_time

        # Compute epoch averages
        for key, values in self.epoch_metrics.items():
            if isinstance(values, list) and values:
                avg_value = sum(values) / len(values)
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(avg_value)

        if self.verbose:
            print(f"  Time: {epoch_time:.1f}s")
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Called at training end."""
        if self.verbose:
            print("\nTraining completed!")

        if self.log_dir:
            self._save_history()

    def _save_history(self) -> None:
        """Save training history to file."""
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


class TensorBoardCallback(AbstractCallback):
    """TensorBoard logging callback."""

    def __init__(
        self,
        log_dir: str,
        log_interval: int = 10,
    ):
        """
        Args:
            log_dir: TensorBoard log directory
            log_interval: Logging interval
        """
        super().__init__()
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.writer = None
        self._batch_count = 0

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available")

    def on_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Log batch metrics."""
        if self.writer is None:
            return

        logs = logs or {}
        global_step = self.trainer.global_step if self.trainer else batch

        if global_step % self.log_interval == 0:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, global_step)

    def on_validation_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Log validation metrics."""
        if self.writer is None:
            return

        logs = logs or {}
        global_step = self.trainer.global_step if self.trainer else 0

        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'val/{key}', value, global_step)

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class WandBCallback(AbstractCallback):
    """Weights & Biases logging callback."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[dict] = None,
        log_interval: int = 10,
    ):
        """
        Args:
            project: W&B project name
            name: Run name
            config: Configuration to log
            log_interval: Logging interval
        """
        super().__init__()
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_interval = log_interval
        self._wandb = None

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Initialize W&B run."""
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
            )
        except ImportError:
            print("wandb not available")

    def on_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Log batch metrics."""
        if self._wandb is None:
            return

        logs = logs or {}
        global_step = self.trainer.global_step if self.trainer else batch

        if global_step % self.log_interval == 0:
            self._wandb.log({f'train/{k}': v for k, v in logs.items()
                           if isinstance(v, (int, float))}, step=global_step)

    def on_validation_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Log validation metrics."""
        if self._wandb is None:
            return

        logs = logs or {}
        global_step = self.trainer.global_step if self.trainer else 0
        self._wandb.log({f'val/{k}': v for k, v in logs.items()
                       if isinstance(v, (int, float))}, step=global_step)

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Finish W&B run."""
        if self._wandb:
            self._wandb.finish()


__all__ = [
    'LoggingCallback',
    'TensorBoardCallback',
    'WandBCallback',
]
