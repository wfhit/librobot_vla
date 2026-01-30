"""Checkpoint callback for saving models."""

from pathlib import Path
from typing import Any, Optional

from .base import AbstractCallback


class CheckpointCallback(AbstractCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_dir: str,
        save_freq: int = 1,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        max_checkpoints: int = 5,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_freq: Save frequency (epochs)
            save_best: Whether to save best model
            save_last: Whether to save last model
            monitor: Metric to monitor for best
            mode: "min" or "max" for best metric
            max_checkpoints: Maximum checkpoints to keep
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode
        self.max_checkpoints = max_checkpoints

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.checkpoints = []

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Save checkpoint at epoch end."""
        logs = logs or {}

        # Save periodic checkpoint
        if (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(f"epoch_{epoch+1}.pt", epoch, logs)

        # Save best checkpoint
        if self.save_best and self.monitor in logs:
            current_value = logs[self.monitor]
            is_better = (self.mode == "min" and current_value < self.best_value) or (
                self.mode == "max" and current_value > self.best_value
            )

            if is_better:
                self.best_value = current_value
                self._save_checkpoint("best.pt", epoch, logs)

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Save last checkpoint."""
        if self.save_last and self.trainer:
            self._save_checkpoint("last.pt", self.trainer.current_epoch, logs or {})

    def _save_checkpoint(self, filename: str, epoch: int, logs: dict[str, Any]) -> None:
        """Save a checkpoint."""
        if self.trainer is None:
            return

        path = self.save_dir / filename

        try:
            import torch

            checkpoint = {
                "epoch": epoch,
                "global_step": self.trainer.global_step,
                "model_state_dict": self.trainer.model.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "metrics": logs,
            }

            if hasattr(self.trainer, "scheduler") and self.trainer.scheduler:
                checkpoint["scheduler_state_dict"] = self.trainer.scheduler.state_dict()

            torch.save(checkpoint, path)

            # Track checkpoints for cleanup
            if "epoch_" in filename:
                self.checkpoints.append(path)
                self._cleanup_old_checkpoints()

        except ImportError:
            pass

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max."""
        while len(self.checkpoints) > self.max_checkpoints:
            old_path = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()


class ModelCheckpoint(CheckpointCallback):
    """Alias for CheckpointCallback."""

    pass


__all__ = ["CheckpointCallback", "ModelCheckpoint"]
