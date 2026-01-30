"""Base trainer class for VLA models."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import time
import numpy as np


class BaseTrainer(ABC):
    """
    Base trainer class for VLA models.

    Provides common training loop functionality and hooks for customization.
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = 1.0,
        log_interval: int = 10,
        val_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        seed: int = 42,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            callbacks: List of callback instances
            max_epochs: Maximum training epochs
            max_steps: Maximum training steps (overrides epochs)
            gradient_accumulation_steps: Steps for gradient accumulation
            gradient_clip_val: Gradient clipping value
            log_interval: Logging interval (steps)
            val_interval: Validation interval (epochs)
            checkpoint_dir: Directory for checkpoints
            device: Training device
            mixed_precision: Use mixed precision training
            seed: Random seed
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks or []
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_val = gradient_clip_val
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device
        self.mixed_precision = mixed_precision
        self.seed = seed

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self._stop_training = False

        # Metrics tracking
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

        # Initialize callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Training history dictionary
        """
        self._call_callbacks('on_train_begin')

        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                if self._stop_training:
                    break

                self.current_epoch = epoch
                self._call_callbacks('on_epoch_begin', epoch=epoch)

                # Training epoch
                train_loss = self._train_epoch()

                # Validation
                if self.val_dataloader and epoch % self.val_interval == 0:
                    val_loss = self._validate()

                    # Check for best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.checkpoint_dir:
                            self.save_checkpoint('best.pt')

                self._call_callbacks('on_epoch_end', epoch=epoch)

                # Check max steps
                if self.max_steps and self.global_step >= self.max_steps:
                    break

        finally:
            self._call_callbacks('on_train_end')

        return self.get_training_history()

    @abstractmethod
    def _train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss
        """
        pass

    @abstractmethod
    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform single training step.

        Args:
            batch: Training batch

        Returns:
            Dictionary with loss and metrics
        """
        pass

    def _validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        self._call_callbacks('on_validation_begin')

        total_loss = 0
        num_batches = 0

        try:
            import torch
            with torch.no_grad():
                for batch in self.val_dataloader:
                    loss = self._validate_step(batch)
                    total_loss += loss
                    num_batches += 1
        except ImportError:
            pass

        avg_loss = total_loss / max(num_batches, 1)
        self._call_callbacks('on_validation_end', logs={'val_loss': avg_loss})

        self.model.train()
        return avg_loss

    def _validate_step(self, batch: Dict[str, Any]) -> float:
        """
        Perform single validation step.

        Args:
            batch: Validation batch

        Returns:
            Validation loss value
        """
        # Default implementation - override for custom validation
        result = self._train_step(batch)
        return result.get('loss', 0)

    def _call_callbacks(self, method: str, **kwargs) -> None:
        """Call callback method on all callbacks."""
        for callback in self.callbacks:
            fn = getattr(callback, method, None)
            if fn:
                fn(**kwargs)

    def save_checkpoint(self, filename: str) -> None:
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename

        try:
            import torch
            checkpoint = {
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
            }
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            torch.save(checkpoint, path)
        except ImportError:
            pass

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        try:
            import torch
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (ImportError, FileNotFoundError):
            pass

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.

        Returns:
            Dictionary with training metrics
        """
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
        }

    def stop_training(self) -> None:
        """Signal to stop training."""
        self._stop_training = True


__all__ = ['BaseTrainer']
