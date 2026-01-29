"""Early stopping callback."""

from typing import Any, Dict, Optional

from .base import AbstractCallback


class EarlyStopping(AbstractCallback):
    """Early stopping callback to stop training when metric stops improving."""
    
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            monitor: Metric to monitor
            patience: Number of epochs to wait
            mode: "min" or "max" for improvement direction
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Restore best model weights on stop
            verbose: Print messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset state at training start."""
        self.wait = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for improvement at epoch end."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return
        
        # Check if improved
        if self._is_improvement(current):
            self.best_value = current
            self.wait = 0
            self.best_epoch = epoch
            
            if self.restore_best_weights and self.trainer:
                self.best_weights = {
                    k: v.clone() for k, v in self.trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.trainer:
                    self.trainer.stop_training()
                
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    print(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1}")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Restore best weights if needed."""
        if self.restore_best_weights and self.best_weights and self.trainer:
            self.trainer.model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch + 1}")
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == 'min':
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta


class LearningRateScheduler(AbstractCallback):
    """Callback for learning rate scheduling."""
    
    def __init__(
        self,
        schedule_type: str = "cosine",
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        """
        Args:
            schedule_type: Type of schedule ("cosine", "linear", "step")
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
            verbose: Print LR changes
        """
        super().__init__()
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.initial_lr = None
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Store initial learning rate."""
        if self.trainer and self.trainer.optimizer:
            self.initial_lr = self.trainer.optimizer.param_groups[0]['lr']
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Update learning rate at epoch start."""
        if self.trainer is None or self.initial_lr is None:
            return
        
        new_lr = self._compute_lr(epoch)
        
        for param_group in self.trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        if self.verbose:
            print(f"Learning rate: {new_lr:.2e}")
    
    def _compute_lr(self, epoch: int) -> float:
        """Compute learning rate for epoch."""
        # Warmup
        if epoch < self.warmup_epochs:
            return self.initial_lr * (epoch + 1) / self.warmup_epochs
        
        # Post-warmup schedule
        effective_epoch = epoch - self.warmup_epochs
        max_epochs = self.trainer.max_epochs - self.warmup_epochs if self.trainer else 100
        
        if self.schedule_type == "cosine":
            import math
            return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                   (1 + math.cos(math.pi * effective_epoch / max_epochs))
        
        elif self.schedule_type == "linear":
            return self.initial_lr - (self.initial_lr - self.min_lr) * effective_epoch / max_epochs
        
        else:  # step
            return self.initial_lr * (0.1 ** (effective_epoch // 30))


class GradientMonitor(AbstractCallback):
    """Callback to monitor gradient statistics."""
    
    def __init__(
        self,
        log_interval: int = 100,
        verbose: bool = False,
    ):
        """
        Args:
            log_interval: Interval for logging gradient stats
            verbose: Print gradient stats
        """
        super().__init__()
        self.log_interval = log_interval
        self.verbose = verbose
        self._step = 0
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log gradient statistics."""
        self._step += 1
        
        if self._step % self.log_interval != 0:
            return
        
        if self.trainer is None:
            return
        
        try:
            import torch
            
            total_norm = 0.0
            max_grad = 0.0
            num_params = 0
            
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
                    num_params += 1
            
            total_norm = total_norm ** 0.5
            
            if self.verbose:
                print(f"Grad norm: {total_norm:.4f}, Max: {max_grad:.4f}")
                
        except ImportError:
            pass


__all__ = [
    'EarlyStopping',
    'LearningRateScheduler',
    'GradientMonitor',
]
