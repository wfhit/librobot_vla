"""Main trainer class with training loop orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from librobot.training.callbacks.base import AbstractCallback
from librobot.training.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    wrap_model_distributed,
)
from librobot.training.losses.base import AbstractLoss
from librobot.utils.checkpoint import Checkpoint
from librobot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for Trainer.

    Args:
        output_dir: Directory for outputs (checkpoints, logs)
        max_epochs: Maximum number of training epochs
        max_steps: Maximum number of training steps (overrides max_epochs)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        gradient_clip_norm: Maximum gradient norm (None to disable)
        gradient_clip_value: Maximum gradient value (None to disable)
        mixed_precision: Whether to use mixed precision training
        log_interval: Log metrics every N steps
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps
        save_total_limit: Keep only N most recent checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
        seed: Random seed for reproducibility
        dataloader_num_workers: Number of workers for data loading
        dataloader_pin_memory: Whether to pin memory for data loading

        # Distributed training
        distributed_strategy: Strategy for distributed training ('ddp', 'deepspeed', 'fsdp')
        distributed_backend: Backend for distributed training ('nccl', 'gloo')
        find_unused_parameters: Find unused parameters in DDP

        # Logging
        use_wandb: Whether to use Weights & Biases logging
        use_tensorboard: Whether to use TensorBoard logging
        logging_dir: Directory for logging outputs
        project_name: Project name for logging
        run_name: Run name for logging
    """

    # Training
    output_dir: str = "./outputs"
    max_epochs: int = 10
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: Optional[float] = 1.0
    gradient_clip_value: Optional[float] = None
    mixed_precision: bool = True

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: Optional[int] = None
    save_interval: int = 1000  # Save checkpoint every N steps (during training) or epochs (during epoch-based training)
    save_total_limit: Optional[int] = 5
    resume_from_checkpoint: Optional[str] = None

    # Reproducibility
    seed: int = 42

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # Distributed
    distributed_strategy: str = "ddp"
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = False

    # Logging integrations
    use_wandb: bool = False
    use_tensorboard: bool = False
    logging_dir: Optional[str] = None
    project_name: str = "librobot"
    run_name: Optional[str] = None


class Trainer:
    """
    Main trainer class for model training with PyTorch Lightning-style API.

    Features:
    - Automatic mixed precision training
    - Gradient accumulation and clipping
    - Distributed training (DDP, DeepSpeed, FSDP)
    - Checkpointing and resuming
    - Callbacks for custom behavior
    - Logging (WandB, TensorBoard)
    - Validation during training

    Examples:
        >>> config = TrainerConfig(max_epochs=10, mixed_precision=True)
        >>> trainer = Trainer(
        ...     config=config,
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     loss_fn=loss_fn,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        loss_fn: Optional[Union[nn.Module, AbstractLoss]] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        callbacks: Optional[list[AbstractCallback]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Trainer configuration
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer (created if None)
            scheduler: Learning rate scheduler
            callbacks: List of training callbacks
            device: Device to train on
        """
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup distributed training
        self.distributed_config = setup_distributed(backend=config.distributed_backend)

        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = f"cuda:{self.distributed_config.local_rank}"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Wrap model for distributed training
        if self.distributed_config.is_distributed:
            self.model = wrap_model_distributed(
                self.model,
                strategy=config.distributed_strategy,
                config=self.distributed_config,
                find_unused_parameters=config.find_unused_parameters,
            )

        # Setup optimizer
        if optimizer is None:
            # TODO: Implement default optimizer creation from config
            logger.warning("No optimizer provided - must be set before training")
        self.optimizer = optimizer

        # Setup scheduler
        self.scheduler = scheduler

        # Setup mixed precision
        # Note: scaler is None when mixed_precision is False
        if config.mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Setup checkpointing
        self.checkpoint_manager = Checkpoint(
            save_dir=self.output_dir / "checkpoints",
            keep_last_n=config.save_total_limit,
            save_best=True,
            metric_name="val_loss",
            mode="min",
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')

        # Setup callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

        # Setup logging
        self._setup_logging()

        # Resume from checkpoint if specified
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def _setup_logging(self) -> None:
        """Setup logging integrations (WandB, TensorBoard)."""
        # TODO: Implement WandB integration
        if self.config.use_wandb:
            logger.info("WandB logging requested but not yet implemented")

        # TODO: Implement TensorBoard integration
        if self.config.use_tensorboard:
            logger.info("TensorBoard logging requested but not yet implemented")

    def train(self) -> dict[str, Any]:
        """
        Run training loop.

        Returns:
            Dictionary containing training metrics and final state
        """
        logger.info("Starting training")
        self._call_callbacks("on_train_begin")

        try:
            if self.config.max_steps:
                # Step-based training
                self._train_steps()
            else:
                # Epoch-based training
                self._train_epochs()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        except Exception as e:
            logger.exception(f"Training failed with error: {e}")
            raise

        finally:
            self._call_callbacks("on_train_end")
            cleanup_distributed()

        logger.info("Training completed")
        return {"global_step": self.global_step, "epoch": self.current_epoch}

    def _train_epochs(self) -> None:
        """Train for specified number of epochs."""
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            self._call_callbacks("on_epoch_begin", epoch=epoch)

            # Train one epoch
            epoch_metrics = self._train_epoch()

            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.validate()
                epoch_metrics.update(val_metrics)

            # Save checkpoint
            if self.config.save_interval and (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch=epoch, metrics=epoch_metrics)

            self._call_callbacks("on_epoch_end", epoch=epoch, logs=epoch_metrics)

            logger.info(f"Epoch {epoch} completed - Metrics: {epoch_metrics}")

    def _train_steps(self) -> None:
        """Train for specified number of steps."""
        while self.global_step < self.config.max_steps:
            # Train one epoch (may break early if max_steps reached)
            self._train_epoch()

            if self.global_step >= self.config.max_steps:
                break

    def _train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Check if max steps reached
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

            self._call_callbacks("on_batch_begin", batch=batch_idx)

            # Training step
            loss = self._training_step(batch)
            epoch_loss += loss
            num_batches += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                logger.info(
                    f"Step {self.global_step} | "
                    f"Epoch {self.current_epoch} | "
                    f"Loss: {loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

            # Validation
            if (self.config.eval_interval and
                self.val_dataloader is not None and
                self.global_step % self.config.eval_interval == 0):
                val_metrics = self.validate()
                logger.info(f"Validation metrics: {val_metrics}")

            # Checkpointing
            if (self.config.save_interval and
                self.global_step % self.config.save_interval == 0):
                self.save_checkpoint(step=self.global_step, metrics={"loss": loss})

            self._call_callbacks("on_batch_end", batch=batch_idx, logs={"loss": loss})

        avg_loss = epoch_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    def _training_step(self, batch: Any) -> float:
        """
        Execute one training step.

        Args:
            batch: Training batch

        Returns:
            Loss value
        """
        # Move batch to device
        batch = self._move_to_device(batch)

        # Determine device type for autocast
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Forward pass with automatic mixed precision
        with torch.amp.autocast(device_type=device_type, enabled=self.config.mixed_precision):
            # TODO: Implement flexible forward pass
            # This is a placeholder - actual implementation depends on model interface
            if isinstance(batch, dict):
                outputs = self.model(**batch)
            elif isinstance(batch, (tuple, list)):
                outputs = self.model(*batch)
            else:
                outputs = self.model(batch)

            # Compute loss
            if self.loss_fn is not None:
                if isinstance(outputs, dict) and isinstance(batch, dict):
                    loss = self.loss_fn(outputs, batch)
                else:
                    # Assume outputs contains loss
                    loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs
            else:
                # Assume model returns loss directly
                loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )

            if self.config.gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(),
                    self.config.gradient_clip_value
                )

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

        self.global_step += 1

        # Return unscaled loss
        return loss.item() * self.config.gradient_accumulation_steps

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        self._call_callbacks("on_validation_begin")

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            batch = self._move_to_device(batch)

            # Determine device type for autocast
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

            # Forward pass
            with torch.amp.autocast(device_type=device_type, enabled=self.config.mixed_precision):
                # TODO: Implement flexible forward pass (same as training)
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                elif isinstance(batch, (tuple, list)):
                    outputs = self.model(*batch)
                else:
                    outputs = self.model(batch)

                # Compute loss
                if self.loss_fn is not None:
                    if isinstance(outputs, dict) and isinstance(batch, dict):
                        loss = self.loss_fn(outputs, batch)
                    else:
                        loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs
                else:
                    loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"val_loss": avg_loss}

        self._call_callbacks("on_validation_end", logs=metrics)
        self.model.train()

        return metrics

    def save_checkpoint(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Current metrics
        """
        if not is_main_process():
            return

        # Get unwrapped model for saving
        if hasattr(self.model, 'unwrap'):
            model = self.model.unwrap()
        elif hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model

        self.checkpoint_manager.save(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch or self.current_epoch,
            step=step or self.global_step,
            metrics=metrics,
            metadata={
                "config": self.config.__dict__,
            }
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Get unwrapped model for loading
        if hasattr(self.model, 'unwrap'):
            model = self.model.unwrap()
        elif hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model

        checkpoint_data = self.checkpoint_manager.load(
            checkpoint_path,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )

        # Restore training state
        if 'epoch' in checkpoint_data:
            self.current_epoch = checkpoint_data['epoch'] + 1
        if 'step' in checkpoint_data:
            self.global_step = checkpoint_data['step'] + 1

        logger.info(
            f"Checkpoint loaded - resuming from epoch {self.current_epoch}, "
            f"step {self.global_step}"
        )

    def _move_to_device(self, batch: Any) -> Any:
        """
        Move batch to device.

        Args:
            batch: Input batch

        Returns:
            Batch on device
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(v) for v in batch)
        else:
            return batch

    def _call_callbacks(self, hook: str, **kwargs) -> None:
        """
        Call callbacks for a specific hook.

        Args:
            hook: Hook name
            **kwargs: Arguments to pass to callback
        """
        for callback in self.callbacks:
            getattr(callback, hook)(**kwargs)


def create_trainer(
    config: Union[TrainerConfig, dict[str, Any]],
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    **kwargs
) -> Trainer:
    """
    Convenience function to create a trainer.

    Args:
        config: Trainer configuration
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        **kwargs: Additional trainer arguments

    Returns:
        Trainer: Configured trainer instance

    Examples:
        >>> config = TrainerConfig(max_epochs=10)
        >>> trainer = create_trainer(config, model, train_loader, val_loader)
        >>> trainer.train()
    """
    if isinstance(config, dict):
        config = TrainerConfig(**config)

    return Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **kwargs
    )
