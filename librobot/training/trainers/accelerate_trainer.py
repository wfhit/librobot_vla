"""Accelerate-based trainer for distributed training."""

from typing import Any, Callable, Optional

from .base_trainer import BaseTrainer


class AccelerateTrainer(BaseTrainer):
    """
    Trainer using HuggingFace Accelerate for distributed training.

    Supports:
    - Multi-GPU training
    - Mixed precision (fp16, bf16)
    - Gradient accumulation
    - DeepSpeed integration
    - FSDP support
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[Callable] = None,
        train_dataloader: Optional[Any] = None,
        val_dataloader: Optional[Any] = None,
        callbacks: Optional[list[Any]] = None,
        max_epochs: int = 100,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: Optional[float] = 1.0,
        log_interval: int = 10,
        val_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        device: str = "cuda",
        mixed_precision: str = "fp16",
        seed: int = 42,
        # Accelerate-specific
        accelerate_config: Optional[dict] = None,
        logging_dir: Optional[str] = None,
        project_name: str = "vla_training",
    ):
        """
        Initialize Accelerate trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            callbacks: List of callbacks
            max_epochs: Maximum epochs
            max_steps: Maximum steps
            gradient_accumulation_steps: Gradient accumulation
            gradient_clip_val: Gradient clipping
            log_interval: Logging interval
            val_interval: Validation interval
            checkpoint_dir: Checkpoint directory
            device: Device (managed by accelerate)
            mixed_precision: Precision mode ("no", "fp16", "bf16")
            seed: Random seed
            accelerate_config: Accelerate configuration
            logging_dir: Directory for logs
            project_name: Project name for tracking
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            callbacks=callbacks,
            max_epochs=max_epochs,
            max_steps=max_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_clip_val=gradient_clip_val,
            log_interval=log_interval,
            val_interval=val_interval,
            checkpoint_dir=checkpoint_dir,
            device=device,
            mixed_precision=mixed_precision != "no",
            seed=seed,
        )

        self.mixed_precision_mode = mixed_precision
        self.accelerate_config = accelerate_config or {}
        self.logging_dir = logging_dir
        self.project_name = project_name

        # Initialize Accelerate
        self.accelerator = None
        self._setup_accelerate()

    def _setup_accelerate(self) -> None:
        """Setup Accelerate for distributed training."""
        try:
            from accelerate import Accelerator
            from accelerate.utils import set_seed

            set_seed(self.seed)

            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision_mode,
                log_with="tensorboard" if self.logging_dir else None,
                project_dir=self.logging_dir,
                **self.accelerate_config,
            )

            # Prepare model, optimizer, dataloaders
            if self.train_dataloader:
                self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
                    self.model, self.optimizer, self.train_dataloader
                )
            else:
                self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

            if self.val_dataloader:
                self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

            if self.scheduler:
                self.scheduler = self.accelerator.prepare(self.scheduler)

            self.device = self.accelerator.device

        except ImportError:
            print("Warning: accelerate not installed, falling back to basic training")
            self.accelerator = None

    def _train_epoch(self) -> float:
        """Train for one epoch using Accelerate."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        if self.train_dataloader is None:
            return 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            self._call_callbacks("on_batch_begin", batch=batch_idx)

            # Training step
            result = self._train_step(batch)
            loss = result.get("loss", 0)
            total_loss += loss
            num_batches += 1

            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics(result)

            self._call_callbacks("on_batch_end", batch=batch_idx, logs=result)

            # Check max steps
            if self.max_steps and self.global_step >= self.max_steps:
                break

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Perform single training step with Accelerate."""
        try:
            import torch

            if self.accelerator:
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

                    # Compute loss
                    if isinstance(outputs, dict) and "loss" in outputs:
                        loss = outputs["loss"]
                    elif self.loss_fn:
                        loss = self.loss_fn(outputs, batch)
                    else:
                        loss = outputs

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.gradient_clip_val:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_val
                        )

                    # Optimizer step
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()

                    return {"loss": loss.item()}
            else:
                # Fallback without accelerate
                return self._train_step_basic(batch)

        except ImportError:
            return self._train_step_basic(batch)

    def _train_step_basic(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Basic training step without Accelerate."""
        try:
            import torch

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

            # Forward
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif self.loss_fn:
                loss = self.loss_fn(outputs, batch)
            else:
                loss = outputs

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward
            loss.backward()

            # Step optimizer every N accumulation steps
            if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            return {"loss": loss.item() * self.gradient_accumulation_steps}

        except ImportError:
            return {"loss": 0.0}

    def _log_metrics(self, metrics: dict[str, Any]) -> None:
        """Log metrics using Accelerate's tracker."""
        if self.accelerator and self.accelerator.is_main_process:
            self.accelerator.log(metrics, step=self.global_step)

            # Store in history
            for key, value in metrics.items():
                if key not in self.train_metrics:
                    self.train_metrics[key] = []
                self.train_metrics[key].append(value)

    def save_checkpoint(self, filename: str) -> None:
        """Save checkpoint using Accelerate."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.accelerator:
            self.accelerator.wait_for_everyone()
            path = self.checkpoint_dir / filename
            self.accelerator.save_state(str(path))
        else:
            super().save_checkpoint(filename)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint using Accelerate."""
        if self.accelerator:
            self.accelerator.load_state(path)
        else:
            super().load_checkpoint(path)

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        if self.accelerator:
            return self.accelerator.is_main_process
        return True

    def print(self, *args, **kwargs) -> None:
        """Print only on main process."""
        if self.is_main_process:
            print(*args, **kwargs)


__all__ = ["AccelerateTrainer"]
