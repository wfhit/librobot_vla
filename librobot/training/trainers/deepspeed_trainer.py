"""DeepSpeed trainer for large-scale distributed training."""

from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
import json

from .base_trainer import BaseTrainer


class DeepSpeedTrainer(BaseTrainer):
    """
    Trainer using DeepSpeed for large-scale distributed training.

    Supports:
    - ZeRO stages 1, 2, 3
    - Offloading to CPU/NVMe
    - Gradient checkpointing
    - Mixed precision (fp16, bf16)
    - Large batch training
    """

    def __init__(
        self,
        model: Any,
        optimizer: Any = None,  # DeepSpeed creates its own
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
        seed: int = 42,
        # DeepSpeed-specific
        deepspeed_config: Optional[Union[str, Dict]] = None,
        zero_stage: int = 2,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        local_rank: int = -1,
    ):
        """
        Initialize DeepSpeed trainer.

        Args:
            model: Model to train
            optimizer: Optimizer config (DeepSpeed manages optimizer)
            scheduler: Scheduler config
            loss_fn: Loss function
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            callbacks: Callbacks
            max_epochs: Maximum epochs
            max_steps: Maximum steps
            gradient_accumulation_steps: Gradient accumulation
            gradient_clip_val: Gradient clipping
            log_interval: Logging interval
            val_interval: Validation interval
            checkpoint_dir: Checkpoint directory
            device: Device
            seed: Random seed
            deepspeed_config: DeepSpeed config dict or path
            zero_stage: ZeRO optimization stage (1, 2, or 3)
            offload_optimizer: Offload optimizer to CPU
            offload_param: Offload parameters to CPU
            local_rank: Local rank for distributed training
        """
        # Don't call super().__init__ with optimizer yet
        self.model = model
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
        self.seed = seed

        self.zero_stage = zero_stage
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.local_rank = local_rank

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self._stop_training = False
        self.train_metrics: Dict[str, List[float]] = {}
        self.val_metrics: Dict[str, List[float]] = {}

        # Initialize callbacks
        for callback in self.callbacks:
            callback.set_trainer(self)

        # Setup DeepSpeed
        self.deepspeed_config = self._build_config(deepspeed_config)
        self.model_engine = None
        self.optimizer = None
        self.scheduler = scheduler
        self._setup_deepspeed()

    def _build_config(self, config: Optional[Union[str, Dict]]) -> Dict:
        """Build DeepSpeed configuration."""
        if isinstance(config, str):
            with open(config, 'r') as f:
                return json.load(f)

        if config is not None:
            return config

        # Default configuration
        ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clip_val or 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
            "zero_optimization": {
                "stage": self.zero_stage,
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
            },
            "wall_clock_breakdown": False,
        }

        # Add offloading if requested
        if self.offload_optimizer:
            ds_config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        if self.offload_param:
            ds_config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True,
            }

        return ds_config

    def _setup_deepspeed(self) -> None:
        """Initialize DeepSpeed engine."""
        try:
            import deepspeed
            import torch.distributed as dist

            # Initialize distributed if not already
            if not dist.is_initialized():
                deepspeed.init_distributed()

            # Initialize DeepSpeed
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=self.deepspeed_config,
                model_parameters=self.model.parameters(),
            )

            self.device = self.model_engine.device

        except ImportError:
            print("Warning: deepspeed not installed, falling back to basic training")
            self.model_engine = None
            import torch
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def _train_epoch(self) -> float:
        """Train for one epoch using DeepSpeed."""
        if self.model_engine:
            self.model_engine.train()
        else:
            self.model.train()

        total_loss = 0
        num_batches = 0

        if self.train_dataloader is None:
            return 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            self._call_callbacks('on_batch_begin', batch=batch_idx)

            result = self._train_step(batch)
            loss = result.get('loss', 0)
            total_loss += loss
            num_batches += 1

            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                self._log_metrics(result)

            self._call_callbacks('on_batch_end', batch=batch_idx, logs=result)

            if self.max_steps and self.global_step >= self.max_steps:
                break

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform training step with DeepSpeed."""
        try:
            import torch

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if hasattr(v, 'to') else v 
                        for k, v in batch.items()}

            if self.model_engine:
                # DeepSpeed forward
                outputs = self.model_engine(**batch) if isinstance(batch, dict) \
                    else self.model_engine(batch)

                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                elif self.loss_fn:
                    loss = self.loss_fn(outputs, batch)
                else:
                    loss = outputs

                # DeepSpeed backward and step
                self.model_engine.backward(loss)
                self.model_engine.step()

                return {'loss': loss.item()}
            else:
                # Fallback
                return self._train_step_basic(batch)

        except ImportError:
            return {'loss': 0.0}

    def _train_step_basic(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Basic training step without DeepSpeed."""
        try:
            import torch

            self.optimizer.zero_grad()

            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)

            if isinstance(outputs, dict) and 'loss' in outputs:
                loss = outputs['loss']
            elif self.loss_fn:
                loss = self.loss_fn(outputs, batch)
            else:
                loss = outputs

            loss.backward()

            if self.gradient_clip_val:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_val
                )

            self.optimizer.step()

            return {'loss': loss.item()}

        except ImportError:
            return {'loss': 0.0}

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log training metrics."""
        for key, value in metrics.items():
            if key not in self.train_metrics:
                self.train_metrics[key] = []
            self.train_metrics[key].append(value)

    def save_checkpoint(self, filename: str) -> None:
        """Save DeepSpeed checkpoint."""
        if self.checkpoint_dir is None:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename

        if self.model_engine:
            self.model_engine.save_checkpoint(str(self.checkpoint_dir), filename)
        else:
            try:
                import torch
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                }, path)
            except ImportError:
                pass

    def load_checkpoint(self, path: str) -> None:
        """Load DeepSpeed checkpoint."""
        if self.model_engine:
            self.model_engine.load_checkpoint(path)
        else:
            try:
                import torch
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.current_epoch = checkpoint.get('epoch', 0)
                self.global_step = checkpoint.get('global_step', 0)
            except (ImportError, FileNotFoundError):
                pass


__all__ = ['DeepSpeedTrainer']
