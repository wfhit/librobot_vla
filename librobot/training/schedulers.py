"""Learning rate scheduler builders with registry support."""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from librobot.utils.registry import Registry

# Global scheduler registry
SCHEDULER_REGISTRY = Registry("schedulers")


def register_scheduler(name: str, **kwargs):
    """
    Decorator to register a scheduler builder.

    Args:
        name: Name to register the scheduler under
        **kwargs: Additional metadata

    Examples:
        >>> @register_scheduler("linear")
        >>> def build_linear_scheduler(optimizer, num_training_steps, **kwargs):
        ...     return LinearScheduler(optimizer, num_training_steps, **kwargs)
    """
    return SCHEDULER_REGISTRY.register(name, **kwargs)


class SchedulerBuilder:
    """
    Builder class for creating learning rate schedulers.

    Supports common scheduler types with warmup and other features.

    Examples:
        >>> builder = SchedulerBuilder(
        ...     "cosine",
        ...     num_training_steps=10000,
        ...     num_warmup_steps=1000
        ... )
        >>> scheduler = builder.build(optimizer)
    """

    def __init__(
        self,
        scheduler_type: str,
        num_training_steps: Optional[int] = None,
        num_warmup_steps: int = 0,
        **scheduler_kwargs,
    ):
        """
        Initialize scheduler builder.

        Args:
            scheduler_type: Type of scheduler (registered name)
            num_training_steps: Total number of training steps
            num_warmup_steps: Number of warmup steps
            **scheduler_kwargs: Additional scheduler-specific arguments
        """
        self.scheduler_type = scheduler_type
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.scheduler_kwargs = scheduler_kwargs

    def build(self, optimizer: Optimizer) -> LRScheduler:
        """
        Build scheduler instance.

        Args:
            optimizer: Optimizer to schedule

        Returns:
            LRScheduler: Configured scheduler instance
        """
        # Get scheduler builder from registry
        scheduler_fn = SCHEDULER_REGISTRY.get(self.scheduler_type)
        if scheduler_fn is None:
            raise ValueError(
                f"Scheduler '{self.scheduler_type}' not found in registry. "
                f"Available: {SCHEDULER_REGISTRY.list()}"
            )

        return scheduler_fn(
            optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
            **self.scheduler_kwargs,
        )


class LinearWarmupScheduler(LRScheduler):
    """
    Linear warmup learning rate scheduler.

    Linearly increases learning rate from 0 to initial LR over warmup steps,
    then holds constant.

    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Number of steps to warm up
        last_epoch: The index of last epoch
    """

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.num_warmup_steps for base_lr in self.base_lrs
            ]
        # Constant after warmup
        return self.base_lrs


class CosineAnnealingWarmupScheduler(LRScheduler):
    """
    Cosine annealing with linear warmup.

    Linearly increases LR during warmup, then decays using cosine annealing.

    Args:
        optimizer: Wrapped optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of base LR
        num_cycles: Number of cosine cycles
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.num_warmup_steps for base_lr in self.base_lrs
            ]

        # Cosine annealing
        progress = (self.last_epoch - self.num_warmup_steps) / (
            self.num_training_steps - self.num_warmup_steps
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.num_cycles * 2 * progress))

        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)
            for base_lr in self.base_lrs
        ]


class LinearDecayScheduler(LRScheduler):
    """
    Linear decay with optional warmup.

    Linearly increases LR during warmup, then linearly decays to min_lr.

    Args:
        optimizer: Wrapped optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of base LR
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.num_warmup_steps for base_lr in self.base_lrs
            ]

        # Linear decay
        progress = (self.last_epoch - self.num_warmup_steps) / (
            self.num_training_steps - self.num_warmup_steps
        )
        decay_factor = max(0.0, 1.0 - progress)

        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor)
            for base_lr in self.base_lrs
        ]


class ConstantScheduler(LRScheduler):
    """
    Constant learning rate with optional warmup.

    Args:
        optimizer: Wrapped optimizer
        num_warmup_steps: Number of warmup steps
        last_epoch: The index of last epoch
    """

    def __init__(self, optimizer: Optimizer, num_warmup_steps: int = 0, last_epoch: int = -1):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.num_warmup_steps for base_lr in self.base_lrs
            ]
        # Constant after warmup
        return self.base_lrs


class PolynomialDecayScheduler(LRScheduler):
    """
    Polynomial decay with optional warmup.

    Args:
        optimizer: Wrapped optimizer
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        power: Polynomial power
        min_lr_ratio: Minimum LR as ratio of base LR
        last_epoch: The index of last epoch
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        num_warmup_steps: int = 0,
        power: float = 1.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate."""
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.num_warmup_steps for base_lr in self.base_lrs
            ]

        # Polynomial decay
        progress = (self.last_epoch - self.num_warmup_steps) / (
            self.num_training_steps - self.num_warmup_steps
        )
        decay_factor = (1.0 - progress) ** self.power

        return [
            base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * decay_factor)
            for base_lr in self.base_lrs
        ]


# Register schedulers
@register_scheduler("constant")
def build_constant_scheduler(
    optimizer: Optimizer,
    num_training_steps: Optional[int] = None,
    num_warmup_steps: int = 0,
    **kwargs,
) -> LRScheduler:
    """
    Build constant LR scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Not used for constant scheduler
        num_warmup_steps: Number of warmup steps
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Constant scheduler instance
    """
    return ConstantScheduler(optimizer, num_warmup_steps=num_warmup_steps)


@register_scheduler("linear")
def build_linear_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> LRScheduler:
    """
    Build linear decay scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of base LR
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Linear decay scheduler instance
    """
    return LinearDecayScheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )


@register_scheduler("cosine")
def build_cosine_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    **kwargs,
) -> LRScheduler:
    """
    Build cosine annealing scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total training steps
        num_warmup_steps: Number of warmup steps
        min_lr_ratio: Minimum LR as ratio of base LR
        num_cycles: Number of cosine cycles
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Cosine annealing scheduler instance
    """
    return CosineAnnealingWarmupScheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )


@register_scheduler("polynomial")
def build_polynomial_scheduler(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    power: float = 1.0,
    min_lr_ratio: float = 0.0,
    **kwargs,
) -> LRScheduler:
    """
    Build polynomial decay scheduler with optional warmup.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Total training steps
        num_warmup_steps: Number of warmup steps
        power: Polynomial power
        min_lr_ratio: Minimum LR as ratio of base LR
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Polynomial decay scheduler instance
    """
    return PolynomialDecayScheduler(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        power=power,
        min_lr_ratio=min_lr_ratio,
    )


@register_scheduler("step")
def build_step_scheduler(
    optimizer: Optimizer,
    num_training_steps: Optional[int] = None,
    step_size: int = 1000,
    gamma: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """
    Build step decay scheduler.

    Decays learning rate by gamma every step_size steps.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Not used for step scheduler
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Step scheduler instance
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


@register_scheduler("multistep")
def build_multistep_scheduler(
    optimizer: Optimizer,
    num_training_steps: Optional[int] = None,
    milestones: list[int] = None,
    gamma: float = 0.1,
    **kwargs,
) -> LRScheduler:
    """
    Build multi-step decay scheduler.

    Decays learning rate by gamma at specified milestones.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Not used for multistep scheduler
        milestones: List of step indices for LR decay
        gamma: Multiplicative factor of learning rate decay
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: MultiStep scheduler instance
    """
    if milestones is None:
        milestones = []

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@register_scheduler("exponential")
def build_exponential_scheduler(
    optimizer: Optimizer, num_training_steps: Optional[int] = None, gamma: float = 0.95, **kwargs
) -> LRScheduler:
    """
    Build exponential decay scheduler.

    Args:
        optimizer: Optimizer to schedule
        num_training_steps: Not used for exponential scheduler
        gamma: Multiplicative factor of learning rate decay
        **kwargs: Additional arguments (ignored)

    Returns:
        LRScheduler: Exponential scheduler instance
    """
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


def build_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_training_steps: Optional[int] = None,
    num_warmup_steps: int = 0,
    **scheduler_kwargs,
) -> LRScheduler:
    """
    Convenience function to build a learning rate scheduler.

    Args:
        scheduler_type: Type of scheduler
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        **scheduler_kwargs: Additional scheduler arguments

    Returns:
        LRScheduler: Configured scheduler instance

    Examples:
        >>> scheduler = build_scheduler(
        ...     "cosine",
        ...     optimizer,
        ...     num_training_steps=10000,
        ...     num_warmup_steps=1000
        ... )
    """
    builder = SchedulerBuilder(
        scheduler_type=scheduler_type,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        **scheduler_kwargs,
    )
    return builder.build(optimizer)


def get_scheduler_names() -> list[str]:
    """
    Get list of registered scheduler names.

    Returns:
        List of scheduler names
    """
    return SCHEDULER_REGISTRY.list()
