"""Training package for LibroBot VLA."""

from .losses import AbstractLoss
from .callbacks import AbstractCallback
from .trainer import Trainer, TrainerConfig, create_trainer
from .optimizers import (
    OptimizerBuilder,
    build_optimizer,
    get_optimizer_names,
    OPTIMIZER_REGISTRY,
)
from .schedulers import (
    SchedulerBuilder,
    build_scheduler,
    get_scheduler_names,
    SCHEDULER_REGISTRY,
    LinearWarmupScheduler,
    CosineAnnealingWarmupScheduler,
    LinearDecayScheduler,
    ConstantScheduler,
    PolynomialDecayScheduler,
)
from .distributed import (
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    DDPWrapper,
    DeepSpeedWrapper,
    FSDPWrapper,
    is_distributed,
    get_world_size,
    get_rank,
    is_main_process,
    barrier,
    all_reduce,
    all_gather,
    broadcast,
)

__all__ = [
    # Base classes
    'AbstractLoss',
    'AbstractCallback',
    
    # Trainer
    'Trainer',
    'TrainerConfig',
    'create_trainer',
    
    # Optimizers
    'OptimizerBuilder',
    'build_optimizer',
    'get_optimizer_names',
    'OPTIMIZER_REGISTRY',
    
    # Schedulers
    'SchedulerBuilder',
    'build_scheduler',
    'get_scheduler_names',
    'SCHEDULER_REGISTRY',
    'LinearWarmupScheduler',
    'CosineAnnealingWarmupScheduler',
    'LinearDecayScheduler',
    'ConstantScheduler',
    'PolynomialDecayScheduler',
    
    # Distributed
    'DistributedConfig',
    'setup_distributed',
    'cleanup_distributed',
    'wrap_model_distributed',
    'DDPWrapper',
    'DeepSpeedWrapper',
    'FSDPWrapper',
    'is_distributed',
    'get_world_size',
    'get_rank',
    'is_main_process',
    'barrier',
    'all_reduce',
    'all_gather',
    'broadcast',
]
