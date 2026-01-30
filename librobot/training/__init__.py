"""Training package for LibroBot VLA."""

from .losses import AbstractLoss
from .callbacks import AbstractCallback
from .trainers import BaseTrainer, AccelerateTrainer, DeepSpeedTrainer
from .distributed import (
    DistributedConfig,
    DDPWrapper,
    FSDPWrapper,
    DeepSpeedWrapper,
    setup_distributed,
    cleanup_distributed,
    wrap_model_distributed,
    is_distributed,
    get_world_size,
    get_rank,
    is_main_process,
    barrier,
    all_reduce,
    all_gather,
    broadcast,
)
from .experiment_tracking import (
    ExperimentConfig,
    AbstractExperimentTracker,
    WandbTracker,
    MLflowTracker,
    MultiTracker,
    create_tracker,
)
from .hyperparameter_tuning import (
    SearchSpace,
    TuningConfig,
    AbstractTuner,
    RayTuner,
    OptunaTuner,
    create_tuner,
    get_vla_search_space,
)
from .advanced_learning import (
    RLConfig,
    RLPolicyWrapper,
    RewardShaping,
    VideoImitationLearner,
    MultiRobotConfig,
    MultiRobotCoordinator,
    SimToRealConfig,
    SimToRealAdapter,
    OnlineLearner,
    ZeroShotAdapter,
    FewShotAdapter,
    EdgeDeployer,
)

# Import submodules
from . import losses
from . import callbacks
from . import trainers
from . import distributed
from . import experiment_tracking
from . import hyperparameter_tuning
from . import advanced_learning

__all__ = [
    # Base classes
    'AbstractLoss',
    'AbstractCallback',
    # Trainers
    'BaseTrainer',
    'AccelerateTrainer',
    'DeepSpeedTrainer',
    # Distributed
    'DistributedConfig',
    'DDPWrapper',
    'FSDPWrapper',
    'DeepSpeedWrapper',
    'setup_distributed',
    'cleanup_distributed',
    'wrap_model_distributed',
    'is_distributed',
    'get_world_size',
    'get_rank',
    'is_main_process',
    'barrier',
    'all_reduce',
    'all_gather',
    'broadcast',
    # Experiment Tracking
    'ExperimentConfig',
    'AbstractExperimentTracker',
    'WandbTracker',
    'MLflowTracker',
    'MultiTracker',
    'create_tracker',
    # Hyperparameter Tuning
    'SearchSpace',
    'TuningConfig',
    'AbstractTuner',
    'RayTuner',
    'OptunaTuner',
    'create_tuner',
    'get_vla_search_space',
    # Advanced Learning
    'RLConfig',
    'RLPolicyWrapper',
    'RewardShaping',
    'VideoImitationLearner',
    'MultiRobotConfig',
    'MultiRobotCoordinator',
    'SimToRealConfig',
    'SimToRealAdapter',
    'OnlineLearner',
    'ZeroShotAdapter',
    'FewShotAdapter',
    'EdgeDeployer',
    # Submodules
    'losses',
    'callbacks',
    'trainers',
    'distributed',
    'experiment_tracking',
    'hyperparameter_tuning',
    'advanced_learning',
]
