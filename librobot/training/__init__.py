"""Training package for LibroBot VLA."""

# Import submodules
from . import (
    advanced_learning,
    callbacks,
    distributed,
    experiment_tracking,
    hyperparameter_tuning,
    losses,
    trainers,
)
from .advanced_learning import (
    EdgeDeployer,
    FewShotAdapter,
    MultiRobotConfig,
    MultiRobotCoordinator,
    OnlineLearner,
    RewardShaping,
    RLConfig,
    RLPolicyWrapper,
    SimToRealAdapter,
    SimToRealConfig,
    VideoImitationLearner,
    ZeroShotAdapter,
)
from .callbacks import AbstractCallback
from .distributed import (
    DDPWrapper,
    DeepSpeedWrapper,
    DistributedConfig,
    FSDPWrapper,
    all_gather,
    all_reduce,
    barrier,
    broadcast,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    setup_distributed,
    wrap_model_distributed,
)
from .experiment_tracking import (
    AbstractExperimentTracker,
    ExperimentConfig,
    MLflowTracker,
    MultiTracker,
    WandbTracker,
    create_tracker,
)
from .hyperparameter_tuning import (
    AbstractTuner,
    OptunaTuner,
    RayTuner,
    SearchSpace,
    TuningConfig,
    create_tuner,
    get_vla_search_space,
)
from .losses import AbstractLoss
from .trainers import AccelerateTrainer, BaseTrainer, DeepSpeedTrainer

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
