"""
LibroBot VLA Framework
======================

A comprehensive framework for Vision-Language-Action models in robot learning.

Modules:
--------
- models: VLMs, action heads, encoders, and VLA frameworks
- data: Datasets and tokenizers
- robots: Robot interface abstractions
- training: Loss functions and training callbacks
- inference: Inference servers
- utils: Utilities for config, logging, checkpointing, profiling, etc.

Quick Start:
------------
>>> import librobot
>>> from librobot.models import register_vlm, create_vla
>>> from librobot.utils import setup_logging, set_seed
>>>
>>> # Setup
>>> setup_logging(level=librobot.utils.INFO)
>>> set_seed(42)
>>>
>>> # Register and create models
>>> # (Register your custom VLM, action head, etc.)
>>> # model = create_vla("my_vla", ...)
"""

# Import key modules
from . import data, inference, models, robots, training, utils
from .data import AbstractDataset, AbstractTokenizer, register_dataset, register_tokenizer
from .inference import AbstractServer

# Import commonly used classes and functions
from .models import (
    AbstractActionHead,
    AbstractEncoder,
    AbstractVLA,
    AbstractVLM,
    register_action_head,
    register_encoder,
    register_vla,
    register_vlm,
)
from .robots import AbstractRobot, register_robot
from .training import AbstractCallback, AbstractLoss
from .utils import (
    Checkpoint,
    Config,
    Logger,
    Registry,
    get_logger,
    load_config,
    set_seed,
    setup_logging,
)
from .version import __author__, __license__, __version__, get_version, get_version_info

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    "get_version",
    "get_version_info",
    # Modules
    "models",
    "data",
    "robots",
    "training",
    "inference",
    "utils",
    # Abstract base classes
    "AbstractVLM",
    "AbstractActionHead",
    "AbstractEncoder",
    "AbstractVLA",
    "AbstractDataset",
    "AbstractTokenizer",
    "AbstractRobot",
    "AbstractLoss",
    "AbstractCallback",
    "AbstractServer",
    # Registration functions
    "register_vlm",
    "register_action_head",
    "register_encoder",
    "register_vla",
    "register_dataset",
    "register_tokenizer",
    "register_robot",
    # Utilities
    "Config",
    "Logger",
    "Registry",
    "Checkpoint",
    "setup_logging",
    "set_seed",
    "get_logger",
    "load_config",
]
