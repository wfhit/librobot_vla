"""LibroBot VLA Framework.

A comprehensive, extensible Vision-Language-Action (VLA) framework for robotics.
"""

__version__ = "0.1.0"

from librobot.utils.registry import REGISTRY
from librobot.utils.config import load_config

__all__ = ["__version__", "REGISTRY", "load_config"]
