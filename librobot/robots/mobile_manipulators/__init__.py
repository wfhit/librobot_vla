"""Mobile manipulator implementations."""

from .base import MobileManipulator
from .wheel_loader_base import WheelLoaderRobot
from .robots import FetchRobot, TIAGoRobot
from .wheel_loader import WheelLoader

__all__ = [
    # Base classes
    'MobileManipulator',
    'WheelLoaderRobot',
    # Comprehensive implementations
    'WheelLoader',
    # Other implementations
    'FetchRobot',
    'TIAGoRobot',
]
