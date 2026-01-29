"""Robots package."""

from .base import AbstractRobot
from .registry import (
    ROBOT_REGISTRY,
    register_robot,
    get_robot,
    create_robot,
    list_robots,
)

# Import robot implementations to register them
from .wheel_loader import WheelLoader
from .so100_arm import SO100Arm
from .humanoid import Humanoid

__all__ = [
    'AbstractRobot',
    'ROBOT_REGISTRY',
    'register_robot',
    'get_robot',
    'create_robot',
    'list_robots',
    'WheelLoader',
    'SO100Arm',
    'Humanoid',
]
