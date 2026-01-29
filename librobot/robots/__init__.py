"""Robot definitions."""

from librobot.robots.base import BaseRobot, RobotConfig
from librobot.robots.so100_arm import SO100ArmRobot
from librobot.robots.wheel_loader import WheelLoaderRobot

__all__ = [
    "BaseRobot",
    "RobotConfig",
    "WheelLoaderRobot",
    "SO100ArmRobot",
]
