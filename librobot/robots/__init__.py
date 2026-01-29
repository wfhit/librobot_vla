"""Robots package."""

from .base import AbstractRobot
from .registry import (
    ROBOT_REGISTRY,
    register_robot,
    get_robot,
    create_robot,
    list_robots,
)

__all__ = [
    'AbstractRobot',
    'ROBOT_REGISTRY',
    'register_robot',
    'get_robot',
    'create_robot',
    'list_robots',
]
