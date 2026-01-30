"""Teleoperation interfaces for data collection."""

from .base import (
    TELEOP_REGISTRY,
    AbstractTeleop,
    create_teleop,
    get_teleop,
    list_teleoperation,
    register_teleop,
)
from .keyboard import KeyboardTeleop
from .leader_follower import LeaderFollowerTeleop
from .mocap import MocapTeleop
from .spacemouse import SpaceMouseTeleop
from .vr import VRTeleop

__all__ = [
    "AbstractTeleop",
    "TELEOP_REGISTRY",
    "register_teleop",
    "get_teleop",
    "create_teleop",
    "list_teleoperation",
    "KeyboardTeleop",
    "SpaceMouseTeleop",
    "VRTeleop",
    "MocapTeleop",
    "LeaderFollowerTeleop",
]
