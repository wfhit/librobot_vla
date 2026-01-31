"""Data collection utilities for LibroBot VLA.

This module contains data collection utilities including:
- Teleoperation interfaces (keyboard, spacemouse, VR, mocap, leader-follower)
- Recording utilities (data buffers, camera recording, episode recording)
- Data format converters (LeRobot, HDF5, Zarr, RLDS)
- Main DataCollector orchestrator
"""

# Base classes
from .base import AbstractConverter, AbstractTeleop

# Main collector
from .collector import DataCollector

# Converters
from .converters import (
    CONVERTER_REGISTRY,
    HDF5Converter,
    LeRobotConverter,
    RLDSConverter,
    ZarrConverter,
    create_converter,
    get_converter,
    list_converters,
    register_converter,
)

# Recording
from .recording import CameraRecorder, DataBuffer, EpisodeRecorder

# Teleoperation
from .teleoperation import (
    TELEOP_REGISTRY,
    KeyboardTeleop,
    LeaderFollowerTeleop,
    MocapTeleop,
    SpaceMouseTeleop,
    VRTeleop,
    create_teleop,
    get_teleop,
    list_teleoperation,
    register_teleop,
)

__all__ = [
    # Base classes
    "AbstractTeleop",
    "AbstractConverter",
    # Main collector
    "DataCollector",
    # Teleoperation
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
    # Recording
    "DataBuffer",
    "CameraRecorder",
    "EpisodeRecorder",
    # Converters
    "CONVERTER_REGISTRY",
    "register_converter",
    "get_converter",
    "create_converter",
    "list_converters",
    "LeRobotConverter",
    "HDF5Converter",
    "ZarrConverter",
    "RLDSConverter",
]
