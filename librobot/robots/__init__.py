"""Robots package for LibroBot VLA."""

from .base import AbstractRobot
from .registry import (
    ROBOT_REGISTRY,
    register_robot,
    get_robot,
    create_robot,
    list_robots,
)

# Import submodules
from . import arms
from . import mobile
from . import mobile_manipulators
from . import humanoids
from . import sensors

# Convenience imports
from .arms import SO100Arm, FrankaArm, UR5Arm, xArmRobot, WidowXArm
from .mobile import LeKiwiRobot, DifferentialDriveRobot
from .mobile_manipulators import FetchRobot, TIAGoRobot, WheelLoaderRobot
from .humanoids import Figure01Robot, GR1Robot, UnitreeH1Robot
from .sensors import Camera, DepthCamera, ForceTorqueSensor, IMU, Lidar

__all__ = [
    # Base
    'AbstractRobot',
    'ROBOT_REGISTRY',
    # Registry
    'register_robot',
    'get_robot',
    'create_robot',
    'list_robots',
    # Arms
    'SO100Arm',
    'FrankaArm',
    'UR5Arm',
    'xArmRobot',
    'WidowXArm',
    # Mobile
    'LeKiwiRobot',
    'DifferentialDriveRobot',
    # Mobile Manipulators
    'FetchRobot',
    'TIAGoRobot',
    'WheelLoaderRobot',
    # Humanoids
    'Figure01Robot',
    'GR1Robot',
    'UnitreeH1Robot',
    # Sensors
    'Camera',
    'DepthCamera',
    'ForceTorqueSensor',
    'IMU',
    'Lidar',
    # Submodules
    'arms',
    'mobile',
    'mobile_manipulators',
    'humanoids',
    'sensors',
]
