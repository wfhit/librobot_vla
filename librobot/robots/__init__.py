"""Robots package for LibroBot VLA.

Architecture:
    Each robot type subfolder follows a consistent structure:
    - base.py: Base class for that robot type
    - robots.py: Specific robot implementations
    - Optional: Comprehensive implementations in separate files (e.g., so100_arm.py)

    arms/
        base.py         -> Arm (base class)
        robots.py       -> FrankaArm, UR5Arm, xArmRobot, WidowXArm
        so100_arm.py    -> SO100Arm (comprehensive)

    mobile/
        base.py         -> MobileRobot (base class)
        robots.py       -> LeKiwiRobot, DifferentialDriveRobot

    mobile_manipulators/
        base.py              -> MobileManipulator (base class)
        wheel_loader_base.py -> WheelLoaderRobot (base class for loaders)
        robots.py            -> FetchRobot, TIAGoRobot
        wheel_loader.py      -> WheelLoader (comprehensive)

    humanoids/
        base.py         -> Humanoid (base class)
        robots.py       -> Figure01Robot, GR1Robot, UnitreeH1Robot
"""

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

# Base classes
from .arms import Arm
from .mobile import MobileRobot
from .mobile_manipulators import MobileManipulator, WheelLoaderRobot
from .humanoids import Humanoid

# Robot implementations
from .arms import SO100Arm, FrankaArm, UR5Arm, xArmRobot, WidowXArm
from .mobile import LeKiwiRobot, DifferentialDriveRobot
from .mobile_manipulators import FetchRobot, TIAGoRobot, WheelLoader
from .humanoids import Figure01Robot, GR1Robot, UnitreeH1Robot
from .sensors import Camera, DepthCamera, ForceTorqueSensor, IMU, Lidar

__all__ = [
    # Abstract base
    'AbstractRobot',
    # Registry
    'ROBOT_REGISTRY',
    'register_robot',
    'get_robot',
    'create_robot',
    'list_robots',
    # Base classes
    'Arm',
    'MobileRobot',
    'MobileManipulator',
    'WheelLoaderRobot',
    'Humanoid',
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
    'WheelLoader',
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
