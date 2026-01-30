"""Robots package for LibroBot VLA.

Architecture:
    Each robot type subfolder follows a consistent structure:
    - <type>.py: Base class for that robot type (e.g., arm.py, humanoid.py)
    - <type>_robot.py: Specific robot implementations

    arms/
        arm.py       -> Arm (base class)
        arm_robot.py -> FrankaArm, UR5Arm, xArmRobot, WidowXArm, SO100Arm

    mobile/
        mobile.py       -> MobileRobot (base class)
        mobile_robot.py -> LeKiwiRobot, DifferentialDriveRobot

    mobile_manipulators/
        mobile_manipulator.py       -> MobileManipulator (base class)
        mobile_manipulator_robot.py -> FetchRobot, TIAGoRobot

    humanoids/
        humanoid.py         -> Humanoid (base class)
        humanoid_robot.py   -> Figure01Robot, GR1Robot, UnitreeH1Robot

    wheel_loaders/
        wheel_loader.py       -> WheelLoader (base class)
        wheel_loader_robot.py -> WheelLoaderRobot (comprehensive)

    excavators/
        excavator.py       -> Excavator (base class)
        excavator_robot.py -> ExcavatorRobot (comprehensive)

    articulated_trucks/
        articulated_truck.py       -> ArticulatedTruck (base class)
        articulated_truck_robot.py -> ArticulatedTruckRobot (comprehensive)
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
from . import wheel_loaders
from . import excavators
from . import articulated_trucks
from . import sensors

# Base classes
from .arms import Arm
from .mobile import MobileRobot
from .mobile_manipulators import MobileManipulator
from .humanoids import Humanoid
from .wheel_loaders import WheelLoader
from .excavators import Excavator
from .articulated_trucks import ArticulatedTruck

# Robot implementations
from .arms import SO100Arm, FrankaArm, UR5Arm, xArmRobot, WidowXArm
from .mobile import LeKiwiRobot, DifferentialDriveRobot
from .mobile_manipulators import FetchRobot, TIAGoRobot
from .humanoids import Figure01Robot, GR1Robot, UnitreeH1Robot
from .wheel_loaders import WheelLoaderRobot
from .excavators import ExcavatorRobot
from .articulated_trucks import ArticulatedTruckRobot
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
    'Humanoid',
    'WheelLoader',
    'Excavator',
    'ArticulatedTruck',
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
    # Humanoids
    'Figure01Robot',
    'GR1Robot',
    'UnitreeH1Robot',
    # Wheel Loaders
    'WheelLoaderRobot',
    # Excavators
    'ExcavatorRobot',
    # Articulated Trucks
    'ArticulatedTruckRobot',
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
    'wheel_loaders',
    'excavators',
    'articulated_trucks',
    'sensors',
]
