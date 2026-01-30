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

# Import submodules
from . import (
    arms,
    articulated_trucks,
    excavators,
    humanoids,
    mobile,
    mobile_manipulators,
    sensors,
    wheel_loaders,
)

# Base classes
# Robot implementations
from .arms import Arm, FrankaArm, SO100Arm, UR5Arm, WidowXArm, xArmRobot
from .articulated_trucks import ArticulatedTruck, ArticulatedTruckRobot
from .base import AbstractRobot
from .excavators import Excavator, ExcavatorRobot
from .humanoids import Figure01Robot, GR1Robot, Humanoid, UnitreeH1Robot
from .mobile import DifferentialDriveRobot, LeKiwiRobot, MobileRobot
from .mobile_manipulators import FetchRobot, MobileManipulator, TIAGoRobot
from .registry import (
    ROBOT_REGISTRY,
    create_robot,
    get_robot,
    list_robots,
    register_robot,
)
from .sensors import IMU, Camera, DepthCamera, ForceTorqueSensor, Lidar
from .wheel_loaders import WheelLoader, WheelLoaderRobot

__all__ = [
    # Abstract base
    "AbstractRobot",
    # Registry
    "ROBOT_REGISTRY",
    "register_robot",
    "get_robot",
    "create_robot",
    "list_robots",
    # Base classes
    "Arm",
    "MobileRobot",
    "MobileManipulator",
    "Humanoid",
    "WheelLoader",
    "Excavator",
    "ArticulatedTruck",
    # Arms
    "SO100Arm",
    "FrankaArm",
    "UR5Arm",
    "xArmRobot",
    "WidowXArm",
    # Mobile
    "LeKiwiRobot",
    "DifferentialDriveRobot",
    # Mobile Manipulators
    "FetchRobot",
    "TIAGoRobot",
    # Humanoids
    "Figure01Robot",
    "GR1Robot",
    "UnitreeH1Robot",
    # Wheel Loaders
    "WheelLoaderRobot",
    # Excavators
    "ExcavatorRobot",
    # Articulated Trucks
    "ArticulatedTruckRobot",
    # Sensors
    "Camera",
    "DepthCamera",
    "ForceTorqueSensor",
    "IMU",
    "Lidar",
    # Submodules
    "arms",
    "mobile",
    "mobile_manipulators",
    "humanoids",
    "wheel_loaders",
    "excavators",
    "articulated_trucks",
    "sensors",
]
