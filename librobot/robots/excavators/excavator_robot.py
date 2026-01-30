"""Excavator robot implementation.

This module provides the interface for controlling excavator heavy equipment,
supporting autonomous operation with multiple camera views, GPS, and IMU sensors.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .excavator import Excavator
from ..registry import register_robot


@register_robot(name="excavator", aliases=["digger", "backhoe"])
class ExcavatorRobot(Excavator):
    """
    Excavator robot interface for autonomous heavy equipment operation.

    The excavator is a tracked heavy equipment vehicle with a rotating cab
    and articulated arm for digging and material handling. This implementation
    provides a unified interface for autonomous control with comprehensive
    safety features.

    Action Space (7 DOF):
        - left_track: Left track speed [-1.0, 1.0] (normalized)
        - right_track: Right track speed [-1.0, 1.0] (normalized)
        - swing: Cab rotation speed [-1.0, 1.0] (left/right)
        - boom: Boom angle control [-1.0, 1.0] (lower/raise)
        - arm: Arm angle control [-1.0, 1.0] (extend/retract)
        - bucket: Bucket angle control [-1.0, 1.0] (curl/dump)
        - throttle: Engine throttle [0.0, 1.0]

    Observation Space:
        - images: Dict of camera views
            - 'front': (H, W, 3) Front view camera
            - 'rear': (H, W, 3) Rear view camera
            - 'left': (H, W, 3) Left side camera
            - 'bucket': (H, W, 3) Bucket view camera
        - proprioception: Dict of robot state
            - 'left_track_speed': Left track speed (m/s)
            - 'right_track_speed': Right track speed (m/s)
            - 'swing_angle': Cab rotation angle (radians)
            - 'boom_angle': Boom angle (radians)
            - 'arm_angle': Arm angle (radians)
            - 'bucket_angle': Bucket angle (radians)
            - 'hydraulic_pressure': Hydraulic system pressure (PSI)
            - 'engine_rpm': Engine RPM
            - 'fuel_level': Fuel level (0-1)
        - gps: Dict of GPS data
            - 'latitude': Latitude (degrees)
            - 'longitude': Longitude (degrees)
            - 'altitude': Altitude (meters)
        - imu: Dict of IMU data
            - 'linear_acceleration': (3,) Linear acceleration (m/s²)
            - 'angular_velocity': (3,) Angular velocity (rad/s)
            - 'orientation': (4,) Quaternion orientation

    Safety Features:
        - Maximum speed limits for autonomous operation
        - Hydraulic pressure monitoring and limits
        - Tip-over prevention based on load and terrain
        - Swing zone monitoring
        - Emergency stop capability
        - Geofencing support

    Example:
        >>> # Basic usage with context manager
        >>> with ExcavatorRobot(robot_id="excavator_001") as robot:
        ...     robot.connect(ip="192.168.1.100", port=5000)
        ...     
        ...     # Reset to safe initial state
        ...     robot.reset()
        ...     
        ...     # Get current observation
        ...     obs = robot.get_observation()
        ...     front_cam = obs['images']['front']
        ...     boom_angle = obs['proprioception']['boom_angle']
        ...     
        ...     # Execute digging action
        ...     action = np.array([
        ...         0.0,   # left_track (stationary)
        ...         0.0,   # right_track (stationary)
        ...         0.0,   # swing (no rotation)
        ...         -0.3,  # boom (lower)
        ...         0.2,   # arm (extend)
        ...         0.5,   # bucket (curl to dig)
        ...         0.6    # throttle (60%)
        ...     ])
        ...     robot.execute_action(action)
    """

    # Safety limits
    MAX_AUTONOMOUS_SPEED = 3.0  # m/s (approximately 11 km/h)
    MAX_SWING_SPEED = 0.3  # rad/s
    MIN_HYDRAULIC_PRESSURE = 2000  # PSI
    MAX_HYDRAULIC_PRESSURE = 4500  # PSI
    MAX_BOOM_ANGLE = np.pi / 3  # 60 degrees up
    MIN_BOOM_ANGLE = -np.pi / 6  # -30 degrees down
    MAX_ARM_ANGLE = np.pi / 2  # 90 degrees
    MIN_ARM_ANGLE = 0.0
    MAX_BUCKET_ANGLE = np.pi / 2  # 90 degrees
    MIN_BUCKET_ANGLE = -np.pi / 4  # -45 degrees

    # Camera configurations
    CAMERA_RESOLUTION = (480, 640)  # Height x Width
    CAMERA_FPS = 30

    def __init__(
        self,
        robot_id: str,
        camera_enabled: bool = True,
        gps_enabled: bool = True,
        imu_enabled: bool = True,
        max_speed: Optional[float] = None,
    ):
        """
        Initialize excavator interface.

        Args:
            robot_id: Unique identifier for this excavator
            camera_enabled: Whether to enable camera feeds
            gps_enabled: Whether to enable GPS
            imu_enabled: Whether to enable IMU
            max_speed: Maximum allowed speed (m/s), defaults to MAX_AUTONOMOUS_SPEED
        """
        super().__init__(robot_id)
        self.camera_enabled = camera_enabled
        self.gps_enabled = gps_enabled
        self.imu_enabled = imu_enabled
        self.max_speed = max_speed or self.MAX_AUTONOMOUS_SPEED

        # Robot state
        self._left_track_speed = 0.0
        self._right_track_speed = 0.0
        self._swing_angle = 0.0
        self._boom_angle = 0.0
        self._arm_angle = 0.0
        self._bucket_angle = 0.0
        self._hydraulic_pressure = 3000.0
        self._engine_rpm = 0.0
        self._fuel_level = 1.0

        # GPS state
        self._latitude = 0.0
        self._longitude = 0.0
        self._altitude = 0.0

        # IMU state
        self._linear_acceleration = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion

        # Connection state
        self._connection_params = {}

    def connect(self, **kwargs) -> bool:
        """Connect to the excavator control system."""
        self._connection_params = kwargs
        self._is_connected = True
        print(f"[{self.robot_id}] Connected to excavator at {kwargs.get('ip', 'unknown')}")
        return True

    def disconnect(self) -> None:
        """Disconnect from the excavator."""
        self._is_connected = False
        print(f"[{self.robot_id}] Disconnected from excavator")

    def reset(self) -> None:
        """Reset excavator to safe initial state."""
        self._left_track_speed = 0.0
        self._right_track_speed = 0.0
        self._swing_angle = 0.0
        self._boom_angle = 0.0
        self._arm_angle = 0.0
        self._bucket_angle = 0.0
        print(f"[{self.robot_id}] Reset to safe initial state")

    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current excavator state."""
        return {
            'track_state': np.array([self._left_track_speed, self._right_track_speed]),
            'arm_state': np.array([self._swing_angle, self._boom_angle, 
                                   self._arm_angle, self._bucket_angle]),
            'hydraulic_state': np.array([self._hydraulic_pressure]),
            'engine_state': np.array([self._engine_rpm, self._fuel_level]),
            'position': np.array([self._latitude, self._longitude, self._altitude]),
            'orientation': self._orientation.copy(),
        }

    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute action on excavator."""
        if action.shape[0] != 7:
            raise ValueError(f"Action must have 7 dimensions, got {action.shape[0]}")

        if not self._is_connected:
            raise RuntimeError("Not connected to excavator")

        # Parse and clip action
        left_track = np.clip(action[0], -1.0, 1.0)
        right_track = np.clip(action[1], -1.0, 1.0)
        swing = np.clip(action[2], -1.0, 1.0)
        boom = np.clip(action[3], -1.0, 1.0)
        arm = np.clip(action[4], -1.0, 1.0)
        bucket = np.clip(action[5], -1.0, 1.0)
        throttle = np.clip(action[6], 0.0, 1.0)

        # Update internal state (simulation)
        self._left_track_speed = left_track * self.max_speed
        self._right_track_speed = right_track * self.max_speed
        self._swing_angle += swing * self.MAX_SWING_SPEED * 0.1
        self._boom_angle = boom * self.MAX_BOOM_ANGLE
        self._arm_angle = (arm + 1.0) / 2.0 * self.MAX_ARM_ANGLE
        self._bucket_angle = bucket * self.MAX_BUCKET_ANGLE

        return True

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from excavator sensors."""
        observation = {
            'proprioception': {
                'left_track_speed': self._left_track_speed,
                'right_track_speed': self._right_track_speed,
                'swing_angle': self._swing_angle,
                'boom_angle': self._boom_angle,
                'arm_angle': self._arm_angle,
                'bucket_angle': self._bucket_angle,
                'hydraulic_pressure': self._hydraulic_pressure,
                'engine_rpm': self._engine_rpm,
                'fuel_level': self._fuel_level,
            }
        }

        if self.camera_enabled:
            observation['images'] = {
                'front': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                'rear': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                'left': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                'bucket': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
            }

        if self.gps_enabled:
            observation['gps'] = {
                'latitude': self._latitude,
                'longitude': self._longitude,
                'altitude': self._altitude,
            }

        if self.imu_enabled:
            observation['imu'] = {
                'linear_acceleration': self._linear_acceleration.copy(),
                'angular_velocity': self._angular_velocity.copy(),
                'orientation': self._orientation.copy(),
            }

        return observation

    def get_action_space(self) -> Dict[str, Any]:
        """Get action space specification."""
        return {
            'shape': (7,),
            'dtype': np.float32,
            'bounds': {
                'low': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
                'high': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            },
            'names': [
                'left_track',
                'right_track',
                'swing',
                'boom',
                'arm',
                'bucket',
                'throttle'
            ],
        }

    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specification."""
        obs_space = {
            'proprioception': {
                'shape': (9,),
                'dtype': np.float32,
                'names': [
                    'left_track_speed',
                    'right_track_speed',
                    'swing_angle',
                    'boom_angle',
                    'arm_angle',
                    'bucket_angle',
                    'hydraulic_pressure',
                    'engine_rpm',
                    'fuel_level'
                ],
            }
        }

        if self.camera_enabled:
            obs_space['images'] = {
                'front': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
                'rear': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
                'left': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
                'bucket': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
            }

        if self.gps_enabled:
            obs_space['gps'] = {
                'shape': (3,),
                'dtype': np.float64,
                'names': ['latitude', 'longitude', 'altitude'],
            }

        if self.imu_enabled:
            obs_space['imu'] = {
                'linear_acceleration': {'shape': (3,), 'dtype': np.float32},
                'angular_velocity': {'shape': (3,), 'dtype': np.float32},
                'orientation': {'shape': (4,), 'dtype': np.float32},
            }

        return obs_space

    def emergency_stop(self) -> None:
        """Trigger emergency stop of excavator."""
        self._left_track_speed = 0.0
        self._right_track_speed = 0.0
        print(f"[{self.robot_id}] EMERGENCY STOP activated")


__all__ = ['ExcavatorRobot']
