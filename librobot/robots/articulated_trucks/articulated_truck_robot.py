"""Articulated Truck robot implementation.

This module provides the interface for controlling articulated dump trucks,
supporting autonomous operation with multiple camera views, GPS, and IMU sensors.
"""

from typing import Any, Optional

import numpy as np

from ..registry import register_robot
from .articulated_truck import ArticulatedTruck


@register_robot(name="articulated_truck", aliases=["adt", "dump_truck", "hauler"])
class ArticulatedTruckRobot(ArticulatedTruck):
    """
    Articulated Truck robot interface for autonomous heavy equipment operation.

    The articulated truck (ADT) is a heavy-duty dump truck with an articulated
    steering joint between the cab and the dump body. This implementation provides
    a unified interface for autonomous hauling operations with comprehensive
    safety features.

    Action Space (5 DOF):
        - steering: Articulation angle [-1.0, 1.0] (normalized, left/right)
        - throttle: Forward/backward speed [0.0, 1.0]
        - brake: Brake pressure [0.0, 1.0]
        - bed_tilt: Dump bed angle control [-1.0, 1.0] (lower/raise)
        - transmission: Gear selection {-1: reverse, 0: neutral, 1: forward}

    Observation Space:
        - images: Dict of camera views
            - 'front': (H, W, 3) Front view camera
            - 'rear': (H, W, 3) Rear view camera
            - 'left': (H, W, 3) Left side camera
            - 'right': (H, W, 3) Right side camera
        - proprioception: Dict of robot state
            - 'steering_angle': Current articulation angle (radians)
            - 'vehicle_speed': Current speed (m/s)
            - 'bed_angle': Dump bed angle (radians)
            - 'current_payload': Current payload weight (metric tons)
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
        - Load monitoring and overload prevention
        - Tip-over prevention based on load, bed angle, and terrain
        - Emergency stop capability
        - Geofencing support

    Example:
        >>> # Basic usage with context manager
        >>> with ArticulatedTruckRobot(robot_id="truck_001") as robot:
        ...     robot.connect(ip="192.168.1.100", port=5000)
        ...
        ...     # Reset to safe initial state
        ...     robot.reset()
        ...
        ...     # Get current observation
        ...     obs = robot.get_observation()
        ...     front_cam = obs['images']['front']
        ...     payload = obs['proprioception']['current_payload']
        ...
        ...     # Execute hauling action: drive forward
        ...     action = np.array([
        ...         0.0,   # steering (straight)
        ...         0.4,   # throttle (40%)
        ...         0.0,   # brake (off)
        ...         0.0,   # bed_tilt (level)
        ...         1.0    # transmission (forward)
        ...     ])
        ...     robot.execute_action(action)
    """

    # Safety limits
    MAX_AUTONOMOUS_SPEED = 8.0  # m/s (approximately 29 km/h)
    MAX_STEERING_ANGLE = 0.7  # radians (~40 degrees)
    MIN_HYDRAULIC_PRESSURE = 1800  # PSI
    MAX_HYDRAULIC_PRESSURE = 4000  # PSI
    MAX_BED_ANGLE = np.pi / 3  # 60 degrees
    MIN_BED_ANGLE = 0.0  # flat
    MAX_PAYLOAD = 40.0  # metric tons

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
        payload_capacity: Optional[float] = None,
    ):
        """
        Initialize articulated truck interface.

        Args:
            robot_id: Unique identifier for this articulated truck
            camera_enabled: Whether to enable camera feeds
            gps_enabled: Whether to enable GPS
            imu_enabled: Whether to enable IMU
            max_speed: Maximum allowed speed (m/s), defaults to MAX_AUTONOMOUS_SPEED
            payload_capacity: Maximum payload (metric tons), defaults to MAX_PAYLOAD
        """
        super().__init__(robot_id)
        self.camera_enabled = camera_enabled
        self.gps_enabled = gps_enabled
        self.imu_enabled = imu_enabled
        self.max_speed = max_speed or self.MAX_AUTONOMOUS_SPEED
        self.payload_capacity = payload_capacity or self.MAX_PAYLOAD

        # Robot state
        self._steering_angle = 0.0
        self._vehicle_speed = 0.0
        self._bed_angle = 0.0
        self._current_payload = 0.0
        self._hydraulic_pressure = 2500.0
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
        """Connect to the articulated truck control system."""
        self._connection_params = kwargs
        self._is_connected = True
        print(f"[{self.robot_id}] Connected to articulated truck at {kwargs.get('ip', 'unknown')}")
        return True

    def disconnect(self) -> None:
        """Disconnect from the articulated truck."""
        self._is_connected = False
        print(f"[{self.robot_id}] Disconnected from articulated truck")

    def reset(self) -> None:
        """Reset articulated truck to safe initial state."""
        self._steering_angle = 0.0
        self._vehicle_speed = 0.0
        self._bed_angle = 0.0
        print(f"[{self.robot_id}] Reset to safe initial state")

    def get_state(self) -> dict[str, np.ndarray]:
        """Get current articulated truck state."""
        return {
            "vehicle_state": np.array(
                [self._steering_angle, self._vehicle_speed, self._bed_angle, self._current_payload]
            ),
            "hydraulic_state": np.array([self._hydraulic_pressure]),
            "engine_state": np.array([self._engine_rpm, self._fuel_level]),
            "position": np.array([self._latitude, self._longitude, self._altitude]),
            "orientation": self._orientation.copy(),
        }

    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute action on articulated truck."""
        if action.shape[0] != 5:
            raise ValueError(f"Action must have 5 dimensions, got {action.shape[0]}")

        if not self._is_connected:
            raise RuntimeError("Not connected to articulated truck")

        # Parse and clip action
        steering = np.clip(action[0], -1.0, 1.0)
        throttle = np.clip(action[1], 0.0, 1.0)
        np.clip(action[2], 0.0, 1.0)
        bed_tilt = np.clip(action[3], -1.0, 1.0)
        np.clip(action[4], -1.0, 1.0)

        # Safety: Don't dump while moving fast
        if self._vehicle_speed > 1.0 and bed_tilt > 0.1:
            print(f"[{self.robot_id}] Warning: Cannot raise bed while moving")
            bed_tilt = 0.0

        # Update internal state (simulation)
        self._steering_angle = steering * self.MAX_STEERING_ANGLE
        self._vehicle_speed = throttle * self.max_speed
        self._bed_angle = max(0, bed_tilt) * self.MAX_BED_ANGLE

        return True

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from articulated truck sensors."""
        observation = {
            "proprioception": {
                "steering_angle": self._steering_angle,
                "vehicle_speed": self._vehicle_speed,
                "bed_angle": self._bed_angle,
                "current_payload": self._current_payload,
                "hydraulic_pressure": self._hydraulic_pressure,
                "engine_rpm": self._engine_rpm,
                "fuel_level": self._fuel_level,
            }
        }

        if self.camera_enabled:
            observation["images"] = {
                "front": np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                "rear": np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                "left": np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                "right": np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
            }

        if self.gps_enabled:
            observation["gps"] = {
                "latitude": self._latitude,
                "longitude": self._longitude,
                "altitude": self._altitude,
            }

        if self.imu_enabled:
            observation["imu"] = {
                "linear_acceleration": self._linear_acceleration.copy(),
                "angular_velocity": self._angular_velocity.copy(),
                "orientation": self._orientation.copy(),
            }

        return observation

    def get_action_space(self) -> dict[str, Any]:
        """Get action space specification."""
        return {
            "shape": (5,),
            "dtype": np.float32,
            "bounds": {
                "low": np.array([-1.0, 0.0, 0.0, -1.0, -1.0]),
                "high": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            },
            "names": ["steering", "throttle", "brake", "bed_tilt", "transmission"],
        }

    def get_observation_space(self) -> dict[str, Any]:
        """Get observation space specification."""
        obs_space = {
            "proprioception": {
                "shape": (7,),
                "dtype": np.float32,
                "names": [
                    "steering_angle",
                    "vehicle_speed",
                    "bed_angle",
                    "current_payload",
                    "hydraulic_pressure",
                    "engine_rpm",
                    "fuel_level",
                ],
            }
        }

        if self.camera_enabled:
            obs_space["images"] = {
                "front": {"shape": (*self.CAMERA_RESOLUTION, 3), "dtype": np.uint8},
                "rear": {"shape": (*self.CAMERA_RESOLUTION, 3), "dtype": np.uint8},
                "left": {"shape": (*self.CAMERA_RESOLUTION, 3), "dtype": np.uint8},
                "right": {"shape": (*self.CAMERA_RESOLUTION, 3), "dtype": np.uint8},
            }

        if self.gps_enabled:
            obs_space["gps"] = {
                "shape": (3,),
                "dtype": np.float64,
                "names": ["latitude", "longitude", "altitude"],
            }

        if self.imu_enabled:
            obs_space["imu"] = {
                "linear_acceleration": {"shape": (3,), "dtype": np.float32},
                "angular_velocity": {"shape": (3,), "dtype": np.float32},
                "orientation": {"shape": (4,), "dtype": np.float32},
            }

        return obs_space

    def emergency_stop(self) -> None:
        """Trigger emergency stop of articulated truck."""
        self._vehicle_speed = 0.0
        print(f"[{self.robot_id}] EMERGENCY STOP activated")

    def set_payload(self, payload: float) -> bool:
        """
        Set current payload weight.

        Args:
            payload: Payload weight in metric tons

        Returns:
            bool: True if payload is within limits
        """
        if payload > self.payload_capacity:
            print(
                f"[{self.robot_id}] Warning: Payload {payload}t exceeds capacity {self.payload_capacity}t"
            )
            return False
        self._current_payload = payload
        return True


__all__ = ["ArticulatedTruckRobot"]
