"""Base class for articulated truck implementations.

This module provides the base class for articulated truck robot implementations.
Specific articulated truck platforms should inherit from ArticulatedTruck
and override methods as needed for their hardware.
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class ArticulatedTruck(AbstractRobot):
    """Base class for articulated truck robots.

    Provides common functionality for articulated truck implementations.
    For a comprehensive reference implementation with full features,
    see ArticulatedTruckRobot in articulated_truck_robot.py.
    """

    # Common articulated truck specifications
    NUM_CAMERAS = 4
    HAS_GPS = True
    HAS_IMU = True

    def __init__(
        self,
        robot_id: str,
        max_speed: float = 15.0,
        max_steering_angle: float = 0.7,
        payload_capacity: float = 40.0,
    ):
        """
        Args:
            robot_id: Robot identifier
            max_speed: Maximum vehicle speed (m/s)
            max_steering_angle: Maximum articulation angle (radians)
            payload_capacity: Maximum payload capacity (metric tons)
        """
        super().__init__(robot_id)
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.payload_capacity = payload_capacity

        # State
        self._steering_angle = 0.0
        self._vehicle_speed = 0.0
        self._bed_angle = 0.0
        self._current_payload = 0.0
        self._engine_rpm = 0.0
        self._fuel_level = 1.0

    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "shape": (5,),  # steering, throttle, brake, bed_tilt, gear
            "low": [-1.0, 0.0, 0.0, -1.0, -1.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "steering_angle": {"shape": (), "dtype": "float32"},
            "vehicle_speed": {"shape": (), "dtype": "float32"},
            "bed_angle": {"shape": (), "dtype": "float32"},
            "current_payload": {"shape": (), "dtype": "float32"},
            "engine_rpm": {"shape": (), "dtype": "float32"},
            "fuel_level": {"shape": (), "dtype": "float32"},
        }


__all__ = ['ArticulatedTruck']
