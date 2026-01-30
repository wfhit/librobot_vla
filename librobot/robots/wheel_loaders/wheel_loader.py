"""Base class for wheel loader implementations.

This module provides the base class for wheel loader robot implementations.
Specific wheel loader platforms should inherit from WheelLoaderRobot and override
methods as needed for their hardware.
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class WheelLoaderRobot(AbstractRobot):
    """Base class for wheel loader robots.
    
    Provides common functionality for wheel loader implementations.
    For a comprehensive reference implementation with full features,
    see WheelLoader in wheel_loader.py.
    """
    
    # Common wheel loader specifications
    NUM_CAMERAS = 3
    HAS_GPS = True
    HAS_IMU = True
    
    def __init__(
        self,
        robot_id: str,
        max_speed: float = 10.0,
        max_steering_angle: float = 0.6,
        bucket_capacity: float = 3.0,
    ):
        """
        Args:
            robot_id: Robot identifier
            max_speed: Maximum vehicle speed (m/s)
            max_steering_angle: Maximum steering angle (radians)
            bucket_capacity: Bucket capacity (cubic meters)
        """
        super().__init__(robot_id)
        self.max_speed = max_speed
        self.max_steering_angle = max_steering_angle
        self.bucket_capacity = bucket_capacity
        
        # State
        self._steering_angle = 0.0
        self._vehicle_speed = 0.0
        self._bucket_angle = 0.0
        self._boom_height = 0.0
        self._engine_rpm = 0.0
        self._fuel_level = 1.0
    
    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "shape": (6,),  # steering, throttle, brake, bucket_tilt, boom_lift, gear
            "low": [-1.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "steering_angle": {"shape": (), "dtype": "float32"},
            "vehicle_speed": {"shape": (), "dtype": "float32"},
            "bucket_angle": {"shape": (), "dtype": "float32"},
            "boom_height": {"shape": (), "dtype": "float32"},
            "engine_rpm": {"shape": (), "dtype": "float32"},
            "fuel_level": {"shape": (), "dtype": "float32"},
        }


__all__ = ['WheelLoaderRobot']
