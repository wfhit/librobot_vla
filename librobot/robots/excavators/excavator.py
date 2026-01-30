"""Base class for excavator implementations.

This module provides the base class for excavator robot implementations.
Specific excavator platforms should inherit from Excavator and override
methods as needed for their hardware.
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class Excavator(AbstractRobot):
    """Base class for excavator robots.
    
    Provides common functionality for excavator implementations.
    For a comprehensive reference implementation with full features,
    see ExcavatorRobot in excavator_robot.py.
    """
    
    # Common excavator specifications
    NUM_CAMERAS = 4
    HAS_GPS = True
    HAS_IMU = True
    
    def __init__(
        self,
        robot_id: str,
        max_speed: float = 6.0,
        max_swing_speed: float = 0.5,
        bucket_capacity: float = 1.5,
    ):
        """
        Args:
            robot_id: Robot identifier
            max_speed: Maximum track speed (m/s)
            max_swing_speed: Maximum swing rotation speed (rad/s)
            bucket_capacity: Bucket capacity (cubic meters)
        """
        super().__init__(robot_id)
        self.max_speed = max_speed
        self.max_swing_speed = max_swing_speed
        self.bucket_capacity = bucket_capacity
        
        # State
        self._left_track_speed = 0.0
        self._right_track_speed = 0.0
        self._swing_angle = 0.0
        self._boom_angle = 0.0
        self._arm_angle = 0.0
        self._bucket_angle = 0.0
        self._engine_rpm = 0.0
        self._fuel_level = 1.0
    
    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "shape": (7,),  # left_track, right_track, swing, boom, arm, bucket, throttle
            "low": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
            "high": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "left_track_speed": {"shape": (), "dtype": "float32"},
            "right_track_speed": {"shape": (), "dtype": "float32"},
            "swing_angle": {"shape": (), "dtype": "float32"},
            "boom_angle": {"shape": (), "dtype": "float32"},
            "arm_angle": {"shape": (), "dtype": "float32"},
            "bucket_angle": {"shape": (), "dtype": "float32"},
            "engine_rpm": {"shape": (), "dtype": "float32"},
            "fuel_level": {"shape": (), "dtype": "float32"},
        }


__all__ = ['Excavator']
