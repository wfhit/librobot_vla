"""Base class for mobile robot implementations.

This module provides the base class for all mobile robot implementations.
Specific mobile robot platforms should inherit from MobileRobot and override
methods as needed for their hardware.
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class MobileRobot(AbstractRobot):
    """Base class for mobile robots."""

    def __init__(
        self,
        robot_id: str,
        drive_type: str = "differential",
        max_linear_velocity: float = 1.0,
        max_angular_velocity: float = 2.0,
    ):
        """
        Args:
            robot_id: Robot identifier
            drive_type: Drive type ("differential", "omni", "ackermann")
            max_linear_velocity: Maximum linear velocity (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        super().__init__(robot_id)
        self.drive_type = drive_type
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        # State
        self._position = np.zeros(3)  # x, y, z
        self._orientation = np.array([0, 0, 0, 1])  # quaternion
        self._velocity = np.zeros(2)  # linear, angular

    def get_action_space(self) -> Dict[str, Any]:
        if self.drive_type == "differential":
            return {
                "type": "continuous",
                "shape": (2,),  # linear_vel, angular_vel
                "low": [-self.max_linear_velocity, -self.max_angular_velocity],
                "high": [self.max_linear_velocity, self.max_angular_velocity],
            }
        elif self.drive_type == "omni":
            return {
                "type": "continuous",
                "shape": (3,),  # vx, vy, angular
                "low": -1.0,
                "high": 1.0,
            }
        return {}

    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "position": {"shape": (3,)},
            "orientation": {"shape": (4,)},
            "velocity": {"shape": (2,)},
        }


__all__ = ['MobileRobot']
