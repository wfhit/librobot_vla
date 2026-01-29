"""Mobile robot implementations."""

from typing import Any, Dict, Optional
import numpy as np

from ..base import AbstractRobot
from ..registry import register_robot


class BaseMobileRobot(AbstractRobot):
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


@register_robot(name="lekiwi", aliases=["le_kiwi"])
class LeKiwiRobot(BaseMobileRobot):
    """LeKiwi holonomic mobile robot."""
    
    def __init__(self, robot_id: str = "lekiwi"):
        super().__init__(
            robot_id,
            drive_type="omni",
            max_linear_velocity=0.5,
            max_angular_velocity=1.5,
        )
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._position = np.zeros(3)
        self._velocity = np.zeros(2)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "position": self._position.copy(),
            "orientation": self._orientation.copy(),
            "velocity": self._velocity.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        # Omni-directional: vx, vy, angular
        dt = kwargs.get("dt", 0.1)
        self._position[:2] += action[:2] * dt
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "proprioception": np.concatenate([
                self._position,
                self._orientation,
                self._velocity,
            ]),
        }


@register_robot(name="differential_drive", aliases=["diff_drive"])
class DifferentialDriveRobot(BaseMobileRobot):
    """Generic differential drive mobile robot."""
    
    def __init__(
        self,
        robot_id: str = "diff_drive",
        wheel_radius: float = 0.05,
        wheel_base: float = 0.3,
    ):
        super().__init__(robot_id, drive_type="differential")
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._position = np.zeros(3)
        self._velocity = np.zeros(2)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "position": self._position.copy(),
            "velocity": self._velocity.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        # Differential drive: linear_vel, angular_vel
        linear_vel, angular_vel = action[:2]
        dt = kwargs.get("dt", 0.1)
        
        # Update position
        theta = np.arctan2(self._orientation[2], self._orientation[3]) * 2
        self._position[0] += linear_vel * np.cos(theta) * dt
        self._position[1] += linear_vel * np.sin(theta) * dt
        
        self._velocity = action[:2].copy()
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": np.concatenate([self._position, self._velocity])}


__all__ = [
    'BaseMobileRobot',
    'LeKiwiRobot',
    'DifferentialDriveRobot',
]
