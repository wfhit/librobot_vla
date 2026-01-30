"""Mobile robot implementations."""

from typing import Any, Dict
import numpy as np

from .mobile import MobileRobot
from ..registry import register_robot


@register_robot(name="lekiwi", aliases=["le_kiwi"])
class LeKiwiRobot(MobileRobot):
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
class DifferentialDriveRobot(MobileRobot):
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
    'LeKiwiRobot',
    'DifferentialDriveRobot',
]
