"""Mobile manipulator implementations."""

from typing import Any, Dict
import numpy as np

from .base import MobileManipulator
from ..registry import register_robot


@register_robot(name="fetch", aliases=["fetch_robot"])
class FetchRobot(MobileManipulator):
    """Fetch mobile manipulator."""
    
    def __init__(self, robot_id: str = "fetch"):
        super().__init__(robot_id, arm_joints=7, gripper_dof=1, base_dof=2)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._arm_positions = np.zeros(self.arm_joints)
        self._base_position = np.zeros(3)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "arm_positions": self._arm_positions.copy(),
            "gripper_state": self._gripper_state.copy(),
            "base_position": self._base_position.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        # Parse action: [base_vel (2), arm_joints (7), gripper (1)]
        self._base_velocity = action[:2]
        self._arm_positions = action[2:9]
        self._gripper_state = action[9:]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "proprioception": np.concatenate([
                self._base_position,
                self._base_velocity,
                self._arm_positions,
                self._gripper_state,
            ]),
        }


@register_robot(name="tiago", aliases=["tiago_robot"])
class TIAGoRobot(MobileManipulator):
    """PAL Robotics TIAGo mobile manipulator."""
    
    def __init__(self, robot_id: str = "tiago"):
        super().__init__(robot_id, arm_joints=7, gripper_dof=2, base_dof=3)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._arm_positions = np.zeros(self.arm_joints)
        self._base_position = np.zeros(3)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "arm_positions": self._arm_positions.copy(),
            "gripper_state": self._gripper_state.copy(),
            "base_position": self._base_position.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": np.concatenate([
            self._base_position, self._arm_positions
        ])}


__all__ = [
    'FetchRobot',
    'TIAGoRobot',
]
