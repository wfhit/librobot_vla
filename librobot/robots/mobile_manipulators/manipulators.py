"""Mobile manipulator implementations."""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot
from ..registry import register_robot


class BaseMobileManipulator(AbstractRobot):
    """Base class for mobile manipulators."""
    
    def __init__(
        self,
        robot_id: str,
        arm_joints: int = 6,
        gripper_dof: int = 1,
        base_dof: int = 3,
    ):
        super().__init__(robot_id)
        self.arm_joints = arm_joints
        self.gripper_dof = gripper_dof
        self.base_dof = base_dof
        self.action_dim = arm_joints + gripper_dof + base_dof
        
        self._arm_positions = np.zeros(arm_joints)
        self._gripper_state = np.zeros(gripper_dof)
        self._base_position = np.zeros(3)
        self._base_velocity = np.zeros(base_dof)
    
    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "shape": (self.action_dim,),
            "low": -1.0,
            "high": 1.0,
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "arm_positions": {"shape": (self.arm_joints,)},
            "gripper_state": {"shape": (self.gripper_dof,)},
            "base_position": {"shape": (3,)},
            "base_velocity": {"shape": (self.base_dof,)},
        }


@register_robot(name="fetch", aliases=["fetch_robot"])
class FetchRobot(BaseMobileManipulator):
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
class TIAGoRobot(BaseMobileManipulator):
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


@register_robot(name="wheel_loader", aliases=["loader"])
class WheelLoaderRobot(BaseMobileManipulator):
    """Wheel loader / construction vehicle."""
    
    def __init__(self, robot_id: str = "wheel_loader"):
        # Loader arm (boom, bucket), steering
        super().__init__(robot_id, arm_joints=2, gripper_dof=1, base_dof=2)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._arm_positions = np.zeros(self.arm_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "boom_angle": self._arm_positions[0:1],
            "bucket_angle": self._arm_positions[1:2],
            "base_position": self._base_position.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._base_velocity = action[:2]  # steering, throttle
        self._arm_positions = action[2:4]  # boom, bucket
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": np.concatenate([
            self._base_position, self._arm_positions
        ])}


__all__ = [
    'BaseMobileManipulator',
    'FetchRobot',
    'TIAGoRobot',
    'WheelLoaderRobot',
]
