"""Humanoid robot implementations."""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot
from ..registry import register_robot


class BaseHumanoid(AbstractRobot):
    """Base class for humanoid robots."""
    
    def __init__(
        self,
        robot_id: str,
        num_joints: int = 32,
        has_hands: bool = True,
        hand_dof: int = 12,
    ):
        super().__init__(robot_id)
        self.num_joints = num_joints
        self.has_hands = has_hands
        self.hand_dof = hand_dof if has_hands else 0
        self.action_dim = num_joints + hand_dof * 2  # Two hands
        
        self._joint_positions = np.zeros(num_joints)
        self._joint_velocities = np.zeros(num_joints)
        self._hand_positions = np.zeros(hand_dof * 2)
        self._torso_pose = np.zeros(7)
    
    def get_action_space(self) -> Dict[str, Any]:
        return {
            "type": "continuous",
            "shape": (self.action_dim,),
            "low": -1.0,
            "high": 1.0,
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        return {
            "joint_positions": {"shape": (self.num_joints,)},
            "joint_velocities": {"shape": (self.num_joints,)},
            "torso_pose": {"shape": (7,)},
            "hand_positions": {"shape": (self.hand_dof * 2,)},
        }


@register_robot(name="figure_01", aliases=["figure", "figure01"])
class Figure01Robot(BaseHumanoid):
    """Figure 01 humanoid robot."""
    
    def __init__(self, robot_id: str = "figure_01"):
        super().__init__(robot_id, num_joints=43, has_hands=True, hand_dof=12)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "torso_pose": self._torso_pose.copy(),
            "hand_positions": self._hand_positions.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        if self.has_hands:
            self._hand_positions = action[self.num_joints:]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "proprioception": np.concatenate([
                self._joint_positions,
                self._joint_velocities,
                self._hand_positions,
            ]),
        }


@register_robot(name="gr1", aliases=["fourier_gr1"])
class GR1Robot(BaseHumanoid):
    """Fourier GR-1 humanoid robot."""
    
    def __init__(self, robot_id: str = "gr1"):
        super().__init__(robot_id, num_joints=40, has_hands=True, hand_dof=10)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "torso_pose": self._torso_pose.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": self._joint_positions.copy()}


@register_robot(name="unitree_h1", aliases=["h1", "unitree_humanoid"])
class UnitreeH1Robot(BaseHumanoid):
    """Unitree H1 humanoid robot."""
    
    def __init__(self, robot_id: str = "unitree_h1"):
        super().__init__(robot_id, num_joints=26, has_hands=True, hand_dof=6)
    
    def connect(self, **kwargs) -> bool:
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "torso_pose": self._torso_pose.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": self._joint_positions.copy()}


__all__ = [
    'BaseHumanoid',
    'Figure01Robot',
    'GR1Robot',
    'UnitreeH1Robot',
]
