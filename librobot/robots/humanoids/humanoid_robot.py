"""Humanoid robot implementations.

Specific humanoid platform implementations that extend the base Humanoid template.
Each implementation customizes joint counts, limits, and hardware interfaces.
"""

from typing import Any, Dict
import numpy as np

from .humanoid import Humanoid
from ..registry import register_robot


@register_robot(name="figure_01", aliases=["figure", "figure01"])
class Figure01Robot(Humanoid):
    """Figure 01 humanoid robot.
    
    Figure's humanoid robot with 43 joints and dexterous hands.
    """
    
    NUM_JOINTS = 43
    HAND_DOF = 12
    
    def __init__(self, robot_id: str = "figure_01"):
        super().__init__(robot_id)
        self._num_joints = self.NUM_JOINTS
        self._hand_dof = self.HAND_DOF
    
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
class GR1Robot(Humanoid):
    """Fourier GR-1 humanoid robot.
    
    Fourier Intelligence's general-purpose humanoid with 40 joints.
    """
    
    NUM_JOINTS = 40
    HAND_DOF = 10
    
    def __init__(self, robot_id: str = "gr1"):
        super().__init__(robot_id)
        self._num_joints = self.NUM_JOINTS
        self._hand_dof = self.HAND_DOF
    
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
class UnitreeH1Robot(Humanoid):
    """Unitree H1 humanoid robot.
    
    Unitree's agile humanoid with 26 joints optimized for locomotion.
    """
    
    NUM_JOINTS = 26
    HAND_DOF = 6
    
    def __init__(self, robot_id: str = "unitree_h1"):
        super().__init__(robot_id)
        self._num_joints = self.NUM_JOINTS
        self._hand_dof = self.HAND_DOF
    
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
    'Figure01Robot',
    'GR1Robot',
    'UnitreeH1Robot',
]
