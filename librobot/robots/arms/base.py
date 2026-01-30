"""Base class for robot arm implementations.

This module provides the base class for all robot arm implementations.
Specific robot arm platforms should inherit from Arm and override
methods as needed for their hardware.
"""
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class Arm(AbstractRobot):
    """Base class for robot arms.
    
    Provides common functionality for robotic arm implementations.
    For a comprehensive reference implementation with full features,
    see SO100Arm in so100_arm.py.
    """
    
    def __init__(
        self,
        robot_id: str,
        num_joints: int = 6,
        gripper_dof: int = 1,
        control_mode: str = "position",
    ):
        """
        Args:
            robot_id: Robot identifier
            num_joints: Number of arm joints
            gripper_dof: Gripper degrees of freedom
            control_mode: Control mode ("position", "velocity", "torque")
        """
        super().__init__(robot_id)
        self.num_joints = num_joints
        self.gripper_dof = gripper_dof
        self.control_mode = control_mode
        self.action_dim = num_joints + gripper_dof
        
        # State
        self._joint_positions = np.zeros(num_joints)
        self._joint_velocities = np.zeros(num_joints)
        self._gripper_state = np.zeros(gripper_dof)
        self._ee_pose = np.zeros(7)  # xyz + quaternion
    
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
            "end_effector_pose": {"shape": (7,)},
            "gripper_state": {"shape": (self.gripper_dof,)},
        }


__all__ = ['Arm']
