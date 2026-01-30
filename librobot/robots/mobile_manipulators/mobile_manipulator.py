"""Base class for mobile manipulator implementations.

This module provides the base class for all mobile manipulator implementations.
Specific mobile manipulator platforms should inherit from MobileManipulator 
and override methods as needed for their hardware.
"""

from typing import Any, Dict
import numpy as np

from ..base import AbstractRobot


class MobileManipulator(AbstractRobot):
    """Base class for mobile manipulators."""

    def __init__(
        self,
        robot_id: str,
        arm_joints: int = 6,
        gripper_dof: int = 1,
        base_dof: int = 3,
    ):
        """
        Args:
            robot_id: Robot identifier
            arm_joints: Number of arm joints
            gripper_dof: Gripper degrees of freedom
            base_dof: Mobile base degrees of freedom
        """
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


__all__ = ['MobileManipulator']
