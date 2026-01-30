"""Robot arm implementations.

Specific robot arm implementations that extend the base arm pattern.
Each implementation customizes joint counts, limits, and hardware interfaces.

For the comprehensive reference implementation, see SO100Arm in so100_arm_robot.py.
"""

from typing import Any, Dict
import numpy as np

from .arm import Arm
from ..registry import register_robot


@register_robot(name="franka", aliases=["franka_panda", "panda"])
class FrankaArm(Arm):
    """Franka Emika Panda robot arm."""
    
    def __init__(self, robot_id: str = "franka"):
        super().__init__(robot_id, num_joints=7, gripper_dof=1)
        self.joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
    
    def connect(self, **kwargs) -> bool:
        ip = kwargs.get("ip", "172.16.0.2")
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        # Home position
        self._joint_positions = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        self._gripper_state = np.array([0.04])  # Open
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "joint_velocities": self._joint_velocities.copy(),
            "end_effector_pose": self._ee_pose.copy(),
            "gripper_state": self._gripper_state.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        self._gripper_state = action[self.num_joints:]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {
            "proprioception": np.concatenate([
                self._joint_positions,
                self._joint_velocities,
                self._gripper_state,
            ]),
        }


@register_robot(name="ur5", aliases=["ur5e", "universal_robot"])
class UR5Arm(Arm):
    """Universal Robots UR5/UR5e arm."""
    
    def __init__(self, robot_id: str = "ur5"):
        super().__init__(robot_id, num_joints=6, gripper_dof=1)
        self.joint_limits = [(-2*np.pi, 2*np.pi)] * 6
    
    def connect(self, **kwargs) -> bool:
        ip = kwargs.get("ip", "192.168.1.100")
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "joint_velocities": self._joint_velocities.copy(),
            "end_effector_pose": self._ee_pose.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        if len(action) > self.num_joints:
            self._gripper_state = action[self.num_joints:]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": np.concatenate([
            self._joint_positions, self._joint_velocities
        ])}


@register_robot(name="xarm", aliases=["xarm7", "xarm6"])
class xArmRobot(Arm):
    """UFactory xArm robot."""
    
    def __init__(self, robot_id: str = "xarm", num_joints: int = 7):
        super().__init__(robot_id, num_joints=num_joints, gripper_dof=1)
    
    def connect(self, **kwargs) -> bool:
        ip = kwargs.get("ip", "192.168.1.111")
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "joint_velocities": self._joint_velocities.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": self._joint_positions.copy()}


@register_robot(name="widowx", aliases=["widowx_250"])
class WidowXArm(Arm):
    """Trossen Robotics WidowX 250 arm."""
    
    def __init__(self, robot_id: str = "widowx"):
        super().__init__(robot_id, num_joints=6, gripper_dof=1)
    
    def connect(self, **kwargs) -> bool:
        port = kwargs.get("port", "/dev/ttyUSB0")
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {"joint_positions": self._joint_positions.copy()}
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        self._joint_positions = action[:self.num_joints]
        return True
    
    def get_observation(self) -> Dict[str, Any]:
        return {"proprioception": self._joint_positions.copy()}


__all__ = [
    'FrankaArm',
    'UR5Arm',
    'xArmRobot',
    'WidowXArm',
]
