"""Robot arm implementations."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..base import AbstractRobot
from ..registry import register_robot


class BaseArm(AbstractRobot):
    """Base class for robot arms."""
    
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


@register_robot(name="so100", aliases=["so-100", "so100_arm"])
class SO100Arm(BaseArm):
    """SO-100 robot arm from SO Robotics."""
    
    def __init__(self, robot_id: str = "so100"):
        super().__init__(robot_id, num_joints=6, gripper_dof=1)
        self.joint_limits = [
            (-2.96, 2.96),  # Joint 1
            (-1.74, 1.74),  # Joint 2
            (-2.09, 2.09),  # Joint 3
            (-2.96, 2.96),  # Joint 4
            (-2.09, 2.09),  # Joint 5
            (-2.96, 2.96),  # Joint 6
        ]
    
    def connect(self, **kwargs) -> bool:
        port = kwargs.get("port", "/dev/ttyUSB0")
        self._is_connected = True
        return True
    
    def disconnect(self) -> None:
        self._is_connected = False
    
    def reset(self) -> None:
        self._joint_positions = np.zeros(self.num_joints)
        self._gripper_state = np.zeros(self.gripper_dof)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            "joint_positions": self._joint_positions.copy(),
            "joint_velocities": self._joint_velocities.copy(),
            "end_effector_pose": self._ee_pose.copy(),
            "gripper_state": self._gripper_state.copy(),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        if len(action) != self.action_dim:
            return False
        
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
            "images": {},  # Camera would add here
        }


@register_robot(name="franka", aliases=["franka_panda", "panda"])
class FrankaArm(BaseArm):
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
class UR5Arm(BaseArm):
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
class xArmRobot(BaseArm):
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
class WidowXArm(BaseArm):
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
    'BaseArm',
    'SO100Arm',
    'FrankaArm',
    'UR5Arm',
    'xArmRobot',
    'WidowXArm',
]
