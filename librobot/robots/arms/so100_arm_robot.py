"""SO100 robotic arm implementation.

This module provides the interface for controlling the SO100 7-DOF manipulator
with gripper, supporting precise manipulation tasks with visual feedback.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..base import AbstractRobot
from ..registry import register_robot


@register_robot(name="so100", aliases=["so100_arm", "so-100"])
class SO100Arm(AbstractRobot):
    """
    SO100 7-DOF robotic arm interface for manipulation tasks.
    
    The SO100 is a collaborative robotic arm designed for precision manipulation
    with integrated gripper and multi-camera vision system. This implementation
    provides a unified interface for control with comprehensive safety features.
    
    Action Space (8 DOF):
        - joint_positions: 7D vector of target joint positions (radians)
          - joint_0: Base rotation [-π, π]
          - joint_1: Shoulder [-π/2, π/2]
          - joint_2: Elbow [-π, π]
          - joint_3: Wrist 1 [-π, π]
          - joint_4: Wrist 2 [-π/2, π/2]
          - joint_5: Wrist 3 [-π, π]
          - joint_6: Flange rotation [-π, π]
        - gripper: Gripper position [0.0, 1.0] (0=fully open, 1=fully closed)
    
    Alternatively, actions can be specified as:
        - joint_velocities: 7D velocity control (rad/s)
        - joint_torques: 7D torque control (Nm)
        - end_effector_pose: 6D pose control (x, y, z, roll, pitch, yaw)
    
    Observation Space:
        - images: Dict of camera views
            - 'wrist_camera': (H, W, 3) Wrist-mounted camera
            - 'external_camera_1': (H, W, 3) Fixed external view
            - 'external_camera_2': (H, W, 3) Optional second external view
        - proprioception: Dict of robot state
            - 'joint_positions': (7,) Joint positions (radians)
            - 'joint_velocities': (7,) Joint velocities (rad/s)
            - 'joint_torques': (7,) Joint torques (Nm)
            - 'joint_temperatures': (7,) Joint motor temperatures (°C)
            - 'gripper_position': Gripper opening (0-1)
            - 'gripper_force': Measured gripper force (N)
            - 'end_effector_pose': (6,) EE pose [x, y, z, roll, pitch, yaw]
            - 'end_effector_wrench': (6,) Force/torque at EE [fx, fy, fz, tx, ty, tz]
    
    Safety Features:
        - Joint position, velocity, and torque limits
        - Collision detection and avoidance
        - Singularity avoidance
        - Temperature monitoring
        - Force limiting for safe human collaboration
        - Workspace boundaries
        - Emergency stop
    
    Example:
        >>> # Basic usage with position control
        >>> with SO100Arm(robot_id="arm_001") as robot:
        ...     robot.connect(ip="192.168.1.100", port=30001)
        ...     
        ...     # Reset to home position
        ...     robot.reset()
        ...     
        ...     # Get current observation
        ...     obs = robot.get_observation()
        ...     wrist_cam = obs['images']['wrist_camera']
        ...     joint_pos = obs['proprioception']['joint_positions']
        ...     
        ...     # Execute joint position action
        ...     action = np.array([
        ...         0.0,    # joint_0 (base)
        ...         -0.5,   # joint_1 (shoulder)
        ...         1.0,    # joint_2 (elbow)
        ...         0.0,    # joint_3 (wrist_1)
        ...         0.5,    # joint_4 (wrist_2)
        ...         0.0,    # joint_5 (wrist_3)
        ...         0.0,    # joint_6 (flange)
        ...         0.5     # gripper (half closed)
        ...     ])
        ...     robot.execute_action(action)
        
        >>> # Advanced usage with end-effector control
        >>> robot = SO100Arm(robot_id="arm_002", control_mode="end_effector")
        >>> robot.connect(ip="192.168.1.101")
        >>> 
        >>> # Move end-effector to target pose
        >>> ee_pose = np.array([0.3, 0.2, 0.4, 0.0, np.pi/2, 0.0])  # x, y, z, r, p, y
        >>> robot.move_to_pose(ee_pose)
        >>> 
        >>> robot.disconnect()
    """
    
    # Physical specifications
    NUM_JOINTS = 7
    GRIPPER_DOF = 1
    
    # Joint limits (radians)
    JOINT_POSITION_LIMITS = np.array([
        [-np.pi, np.pi],           # joint_0: base
        [-np.pi/2, np.pi/2],       # joint_1: shoulder
        [-np.pi, np.pi],           # joint_2: elbow
        [-np.pi, np.pi],           # joint_3: wrist_1
        [-np.pi/2, np.pi/2],       # joint_4: wrist_2
        [-np.pi, np.pi],           # joint_5: wrist_3
        [-np.pi, np.pi],           # joint_6: flange
    ])
    
    # Joint velocity limits (rad/s)
    JOINT_VELOCITY_LIMITS = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    
    # Joint torque limits (Nm)
    JOINT_TORQUE_LIMITS = np.array([50.0, 50.0, 30.0, 20.0, 15.0, 10.0, 10.0])
    
    # Temperature limits (°C)
    MAX_JOINT_TEMPERATURE = 80.0
    WARNING_JOINT_TEMPERATURE = 70.0
    
    # Gripper specifications
    GRIPPER_MIN_POSITION = 0.0  # Fully open (meters)
    GRIPPER_MAX_POSITION = 0.085  # Fully closed (meters)
    GRIPPER_MAX_FORCE = 100.0  # Maximum gripping force (N)
    
    # Workspace limits (meters, in base frame)
    WORKSPACE_MIN = np.array([-0.6, -0.6, -0.1])
    WORKSPACE_MAX = np.array([0.6, 0.6, 0.8])
    
    # Camera configurations
    CAMERA_RESOLUTION = (480, 640)  # Height x Width
    CAMERA_FPS = 30
    
    # Home position (safe starting pose)
    HOME_JOINT_POSITIONS = np.array([0.0, -0.785, 1.571, 0.0, 0.785, 0.0, 0.0])
    
    def __init__(
        self,
        robot_id: str,
        control_mode: str = "position",
        camera_enabled: bool = True,
        force_torque_sensor: bool = True,
        collision_detection: bool = True,
    ):
        """
        Initialize SO100 arm interface.
        
        Args:
            robot_id: Unique identifier for this arm
            control_mode: Control mode ("position", "velocity", "torque", "end_effector")
            camera_enabled: Whether to enable camera feeds
            force_torque_sensor: Whether force/torque sensing is available
            collision_detection: Whether to enable collision detection
        """
        super().__init__(robot_id)
        
        if control_mode not in ["position", "velocity", "torque", "end_effector"]:
            raise ValueError(f"Invalid control mode: {control_mode}")
        
        self.control_mode = control_mode
        self.camera_enabled = camera_enabled
        self.force_torque_sensor = force_torque_sensor
        self.collision_detection = collision_detection
        
        # Robot state
        self._joint_positions = self.HOME_JOINT_POSITIONS.copy()
        self._joint_velocities = np.zeros(self.NUM_JOINTS)
        self._joint_torques = np.zeros(self.NUM_JOINTS)
        self._joint_temperatures = np.full(self.NUM_JOINTS, 25.0)  # Room temperature
        
        # Gripper state
        self._gripper_position = 0.0  # Fully open
        self._gripper_force = 0.0
        
        # End-effector state
        self._ee_pose = np.zeros(6)  # x, y, z, roll, pitch, yaw
        self._ee_wrench = np.zeros(6)  # fx, fy, fz, tx, ty, tz
        
        # Safety flags
        self._emergency_stop_triggered = False
        self._collision_detected = False
        
        # Connection state
        self._connection_params = {}
    
    def connect(self, **kwargs) -> bool:
        """
        Connect to the SO100 arm control system.
        
        Args:
            **kwargs: Connection parameters
                - ip: IP address of the arm controller
                - port: Port number (default: 30001)
                - timeout: Connection timeout in seconds
                - api_key: Optional API key for authentication
        
        Returns:
            bool: True if connection successful
        """
        # TODO: Implement connection to SO100 control system
        # This should establish connection to:
        # - Arm controller (TCP/IP socket or proprietary protocol)
        # - Camera servers (RTSP/HTTP streams)
        # - Force/torque sensor
        # - Safety controller
        
        self._connection_params = kwargs
        self._is_connected = True
        print(f"[{self.robot_id}] Connected to SO100 arm at {kwargs.get('ip', 'unknown')}")
        return True
    
    def disconnect(self) -> None:
        """Disconnect from the arm and cleanup resources."""
        # TODO: Implement disconnection
        # - Close camera streams
        # - Disconnect from control system
        # - Release hardware resources
        # - Send safe shutdown command
        
        self._is_connected = False
        print(f"[{self.robot_id}] Disconnected from SO100 arm")
    
    def reset(self) -> None:
        """
        Reset arm to home position.
        
        This moves all joints to a safe home configuration and opens the gripper.
        """
        # TODO: Implement reset sequence
        # - Check current position for safe trajectory
        # - Plan path to home position avoiding obstacles
        # - Execute motion with safety monitoring
        # - Open gripper
        # - Verify final position
        
        self._joint_positions = self.HOME_JOINT_POSITIONS.copy()
        self._joint_velocities = np.zeros(self.NUM_JOINTS)
        self._joint_torques = np.zeros(self.NUM_JOINTS)
        self._gripper_position = 0.0
        self._emergency_stop_triggered = False
        
        print(f"[{self.robot_id}] Reset to home position")
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current arm state.
        
        Returns:
            Dictionary containing:
                - 'joint_positions': (7,) Joint positions (radians)
                - 'joint_velocities': (7,) Joint velocities (rad/s)
                - 'joint_torques': (7,) Joint torques (Nm)
                - 'end_effector_pose': (6,) EE pose [x, y, z, roll, pitch, yaw]
                - 'gripper_state': [position, force]
        """
        # TODO: Implement state retrieval from hardware
        # - Query arm controller for joint states
        # - Read force/torque sensor
        # - Compute forward kinematics for end-effector pose
        # - Get gripper measurements
        
        return {
            'joint_positions': self._joint_positions.copy(),
            'joint_velocities': self._joint_velocities.copy(),
            'joint_torques': self._joint_torques.copy(),
            'end_effector_pose': self._ee_pose.copy(),
            'gripper_state': np.array([self._gripper_position, self._gripper_force]),
        }
    
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        """
        Execute action on SO100 arm.
        
        Args:
            action: Action vector, shape depends on control_mode:
                - position: (8,) [joint_positions (7), gripper (1)]
                - velocity: (8,) [joint_velocities (7), gripper_velocity (1)]
                - torque: (8,) [joint_torques (7), gripper_force (1)]
                - end_effector: (7,) [ee_pose (6), gripper (1)]
            **kwargs: Additional parameters
                - timeout: Action timeout in seconds
                - blocking: Whether to wait for completion (default: True)
                - velocity_scaling: Scale factor for velocity (0-1)
        
        Returns:
            bool: True if action executed successfully
        
        Raises:
            ValueError: If action is invalid or unsafe
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to SO100 arm")
        
        if self._emergency_stop_triggered:
            raise RuntimeError("Emergency stop is active")
        
        # Validate action dimensions
        expected_dim = 7 if self.control_mode == "end_effector" else 8
        if action.shape[0] != expected_dim:
            raise ValueError(f"Action must have {expected_dim} dimensions for {self.control_mode} mode, got {action.shape[0]}")
        
        # Safety checks
        if not self._check_action_safety(action):
            print(f"[{self.robot_id}] Action rejected due to safety constraints")
            return False
        
        # Execute based on control mode
        if self.control_mode == "position":
            return self._execute_position_action(action, **kwargs)
        elif self.control_mode == "velocity":
            return self._execute_velocity_action(action, **kwargs)
        elif self.control_mode == "torque":
            return self._execute_torque_action(action, **kwargs)
        elif self.control_mode == "end_effector":
            return self._execute_ee_action(action, **kwargs)
        
        return False
    
    def _execute_position_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute position control action."""
        joint_targets = action[:7]
        gripper_target = np.clip(action[7], 0.0, 1.0)
        
        # Clip joint positions to limits
        for i in range(self.NUM_JOINTS):
            joint_targets[i] = np.clip(
                joint_targets[i],
                self.JOINT_POSITION_LIMITS[i, 0],
                self.JOINT_POSITION_LIMITS[i, 1]
            )
        
        # TODO: Implement position control
        # - Send joint position targets to controller
        # - Control gripper position
        # - Monitor execution
        # - Handle errors
        
        self._joint_positions = joint_targets.copy()
        self._gripper_position = gripper_target
        
        return True
    
    def _execute_velocity_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute velocity control action."""
        joint_velocities = action[:7]
        gripper_velocity = action[7]
        
        # Clip velocities to limits
        joint_velocities = np.clip(
            joint_velocities,
            -self.JOINT_VELOCITY_LIMITS,
            self.JOINT_VELOCITY_LIMITS
        )
        
        # TODO: Implement velocity control
        # - Send joint velocity commands
        # - Control gripper velocity
        # - Monitor for position limits
        
        self._joint_velocities = joint_velocities.copy()
        
        return True
    
    def _execute_torque_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute torque control action."""
        joint_torques = action[:7]
        gripper_force = np.clip(action[7], 0.0, self.GRIPPER_MAX_FORCE)
        
        # Clip torques to limits
        joint_torques = np.clip(
            joint_torques,
            -self.JOINT_TORQUE_LIMITS,
            self.JOINT_TORQUE_LIMITS
        )
        
        # TODO: Implement torque control
        # - Send joint torque commands
        # - Control gripper force
        # - Monitor for safety limits
        
        self._joint_torques = joint_torques.copy()
        self._gripper_force = gripper_force
        
        return True
    
    def _execute_ee_action(self, action: np.ndarray, **kwargs) -> bool:
        """Execute end-effector pose control action."""
        ee_target = action[:6]
        gripper_target = np.clip(action[6], 0.0, 1.0)
        
        # Check workspace limits
        if not self._check_workspace_limits(ee_target[:3]):
            print(f"[{self.robot_id}] Target pose outside workspace")
            return False
        
        # TODO: Implement end-effector control
        # - Compute inverse kinematics
        # - Check for singularities
        # - Plan trajectory
        # - Execute motion
        # - Control gripper
        
        self._ee_pose = ee_target.copy()
        self._gripper_position = gripper_target
        
        return True
    
    def _check_action_safety(self, action: np.ndarray) -> bool:
        """Check if action is safe to execute."""
        # TODO: Implement comprehensive safety checks
        # - Check joint limits
        # - Check velocity/acceleration limits
        # - Check for potential collisions
        # - Check temperature thresholds
        # - Verify workspace boundaries
        
        # Check temperature warnings
        if np.any(self._joint_temperatures > self.WARNING_JOINT_TEMPERATURE):
            print(f"[{self.robot_id}] Warning: High joint temperatures detected")
        
        if np.any(self._joint_temperatures > self.MAX_JOINT_TEMPERATURE):
            print(f"[{self.robot_id}] Error: Joint temperature limit exceeded")
            return False
        
        return True
    
    def _check_workspace_limits(self, position: np.ndarray) -> bool:
        """Check if position is within workspace limits."""
        return np.all(position >= self.WORKSPACE_MIN) and np.all(position <= self.WORKSPACE_MAX)
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from arm sensors.
        
        Returns:
            Dictionary containing:
                - 'images': Camera images from all views (if enabled)
                - 'proprioception': Internal robot state
        """
        observation = {
            'proprioception': {
                'joint_positions': self._joint_positions.copy(),
                'joint_velocities': self._joint_velocities.copy(),
                'joint_torques': self._joint_torques.copy(),
                'joint_temperatures': self._joint_temperatures.copy(),
                'gripper_position': self._gripper_position,
                'gripper_force': self._gripper_force,
                'end_effector_pose': self._ee_pose.copy(),
            }
        }
        
        if self.force_torque_sensor:
            # TODO: Implement F/T sensor reading
            observation['proprioception']['end_effector_wrench'] = self._ee_wrench.copy()
        
        if self.camera_enabled:
            # TODO: Implement camera frame capture
            # - Capture frames from all camera feeds
            # - Decode and preprocess images
            # - Synchronize timestamps
            observation['images'] = {
                'wrist_camera': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
                'external_camera_1': np.zeros((*self.CAMERA_RESOLUTION, 3), dtype=np.uint8),
            }
        
        return observation
    
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get action space specification.
        
        Returns:
            Dictionary describing the action space based on control mode
        """
        if self.control_mode == "position":
            low = np.concatenate([self.JOINT_POSITION_LIMITS[:, 0], [0.0]])
            high = np.concatenate([self.JOINT_POSITION_LIMITS[:, 1], [1.0]])
            names = [f'joint_{i}' for i in range(7)] + ['gripper']
            units = ['rad'] * 7 + ['normalized']
        elif self.control_mode == "velocity":
            low = np.concatenate([-self.JOINT_VELOCITY_LIMITS, [-1.0]])
            high = np.concatenate([self.JOINT_VELOCITY_LIMITS, [1.0]])
            names = [f'joint_{i}_vel' for i in range(7)] + ['gripper_vel']
            units = ['rad/s'] * 7 + ['normalized']
        elif self.control_mode == "torque":
            low = np.concatenate([-self.JOINT_TORQUE_LIMITS, [0.0]])
            high = np.concatenate([self.JOINT_TORQUE_LIMITS, [self.GRIPPER_MAX_FORCE]])
            names = [f'joint_{i}_torque' for i in range(7)] + ['gripper_force']
            units = ['Nm'] * 7 + ['N']
        else:  # end_effector
            low = np.concatenate([self.WORKSPACE_MIN, [-np.pi, -np.pi, -np.pi], [0.0]])
            high = np.concatenate([self.WORKSPACE_MAX, [np.pi, np.pi, np.pi], [1.0]])
            names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
            units = ['m', 'm', 'm', 'rad', 'rad', 'rad', 'normalized']
        
        return {
            'shape': (len(low),),
            'dtype': np.float32,
            'bounds': {'low': low, 'high': high},
            'names': names,
            'units': units,
            'control_mode': self.control_mode,
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get observation space specification.
        
        Returns:
            Dictionary describing observation space structure
        """
        obs_space = {
            'proprioception': {
                'joint_positions': {'shape': (7,), 'dtype': np.float32},
                'joint_velocities': {'shape': (7,), 'dtype': np.float32},
                'joint_torques': {'shape': (7,), 'dtype': np.float32},
                'joint_temperatures': {'shape': (7,), 'dtype': np.float32},
                'gripper_position': {'shape': (), 'dtype': np.float32},
                'gripper_force': {'shape': (), 'dtype': np.float32},
                'end_effector_pose': {'shape': (6,), 'dtype': np.float32},
            }
        }
        
        if self.force_torque_sensor:
            obs_space['proprioception']['end_effector_wrench'] = {
                'shape': (6,), 'dtype': np.float32
            }
        
        if self.camera_enabled:
            obs_space['images'] = {
                'wrist_camera': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
                'external_camera_1': {'shape': (*self.CAMERA_RESOLUTION, 3), 'dtype': np.uint8},
            }
        
        return obs_space
    
    def move_to_pose(self, pose: np.ndarray, velocity_scaling: float = 0.5) -> bool:
        """
        Move end-effector to target pose using path planning.
        
        Args:
            pose: Target pose [x, y, z, roll, pitch, yaw]
            velocity_scaling: Motion speed scaling factor (0-1)
        
        Returns:
            bool: True if movement completed successfully
        """
        # TODO: Implement motion planning
        # - Compute inverse kinematics
        # - Plan collision-free trajectory
        # - Execute motion with velocity scaling
        # - Monitor and handle errors
        
        if not self._check_workspace_limits(pose[:3]):
            print(f"[{self.robot_id}] Target pose outside workspace")
            return False
        
        print(f"[{self.robot_id}] Moving to pose: {pose}")
        self._ee_pose = pose.copy()
        return True
    
    def move_to_joint_positions(self, joint_positions: np.ndarray, velocity_scaling: float = 0.5) -> bool:
        """
        Move to target joint configuration.
        
        Args:
            joint_positions: Target joint positions (7,)
            velocity_scaling: Motion speed scaling factor (0-1)
        
        Returns:
            bool: True if movement completed successfully
        """
        # TODO: Implement joint space motion
        # - Validate joint positions
        # - Plan trajectory in joint space
        # - Execute motion
        # - Verify final position
        
        for i in range(self.NUM_JOINTS):
            if not (self.JOINT_POSITION_LIMITS[i, 0] <= joint_positions[i] <= self.JOINT_POSITION_LIMITS[i, 1]):
                print(f"[{self.robot_id}] Joint {i} target outside limits")
                return False
        
        print(f"[{self.robot_id}] Moving to joint positions")
        self._joint_positions = joint_positions.copy()
        return True
    
    def emergency_stop(self) -> None:
        """
        Trigger emergency stop of the arm.
        
        This immediately stops all arm motion and holds current position.
        """
        # TODO: Implement emergency stop
        # - Send emergency stop command to controller
        # - Activate brakes
        # - Log emergency stop event
        # - Disable further motion commands
        
        self._emergency_stop_triggered = True
        self._joint_velocities = np.zeros(self.NUM_JOINTS)
        print(f"[{self.robot_id}] EMERGENCY STOP activated")
    
    def clear_emergency_stop(self) -> None:
        """Clear emergency stop and re-enable motion."""
        # TODO: Implement emergency stop clearing
        # - Verify system is safe
        # - Clear emergency stop flag
        # - Re-enable motion commands
        
        self._emergency_stop_triggered = False
        print(f"[{self.robot_id}] Emergency stop cleared")
    
    def calibrate(self) -> bool:
        """
        Perform arm calibration sequence.
        
        Returns:
            bool: True if calibration successful
        """
        # TODO: Implement calibration
        # - Move to calibration poses
        # - Read encoder offsets
        # - Verify joint limits
        # - Calibrate force/torque sensor
        # - Calibrate gripper
        
        print(f"[{self.robot_id}] Calibration completed")
        return True
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics information.
        
        Returns:
            Dictionary with system health and diagnostic data
        """
        # TODO: Implement diagnostics retrieval
        # - Check all system health indicators
        # - Retrieve error codes and warnings
        # - Get maintenance status
        # - Check sensor calibration
        
        return {
            'joints': {
                f'joint_{i}': {
                    'temperature': float(self._joint_temperatures[i]),
                    'status': 'OK' if self._joint_temperatures[i] < self.WARNING_JOINT_TEMPERATURE else 'WARNING',
                }
                for i in range(self.NUM_JOINTS)
            },
            'gripper': {
                'status': 'OK',
                'position': float(self._gripper_position),
                'force': float(self._gripper_force),
            },
            'safety': {
                'emergency_stop': self._emergency_stop_triggered,
                'collision_detected': self._collision_detected,
            },
            'warnings': [],
            'errors': [],
        }


__all__ = ['SO100Arm']
