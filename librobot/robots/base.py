"""Abstract base class for robot interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class AbstractRobot(ABC):
    """
    Abstract base class for robot interfaces.
    
    Provides a unified interface for controlling different robot platforms.
    """
    
    def __init__(self, robot_id: str):
        """
        Initialize robot interface.
        
        Args:
            robot_id: Unique identifier for this robot
        """
        self.robot_id = robot_id
        self._is_connected = False
    
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """
        Connect to the robot.
        
        Args:
            **kwargs: Connection parameters
            
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset robot to initial state."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get current robot state.
        
        Returns:
            Dictionary containing:
                - 'joint_positions': Joint positions
                - 'joint_velocities': Joint velocities
                - 'end_effector_pose': End effector pose
                - Other state information
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: np.ndarray, **kwargs) -> bool:
        """
        Execute an action on the robot.
        
        Args:
            action: Action vector to execute
            **kwargs: Additional execution parameters
            
        Returns:
            bool: True if action executed successfully
        """
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation.
        
        Returns:
            Dictionary containing:
                - 'images': Camera images
                - 'proprioception': Proprioceptive state
                - Other observation data
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get action space specification.
        
        Returns:
            Dictionary describing the action space
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get observation space specification.
        
        Returns:
            Dictionary describing the observation space
        """
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
