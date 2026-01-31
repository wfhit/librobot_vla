"""Leader-follower robot teleoperation interface."""

from typing import Optional

import numpy as np

from librobot.collection.base import AbstractTeleop
from librobot.collection.teleoperation.base import register_teleop


@register_teleop(name="leader_follower", aliases=["leader-follower", "bilateral"])
class LeaderFollowerTeleop(AbstractTeleop):
    """
    Leader-follower robot teleoperation interface.

    Uses a leader robot (usually lower-cost or kinematic) to control
    a follower robot. Common in systems like ALOHA.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        action_dim: int = 7,
        leader_robot=None,
    ):
        """
        Initialize leader-follower teleoperation.

        Args:
            device_id: Optional device identifier
            action_dim: Dimension of action space
            leader_robot: Leader robot instance
        """
        super().__init__(device_id)
        self.action_dim = action_dim
        self.leader_robot = leader_robot

    def connect(self, **kwargs) -> bool:
        """
        Connect to leader robot.

        Args:
            **kwargs: Connection parameters
                leader_robot: Leader robot instance (if not provided in init)

        Returns:
            bool: True if connection successful
        """
        # Get leader robot from kwargs if not set
        if "leader_robot" in kwargs:
            self.leader_robot = kwargs["leader_robot"]

        if self.leader_robot is None:
            print("Warning: No leader robot provided")
            return False

        try:
            # Connect to leader robot if not already connected
            if hasattr(self.leader_robot, "is_connected"):
                if not self.leader_robot.is_connected:
                    if hasattr(self.leader_robot, "connect"):
                        self.leader_robot.connect()

            self._is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to leader robot: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from leader robot."""
        if self.leader_robot is not None:
            try:
                if hasattr(self.leader_robot, "disconnect"):
                    self.leader_robot.disconnect()
            except Exception:
                pass
        self._is_connected = False

    def get_action(self) -> np.ndarray:
        """
        Get current action from leader robot state.

        Returns:
            np.ndarray: Action vector (leader robot joint positions/velocities)
        """
        if not self._is_connected or self.leader_robot is None:
            return np.zeros(self.action_dim)

        try:
            # Get leader robot state
            if hasattr(self.leader_robot, "get_state"):
                state = self.leader_robot.get_state()

                # Extract joint positions or velocities
                if "joint_positions" in state:
                    action = state["joint_positions"]
                    # Resize if needed
                    if len(action) > self.action_dim:
                        action = action[: self.action_dim]
                    elif len(action) < self.action_dim:
                        action = np.pad(action, (0, self.action_dim - len(action)), mode="constant")
                    return action
                else:
                    return np.zeros(self.action_dim)
            else:
                return np.zeros(self.action_dim)

        except Exception as e:
            print(f"Error reading leader robot state: {e}")
            return np.zeros(self.action_dim)

    def calibrate(self) -> bool:
        """
        Calibrate leader-follower system.

        Returns:
            bool: True if calibration successful
        """
        if not self._is_connected or self.leader_robot is None:
            return False

        try:
            # Calibrate leader robot if supported
            if hasattr(self.leader_robot, "calibrate"):
                return self.leader_robot.calibrate()
            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

    def get_status(self) -> dict:
        """
        Get leader-follower teleoperation status.

        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update(
            {
                "action_dim": self.action_dim,
                "leader_robot": (
                    str(type(self.leader_robot).__name__) if self.leader_robot is not None else None
                ),
            }
        )
        return status


__all__ = ["LeaderFollowerTeleop"]
