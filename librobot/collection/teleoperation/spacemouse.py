"""3Dconnexion SpaceMouse teleoperation interface."""

from typing import Optional

import numpy as np

from librobot.collection.base import AbstractTeleop
from librobot.collection.teleoperation.base import register_teleop


@register_teleop(name="spacemouse", aliases=["space_mouse", "3dmouse"])
class SpaceMouseTeleop(AbstractTeleop):
    """
    3Dconnexion SpaceMouse teleoperation interface.

    Provides 6-DOF control using a SpaceMouse device.
    Handles optional dependency gracefully.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        action_dim: int = 7,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.0,
    ):
        """
        Initialize SpaceMouse teleoperation.

        Args:
            device_id: Optional device identifier
            action_dim: Dimension of action space
            pos_sensitivity: Position control sensitivity
            rot_sensitivity: Rotation control sensitivity
        """
        super().__init__(device_id)
        self.action_dim = action_dim
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self._device = None
        self._spacemouse_available = self._check_spacemouse_available()

    def _check_spacemouse_available(self) -> bool:
        """
        Check if spacemouse library is available.

        Returns:
            bool: True if spacemouse library can be imported
        """
        try:
            import pyspacemouse  # noqa: F401

            return True
        except ImportError:
            return False

    def connect(self, **kwargs) -> bool:
        """
        Connect to SpaceMouse device.

        Args:
            **kwargs: Connection parameters

        Returns:
            bool: True if connection successful
        """
        if not self._spacemouse_available:
            print(
                "Warning: pyspacemouse library not installed. "
                "Install with: pip install pyspacemouse"
            )
            return False

        try:
            import pyspacemouse

            self._device = pyspacemouse.open()
            if self._device is None:
                print("Warning: No SpaceMouse device found")
                return False

            self._is_connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to SpaceMouse: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from SpaceMouse device."""
        if self._device is not None:
            try:
                import pyspacemouse

                pyspacemouse.close()
            except Exception:
                pass
            self._device = None
        self._is_connected = False

    def get_action(self) -> np.ndarray:
        """
        Get current action from SpaceMouse.

        Returns:
            np.ndarray: Action vector [x, y, z, roll, pitch, yaw, gripper]
        """
        if not self._is_connected or self._device is None:
            return np.zeros(self.action_dim)

        try:
            import pyspacemouse

            state = pyspacemouse.read()
            if state is None:
                return np.zeros(self.action_dim)

            # Extract 6-DOF values
            action = np.zeros(self.action_dim)
            action[0] = state.x * self.pos_sensitivity
            action[1] = state.y * self.pos_sensitivity
            action[2] = state.z * self.pos_sensitivity
            action[3] = state.roll * self.rot_sensitivity
            action[4] = state.pitch * self.rot_sensitivity
            action[5] = state.yaw * self.rot_sensitivity

            # Gripper control from buttons
            if hasattr(state, "buttons"):
                if state.buttons[0]:  # Left button
                    action[6] = 1.0
                elif state.buttons[1]:  # Right button
                    action[6] = -1.0

            return action

        except Exception as e:
            print(f"Error reading SpaceMouse: {e}")
            return np.zeros(self.action_dim)

    def calibrate(self) -> bool:
        """
        Calibrate SpaceMouse (zero the current position).

        Returns:
            bool: True if calibration successful
        """
        if not self._is_connected:
            return False

        # Read current state to establish zero position
        # Most SpaceMouse devices auto-zero when released
        return True

    def get_status(self) -> dict:
        """
        Get SpaceMouse status.

        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update(
            {
                "spacemouse_available": self._spacemouse_available,
                "action_dim": self.action_dim,
                "pos_sensitivity": self.pos_sensitivity,
                "rot_sensitivity": self.rot_sensitivity,
            }
        )
        return status


__all__ = ["SpaceMouseTeleop"]
