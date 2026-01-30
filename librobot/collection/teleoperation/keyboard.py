"""Keyboard-based teleoperation interface."""

import sys
from typing import Optional

import numpy as np

from librobot.collection.base import AbstractTeleop
from librobot.collection.teleoperation.base import register_teleop


@register_teleop(name="keyboard", aliases=["kbd"])
class KeyboardTeleop(AbstractTeleop):
    """
    Keyboard-based teleoperation interface.

    Provides control using keyboard input with configurable key mappings.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        action_dim: int = 7,
        speed_scale: float = 0.1,
    ):
        """
        Initialize keyboard teleoperation.

        Args:
            device_id: Optional device identifier
            action_dim: Dimension of action space
            speed_scale: Scaling factor for actions
        """
        super().__init__(device_id)
        self.action_dim = action_dim
        self.speed_scale = speed_scale
        self._current_action = np.zeros(action_dim)
        self._key_bindings = self._setup_key_bindings()

    def _setup_key_bindings(self) -> dict[str, int]:
        """
        Setup default key bindings.

        Returns:
            Dictionary mapping keys to action indices
        """
        return {
            "w": 0,  # Forward
            "s": 0,  # Backward
            "a": 1,  # Left
            "d": 1,  # Right
            "q": 2,  # Up
            "e": 2,  # Down
            "i": 3,  # Roll positive
            "k": 3,  # Roll negative
            "j": 4,  # Pitch positive
            "l": 4,  # Pitch negative
            "u": 5,  # Yaw positive
            "o": 5,  # Yaw negative
            "z": 6,  # Gripper open
            "x": 6,  # Gripper close
        }

    def connect(self, **kwargs) -> bool:
        """
        Connect to keyboard input.

        Args:
            **kwargs: Connection parameters (unused for keyboard)

        Returns:
            bool: True if connection successful
        """
        try:
            # Check if we can read from stdin
            if sys.stdin.isatty():
                self._is_connected = True
                return True
            else:
                # In non-interactive mode, still connect but warn
                self._is_connected = True
                return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Disconnect from keyboard input."""
        self._is_connected = False
        self._current_action = np.zeros(self.action_dim)

    def get_action(self) -> np.ndarray:
        """
        Get current action from keyboard.

        Returns:
            np.ndarray: Action vector
        """
        return self._current_action.copy()

    def update_action_from_key(self, key: str, pressed: bool = True) -> None:
        """
        Update action based on key press/release.

        Args:
            key: Key character
            pressed: True if key is pressed, False if released
        """
        key = key.lower()
        if key not in self._key_bindings:
            return

        action_idx = self._key_bindings[key]

        # Determine direction based on key
        direction = 1.0 if pressed else 0.0
        if key in ["s", "a", "k", "l", "o", "x"]:
            direction = -direction if pressed else 0.0

        self._current_action[action_idx] = direction * self.speed_scale

    def get_status(self) -> dict:
        """
        Get keyboard teleoperation status.

        Returns:
            Dictionary containing status information
        """
        status = super().get_status()
        status.update(
            {
                "action_dim": self.action_dim,
                "speed_scale": self.speed_scale,
                "current_action": self._current_action.tolist(),
            }
        )
        return status


__all__ = ["KeyboardTeleop"]
