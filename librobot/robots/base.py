"""Base robot interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RobotConfig:
    """Robot configuration."""

    name: str
    action_dim: int
    state_dim: int
    action_names: List[str]
    state_names: List[str]
    action_ranges: Optional[np.ndarray] = None  # [action_dim, 2] min/max ranges
    state_ranges: Optional[np.ndarray] = None  # [state_dim, 2] min/max ranges
    control_frequency: float = 10.0  # Hz
    description: str = ""


class BaseRobot(ABC):
    """Base class for robot definitions.

    This class defines the robot's action and state spaces,
    but does not handle actual robot control.
    """

    def __init__(self, config: RobotConfig):
        """Initialize robot.

        Args:
            config: Robot configuration
        """
        self.config = config

    @property
    def name(self) -> str:
        """Get robot name."""
        return self.config.name

    @property
    def action_dim(self) -> int:
        """Get action space dimension."""
        return self.config.action_dim

    @property
    def state_dim(self) -> int:
        """Get state space dimension."""
        return self.config.state_dim

    @property
    def action_names(self) -> List[str]:
        """Get action names."""
        return self.config.action_names

    @property
    def state_names(self) -> List[str]:
        """Get state names."""
        return self.config.state_names

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] range.

        Args:
            action: Raw action

        Returns:
            Normalized action
        """
        if self.config.action_ranges is None:
            return action

        ranges = self.config.action_ranges
        normalized = 2 * (action - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1
        return np.clip(normalized, -1, 1)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1] range.

        Args:
            action: Normalized action

        Returns:
            Raw action
        """
        if self.config.action_ranges is None:
            return action

        ranges = self.config.action_ranges
        denormalized = (action + 1) / 2 * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        return denormalized

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to [-1, 1] range.

        Args:
            state: Raw state

        Returns:
            Normalized state
        """
        if self.config.state_ranges is None:
            return state

        ranges = self.config.state_ranges
        normalized = 2 * (state - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0]) - 1
        return np.clip(normalized, -1, 1)

    def denormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state from [-1, 1] range.

        Args:
            state: Normalized state

        Returns:
            Raw state
        """
        if self.config.state_ranges is None:
            return state

        ranges = self.config.state_ranges
        denormalized = (state + 1) / 2 * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        return denormalized
