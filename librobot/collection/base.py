"""Base classes for the collection module."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class AbstractTeleop(ABC):
    """
    Abstract base class for teleoperation interfaces.

    All teleoperation devices (keyboard, spacemouse, VR, mocap, etc.)
    should inherit from this class and implement the required methods.
    """

    def __init__(self, device_id: Optional[str] = None):
        """
        Initialize teleoperation interface.

        Args:
            device_id: Optional device identifier
        """
        self.device_id = device_id
        self._is_connected = False

    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """
        Connect to the teleoperation device.

        Args:
            **kwargs: Device-specific connection parameters

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the teleoperation device."""
        pass

    @abstractmethod
    def get_action(self) -> np.ndarray:
        """
        Get current action from the teleoperation device.

        Returns:
            np.ndarray: Action vector (velocities or positions)
        """
        pass

    def calibrate(self) -> bool:
        """
        Calibrate the teleoperation device.

        Returns:
            bool: True if calibration successful
        """
        return True

    def get_status(self) -> dict[str, Any]:
        """
        Get device status information.

        Returns:
            Dictionary containing device status information
        """
        return {
            "device_id": self.device_id,
            "connected": self._is_connected,
        }

    @property
    def is_connected(self) -> bool:
        """Check if device is connected."""
        return self._is_connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class AbstractConverter(ABC):
    """
    Abstract base class for dataset format converters.

    Converters handle reading and writing different dataset formats
    (LeRobot, HDF5, Zarr, RLDS, etc.).
    """

    def __init__(self, format_name: str):
        """
        Initialize converter.

        Args:
            format_name: Name of the data format
        """
        self.format_name = format_name

    @abstractmethod
    def read_episode(self, path: str, episode_idx: int) -> dict[str, Any]:
        """
        Read a single episode from dataset.

        Args:
            path: Path to dataset
            episode_idx: Episode index

        Returns:
            Dictionary containing episode data
        """
        pass

    @abstractmethod
    def write_episode(self, path: str, episode_data: dict[str, Any]) -> None:
        """
        Write a single episode to dataset.

        Args:
            path: Path to dataset
            episode_data: Episode data to write
        """
        pass

    @abstractmethod
    def validate_dataset(self, path: str) -> bool:
        """
        Validate dataset integrity.

        Args:
            path: Path to dataset

        Returns:
            bool: True if dataset is valid
        """
        pass

    def get_metadata(self, path: str) -> dict[str, Any]:
        """
        Get dataset metadata.

        Args:
            path: Path to dataset

        Returns:
            Dictionary containing metadata
        """
        return {}

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        """
        Set dataset metadata.

        Args:
            path: Path to dataset
            metadata: Metadata to set
        """
        pass


__all__ = [
    "AbstractTeleop",
    "AbstractConverter",
]
