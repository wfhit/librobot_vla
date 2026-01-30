"""Sensor implementations for robots."""

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np


class BaseSensor(ABC):
    """Base class for sensors."""

    def __init__(
        self,
        sensor_id: str,
        rate: float = 30.0,
    ):
        """
        Args:
            sensor_id: Sensor identifier
            rate: Sensor update rate (Hz)
        """
        self.sensor_id = sensor_id
        self.rate = rate
        self._is_active = False

    @abstractmethod
    def read(self) -> Dict[str, Any]:
        """Read sensor data."""
        pass

    def start(self) -> bool:
        """Start sensor."""
        self._is_active = True
        return True

    def stop(self) -> None:
        """Stop sensor."""
        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active


class Camera(BaseSensor):
    """RGB camera sensor."""

    def __init__(
        self,
        sensor_id: str = "camera",
        resolution: Tuple[int, int] = (640, 480),
        fov: float = 60.0,
        rate: float = 30.0,
    ):
        super().__init__(sensor_id, rate)
        self.resolution = resolution
        self.fov = fov

    def read(self) -> Dict[str, Any]:
        """Read camera image."""
        # Return dummy data
        return {
            "image": np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8),
            "timestamp": 0.0,
        }

    def get_intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix."""
        fx = self.resolution[0] / (2 * np.tan(np.radians(self.fov / 2)))
        fy = fx
        cx = self.resolution[0] / 2
        cy = self.resolution[1] / 2
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])


class DepthCamera(Camera):
    """RGB-D camera sensor."""

    def __init__(
        self,
        sensor_id: str = "depth_camera",
        resolution: Tuple[int, int] = (640, 480),
        fov: float = 60.0,
        rate: float = 30.0,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
    ):
        super().__init__(sensor_id, resolution, fov, rate)
        self.min_depth = min_depth
        self.max_depth = max_depth

    def read(self) -> Dict[str, Any]:
        """Read RGB and depth images."""
        base_data = super().read()
        base_data["depth"] = np.random.uniform(
            self.min_depth, self.max_depth,
            self.resolution
        ).astype(np.float32)
        return base_data


class ForceTorqueSensor(BaseSensor):
    """Force-torque sensor."""

    def __init__(
        self,
        sensor_id: str = "ft_sensor",
        rate: float = 1000.0,
    ):
        super().__init__(sensor_id, rate)
        self.force_range = 100.0  # N
        self.torque_range = 10.0  # Nm

    def read(self) -> Dict[str, Any]:
        """Read force-torque data."""
        return {
            "force": np.random.randn(3) * 0.1,  # fx, fy, fz
            "torque": np.random.randn(3) * 0.01,  # tx, ty, tz
            "timestamp": 0.0,
        }


class JointEncoder(BaseSensor):
    """Joint encoder sensor."""

    def __init__(
        self,
        sensor_id: str = "encoder",
        num_joints: int = 6,
        resolution: int = 4096,
        rate: float = 1000.0,
    ):
        super().__init__(sensor_id, rate)
        self.num_joints = num_joints
        self.resolution = resolution

    def read(self) -> Dict[str, Any]:
        """Read joint encoder data."""
        return {
            "positions": np.zeros(self.num_joints),
            "velocities": np.zeros(self.num_joints),
            "timestamp": 0.0,
        }


class IMU(BaseSensor):
    """Inertial measurement unit."""

    def __init__(
        self,
        sensor_id: str = "imu",
        rate: float = 200.0,
    ):
        super().__init__(sensor_id, rate)

    def read(self) -> Dict[str, Any]:
        """Read IMU data."""
        return {
            "acceleration": np.array([0, 0, 9.81]),  # m/s^2
            "angular_velocity": np.zeros(3),  # rad/s
            "orientation": np.array([0, 0, 0, 1]),  # quaternion
            "timestamp": 0.0,
        }


class Lidar(BaseSensor):
    """LiDAR sensor."""

    def __init__(
        self,
        sensor_id: str = "lidar",
        num_beams: int = 360,
        max_range: float = 30.0,
        rate: float = 10.0,
    ):
        super().__init__(sensor_id, rate)
        self.num_beams = num_beams
        self.max_range = max_range

    def read(self) -> Dict[str, Any]:
        """Read LiDAR scan."""
        return {
            "ranges": np.random.uniform(0.1, self.max_range, self.num_beams),
            "angles": np.linspace(0, 2*np.pi, self.num_beams),
            "timestamp": 0.0,
        }


class Tactile(BaseSensor):
    """Tactile sensor (e.g., GelSight)."""

    def __init__(
        self,
        sensor_id: str = "tactile",
        resolution: Tuple[int, int] = (64, 64),
        rate: float = 30.0,
    ):
        super().__init__(sensor_id, rate)
        self.resolution = resolution

    def read(self) -> Dict[str, Any]:
        """Read tactile data."""
        return {
            "image": np.random.randint(0, 255, (*self.resolution, 3), dtype=np.uint8),
            "contact_force": np.random.rand() * 10,  # N
            "timestamp": 0.0,
        }


__all__ = [
    'BaseSensor',
    'Camera',
    'DepthCamera',
    'ForceTorqueSensor',
    'JointEncoder',
    'IMU',
    'Lidar',
    'Tactile',
]
