"""HDF5 format converter."""

from pathlib import Path
from typing import Any

import numpy as np

from librobot.collection.base import AbstractConverter
from librobot.collection.converters.base import register_converter


@register_converter(name="hdf5", aliases=["HDF5", "h5"])
class HDF5Converter(AbstractConverter):
    """
    Converter for HDF5 dataset format.

    HDF5 format stores data in hierarchical groups with efficient
    compression and chunking.
    """

    def __init__(self):
        """Initialize HDF5 converter."""
        super().__init__(format_name="hdf5")
        self._h5py_available = self._check_h5py_available()

    def _check_h5py_available(self) -> bool:
        """Check if h5py library is available."""
        try:
            import h5py  # noqa: F401

            return True
        except ImportError:
            return False

    def read_episode(self, path: str, episode_idx: int) -> dict[str, Any]:
        """
        Read a single episode from HDF5 dataset.

        Args:
            path: Path to HDF5 file
            episode_idx: Episode index

        Returns:
            Dictionary containing episode data
        """
        if not self._h5py_available:
            raise ImportError("h5py library not installed. Install with: pip install h5py")

        import h5py

        dataset_path = Path(path)
        episode_data = {}

        with h5py.File(dataset_path, "r") as f:
            episode_group = f[f"episode_{episode_idx}"]

            # Read all datasets in the episode group
            for key in episode_group.keys():
                episode_data[key] = episode_group[key][:]

            # Read attributes as metadata
            episode_data["metadata"] = dict(episode_group.attrs)

        return episode_data

    def write_episode(self, path: str, episode_data: dict[str, Any]) -> None:
        """
        Write a single episode to HDF5 dataset.

        Args:
            path: Path to HDF5 file
            episode_data: Episode data to write
        """
        if not self._h5py_available:
            raise ImportError("h5py library not installed. Install with: pip install h5py")

        import h5py

        dataset_path = Path(path)
        metadata = episode_data.get("metadata", {})
        episode_idx = metadata.get("episode_idx", 0)

        with h5py.File(dataset_path, "a") as f:
            # Create episode group
            episode_group = f.create_group(f"episode_{episode_idx}")

            # Write data streams
            for key, value in episode_data.items():
                if key == "metadata":
                    continue

                if isinstance(value, (list, np.ndarray)):
                    episode_group.create_dataset(key, data=np.array(value), compression="gzip")

            # Write metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    episode_group.attrs[key] = value

    def validate_dataset(self, path: str) -> bool:
        """
        Validate HDF5 dataset integrity.

        Args:
            path: Path to HDF5 file

        Returns:
            bool: True if dataset is valid
        """
        if not self._h5py_available:
            return False

        import h5py

        dataset_path = Path(path)
        if not dataset_path.exists():
            return False

        try:
            with h5py.File(dataset_path, "r") as f:
                # Check for at least one episode group
                episode_groups = [k for k in f.keys() if k.startswith("episode_")]
                return len(episode_groups) > 0
        except Exception:
            return False

    def get_metadata(self, path: str) -> dict[str, Any]:
        """
        Get dataset metadata.

        Args:
            path: Path to HDF5 file

        Returns:
            Dictionary containing metadata
        """
        if not self._h5py_available:
            return {}

        import h5py

        dataset_path = Path(path)
        metadata = {"format": "hdf5", "path": str(dataset_path)}

        try:
            with h5py.File(dataset_path, "r") as f:
                # Get dataset-level attributes
                metadata.update(dict(f.attrs))

                # Count episodes
                episode_groups = [k for k in f.keys() if k.startswith("episode_")]
                metadata["num_episodes"] = len(episode_groups)
        except Exception:
            pass

        return metadata

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        """
        Set dataset metadata.

        Args:
            path: Path to HDF5 file
            metadata: Metadata to set
        """
        if not self._h5py_available:
            raise ImportError("h5py library not installed. Install with: pip install h5py")

        import h5py

        dataset_path = Path(path)

        with h5py.File(dataset_path, "a") as f:
            # Write dataset-level attributes
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value


__all__ = ["HDF5Converter"]
