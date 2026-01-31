"""Zarr format converter."""

from pathlib import Path
from typing import Any

import numpy as np

from librobot.collection.base import AbstractConverter
from librobot.collection.converters.base import register_converter


@register_converter(name="zarr", aliases=["Zarr"])
class ZarrConverter(AbstractConverter):
    """
    Converter for Zarr dataset format.

    Zarr format provides cloud-native chunked array storage with
    compression and parallel access.
    """

    def __init__(self):
        """Initialize Zarr converter."""
        super().__init__(format_name="zarr")
        self._zarr_available = self._check_zarr_available()

    def _check_zarr_available(self) -> bool:
        """Check if zarr library is available."""
        try:
            import zarr  # noqa: F401

            return True
        except ImportError:
            return False

    def read_episode(self, path: str, episode_idx: int) -> dict[str, Any]:
        """
        Read a single episode from Zarr dataset.

        Args:
            path: Path to Zarr store
            episode_idx: Episode index

        Returns:
            Dictionary containing episode data
        """
        if not self._zarr_available:
            raise ImportError("zarr library not installed. Install with: pip install zarr")

        import zarr

        store_path = Path(path)
        episode_data = {}

        root = zarr.open(str(store_path), mode="r")
        episode_group = root[f"episode_{episode_idx}"]

        # Read all arrays in the episode group
        for key in episode_group.array_keys():
            episode_data[key] = episode_group[key][:]

        # Read attributes as metadata
        episode_data["metadata"] = dict(episode_group.attrs)

        return episode_data

    def write_episode(self, path: str, episode_data: dict[str, Any]) -> None:
        """
        Write a single episode to Zarr dataset.

        Args:
            path: Path to Zarr store
            episode_data: Episode data to write
        """
        if not self._zarr_available:
            raise ImportError("zarr library not installed. Install with: pip install zarr")

        import zarr

        store_path = Path(path)
        metadata = episode_data.get("metadata", {})
        episode_idx = metadata.get("episode_idx", 0)

        root = zarr.open(str(store_path), mode="a")

        # Create episode group
        episode_group = root.create_group(f"episode_{episode_idx}", overwrite=False)

        # Write data streams
        for key, value in episode_data.items():
            if key == "metadata":
                continue

            if isinstance(value, (list, np.ndarray)):
                data_array = np.array(value)
                episode_group.create_dataset(key, data=data_array, chunks=True, compression="gzip")

        # Write metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                episode_group.attrs[key] = value

    def validate_dataset(self, path: str) -> bool:
        """
        Validate Zarr dataset integrity.

        Args:
            path: Path to Zarr store

        Returns:
            bool: True if dataset is valid
        """
        if not self._zarr_available:
            return False

        import zarr

        store_path = Path(path)
        if not store_path.exists():
            return False

        try:
            root = zarr.open(str(store_path), mode="r")
            # Check for at least one episode group
            episode_groups = [k for k in root.group_keys() if k.startswith("episode_")]
            return len(episode_groups) > 0
        except Exception:
            return False

    def get_metadata(self, path: str) -> dict[str, Any]:
        """
        Get dataset metadata.

        Args:
            path: Path to Zarr store

        Returns:
            Dictionary containing metadata
        """
        if not self._zarr_available:
            return {}

        import zarr

        store_path = Path(path)
        metadata = {"format": "zarr", "path": str(store_path)}

        try:
            root = zarr.open(str(store_path), mode="r")

            # Get dataset-level attributes
            metadata.update(dict(root.attrs))

            # Count episodes
            episode_groups = [k for k in root.group_keys() if k.startswith("episode_")]
            metadata["num_episodes"] = len(episode_groups)
        except Exception:
            pass

        return metadata

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        """
        Set dataset metadata.

        Args:
            path: Path to Zarr store
            metadata: Metadata to set
        """
        if not self._zarr_available:
            raise ImportError("zarr library not installed. Install with: pip install zarr")

        import zarr

        store_path = Path(path)
        root = zarr.open(str(store_path), mode="a")

        # Write dataset-level attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                root.attrs[key] = value


__all__ = ["ZarrConverter"]
