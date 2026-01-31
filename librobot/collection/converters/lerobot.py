"""LeRobot format converter."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from librobot.collection.base import AbstractConverter
from librobot.collection.converters.base import register_converter


@register_converter(name="lerobot", aliases=["LeRobot"])
class LeRobotConverter(AbstractConverter):
    """
    Converter for LeRobot dataset format.

    LeRobot format stores episodes in separate directories with:
    - metadata.json: Episode metadata
    - *.npy: Data streams (images, actions, etc.)
    """

    def __init__(self):
        """Initialize LeRobot converter."""
        super().__init__(format_name="lerobot")

    def read_episode(self, path: str, episode_idx: int) -> dict[str, Any]:
        """
        Read a single episode from LeRobot dataset.

        Args:
            path: Path to dataset root
            episode_idx: Episode index

        Returns:
            Dictionary containing episode data
        """
        dataset_path = Path(path)
        episode_dir = dataset_path / f"episode_{episode_idx:06d}"

        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode {episode_idx} not found at {episode_dir}")

        # Read metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Read data streams
        episode_data = {"metadata": metadata}

        for npy_file in episode_dir.glob("*.npy"):
            stream_name = npy_file.stem
            episode_data[stream_name] = np.load(npy_file)

        return episode_data

    def write_episode(self, path: str, episode_data: dict[str, Any]) -> None:
        """
        Write a single episode to LeRobot dataset.

        Args:
            path: Path to dataset root
            episode_data: Episode data to write
        """
        dataset_path = Path(path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Get episode index
        metadata = episode_data.get("metadata", {})
        episode_idx = metadata.get("episode_idx", 0)

        episode_dir = dataset_path / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Write metadata
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            # Convert numpy types to native Python types
            metadata_json = {}
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata_json[key] = value.item()
                else:
                    metadata_json[key] = value
            json.dump(metadata_json, f, indent=2)

        # Write data streams
        for stream_name, stream_data in episode_data.items():
            if stream_name == "metadata":
                continue

            data_path = episode_dir / f"{stream_name}.npy"
            if isinstance(stream_data, (list, np.ndarray)):
                try:
                    np.save(data_path, np.array(stream_data))
                except Exception as e:
                    print(f"Warning: Could not save {stream_name}: {e}")

    def validate_dataset(self, path: str) -> bool:
        """
        Validate LeRobot dataset integrity.

        Args:
            path: Path to dataset root

        Returns:
            bool: True if dataset is valid
        """
        dataset_path = Path(path)
        if not dataset_path.exists():
            return False

        # Check for at least one episode
        episode_dirs = list(dataset_path.glob("episode_*"))
        if not episode_dirs:
            return False

        # Validate first episode structure
        first_episode = episode_dirs[0]
        metadata_path = first_episode / "metadata.json"

        return metadata_path.exists()

    def get_metadata(self, path: str) -> dict[str, Any]:
        """
        Get dataset metadata.

        Args:
            path: Path to dataset root

        Returns:
            Dictionary containing metadata
        """
        dataset_path = Path(path)
        dataset_metadata_path = dataset_path / "dataset_metadata.json"

        if dataset_metadata_path.exists():
            with open(dataset_metadata_path) as f:
                return json.load(f)

        # If no dataset metadata, aggregate from episodes
        episode_dirs = sorted(dataset_path.glob("episode_*"))
        return {
            "format": "lerobot",
            "num_episodes": len(episode_dirs),
            "path": str(dataset_path),
        }

    def set_metadata(self, path: str, metadata: dict[str, Any]) -> None:
        """
        Set dataset metadata.

        Args:
            path: Path to dataset root
            metadata: Metadata to set
        """
        dataset_path = Path(path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        dataset_metadata_path = dataset_path / "dataset_metadata.json"
        with open(dataset_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


__all__ = ["LeRobotConverter"]
