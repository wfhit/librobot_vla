"""Episode recording utilities for data collection."""

import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from librobot.collection.recording.data_buffer import DataBuffer


class EpisodeRecorder:
    """
    Episode recorder for collecting robot demonstration data.

    Manages recording of complete episodes including images, actions,
    proprioception, and metadata.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        format: str = "lerobot",
        buffer_size: Optional[int] = None,
    ):
        """
        Initialize episode recorder.

        Args:
            output_dir: Directory to save episodes
            format: Data format (lerobot, hdf5, zarr)
            buffer_size: Maximum buffer size for episode data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.buffer_size = buffer_size

        self._buffer = DataBuffer(max_size=buffer_size)
        self._is_recording = False
        self._episode_start_time: Optional[float] = None
        self._episode_metadata: dict[str, Any] = {}
        self._episode_count = 0

    def start_episode(
        self,
        task_name: Optional[str] = None,
        instruction: Optional[str] = None,
        **metadata,
    ) -> bool:
        """
        Start recording a new episode.

        Args:
            task_name: Task name for this episode
            instruction: Language instruction for this episode
            **metadata: Additional metadata

        Returns:
            bool: True if recording started successfully
        """
        if self._is_recording:
            print("Warning: Already recording an episode")
            return False

        # Clear buffer
        self._buffer.clear_all()

        # Setup episode metadata
        self._episode_metadata = {
            "episode_idx": self._episode_count,
            "task_name": task_name,
            "instruction": instruction,
            "start_time": time.time(),
            **metadata,
        }

        self._episode_start_time = time.time()
        self._is_recording = True
        return True

    def stop_episode(self) -> dict[str, Any]:
        """
        Stop recording current episode.

        Returns:
            Dictionary containing episode data
        """
        if not self._is_recording:
            print("Warning: Not currently recording")
            return {}

        # Finalize metadata
        self._episode_metadata["end_time"] = time.time()
        self._episode_metadata["duration"] = (
            time.time() - self._episode_start_time if self._episode_start_time else 0.0
        )

        # Get all buffered data
        episode_data = self._buffer.get_all_streams()
        episode_data["metadata"] = self._episode_metadata

        self._is_recording = False
        self._episode_count += 1

        return episode_data

    def add_timestep(
        self,
        images: Optional[dict[str, np.ndarray]] = None,
        action: Optional[np.ndarray] = None,
        proprioception: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Add a timestep to current episode.

        Args:
            images: Dictionary of camera images
            action: Action vector
            proprioception: Proprioceptive state
            reward: Reward value
            **kwargs: Additional data streams
        """
        if not self._is_recording:
            print("Warning: Not recording, call start_episode() first")
            return

        # Record timestamp
        timestamp = time.time() - (self._episode_start_time or time.time())
        self._buffer.append("timestamp", timestamp)

        # Record images
        if images is not None:
            for camera_name, image in images.items():
                self._buffer.append(f"image_{camera_name}", image)

        # Record action
        if action is not None:
            self._buffer.append("action", action)

        # Record proprioception
        if proprioception is not None:
            self._buffer.append("proprioception", proprioception)

        # Record reward
        if reward is not None:
            self._buffer.append("reward", reward)

        # Record additional streams
        for key, value in kwargs.items():
            self._buffer.append(key, value)

    def save_episode(self, episode_data: Optional[dict[str, Any]] = None) -> Optional[Path]:
        """
        Save episode to disk.

        Args:
            episode_data: Episode data to save (or None to save current episode)

        Returns:
            Path to saved episode file, or None if save failed
        """
        # If no data provided, stop and get current episode
        if episode_data is None:
            if self._is_recording:
                episode_data = self.stop_episode()
            else:
                print("Warning: No episode data to save")
                return None

        # Get episode index
        episode_idx = episode_data.get("metadata", {}).get("episode_idx", self._episode_count - 1)

        # Create episode directory
        episode_dir = self.output_dir / f"episode_{episode_idx:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save based on format
            if self.format == "lerobot":
                return self._save_lerobot_format(episode_data, episode_dir)
            elif self.format == "hdf5":
                return self._save_hdf5_format(episode_data, episode_dir)
            elif self.format == "zarr":
                return self._save_zarr_format(episode_data, episode_dir)
            else:
                print(f"Warning: Unknown format '{self.format}', using JSON")
                return self._save_json_format(episode_data, episode_dir)

        except Exception as e:
            print(f"Failed to save episode: {e}")
            return None

    def _save_lerobot_format(self, episode_data: dict[str, Any], episode_dir: Path) -> Path:
        """Save episode in LeRobot format."""
        # Simplified LeRobot format (JSON + numpy files)
        metadata_path = episode_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            metadata = episode_data.get("metadata", {})
            # Convert numpy types to native Python types for JSON
            metadata_json = {}
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata_json[key] = value.item()
                else:
                    metadata_json[key] = value
            json.dump(metadata_json, f, indent=2)

        # Save data streams
        for stream_name, stream_data in episode_data.items():
            if stream_name == "metadata":
                continue
            data_path = episode_dir / f"{stream_name}.npy"
            if isinstance(stream_data, list) and len(stream_data) > 0:
                try:
                    np.save(data_path, np.array(stream_data))
                except Exception as e:
                    print(f"Warning: Could not save {stream_name}: {e}")

        return metadata_path

    def _save_hdf5_format(self, episode_data: dict[str, Any], episode_dir: Path) -> Path:
        """Save episode in HDF5 format."""
        # Placeholder - would use h5py
        return episode_dir / "episode.h5"

    def _save_zarr_format(self, episode_data: dict[str, Any], episode_dir: Path) -> Path:
        """Save episode in Zarr format."""
        # Placeholder - would use zarr
        return episode_dir / "episode.zarr"

    def _save_json_format(self, episode_data: dict[str, Any], episode_dir: Path) -> Path:
        """Save episode in simple JSON format."""
        data_path = episode_dir / "episode.json"
        # Simplify data for JSON serialization
        simplified_data = {
            "metadata": episode_data.get("metadata", {}),
            "length": len(episode_data.get("timestamp", [])),
        }
        with open(data_path, "w") as f:
            json.dump(simplified_data, f, indent=2)
        return data_path

    def get_status(self) -> dict:
        """
        Get recorder status.

        Returns:
            Dictionary containing status information
        """
        return {
            "output_dir": str(self.output_dir),
            "format": self.format,
            "is_recording": self._is_recording,
            "episode_count": self._episode_count,
            "current_episode_length": self._buffer.get_stream_length("timestamp"),
        }

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording


__all__ = ["EpisodeRecorder"]
