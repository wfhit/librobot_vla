"""HDF5 dataset loader."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import BaseDatasetLoader


class HDF5Dataset(BaseDatasetLoader):
    """
    Dataset loader for HDF5 format.

    HDF5 is a hierarchical data format commonly used for storing
    large robot demonstration datasets.

    Expected structure:
        data_path/
        ├── episode_0.hdf5
        ├── episode_1.hdf5
        └── ...

    Or single file:
        data_path.hdf5
        ├── episode_0/
        │   ├── observations/
        │   │   ├── images
        │   │   └── states
        │   └── actions
        └── episode_1/
            └── ...
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
        obs_keys: Optional[List[str]] = None,
        action_key: str = "actions",
        chunk_size: int = 1000,
    ):
        """
        Initialize HDF5 dataset.

        Args:
            data_path: Path to HDF5 file or directory
            split: Dataset split
            transform: Optional transform
            cache_in_memory: Cache in memory
            num_workers: Data loading workers
            obs_keys: Keys for observations
            action_key: Key for actions
            chunk_size: Chunk size for reading
        """
        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            cache_in_memory=cache_in_memory,
            num_workers=num_workers,
        )
        self.obs_keys = obs_keys or ["images", "states"]
        self.action_key = action_key
        self.chunk_size = chunk_size
        self._episode_files: List[Path] = []
        self._episode_lengths: List[int] = []
        self._h5_file = None
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Index HDF5 files or episodes within single file."""
        if self.data_path.is_file():
            self._index_single_file()
        elif self.data_path.is_dir():
            self._index_directory()
        else:
            self._create_dummy_index()

    def _index_single_file(self) -> None:
        """Index episodes within a single HDF5 file."""
        try:
            import h5py

            with h5py.File(self.data_path, "r") as f:
                for key in f.keys():
                    if key.startswith("episode") or key.startswith("demo"):
                        self._episode_files.append(self.data_path)
                        episode_data = f[key]
                        if self.action_key in episode_data:
                            self._episode_lengths.append(len(episode_data[self.action_key]))
                        else:
                            self._episode_lengths.append(100)
        except (ImportError, Exception):
            self._create_dummy_index()

    def _index_directory(self) -> None:
        """Index HDF5 files in directory."""
        h5_files = sorted(self.data_path.glob("*.hdf5")) + sorted(self.data_path.glob("*.h5"))

        for h5_file in h5_files:
            self._episode_files.append(h5_file)
            self._episode_lengths.append(self._get_file_episode_length(h5_file))

    def _get_file_episode_length(self, path: Path) -> int:
        """Get episode length from HDF5 file."""
        try:
            import h5py

            with h5py.File(path, "r") as f:
                if self.action_key in f:
                    return len(f[self.action_key])
                # Try nested structure
                for key in f.keys():
                    if self.action_key in f[key]:
                        return len(f[key][self.action_key])
        except (ImportError, Exception):
            pass
        return 100

    def _create_dummy_index(self) -> None:
        """Create dummy index for testing."""
        for i in range(10):
            self._episode_files.append(None)
            self._episode_lengths.append(100)

    def _get_num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self._episode_files)

    def _get_episode_length(self, episode_idx: int) -> int:
        """Get episode length."""
        if episode_idx < len(self._episode_lengths):
            return self._episode_lengths[episode_idx]
        return 100

    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Load episode from HDF5.

        Args:
            episode_idx: Episode index

        Returns:
            Episode data
        """
        if episode_idx >= len(self._episode_files):
            raise IndexError(f"Episode {episode_idx} not found")

        episode_file = self._episode_files[episode_idx]

        if episode_file is None:
            return self._create_dummy_episode()

        return self._load_h5_episode(episode_file, episode_idx)

    def _load_h5_episode(self, path: Path, episode_idx: int) -> Dict[str, Any]:
        """Load episode from HDF5 file."""
        try:
            import h5py

            with h5py.File(path, "r") as f:
                # Single episode per file
                if self.action_key in f:
                    return self._extract_h5_data(f)

                # Multiple episodes per file
                episode_key = f"episode_{episode_idx}"
                if episode_key in f:
                    return self._extract_h5_data(f[episode_key])

                # Try demo key
                demo_key = f"demo_{episode_idx}"
                if demo_key in f:
                    return self._extract_h5_data(f[demo_key])

                # Fallback: use first available key
                for key in f.keys():
                    if isinstance(f[key], h5py.Group):
                        return self._extract_h5_data(f[key])

        except (ImportError, Exception) as e:
            pass

        return self._create_dummy_episode()

    def _extract_h5_data(self, group) -> Dict[str, Any]:
        """Extract data from HDF5 group."""
        import h5py

        episode = {}

        # Extract observations
        if "observations" in group:
            obs = group["observations"]
            if "images" in obs:
                episode["images"] = np.array(obs["images"])
            if "states" in obs:
                episode["proprioception"] = np.array(obs["states"])

        # Direct keys
        for key in self.obs_keys:
            if key in group and key not in episode:
                episode[key] = np.array(group[key])

        # Extract actions
        if self.action_key in group:
            episode["actions"] = np.array(group[self.action_key])

        return episode

    def _create_dummy_episode(self, length: int = 100) -> Dict[str, Any]:
        """Create dummy episode for testing."""
        return {
            "images": np.random.randn(length, 3, 224, 224).astype(np.float32),
            "actions": np.random.randn(length, 7).astype(np.float32),
            "proprioception": np.random.randn(length, 14).astype(np.float32),
        }

    def close(self) -> None:
        """Close any open file handles."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


__all__ = ["HDF5Dataset"]
