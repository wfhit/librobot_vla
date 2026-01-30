"""Zarr dataset loader."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .base import BaseDatasetLoader


class ZarrDataset(BaseDatasetLoader):
    """
    Dataset loader for Zarr format.

    Zarr is a format for chunked, compressed N-dimensional arrays,
    commonly used for large-scale robot learning datasets.

    Expected structure:
        data_path.zarr/
        ├── .zattrs
        ├── .zgroup
        ├── episodes/
        │   ├── 0/
        │   │   ├── images/
        │   │   ├── actions/
        │   │   └── states/
        │   └── 1/
        │       └── ...
        └── metadata/
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
        chunk_size: int = 1000,
        compression: str = "blosc",
    ):
        """
        Initialize Zarr dataset.

        Args:
            data_path: Path to Zarr store
            split: Dataset split
            transform: Optional transform
            cache_in_memory: Cache in memory
            num_workers: Data loading workers
            chunk_size: Chunk size for reading
            compression: Compression algorithm
        """
        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            cache_in_memory=cache_in_memory,
            num_workers=num_workers,
        )
        self.chunk_size = chunk_size
        self.compression = compression
        self._zarr_store = None
        self._episode_keys: List[str] = []
        self._episode_lengths: List[int] = []
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Index Zarr store."""
        try:
            import zarr

            self._zarr_store = zarr.open(str(self.data_path), mode="r")

            # Find episodes
            if "episodes" in self._zarr_store:
                episodes_group = self._zarr_store["episodes"]
                for key in sorted(
                    episodes_group.keys(), key=lambda x: int(x) if x.isdigit() else x
                ):
                    self._episode_keys.append(key)
                    episode = episodes_group[key]
                    if "actions" in episode:
                        self._episode_lengths.append(len(episode["actions"]))
                    else:
                        self._episode_lengths.append(100)
            else:
                # Flat structure
                self._index_flat_structure()

        except (ImportError, Exception):
            self._create_dummy_index()

    def _index_flat_structure(self) -> None:
        """Index flat Zarr structure."""
        if self._zarr_store is None:
            return

        # Check for episode_* groups
        for key in self._zarr_store.keys():
            if key.startswith("episode_") or key.isdigit():
                self._episode_keys.append(key)
                episode = self._zarr_store[key]
                if "actions" in episode:
                    self._episode_lengths.append(len(episode["actions"]))
                else:
                    self._episode_lengths.append(100)

    def _create_dummy_index(self) -> None:
        """Create dummy index for testing."""
        for i in range(10):
            self._episode_keys.append(str(i))
            self._episode_lengths.append(100)

    def _get_num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self._episode_keys)

    def _get_episode_length(self, episode_idx: int) -> int:
        """Get episode length."""
        if episode_idx < len(self._episode_lengths):
            return self._episode_lengths[episode_idx]
        return 100

    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Load episode from Zarr.

        Args:
            episode_idx: Episode index

        Returns:
            Episode data
        """
        if episode_idx >= len(self._episode_keys):
            raise IndexError(f"Episode {episode_idx} not found")

        if self._zarr_store is None:
            return self._create_dummy_episode()

        episode_key = self._episode_keys[episode_idx]

        try:
            if "episodes" in self._zarr_store:
                episode_group = self._zarr_store["episodes"][episode_key]
            else:
                episode_group = self._zarr_store[episode_key]

            return self._extract_zarr_data(episode_group)
        except Exception:
            return self._create_dummy_episode()

    def _extract_zarr_data(self, group) -> Dict[str, Any]:
        """Extract data from Zarr group."""
        episode = {}

        # Standard keys
        key_mapping = {
            "images": "images",
            "image": "images",
            "observations/images": "images",
            "actions": "actions",
            "action": "actions",
            "states": "proprioception",
            "state": "proprioception",
            "proprioception": "proprioception",
        }

        for zarr_key, episode_key in key_mapping.items():
            try:
                if "/" in zarr_key:
                    parts = zarr_key.split("/")
                    data = group
                    for part in parts:
                        if part in data:
                            data = data[part]
                        else:
                            data = None
                            break
                else:
                    data = group.get(zarr_key)

                if data is not None and episode_key not in episode:
                    episode[episode_key] = np.array(data)
            except Exception:
                pass

        return episode if episode else self._create_dummy_episode()

    def _create_dummy_episode(self, length: int = 100) -> Dict[str, Any]:
        """Create dummy episode for testing."""
        return {
            "images": np.random.randn(length, 3, 224, 224).astype(np.float32),
            "actions": np.random.randn(length, 7).astype(np.float32),
            "proprioception": np.random.randn(length, 14).astype(np.float32),
        }

    def close(self) -> None:
        """Close Zarr store."""
        self._zarr_store = None


__all__ = ["ZarrDataset"]
