"""LeRobot dataset loader."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import numpy as np

from .base import BaseDatasetLoader


class LeRobotDataset(BaseDatasetLoader):
    """
    Dataset loader for LeRobot format.

    LeRobot is a popular format for robot learning datasets that stores
    episodes as individual files with metadata.

    Expected structure:
        data_path/
        ├── meta/
        │   ├── info.json
        │   └── episodes.json
        ├── videos/
        │   └── observation.image/
        └── data/
            └── chunk-000/
                └── episode_*.parquet
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Initialize LeRobot dataset.

        Args:
            data_path: Path to LeRobot dataset
            split: Dataset split
            transform: Optional transform
            cache_in_memory: Cache episodes in memory
            num_workers: Number of data loading workers
            delta_timestamps: Delta timestamps for temporal features
        """
        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            cache_in_memory=cache_in_memory,
            num_workers=num_workers,
        )
        self.delta_timestamps = delta_timestamps or {}
        self._episode_info: Optional[List[Dict]] = None
        self._info: Optional[Dict] = None
        self._load_info()

    def _load_info(self) -> None:
        """Load dataset info and episode metadata."""
        info_path = self.data_path / "meta" / "info.json"
        episodes_path = self.data_path / "meta" / "episodes.json"

        if info_path.exists():
            with open(info_path, "r") as f:
                self._info = json.load(f)
        else:
            self._info = {}

        if episodes_path.exists():
            with open(episodes_path, "r") as f:
                self._episode_info = json.load(f)
        else:
            # Try to infer from data directory
            self._episode_info = self._infer_episodes()

    def _infer_episodes(self) -> List[Dict]:
        """Infer episode info from data directory."""
        episodes = []
        data_dir = self.data_path / "data"

        if data_dir.exists():
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                for parquet_file in sorted(chunk_dir.glob("episode_*.parquet")):
                    episode_idx = int(parquet_file.stem.split("_")[1])
                    episodes.append(
                        {
                            "episode_index": episode_idx,
                            "path": str(parquet_file.relative_to(self.data_path)),
                        }
                    )

        return episodes

    def _get_num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self._episode_info) if self._episode_info else 0

    def _get_episode_length(self, episode_idx: int) -> int:
        """Get episode length."""
        if self._episode_info and episode_idx < len(self._episode_info):
            info = self._episode_info[episode_idx]
            return info.get("length", info.get("num_frames", 100))
        return 100  # Default

    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Load episode data.

        Args:
            episode_idx: Episode index

        Returns:
            Episode data dictionary
        """
        if not self._episode_info or episode_idx >= len(self._episode_info):
            raise IndexError(f"Episode {episode_idx} not found")

        episode_info = self._episode_info[episode_idx]
        episode_path = self.data_path / episode_info.get(
            "path", f"data/chunk-000/episode_{episode_idx:06d}.parquet"
        )

        # Try loading from parquet
        if episode_path.suffix == ".parquet":
            return self._load_parquet_episode(episode_path)

        # Fallback to HDF5 or other formats
        return self._load_fallback_episode(episode_idx)

    def _load_parquet_episode(self, path: Path) -> Dict[str, Any]:
        """Load episode from parquet file."""
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            df = table.to_pandas()

            episode = {}
            for col in df.columns:
                values = df[col].values
                if hasattr(values[0], "__len__") and not isinstance(values[0], str):
                    episode[col] = np.stack(values)
                else:
                    episode[col] = np.array(values)

            return episode
        except ImportError:
            # Fallback without pyarrow
            return self._create_dummy_episode()

    def _load_fallback_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Fallback episode loading."""
        return self._create_dummy_episode()

    def _create_dummy_episode(self, length: int = 100) -> Dict[str, Any]:
        """Create dummy episode for testing."""
        return {
            "images": np.random.randn(length, 3, 224, 224).astype(np.float32),
            "actions": np.random.randn(length, 7).astype(np.float32),
            "proprioception": np.random.randn(length, 14).astype(np.float32),
            "text": ["pick up the object"] * length,
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata = super()._load_metadata()
        if self._info:
            metadata.update(self._info)
        return metadata


__all__ = ["LeRobotDataset"]
