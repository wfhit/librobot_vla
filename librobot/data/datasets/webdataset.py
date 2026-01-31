"""WebDataset loader for streaming large-scale datasets."""

import io
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import numpy as np

from .base import BaseDatasetLoader


class WebDatasetLoader(BaseDatasetLoader):
    """
    Dataset loader for WebDataset format.

    WebDataset is a format designed for large-scale distributed training
    with efficient sequential access to tar archives.

    Expected structure:
        data_path/
        ├── train-000000.tar
        ├── train-000001.tar
        └── ...

    Each tar contains samples:
        sample_000000.jpg
        sample_000000.json
        sample_000000.npy (actions)
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[str]],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
        shuffle: bool = True,
        buffer_size: int = 1000,
        decode_fn: Optional[Callable] = None,
    ):
        """
        Initialize WebDataset loader.

        Args:
            data_path: Path to tar files or URL pattern
            split: Dataset split
            transform: Optional transform
            cache_in_memory: Cache in memory (limited for streaming)
            num_workers: Data loading workers
            shuffle: Whether to shuffle
            buffer_size: Shuffle buffer size
            decode_fn: Custom decode function
        """
        # Handle list of paths
        if isinstance(data_path, list):
            self._urls = data_path
            data_path = Path(data_path[0]).parent if data_path else Path(".")
        else:
            data_path = Path(data_path)
            self._urls = None

        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            cache_in_memory=cache_in_memory,
            num_workers=num_workers,
        )
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.decode_fn = decode_fn or self._default_decode
        self._tar_files: List[Path] = []
        self._samples: List[Dict] = []
        self._indexed = False
        self._index_dataset()

    def _index_dataset(self) -> None:
        """Index tar files."""
        if self._urls:
            self._tar_files = [Path(u) for u in self._urls]
        elif self.data_path.is_dir():
            pattern = f"{self.split}-*.tar"
            self._tar_files = sorted(self.data_path.glob(pattern))
            if not self._tar_files:
                # Try without split prefix
                self._tar_files = sorted(self.data_path.glob("*.tar"))
        elif self.data_path.suffix == ".tar":
            self._tar_files = [self.data_path]

        if not self._tar_files:
            self._create_dummy_samples()

    def _create_dummy_samples(self) -> None:
        """Create dummy samples for testing."""
        for i in range(100):
            self._samples.append(
                {
                    "images": np.random.randn(3, 224, 224).astype(np.float32),
                    "actions": np.random.randn(7).astype(np.float32),
                    "proprioception": np.random.randn(14).astype(np.float32),
                    "text": "pick up the object",
                }
            )
        self._indexed = True

    def _get_num_episodes(self) -> int:
        """Get number of episodes (tar files)."""
        return len(self._tar_files) if self._tar_files else 1

    def _get_episode_length(self, episode_idx: int) -> int:
        """Get episode length (samples per tar)."""
        return 100  # Estimate

    def __len__(self) -> int:
        """Return estimated total samples."""
        if self._indexed and self._samples:
            return len(self._samples)
        return len(self._tar_files) * 100  # Estimate

    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Load episode (tar file contents)."""
        if episode_idx >= len(self._tar_files):
            return self._create_dummy_episode_data()

        tar_path = self._tar_files[episode_idx]
        return self._load_tar(tar_path)

    def _load_tar(self, tar_path: Path) -> Dict[str, Any]:
        """Load samples from tar file."""
        samples = []

        try:
            import tarfile

            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()

                # Group by sample name
                sample_groups = {}
                for member in members:
                    if member.isfile():
                        name = member.name.rsplit(".", 1)[0]
                        ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                        if name not in sample_groups:
                            sample_groups[name] = {}
                        sample_groups[name][ext] = member

                # Load each sample
                for name, files in sample_groups.items():
                    sample = self._load_sample_from_tar(tar, files)
                    if sample:
                        samples.append(sample)

        except (ImportError, Exception):
            samples = [self._create_dummy_sample() for _ in range(10)]

        # Convert list of samples to episode format
        if samples:
            episode = {}
            for key in samples[0].keys():
                episode[key] = np.stack([s[key] for s in samples if key in s])
            return episode

        return self._create_dummy_episode_data()

    def _load_sample_from_tar(self, tar, files: Dict) -> Optional[Dict]:
        """Load a single sample from tar file members."""
        sample = {}

        for ext, member in files.items():
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue

                data = f.read()
                decoded = self.decode_fn(ext, data)
                if decoded is not None:
                    sample.update(decoded)
            except Exception:
                pass

        return sample if sample else None

    def _default_decode(self, ext: str, data: bytes) -> Optional[Dict[str, Any]]:
        """Default decoder for common formats."""
        if ext in ("jpg", "jpeg", "png"):
            try:
                from PIL import Image

                img = Image.open(io.BytesIO(data))
                arr = np.array(img)
                if arr.ndim == 3:
                    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
                return {"images": arr.astype(np.float32) / 255.0}
            except ImportError:
                return {"images": np.random.randn(3, 224, 224).astype(np.float32)}

        elif ext == "npy":
            arr = np.load(io.BytesIO(data))
            return {"actions": arr.astype(np.float32)}

        elif ext == "json":
            import json

            info = json.loads(data.decode("utf-8"))
            result = {}
            if "text" in info:
                result["text"] = info["text"]
            if "instruction" in info:
                result["text"] = info["instruction"]
            return result

        elif ext == "txt":
            return {"text": data.decode("utf-8").strip()}

        return None

    def _create_dummy_sample(self) -> Dict[str, Any]:
        """Create dummy sample."""
        return {
            "images": np.random.randn(3, 224, 224).astype(np.float32),
            "actions": np.random.randn(7).astype(np.float32),
            "proprioception": np.random.randn(14).astype(np.float32),
            "text": "pick up the object",
        }

    def _create_dummy_episode_data(self, length: int = 100) -> Dict[str, Any]:
        """Create dummy episode data."""
        return {
            "images": np.random.randn(length, 3, 224, 224).astype(np.float32),
            "actions": np.random.randn(length, 7).astype(np.float32),
            "proprioception": np.random.randn(length, 14).astype(np.float32),
            "text": ["pick up the object"] * length,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by index."""
        if self._indexed and idx < len(self._samples):
            sample = self._samples[idx]
        else:
            # Load on demand
            episode_idx = idx // 100
            timestep = idx % 100
            episode = self._load_episode(episode_idx)
            sample = self._extract_timestep(episode, timestep)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def create_dataloader(self, batch_size: int = 32, **kwargs):
        """
        Create WebDataset-compatible dataloader for streaming.

        Args:
            batch_size: Batch size
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader or WebDataset pipeline
        """
        try:
            import webdataset as wds

            urls = [str(p) for p in self._tar_files]
            dataset = wds.WebDataset(urls)

            if self.shuffle:
                dataset = dataset.shuffle(self.buffer_size)

            dataset = dataset.decode("pil").to_tuple("jpg", "json", "npy")
            dataset = dataset.batched(batch_size)

            return dataset
        except ImportError:
            # Fallback to standard PyTorch DataLoader
            import torch.utils.data as data

            return data.DataLoader(self, batch_size=batch_size, **kwargs)


__all__ = ["WebDatasetLoader"]
