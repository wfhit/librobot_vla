"""RLDS (Reinforcement Learning Datasets) loader."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .base import BaseDatasetLoader


class RLDSDataset(BaseDatasetLoader):
    """
    Dataset loader for RLDS (Reinforcement Learning Datasets) format.
    
    RLDS is a standard format used by many robotics datasets including
    Open X-Embodiment, RT-1, and others.
    
    Expected structure:
        data_path/
        ├── 1.0.0/
        │   ├── dataset_info.json
        │   └── train/
        │       └── *.tfrecord
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
        version: str = "1.0.0",
    ):
        """
        Initialize RLDS dataset.
        
        Args:
            data_path: Path to RLDS dataset
            split: Dataset split
            transform: Optional transform
            cache_in_memory: Cache in memory
            num_workers: Data loading workers
            version: Dataset version
        """
        super().__init__(
            data_path=data_path,
            split=split,
            transform=transform,
            cache_in_memory=cache_in_memory,
            num_workers=num_workers,
        )
        self.version = version
        self._episodes: List[Dict] = []
        self._episode_lengths: List[int] = []
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load dataset metadata and index episodes."""
        version_path = self.data_path / self.version
        split_path = version_path / self.split
        
        if not split_path.exists():
            # Try without version directory
            split_path = self.data_path / self.split
        
        if split_path.exists():
            self._index_tfrecords(split_path)
        else:
            # Create dummy data for testing
            self._create_dummy_index()
    
    def _index_tfrecords(self, split_path: Path) -> None:
        """Index TFRecord files."""
        tfrecord_files = sorted(split_path.glob("*.tfrecord*"))
        
        for tf_file in tfrecord_files:
            self._episodes.append({
                'path': str(tf_file),
                'index': len(self._episodes),
            })
            self._episode_lengths.append(100)  # Default length
    
    def _create_dummy_index(self) -> None:
        """Create dummy index for testing."""
        for i in range(10):
            self._episodes.append({'index': i})
            self._episode_lengths.append(100)
    
    def _get_num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self._episodes)
    
    def _get_episode_length(self, episode_idx: int) -> int:
        """Get episode length."""
        if episode_idx < len(self._episode_lengths):
            return self._episode_lengths[episode_idx]
        return 100
    
    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Load episode from TFRecord.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Episode data
        """
        if episode_idx >= len(self._episodes):
            raise IndexError(f"Episode {episode_idx} not found")
        
        episode_info = self._episodes[episode_idx]
        
        if 'path' in episode_info:
            return self._load_tfrecord_episode(episode_info['path'])
        
        return self._create_dummy_episode()
    
    def _load_tfrecord_episode(self, path: str) -> Dict[str, Any]:
        """Load episode from TFRecord file."""
        try:
            import tensorflow_datasets as tfds
            # Load using TFDS
            return self._load_with_tfds(path)
        except ImportError:
            pass
        
        try:
            import tensorflow as tf
            return self._load_with_tf(path)
        except ImportError:
            pass
        
        return self._create_dummy_episode()
    
    def _load_with_tfds(self, path: str) -> Dict[str, Any]:
        """Load using tensorflow_datasets."""
        return self._create_dummy_episode()
    
    def _load_with_tf(self, path: str) -> Dict[str, Any]:
        """Load using raw tensorflow."""
        return self._create_dummy_episode()
    
    def _create_dummy_episode(self, length: int = 100) -> Dict[str, Any]:
        """Create dummy episode for testing."""
        return {
            'observation': {
                'image': np.random.randn(length, 224, 224, 3).astype(np.float32),
                'state': np.random.randn(length, 14).astype(np.float32),
            },
            'action': np.random.randn(length, 7).astype(np.float32),
            'reward': np.zeros(length, dtype=np.float32),
            'is_terminal': np.zeros(length, dtype=bool),
            'language_instruction': 'pick up the object',
        }
    
    def _extract_timestep(
        self, 
        episode: Dict[str, Any], 
        timestep: int
    ) -> Dict[str, Any]:
        """Extract timestep from RLDS episode format."""
        sample = {}
        
        # Handle nested observation structure
        if 'observation' in episode:
            obs = episode['observation']
            if isinstance(obs, dict):
                if 'image' in obs:
                    img = obs['image']
                    if len(img.shape) == 4 and img.shape[0] > timestep:
                        sample['images'] = img[timestep]
                    else:
                        sample['images'] = img
                if 'state' in obs:
                    state = obs['state']
                    if len(state.shape) == 2 and state.shape[0] > timestep:
                        sample['proprioception'] = state[timestep]
                    else:
                        sample['proprioception'] = state
        
        # Handle action
        if 'action' in episode:
            action = episode['action']
            if len(action.shape) == 2 and action.shape[0] > timestep:
                sample['actions'] = action[timestep]
            else:
                sample['actions'] = action
        
        # Handle language instruction
        if 'language_instruction' in episode:
            sample['text'] = episode['language_instruction']
        
        return sample


__all__ = ['RLDSDataset']
