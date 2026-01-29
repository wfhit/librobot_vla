"""Base dataset loader class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np


class BaseDatasetLoader(ABC):
    """
    Base class for dataset loaders.
    
    Provides common functionality for loading robot learning datasets
    from various formats.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        cache_in_memory: bool = False,
        num_workers: int = 0,
    ):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to dataset
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply
            cache_in_memory: Whether to cache data in memory
            num_workers: Number of workers for data loading
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.num_workers = num_workers
        self._cache: Dict[int, Any] = {}
        self._metadata: Optional[Dict[str, Any]] = None
        
    @abstractmethod
    def _load_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Load a single episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Episode data dictionary
        """
        pass
    
    @abstractmethod
    def _get_episode_length(self, episode_idx: int) -> int:
        """
        Get length of an episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Number of timesteps in episode
        """
        pass
    
    @abstractmethod
    def _get_num_episodes(self) -> int:
        """
        Get total number of episodes.
        
        Returns:
            Number of episodes
        """
        pass
    
    def __len__(self) -> int:
        """Return total number of samples across all episodes."""
        return sum(
            self._get_episode_length(i) 
            for i in range(self._get_num_episodes())
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample by global index.
        
        Args:
            idx: Global sample index
            
        Returns:
            Sample dictionary
        """
        episode_idx, timestep = self._global_to_local_index(idx)
        
        if self.cache_in_memory and episode_idx in self._cache:
            episode = self._cache[episode_idx]
        else:
            episode = self._load_episode(episode_idx)
            if self.cache_in_memory:
                self._cache[episode_idx] = episode
        
        sample = self._extract_timestep(episode, timestep)
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample
    
    def _global_to_local_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global index to (episode_idx, timestep).
        
        Args:
            global_idx: Global sample index
            
        Returns:
            Tuple of (episode_idx, timestep)
        """
        cumsum = 0
        for episode_idx in range(self._get_num_episodes()):
            episode_len = self._get_episode_length(episode_idx)
            if cumsum + episode_len > global_idx:
                return episode_idx, global_idx - cumsum
            cumsum += episode_len
        raise IndexError(f"Index {global_idx} out of range")
    
    def _extract_timestep(
        self, 
        episode: Dict[str, Any], 
        timestep: int
    ) -> Dict[str, Any]:
        """
        Extract a single timestep from episode data.
        
        Args:
            episode: Episode data
            timestep: Timestep index
            
        Returns:
            Single timestep data
        """
        sample = {}
        for key, value in episode.items():
            if isinstance(value, np.ndarray) and len(value.shape) > 0:
                if value.shape[0] > timestep:
                    sample[key] = value[timestep]
                else:
                    sample[key] = value
            else:
                sample[key] = value
        return sample
    
    def get_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get full episode data.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Full episode data
        """
        return self._load_episode(episode_idx)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Metadata dictionary
        """
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load dataset metadata.
        
        Returns:
            Metadata dictionary
        """
        return {
            'num_episodes': self._get_num_episodes(),
            'total_samples': len(self),
            'split': self.split,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics for normalization.
        
        Returns:
            Statistics dictionary with means and stds
        """
        all_actions = []
        all_states = []
        
        for episode_idx in range(min(100, self._get_num_episodes())):
            episode = self._load_episode(episode_idx)
            if 'actions' in episode:
                all_actions.append(episode['actions'])
            if 'proprioception' in episode:
                all_states.append(episode['proprioception'])
        
        stats = {}
        if all_actions:
            actions = np.concatenate(all_actions, axis=0)
            stats['action_mean'] = np.mean(actions, axis=0)
            stats['action_std'] = np.std(actions, axis=0) + 1e-6
        if all_states:
            states = np.concatenate(all_states, axis=0)
            stats['state_mean'] = np.mean(states, axis=0)
            stats['state_std'] = np.std(states, axis=0) + 1e-6
            
        return stats
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all samples."""
        for i in range(len(self)):
            yield self[i]


__all__ = ['BaseDatasetLoader']
