"""Abstract base class for robotics datasets."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class AbstractDataset(ABC, Dataset):
    """
    Abstract base class for robotics datasets.
    
    All dataset implementations must inherit from this class and implement
    the required abstract methods. This class follows PyTorch Dataset conventions
    and can be used with PyTorch DataLoader.
    
    The dataset should provide:
    - Observation data (images, proprioception, etc.)
    - Action data (joint positions, velocities, gripper states, etc.)
    - Metadata (episode boundaries, task descriptions, etc.)
    
    See docs/design/data_pipeline.md for detailed design documentation.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to dataset directory
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply to samples
            **kwargs: Additional dataset-specific arguments
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Number of samples in dataset
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'observation': Dict with observation data
                    - 'image': Image tensor [C, H, W] or dict of images
                    - 'state': Proprioceptive state [state_dim]
                    - Other observation modalities
                - 'action': Action tensor [action_dim]
                - 'reward': Reward scalar (optional)
                - 'done': Episode termination flag (optional)
                - 'task': Task description string (optional)
                - 'metadata': Additional metadata dict (optional)
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Get dataset statistics for normalization.
        
        Returns:
            Dictionary containing statistics:
                - 'action': {'mean': tensor, 'std': tensor, 'min': tensor, 'max': tensor}
                - 'state': {'mean': tensor, 'std': tensor, 'min': tensor, 'max': tensor}
                - Other modalities as needed
        """
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Dict[str, Any]:
        """
        Get observation space specification.
        
        Returns:
            Dictionary describing observation space:
                - 'image': {'shape': tuple, 'dtype': str, 'channels': int}
                - 'state': {'shape': tuple, 'dtype': str}
                - Other observation modalities
        """
        pass
    
    @abstractmethod
    def get_action_space(self) -> Dict[str, Any]:
        """
        Get action space specification.
        
        Returns:
            Dictionary describing action space:
                - 'shape': tuple
                - 'dtype': str
                - 'low': tensor (optional)
                - 'high': tensor (optional)
        """
        pass
    
    def get_episode_boundaries(self) -> Optional[List[Tuple[int, int]]]:
        """
        Get episode boundaries.
        
        Returns:
            List of (start_idx, end_idx) tuples for each episode,
            or None if episodes are not tracked
        """
        return None
    
    def get_task_descriptions(self) -> Optional[List[str]]:
        """
        Get task descriptions for all samples.
        
        Returns:
            List of task description strings, or None if not available
        """
        return None
    
    def _apply_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transform to sample.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Transformed sample
        """
        if self.transform is not None:
            return self.transform(sample)
        return sample
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch of samples.
        
        This method can be overridden for custom batching logic.
        Default implementation uses torch.utils.data.default_collate behavior.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched dictionary
        """
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


class EpisodicDataset(AbstractDataset):
    """
    Base class for episodic datasets.
    
    This class extends AbstractDataset with episode-aware functionality
    for trajectory-based learning.
    """
    
    @abstractmethod
    def get_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get a full episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Dictionary containing full episode data:
                - 'observations': Dict of observation sequences
                - 'actions': Action sequence [T, action_dim]
                - 'rewards': Reward sequence [T] (optional)
                - 'task': Task description string (optional)
        """
        pass
    
    @abstractmethod
    def get_num_episodes(self) -> int:
        """
        Get number of episodes in dataset.
        
        Returns:
            Number of episodes
        """
        pass


__all__ = [
    'AbstractDataset',
    'EpisodicDataset',
]
