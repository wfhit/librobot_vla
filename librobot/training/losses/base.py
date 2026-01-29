"""Abstract base class for loss functions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class AbstractLoss(ABC, nn.Module):
    """
    Abstract base class for loss functions.
    
    All loss implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize loss function.
        
        Args:
            weight: Weight for this loss component
        """
        super().__init__()
        self.weight = weight
    
    @abstractmethod
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional loss computation arguments
            
        Returns:
            Loss tensor (scalar)
        """
        pass
    
    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Weighted loss tensor
        """
        loss = self.forward(predictions, targets, **kwargs)
        return self.weight * loss
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get loss configuration.
        
        Returns:
            Dictionary containing configuration
        """
        return {
            'type': self.__class__.__name__,
            'weight': self.weight,
        }
