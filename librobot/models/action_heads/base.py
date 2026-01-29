"""Abstract base class for action prediction heads."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class AbstractActionHead(ABC, nn.Module):
    """
    Abstract base class for action prediction heads.
    
    Action heads take embeddings from VLMs and predict robot actions.
    """
    
    def __init__(self, input_dim: int, action_dim: int):
        """
        Initialize action head.
        
        Args:
            input_dim: Dimension of input embeddings
            action_dim: Dimension of action space
        """
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict actions.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - Other head-specific outputs (e.g., 'logits', 'distribution')
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss for action prediction.
        
        Args:
            predictions: Model predictions from forward()
            targets: Target actions [batch_size, action_dim]
            **kwargs: Additional loss computation arguments
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample actions from the predicted distribution.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, input_dim]
            temperature: Sampling temperature
            **kwargs: Additional sampling arguments
            
        Returns:
            Sampled actions [batch_size, action_dim]
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get head configuration.
        
        Returns:
            Dictionary containing configuration
        """
        return {
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'type': self.__class__.__name__,
        }
    
    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
