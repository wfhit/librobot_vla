"""Abstract base class for Vision-Language Models (VLMs)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class AbstractVLM(ABC, nn.Module):
    """
    Abstract base class for Vision-Language Models.
    
    All VLM implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self):
        """Initialize VLM."""
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLM.
        
        Args:
            images: Input images tensor [batch_size, channels, height, width]
            text: Optional text input (string or list of strings)
            attention_mask: Optional attention mask for text
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary containing:
                - 'embeddings': Vision-language embeddings [batch_size, seq_len, hidden_dim]
                - Other model-specific outputs
        """
        pass
    
    @abstractmethod
    def encode_image(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            **kwargs: Additional arguments
            
        Returns:
            Image embeddings [batch_size, num_patches, hidden_dim]
        """
        pass
    
    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Input text (string or list of strings)
            **kwargs: Additional arguments
            
        Returns:
            Text embeddings [batch_size, seq_len, hidden_dim]
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        pass
    
    def freeze(self) -> None:
        """Freeze all parameters in the VLM."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters in the VLM."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters in the model.
        
        Args:
            trainable_only: If True, only counts trainable parameters
            
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def load_pretrained(self, path: str, **kwargs) -> None:
        """
        Load pretrained weights.
        
        Args:
            path: Path to pretrained weights
            **kwargs: Additional loading arguments
            
        Note:
            Uses weights_only=False to support loading complex state dicts.
            Only load checkpoints from trusted sources.
        """
        state_dict = torch.load(path, weights_only=False, **kwargs)
        self.load_state_dict(state_dict, strict=False)
    
    def save_pretrained(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        torch.save(self.state_dict(), path)
