"""Base VLM (Vision-Language Model) interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class BaseVLM(nn.Module, ABC):
    """Base class for Vision-Language Models.

    All VLM implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        freeze: bool = True,
        **kwargs,
    ):
        """Initialize VLM.

        Args:
            model_name: Name/path of the pretrained model
            hidden_dim: Hidden dimension of the model
            freeze: Whether to freeze the model parameters
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze = freeze

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the VLM.

        Args:
            images: Input images [batch_size, num_images, C, H, W]
            text: Optional text prompts
            attention_mask: Optional attention mask
            return_dict: Whether to return dict or tuple

        Returns:
            Dictionary containing:
                - hidden_states: [batch_size, seq_len, hidden_dim]
                - attention_mask: [batch_size, seq_len]
                - (optional) other model-specific outputs
        """
        pass

    @abstractmethod
    def get_image_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Extract image features.

        Args:
            images: Input images [batch_size, num_images, C, H, W]

        Returns:
            Image features [batch_size, num_patches, hidden_dim]
        """
        pass

    @abstractmethod
    def get_text_features(
        self,
        text: List[str],
    ) -> torch.Tensor:
        """Extract text features.

        Args:
            text: List of text prompts

        Returns:
            Text features [batch_size, seq_len, hidden_dim]
        """
        pass

    def freeze_parameters(self) -> None:
        """Freeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
