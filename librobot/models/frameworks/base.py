"""Abstract base class for Vision-Language-Action (VLA) frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch
import torch.nn as nn


class AbstractVLA(ABC, nn.Module):
    """
    Abstract base class for Vision-Language-Action frameworks.

    VLA frameworks combine VLMs with action prediction heads and additional
    components to create end-to-end systems for robot learning.
    """

    def __init__(self):
        """Initialize VLA framework."""
        super().__init__()

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the VLA framework.

        Args:
            images: Input images [batch_size, channels, height, width]
            text: Optional text instructions
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            actions: Optional action targets for training
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions
                - 'loss': Training loss (if actions provided)
                - Other framework-specific outputs
        """
        pass

    @abstractmethod
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict actions for inference.

        Args:
            images: Input images [batch_size, channels, height, width]
            text: Optional text instructions
            proprioception: Optional proprioceptive state
            **kwargs: Additional arguments

        Returns:
            Predicted actions [batch_size, action_dim]
        """
        pass

    @abstractmethod
    def compute_loss(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for training.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional loss computation arguments

        Returns:
            Dictionary containing:
                - 'total_loss': Total loss
                - Other component losses
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        pass

    def freeze_backbone(self) -> None:
        """Freeze VLM backbone parameters."""
        if hasattr(self, "vlm"):
            for param in self.vlm.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze VLM backbone parameters."""
        if hasattr(self, "vlm"):
            for param in self.vlm.parameters():
                param.requires_grad = True

    def freeze_head(self) -> None:
        """Freeze action head parameters."""
        if hasattr(self, "action_head"):
            for param in self.action_head.parameters():
                param.requires_grad = False

    def unfreeze_head(self) -> None:
        """Unfreeze action head parameters."""
        if hasattr(self, "action_head"):
            for param in self.action_head.parameters():
                param.requires_grad = True

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters.

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
