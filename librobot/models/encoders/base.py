"""Abstract base class for encoders."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class AbstractEncoder(ABC, nn.Module):
    """
    Abstract base class for encoders.

    Encoders transform raw inputs (images, text, proprioception, etc.)
    into embeddings that can be used by the VLA framework.
    """

    def __init__(self, output_dim: int):
        """
        Initialize encoder.

        Args:
            output_dim: Dimension of output embeddings
        """
        super().__init__()
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> torch.Tensor:
        """
        Encode inputs to embeddings.

        Args:
            inputs: Input data (format depends on encoder type)
            **kwargs: Additional encoding arguments

        Returns:
            Encoded embeddings [batch_size, seq_len, output_dim]
        """
        pass

    @abstractmethod
    def get_output_shape(self, input_shape: tuple) -> tuple:
        """
        Get output shape for given input shape.

        Args:
            input_shape: Shape of input tensor

        Returns:
            Shape of output tensor
        """
        pass

    @property
    @abstractmethod
    def config(self) -> dict[str, Any]:
        """
        Get encoder configuration.

        Returns:
            Dictionary containing configuration
        """
        pass

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
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
