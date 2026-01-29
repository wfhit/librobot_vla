"""Base encoder interface."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Base class for encoders (state, history, etc.)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs,
    ):
        """Initialize encoder.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional encoder-specific arguments
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor
            mask: Optional mask
            **kwargs: Additional arguments

        Returns:
            Encoded features
        """
        pass


class MLPEncoder(BaseEncoder):
    """Simple MLP encoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize MLP encoder.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            dropout: Dropout probability
        """
        super().__init__(input_dim, output_dim)

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            mask: Optional mask (unused for MLP)

        Returns:
            Encoded features [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """
        return self.encoder(x)
