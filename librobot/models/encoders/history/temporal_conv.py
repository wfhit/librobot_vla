"""Temporal convolutional encoder for history."""

from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import AbstractEncoder


class TemporalConvEncoder(AbstractEncoder):
    """
    Temporal convolutional encoder for history.

    Uses 1D convolutions to capture temporal patterns in robot trajectories.
    Efficient alternative to transformers for fixed-length sequences.

    Args:
        input_dim: Input dimension per timestep
        output_dim: Output embedding dimension
        channels: List of channel dimensions for conv layers
        kernel_sizes: List of kernel sizes for conv layers
        strides: List of strides for conv layers
        activation: Activation function
        dropout: Dropout rate
        pooling: Final pooling method ('mean', 'max', 'adaptive')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        channels: list[int] = [64, 128, 256],
        kernel_sizes: list[int] = [3, 3, 3],
        strides: list[int] = [1, 2, 2],
        activation: str = "relu",
        dropout: float = 0.0,
        pooling: str = "adaptive",
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.activation_name = activation
        self.dropout_rate = dropout
        self.pooling_method = pooling

        # Ensure lists have same length
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have same length")

        # Build convolutional layers
        layers = []
        in_channels = [input_dim] + channels[:-1]

        for i, (in_ch, out_ch, kernel, stride) in enumerate(
            zip(in_channels, channels, kernel_sizes, strides)
        ):
            # Conv layer
            padding = (kernel - 1) // 2
            layers.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

            # Batch normalization
            layers.append(nn.BatchNorm1d(out_ch))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU(inplace=True))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)

        # Pooling
        if pooling == "adaptive":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == "mean":
            self.pool = None
        elif pooling == "max":
            self.pool = None

        # Output projection
        self.output_proj = nn.Linear(channels[-1], output_dim)

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Encode history with temporal convolutions.

        Args:
            inputs: History tensor [batch_size, seq_len, input_dim]
            mask: Optional mask [batch_size, seq_len] (not fully supported)
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, output_dim]
        """
        # Transpose for conv1d: [batch, channels, length]
        x = inputs.transpose(1, 2)

        # Apply convolutions
        x = self.conv_layers(x)

        # Pool
        if self.pooling_method == "adaptive":
            x = self.pool(x).squeeze(-1)
        elif self.pooling_method == "mean":
            x = x.mean(dim=-1)
        elif self.pooling_method == "max":
            x = x.max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Project to output dimension
        output = self.output_proj(x)

        return output

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape."""
        if len(input_shape) == 3:
            return (input_shape[0], self.output_dim)
        else:
            raise ValueError(f"Expected 3D input, got shape: {input_shape}")

    @property
    def config(self) -> dict[str, Any]:
        """Get encoder configuration."""
        return {
            "type": "TemporalConvEncoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "channels": self.channels,
            "kernel_sizes": self.kernel_sizes,
            "strides": self.strides,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "pooling": self.pooling_method,
        }
