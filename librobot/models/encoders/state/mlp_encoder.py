"""MLP encoder for robot state."""

from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import AbstractEncoder


class MLPStateEncoder(AbstractEncoder):
    """
    Multi-Layer Perceptron encoder for robot state.

    Encodes proprioceptive state (joint positions, velocities, etc.)
    using a simple feedforward network.

    Args:
        input_dim: Input state dimension
        output_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        dropout: Dropout rate
        norm: Normalization type ('layer', 'batch', or None)
        residual: Whether to use residual connections
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        norm: Optional[str] = "layer",
        residual: bool = False,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_name = activation
        self.dropout_rate = dropout
        self.norm_type = norm
        self.use_residual = residual

        # Build MLP layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Normalization (except last layer)
            if i < len(dims) - 2 and norm is not None:
                if norm == "layer":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                elif norm == "batch":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Activation (except last layer)
            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "silu":
                    layers.append(nn.SiLU(inplace=True))
                else:
                    raise ValueError(f"Unknown activation: {activation}")

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Residual projection if needed
        if residual and input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode state to embeddings.

        Args:
            inputs: State tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            **kwargs: Additional arguments (ignored)

        Returns:
            Encoded embeddings [batch_size, output_dim] or [batch_size, seq_len, output_dim]
        """

        # Handle both 2D and 3D inputs
        if inputs.dim() == 3:
            batch_size, seq_len, _ = inputs.shape
            inputs = inputs.reshape(batch_size * seq_len, -1)
            flatten = True
        else:
            flatten = False

        # Forward through MLP
        output = self.mlp(inputs)

        # Add residual if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(inputs)
            else:
                residual = inputs
            output = output + residual

        # Reshape back if needed
        if flatten:
            output = output.reshape(batch_size, seq_len, -1)

        return output

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape."""
        if len(input_shape) == 2:
            return (input_shape[0], self.output_dim)
        elif len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    @property
    def config(self) -> dict[str, Any]:
        """Get encoder configuration."""
        return {
            "type": "MLPStateEncoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "norm": self.norm_type,
            "residual": self.use_residual,
        }
