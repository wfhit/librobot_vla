"""MLP encoder for history with temporal pooling."""

from typing import Any, Optional

import torch
import torch.nn as nn

from ..base import AbstractEncoder


class MLPHistoryEncoder(AbstractEncoder):
    """
    MLP encoder for history with temporal pooling.

    Processes each timestep independently with MLP, then pools across time.
    Simple baseline for encoding temporal history.

    Args:
        input_dim: Input dimension per timestep
        output_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions
        pooling: Pooling method ('mean', 'max', 'last', 'attention')
        activation: Activation function
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = [256, 256],
        pooling: str = "mean",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.pooling_method = pooling
        self.activation_name = activation
        self.dropout_rate = dropout

        # Build MLP
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

        # Attention pooling
        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(output_dim, 1),
                nn.Softmax(dim=1),
            )

    def forward(
        self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """
        Encode history.

        Args:
            inputs: History tensor [batch_size, seq_len, input_dim]
            mask: Optional mask [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, output_dim]
        """
        batch_size, seq_len, _ = inputs.shape

        # Process each timestep
        x = inputs.reshape(batch_size * seq_len, -1)
        x = self.mlp(x)
        x = x.reshape(batch_size, seq_len, -1)

        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # Pool across time
        if self.pooling_method == "mean":
            if mask is not None:
                output = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                output = x.mean(dim=1)

        elif self.pooling_method == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            output = x.max(dim=1)[0]

        elif self.pooling_method == "last":
            if mask is not None:
                # Get last valid timestep for each batch
                lengths = mask.sum(dim=1).long() - 1
                output = x[torch.arange(batch_size), lengths]
            else:
                output = x[:, -1]

        elif self.pooling_method == "attention":
            # Attention-based pooling
            attn_weights = self.attn_pool(x)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1).bool(), 0)
            output = (x * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

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
            "type": "MLPHistoryEncoder",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "pooling": self.pooling_method,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
        }
