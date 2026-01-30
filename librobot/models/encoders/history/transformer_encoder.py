"""Transformer encoder for history."""

from typing import Any, Optional

import torch
import torch.nn as nn

from ...components.attention import StandardAttention
from ...components.normalization import LayerNorm
from ...components.positional import SinusoidalPositionalEncoding
from ..base import AbstractEncoder


class TransformerHistoryEncoder(AbstractEncoder):
    """
    Transformer encoder for history sequences.

    Uses self-attention to model temporal dependencies in robot trajectories.
    Better at capturing long-range dependencies than RNNs.

    Args:
        input_dim: Input dimension per timestep
        output_dim: Output embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ffn_dim: Feedforward network dimension
        dropout: Dropout rate
        activation: Activation function
        pooling: Output pooling ('mean', 'cls', 'last')
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pooling: str = 'mean',
        max_seq_len: int = 512,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.activation_name = activation
        self.pooling_method = pooling
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)

        # CLS token for pooling
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(
            output_dim,
            max_len=max_seq_len + 1,  # +1 for CLS token
            dropout=dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=output_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])

        self.norm = LayerNorm(output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode history sequence.

        Args:
            inputs: History tensor [batch_size, seq_len, input_dim]
            mask: Optional mask [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, output_dim]
        """
        batch_size = inputs.size(0)

        # Project input
        x = self.input_proj(inputs)

        # Add CLS token if using CLS pooling
        if self.pooling_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        # Pool output
        if self.pooling_method == 'cls':
            output = x[:, 0]

        elif self.pooling_method == 'mean':
            if mask is not None:
                output = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                output = x.mean(dim=1)

        elif self.pooling_method == 'last':
            if mask is not None:
                lengths = mask.sum(dim=1).long() - 1
                output = x[torch.arange(batch_size), lengths]
            else:
                output = x[:, -1]

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
            'type': 'TransformerHistoryEncoder',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'pooling': self.pooling_method,
            'max_seq_len': self.max_seq_len,
        }


class TransformerLayer(nn.Module):
    """Single transformer layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        self.self_attn = StandardAttention(
            dim=d_model,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x
