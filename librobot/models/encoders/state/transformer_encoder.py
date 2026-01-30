"""Transformer encoder for robot state."""

from typing import Any, Optional

import torch
import torch.nn as nn

from ...components.attention import StandardAttention
from ...components.normalization import LayerNorm
from ...components.positional import SinusoidalPositionalEncoding
from ..base import AbstractEncoder


class TransformerStateEncoder(AbstractEncoder):
    """
    Transformer encoder for robot state sequences.

    Processes sequential state information using self-attention,
    useful for encoding state history or multi-agent states.

    Args:
        input_dim: Input state dimension
        output_dim: Output embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        ffn_dim: Feedforward network dimension
        dropout: Dropout rate
        activation: Activation function
        use_pos_encoding: Whether to use positional encoding
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
        use_pos_encoding: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout
        self.activation_name = activation
        self.use_pos_encoding = use_pos_encoding
        self.max_seq_len = max_seq_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Positional encoding
        if use_pos_encoding:
            self.pos_encoding = SinusoidalPositionalEncoding(
                output_dim,
                max_len=max_seq_len,
                dropout=dropout,
            )
        else:
            self.pos_encoding = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
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
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode state sequence.

        Args:
            inputs: State tensor [batch_size, seq_len, input_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, seq_len, output_dim]
        """
        # Project input
        x = self.input_proj(inputs)

        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization
        x = self.norm(x)

        return x

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape."""
        if len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            raise ValueError(f"Expected 3D input, got shape: {input_shape}")

    @property
    def config(self) -> dict[str, Any]:
        """Get encoder configuration."""
        return {
            'type': 'TransformerStateEncoder',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'ffn_dim': self.ffn_dim,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'use_pos_encoding': self.use_pos_encoding,
            'max_seq_len': self.max_seq_len,
        }


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()

        # Self-attention
        self.self_attn = StandardAttention(
            dim=d_model,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-norm architecture."""
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x), attention_mask)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x
