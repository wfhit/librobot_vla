"""Sinusoidal positional encoding."""

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    Generates fixed positional encodings using sine and cosine functions
    of different frequencies. No learned parameters.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Dimension of the model
        max_len: Maximum sequence length
        dropout: Dropout rate
        scale: Whether to scale positional encoding
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
        scale: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not trained)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            offset: Positional offset (for autoregressive decoding)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        if seq_len + offset > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} with offset {offset} exceeds maximum length {self.max_len}"
            )

        pe = self.pe[:, offset : offset + seq_len, :]

        if self.scale:
            x = x * math.sqrt(self.d_model)

        x = x + pe
        x = self.dropout(x)

        return x

    def get_encoding(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Get positional encoding without adding to input.

        Args:
            seq_len: Sequence length
            offset: Positional offset

        Returns:
            Positional encoding [1, seq_len, d_model]
        """
        if seq_len + offset > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} with offset {offset} exceeds maximum length {self.max_len}"
            )

        return self.pe[:, offset : offset + seq_len, :]

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, max_len={self.max_len}, scale={self.scale}"
