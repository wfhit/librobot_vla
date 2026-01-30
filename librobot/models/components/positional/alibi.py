"""ALiBi (Attention with Linear Biases) positional bias."""

import math
from typing import Optional

import torch
import torch.nn as nn


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) from "Train Short, Test Long".

    Instead of adding positional encodings to embeddings, ALiBi adds a bias
    to attention scores based on distance between positions. This allows
    extrapolation to longer sequences than seen during training.

    Args:
        num_heads: Number of attention heads
        max_len: Maximum sequence length for precomputation
        slopes_init: Method for initializing slopes ('linear' or 'geometric')
    """

    def __init__(
        self,
        num_heads: int,
        max_len: int = 2048,
        slopes_init: str = "geometric",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.slopes_init = slopes_init

        # Compute slopes for each head
        slopes = self._get_slopes(num_heads, slopes_init)
        self.register_buffer("slopes", slopes)

        # Precompute bias matrix for max_len
        self._precompute_bias(max_len)

    def _get_slopes(self, num_heads: int, method: str = "geometric") -> torch.Tensor:
        """
        Compute slopes for ALiBi.

        Args:
            num_heads: Number of attention heads
            method: 'geometric' (original paper) or 'linear'

        Returns:
            Slopes tensor [num_heads]
        """
        if method == "geometric":
            # Original ALiBi method: geometric sequence
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * (ratio**i) for i in range(n)]

            if math.log2(num_heads).is_integer():
                slopes = get_slopes_power_of_2(num_heads)
            else:
                # If not power of 2, use closest power of 2 and interpolate
                closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
                slopes = get_slopes_power_of_2(closest_power_of_2)
                extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
                slopes = slopes + extra_slopes[0::2][: num_heads - closest_power_of_2]

            slopes = torch.tensor(slopes, dtype=torch.float32)

        elif method == "linear":
            # Linear slopes
            slopes = torch.linspace(0.1, 1.0, num_heads)

        else:
            raise ValueError(f"Unknown slopes_init method: {method}")

        return slopes

    def _precompute_bias(self, max_len: int):
        """
        Precompute bias matrix for efficiency.

        Args:
            max_len: Maximum sequence length
        """
        # Create relative position matrix
        positions = torch.arange(max_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # ALiBi bias is negative of relative distance
        bias = -torch.abs(relative_positions).float()

        # Apply slopes [num_heads, 1, 1] * [1, max_len, max_len]
        bias = self.slopes.view(-1, 1, 1) * bias.unsqueeze(0)

        self.register_buffer("bias_cached", bias, persistent=False)

    def forward(
        self,
        seq_len: int,
        key_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias for attention computation.

        Args:
            seq_len: Query sequence length
            key_len: Key sequence length (defaults to seq_len)
            device: Device to create bias on

        Returns:
            Bias tensor [num_heads, seq_len, key_len]
        """
        if key_len is None:
            key_len = seq_len

        if device is None:
            device = self.slopes.device

        # Use cached bias if available
        if seq_len <= self.max_len and key_len <= self.max_len:
            bias = self.bias_cached[:, :seq_len, :key_len]
        else:
            # Recompute for longer sequences
            positions_q = torch.arange(seq_len, device=device)
            positions_k = torch.arange(key_len, device=device)
            relative_positions = positions_q.unsqueeze(1) - positions_k.unsqueeze(0)
            bias = -torch.abs(relative_positions).float()
            bias = self.slopes.view(-1, 1, 1) * bias.unsqueeze(0)

        return bias

    def forward_with_offset(
        self,
        seq_len: int,
        offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Get ALiBi bias with position offset (for KV caching).

        Args:
            seq_len: Sequence length
            offset: Position offset
            device: Device to create bias on

        Returns:
            Bias tensor [num_heads, seq_len, seq_len + offset]
        """
        if device is None:
            device = self.slopes.device

        positions_q = torch.arange(offset, offset + seq_len, device=device)
        positions_k = torch.arange(offset + seq_len, device=device)
        relative_positions = positions_q.unsqueeze(1) - positions_k.unsqueeze(0)

        bias = -torch.abs(relative_positions).float()
        bias = self.slopes.view(-1, 1, 1) * bias.unsqueeze(0)

        return bias

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, max_len={self.max_len}, slopes_init={self.slopes_init}"
