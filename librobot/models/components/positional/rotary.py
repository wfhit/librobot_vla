"""Rotary Position Embedding (RoPE)."""

from typing import Optional

import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    Applies rotary transformations to queries and keys based on their positions,
    encoding relative positional information through rotation matrices.

    Args:
        dim: Dimension per attention head
        max_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
        scaling_factor: Scaling factor for extended context
    """

    def __init__(
        self,
        dim: int,
        max_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for max_len
        self._precompute_freqs(max_len)

    def _precompute_freqs(self, max_len: int):
        """Precompute cos and sin values for positions."""
        t = torch.arange(max_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            position_ids: Optional position IDs [batch_size, seq_len]

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.size(2)

        if seq_len > self.max_len:
            self._precompute_freqs(seq_len)

        if position_ids is None:
            # Use sequential positions
            cos = self.cos_cached[:seq_len, ...]
            sin = self.sin_cached[:seq_len, ...]
        else:
            # Use provided position IDs
            cos = self.cos_cached[position_ids, ...]
            sin = self.sin_cached[position_ids, ...]

        # Reshape for broadcasting [1, 1, seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    def forward_with_offset(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embedding with position offset (for KV caching).

        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            offset: Position offset

        Returns:
            Tuple of (rotated_q, rotated_k)
        """
        seq_len = q.size(2)
        position_ids = torch.arange(offset, offset + seq_len, device=q.device).unsqueeze(0)
        return self.forward(q, k, position_ids)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_len={self.max_len}, base={self.base}"
