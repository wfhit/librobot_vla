"""Sliding window attention for long sequences."""

from typing import Optional

import torch
import torch.nn as nn


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention mechanism.

    Each token attends to a fixed window of neighboring tokens,
    reducing computational complexity from O(N^2) to O(N*W) where W is window size.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        window_size: Size of attention window
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 512,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.causal = causal

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _compute_windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention with sliding window.

        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            attention_mask: Optional mask [B, N]

        Returns:
            Output tensor [B, H, N, D]
        """
        B, H, N, D = q.shape
        W = self.window_size

        output = torch.zeros_like(q)

        for i in range(N):
            # Determine window bounds
            if self.causal:
                start = max(0, i - W + 1)
                end = i + 1
            else:
                start = max(0, i - W // 2)
                end = min(N, i + W // 2 + 1)

            # Extract window
            q_i = q[:, :, i:i+1, :]
            k_window = k[:, :, start:end, :]
            v_window = v[:, :, start:end, :]

            # Compute attention
            attn = (q_i @ k_window.transpose(-2, -1)) * self.scale

            # Apply mask
            if attention_mask is not None:
                mask_window = attention_mask[:, start:end].unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(mask_window == 0, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Compute output
            output[:, :, i:i+1, :] = attn @ v_window

        return output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights (not supported)

        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        B, N, C = x.shape

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute windowed attention
        if N <= self.window_size:
            # Use standard attention for short sequences
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))

            if self.causal:
                causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
                attn = attn.masked_fill(~causal_mask, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            # Use sliding window
            x = self._compute_windowed_attention(q, k, v, attention_mask)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}, window_size={self.window_size}, causal={self.causal}'
