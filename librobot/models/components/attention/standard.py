"""Standard multi-head attention implementation."""

from typing import Optional

import torch
import torch.nn as nn


class StandardAttention(nn.Module):
    """
    Standard multi-head attention mechanism.

    Implements the scaled dot-product attention from "Attention Is All You Need".
    Supports causal masking, attention masks, and key-value caching.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        causal: Whether to use causal masking
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
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
        self.scale = self.head_dim**-0.5
        self.causal = causal

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            key_value: Optional cached (key, value) tuple for cross-attention
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch_size, seq_len, dim] or tuple of (output, attention_weights)
        """
        B, N, C = x.shape

        if key_value is None:
            # Self-attention
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            # Cross-attention with cached key-value
            q = nn.functional.linear(
                x,
                self.qkv.weight[: self.dim],
                self.qkv.bias[: self.dim] if self.qkv.bias is not None else None,
            )
            q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k, v = key_value

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Broadcast mask to all heads
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            attn = attn.masked_fill(attention_mask == 0, float("-inf"))

        # Apply causal mask
        if self.causal:
            causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, causal={self.causal}"
