"""Flash Attention wrapper for efficient attention computation."""

from typing import Optional

import torch
import torch.nn as nn


class FlashAttention(nn.Module):
    """
    Flash Attention wrapper for memory-efficient attention.

    Falls back to standard attention if flash-attn is not available.
    Flash Attention provides O(N) memory complexity instead of O(N^2).

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        causal: Whether to use causal masking
        use_flash: Whether to attempt using flash attention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        causal: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.attn_drop_p = attn_drop

        # Try to import flash attention
        self.has_flash = False
        if use_flash:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.has_flash = True
            except ImportError:
                pass

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not self.has_flash:
            self.attn_drop = nn.Dropout(attn_drop)

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
            attention_mask: Optional attention mask (only used in fallback mode)
            return_attention: Whether to return attention weights (only in fallback mode)

        Returns:
            Output tensor [batch_size, seq_len, dim] or tuple of (output, attention_weights)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.has_flash and not return_attention and attention_mask is None:
            # Use flash attention
            q, k, v = qkv.unbind(2)

            # flash_attn expects [batch, seq_len, num_heads, head_dim]
            x = self.flash_attn_func(
                q, k, v,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                causal=self.causal,
            )
            x = x.reshape(B, N, C)
        else:
            # Fallback to standard attention
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                attn = attn.masked_fill(attention_mask == 0, float('-inf'))

            if self.causal:
                causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))
                attn = attn.masked_fill(~causal_mask, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            if return_attention:
                x_out = self.proj(x)
                x_out = self.proj_drop(x_out)
                return x_out, attn

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}, causal={self.causal}, flash={self.has_flash}'
