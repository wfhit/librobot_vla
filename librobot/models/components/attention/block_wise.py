"""Block-wise attention as used in π0 (pi-zero) model."""

from typing import Optional

import torch
import torch.nn as nn


class BlockWiseAttention(nn.Module):
    """
    Block-wise attention mechanism from π0.

    Processes attention in fixed-size blocks to reduce memory complexity
    while maintaining global receptive field through hierarchical processing.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        block_size: Size of each attention block
        qkv_bias: Whether to use bias in QKV projection
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate for output projection
        use_global: Whether to use global attention for first block
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 64,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_global: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.block_size = block_size
        self.use_global = use_global

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Global token for aggregating information across blocks
        if use_global:
            self.global_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with block-wise attention.

        Args:
            x: Input tensor [batch_size, seq_len, dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            Output tensor [batch_size, seq_len, dim] or tuple of (output, attention_weights)
        """
        B, N, C = x.shape

        # Add global token if enabled
        if self.use_global:
            global_tokens = self.global_token.expand(B, -1, -1)
            x = torch.cat([global_tokens, x], dim=1)
            N = N + 1
            if attention_mask is not None:
                global_mask = torch.ones(B, 1, device=x.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([global_mask, attention_mask], dim=1)

        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention in blocks
        num_blocks = (N + self.block_size - 1) // self.block_size
        outputs = []
        attention_weights = [] if return_attention else None

        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, N)

            # Extract block
            q_block = q[:, :, start_idx:end_idx, :]

            # For block-wise attention, attend to current and previous blocks
            k_end = end_idx
            k_block = k[:, :, :k_end, :]
            v_block = v[:, :, :k_end, :]

            # Compute attention
            attn = (q_block @ k_block.transpose(-2, -1)) * self.scale

            # Apply mask if provided
            if attention_mask is not None:
                mask_block = attention_mask[:, start_idx:end_idx]
                mask_k = attention_mask[:, :k_end]
                attn_mask = mask_block.unsqueeze(1).unsqueeze(2) * mask_k.unsqueeze(1).unsqueeze(3)
                attn = attn.masked_fill(attn_mask == 0, float("-inf"))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if return_attention:
                attention_weights.append(attn)

            # Compute output
            out = (attn @ v_block).transpose(1, 2).reshape(B, end_idx - start_idx, C)
            outputs.append(out)

        # Concatenate blocks
        x = torch.cat(outputs, dim=1)

        # Remove global token if added
        if self.use_global:
            x = x[:, 1:, :]

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attention_weights
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, block_size={self.block_size}"
