"""Cross-attention fusion between modalities."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...components.attention import StandardAttention
from ...components.normalization import LayerNorm


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between modalities.
    
    One modality attends to another, allowing information flow between them.
    Commonly used for vision-language fusion.
    
    Args:
        query_dim: Query embedding dimension
        context_dim: Context embedding dimension
        output_dim: Output dimension
        num_heads: Number of attention heads
        num_layers: Number of cross-attention layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        output_dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Project inputs to common dimension
        self.query_proj = nn.Linear(query_dim, output_dim)
        self.context_proj = nn.Linear(context_dim, output_dim)
        
        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=output_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(output_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Fuse query and context via cross-attention.
        
        Args:
            query: Query embeddings [batch_size, query_len, query_dim]
            context: Context embeddings [batch_size, context_len, context_dim]
            query_mask: Optional query mask [batch_size, query_len]
            context_mask: Optional context mask [batch_size, context_len]
            **kwargs: Additional arguments
            
        Returns:
            Fused embeddings [batch_size, query_len, output_dim]
        """
        # Project to common dimension
        q = self.query_proj(query)
        c = self.context_proj(context)
        
        # Apply cross-attention layers
        for layer in self.cross_attn_layers:
            q = layer(q, c, query_mask, context_mask)
        
        # Final normalization
        q = self.norm(q)
        
        return q
    
    def get_config(self) -> Dict[str, Any]:
        """Get fusion configuration."""
        return {
            'type': 'CrossAttentionFusion',
            'query_dim': self.query_dim,
            'context_dim': self.context_dim,
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
        }


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # Cross-attention
        self.cross_attn = StandardAttention(
            dim=d_model,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        # Create combined key-value from context
        kv = context
        
        # Cross-attention with residual
        q_norm = self.norm1(query)
        
        # Use query for Q and context for K,V
        B, N_q, C = query.shape
        N_c = context.shape[1]
        
        # Compute Q from query
        q_proj = self.cross_attn.qkv.weight[:C].t()
        q_bias = self.cross_attn.qkv.bias[:C] if self.cross_attn.qkv.bias is not None else None
        q_out = nn.functional.linear(q_norm, q_proj, q_bias)
        q_out = q_out.reshape(B, N_q, self.cross_attn.num_heads, self.cross_attn.head_dim).permute(0, 2, 1, 3)
        
        # Compute K,V from context
        kv_proj = self.cross_attn.qkv.weight[C:].t()
        kv_bias = self.cross_attn.qkv.bias[C:] if self.cross_attn.qkv.bias is not None else None
        kv_out = nn.functional.linear(context, kv_proj, kv_bias)
        kv_out = kv_out.reshape(B, N_c, 2, self.cross_attn.num_heads, self.cross_attn.head_dim).permute(2, 0, 3, 1, 4)
        k_out, v_out = kv_out[0], kv_out[1]
        
        # Attention
        attn = (q_out @ k_out.transpose(-2, -1)) * self.cross_attn.scale
        
        if context_mask is not None:
            context_mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(context_mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.cross_attn.attn_drop(attn)
        
        out = (attn @ v_out).transpose(1, 2).reshape(B, N_q, C)
        out = self.cross_attn.proj(out)
        out = self.cross_attn.proj_drop(out)
        
        query = query + out
        
        # FFN with residual
        query = query + self.ffn(self.norm2(query))
        
        return query
