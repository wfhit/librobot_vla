"""RMSNorm (Root Mean Square Normalization)."""

from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and faster alternative to LayerNorm that only normalizes by RMS
    without mean centering. Used in models like LLaMA and T5.
    
    RMSNorm(x) = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    
    Args:
        dim: Input dimension
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn scale parameter
    """
    
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor [..., dim]
            
        Returns:
            Normalized tensor [..., dim]
        """
        output = self._norm(x.float()).type_as(x)
        
        if self.elementwise_affine:
            output = output * self.weight
        
        return output
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
