"""LayerNorm implementation with optional bias."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes inputs across the feature dimension. More stable than BatchNorm
    for transformer models and works well with variable batch sizes.

    Args:
        normalized_shape: Input shape (typically embedding dimension)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters
        bias: Whether to use bias (can be disabled for efficiency)
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            x: Input tensor [..., normalized_shape]

        Returns:
            Normalized tensor [..., normalized_shape]
        """
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
