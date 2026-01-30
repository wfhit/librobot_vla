"""GroupNorm for convolutional layers."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm(nn.Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.
    More stable than BatchNorm for small batch sizes and commonly used
    in vision models and diffusion models.

    Args:
        num_groups: Number of groups to divide channels into
        num_channels: Number of channels
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f'num_channels ({num_channels}) must be divisible by num_groups ({num_groups})')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization.

        Args:
            x: Input tensor [batch_size, num_channels, *spatial_dims]

        Returns:
            Normalized tensor [batch_size, num_channels, *spatial_dims]
        """
        return F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
        )

    def extra_repr(self) -> str:
        return f'num_groups={self.num_groups}, num_channels={self.num_channels}, eps={self.eps}, affine={self.affine}'
