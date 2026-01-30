"""Simple concatenation fusion."""

from typing import Any, Optional

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion.

    Concatenates multiple embeddings and optionally projects to target dimension.
    Simplest baseline for multimodal fusion.

    Args:
        input_dims: List of input dimensions for each modality
        output_dim: Output dimension (if None, no projection)
        use_layernorm: Whether to apply layer normalization
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dim: Optional[int] = None,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim if output_dim is not None else sum(input_dims)
        self.use_layernorm = use_layernorm

        concat_dim = sum(input_dims)

        # Projection layer
        if output_dim is not None and concat_dim != output_dim:
            self.proj = nn.Linear(concat_dim, output_dim)
        else:
            self.proj = nn.Identity()

        # Layer normalization
        if use_layernorm:
            self.norm = nn.LayerNorm(self.output_dim)
        else:
            self.norm = nn.Identity()

    def forward(
        self,
        *embeddings: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Fuse embeddings via concatenation.

        Args:
            *embeddings: Variable number of embedding tensors [batch_size, dim_i]
            **kwargs: Additional arguments (ignored)

        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        if len(embeddings) != len(self.input_dims):
            raise ValueError(
                f"Expected {len(self.input_dims)} embeddings, got {len(embeddings)}"
            )

        # Concatenate
        x = torch.cat(embeddings, dim=-1)

        # Project
        x = self.proj(x)

        # Normalize
        x = self.norm(x)

        return x

    def get_config(self) -> dict[str, Any]:
        """Get fusion configuration."""
        return {
            'type': 'ConcatFusion',
            'input_dims': self.input_dims,
            'output_dim': self.output_dim,
            'use_layernorm': self.use_layernorm,
        }
