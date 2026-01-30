"""FiLM (Feature-wise Linear Modulation) fusion."""

from typing import Any

import torch
import torch.nn as nn


class FiLMFusion(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) fusion.

    Modulates one modality with affine transformations learned from another.
    Effective for conditioning visual features on language or state.

    FiLM(x|context) = γ(context) ⊙ x + β(context)

    Args:
        feature_dim: Feature dimension to be modulated
        context_dim: Context dimension for computing modulation
        use_residual: Whether to add residual connection
    """

    def __init__(
        self,
        feature_dim: int,
        context_dim: int,
        use_residual: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.use_residual = use_residual

        # Networks to predict scale (gamma) and shift (beta)
        self.gamma_net = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
            nn.Sigmoid(),
        )

        self.beta_net = nn.Sequential(
            nn.Linear(context_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, context: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modulate features with context via FiLM.

        Args:
            features: Feature embeddings [batch_size, ..., feature_dim]
            context: Context embeddings [batch_size, context_dim] or [batch_size, seq_len, context_dim]
            **kwargs: Additional arguments

        Returns:
            Modulated features [batch_size, ..., feature_dim]
        """
        # Handle context shape
        if context.dim() == 3:
            # Pool context if it's a sequence
            context = context.mean(dim=1)

        # Compute modulation parameters
        gamma = self.gamma_net(context)
        beta = self.beta_net(context)

        # Reshape for broadcasting if features have spatial dimensions
        if features.dim() > 2:
            # Add dimensions for broadcasting: [batch, 1, ..., 1, feature_dim]
            shape = [features.size(0)] + [1] * (features.dim() - 2) + [self.feature_dim]
            gamma = gamma.view(*shape)
            beta = beta.view(*shape)

        # Apply FiLM modulation
        output = gamma * features + beta

        # Add residual if enabled
        if self.use_residual:
            output = output + features

        return output

    def get_config(self) -> dict[str, Any]:
        """Get fusion configuration."""
        return {
            "type": "FiLMFusion",
            "feature_dim": self.feature_dim,
            "context_dim": self.context_dim,
            "use_residual": self.use_residual,
        }
