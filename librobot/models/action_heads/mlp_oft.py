"""MLP-based action head for open-loop trajectory forecasting."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from librobot.models.action_heads.base import BaseActionHead
from librobot.utils.registry import register_action_head


@register_action_head("mlp_oft", aliases=["mlp"])
class MLPOFTHead(BaseActionHead):
    """MLP-based action head for Open-loop Forecast Trajectory (OFT).

    Predicts action chunks in parallel using MLPs.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        action_horizon: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize MLP OFT head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            action_horizon: Number of action steps to predict
            dropout: Dropout probability
        """
        super().__init__(input_dim, action_dim, hidden_dim)
        self.num_layers = num_layers
        self.action_horizon = action_horizon
        self.dropout = dropout

        # Build MLP layers
        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Output layer predicts all actions at once
        layers.append(nn.Linear(in_dim, action_horizon * action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Input features [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            actions: Ground truth actions [batch_size, action_horizon, action_dim]

        Returns:
            Dictionary with 'actions' and optionally 'loss'
        """
        # If features have sequence dimension, use last token
        if features.ndim == 3:
            features = features[:, -1]  # [batch_size, input_dim]

        # Predict actions
        action_flat = self.mlp(features)  # [batch_size, action_horizon * action_dim]
        predicted_actions = action_flat.reshape(
            features.shape[0], self.action_horizon, self.action_dim
        )

        output = {"actions": predicted_actions}

        # Compute loss if targets provided
        if actions is not None:
            output["loss"] = self.compute_loss(predicted_actions, actions)

        return output

    def predict(
        self,
        features: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Predict actions at inference time.

        Args:
            features: Input features [batch_size, input_dim]
            num_samples: Number of samples (ignored for deterministic MLP)

        Returns:
            Predicted actions [batch_size, 1, action_horizon, action_dim]
        """
        with torch.no_grad():
            output = self.forward(features)
            actions = output["actions"]
            # Add sample dimension for consistency
            return actions.unsqueeze(1)
