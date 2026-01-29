"""Base action head interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseActionHead(nn.Module, ABC):
    """Base class for action prediction heads.

    All action head implementations should inherit from this class.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        **kwargs,
    ):
        """Initialize action head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            **kwargs: Additional head-specific arguments
        """
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through action head.

        Args:
            features: Input features [batch_size, seq_len, input_dim]
            actions: Ground truth actions for training [batch_size, action_horizon, action_dim]
            **kwargs: Additional head-specific arguments

        Returns:
            Dictionary containing:
                - actions: Predicted actions [batch_size, action_horizon, action_dim]
                - loss: Training loss (if actions provided)
                - (optional) other head-specific outputs
        """
        pass

    @abstractmethod
    def predict(
        self,
        features: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Predict actions at inference time.

        Args:
            features: Input features [batch_size, seq_len, input_dim]
            num_samples: Number of action samples to generate
            **kwargs: Additional head-specific arguments

        Returns:
            Predicted actions [batch_size, num_samples, action_horizon, action_dim]
        """
        pass

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss for training.

        Args:
            predictions: Predicted actions
            targets: Ground truth actions
            **kwargs: Additional loss arguments

        Returns:
            Scalar loss tensor
        """
        # Default MSE loss, can be overridden
        return torch.nn.functional.mse_loss(predictions, targets)
