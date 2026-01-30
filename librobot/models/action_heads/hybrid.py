"""Hybrid action head combining multiple approaches."""
import torch
import torch.nn as nn

from .base import AbstractActionHead


class HybridActionHead(AbstractActionHead):
    """Hybrid head with multiple prediction modes."""
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__(input_dim, action_dim)
        self.continuous_head = nn.Linear(input_dim, action_dim)
        self.discrete_head = nn.Linear(input_dim, action_dim * 256)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        continuous = self.continuous_head(embeddings)
        return {'actions': continuous}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions['actions'], targets)

    def sample(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.continuous_head(embeddings)
